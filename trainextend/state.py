from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import yaml

from trainextend.experiment import expand_experiment
from trainextend.paths import experiment_config_path, experiment_root, job_config_path, job_root
from trainextend.schemas import (
    ExperimentConfig,
    ExperimentRecord,
    ExperimentStatus,
    JobEvent,
    JobRecord,
    JobStatus,
    TrainingConfig,
    utcnow,
)


class JobStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.db_path = data_dir / "metadata.db"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _migrate_jobs(self, conn: sqlite3.Connection) -> None:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
        for col, ddl in (
            ("experiment_id", "ALTER TABLE jobs ADD COLUMN experiment_id TEXT"),
            ("cutoff", "ALTER TABLE jobs ADD COLUMN cutoff TEXT"),
            ("prediction_length", "ALTER TABLE jobs ADD COLUMN prediction_length INTEGER"),
            ("context_length", "ALTER TABLE jobs ADD COLUMN context_length INTEGER"),
        ):
            if col not in cols:
                conn.execute(ddl)

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    config_path TEXT NOT NULL,
                    latest_checkpoint TEXT,
                    last_good_checkpoint TEXT,
                    global_step INTEGER DEFAULT 0,
                    epoch INTEGER DEFAULT 0,
                    worker_id TEXT,
                    resume_count INTEGER DEFAULT 0,
                    error TEXT,
                    pending_failure INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    t TEXT NOT NULL,
                    event TEXT NOT NULL,
                    worker_id TEXT,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    enqueued_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    job_ids_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            self._migrate_jobs(conn)

    def create_job(
        self,
        config: TrainingConfig,
        *,
        enqueue: bool = True,
        experiment_id: str | None = None,
    ) -> JobRecord:
        job_id = f"job_{uuid.uuid4().hex[:10]}"
        now = utcnow()
        root = job_root(self.data_dir, job_id)
        root.mkdir(parents=True, exist_ok=True)
        cfg_path = job_config_path(self.data_dir, job_id)
        cfg_path.write_text(yaml.safe_dump(config.model_dump()))

        if experiment_id:
            config = config.model_copy(update={"experiment_id": experiment_id})

        record = JobRecord(
            job_id=job_id,
            status=JobStatus.QUEUED,
            config=config,
            config_path=str(cfg_path),
            experiment_id=config.experiment_id,
            cutoff=config.cutoff,
            prediction_length=config.prediction_length if config.task == "forecast" else None,
            context_length=config.context_length if config.task == "forecast" else None,
            created_at=now,
            updated_at=now,
        )
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, status, config_json, config_path,
                    experiment_id, cutoff, prediction_length, context_length,
                    global_step, epoch, resume_count, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    record.status.value,
                    config.model_dump_json(),
                    str(cfg_path),
                    record.experiment_id,
                    record.cutoff,
                    record.prediction_length,
                    record.context_length,
                    0,
                    0,
                    0,
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
            if enqueue:
                conn.execute(
                    "INSERT INTO queue (job_id, enqueued_at) VALUES (?, ?)",
                    (job_id, now.isoformat()),
                )
        self.append_event(job_id, "submitted", payload={"config": config.model_dump()})
        return record

    def create_experiment(self, config: ExperimentConfig, *, enqueue: bool = True) -> ExperimentRecord:
        experiment_id = f"exp_{uuid.uuid4().hex[:10]}"
        now = utcnow()
        exp_root = experiment_root(self.data_dir, experiment_id)
        exp_root.mkdir(parents=True, exist_ok=True)
        experiment_config_path(self.data_dir, experiment_id).write_text(yaml.safe_dump(config.model_dump()))

        job_configs = expand_experiment(config)
        job_ids: list[str] = []
        for job_cfg in job_configs:
            record = self.create_job(job_cfg, enqueue=enqueue, experiment_id=experiment_id)
            job_ids.append(record.job_id)

        record = ExperimentRecord(
            experiment_id=experiment_id,
            name=config.experiment_name,
            config=config,
            job_ids=job_ids,
            status=ExperimentStatus.QUEUED,
            created_at=now,
            updated_at=now,
        )
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO experiments (
                    experiment_id, name, config_json, job_ids_json, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    config.experiment_name,
                    config.model_dump_json(),
                    json.dumps(job_ids),
                    record.status.value,
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
        self.append_event(
            job_ids[0],
            "experiment_created",
            payload={"experiment_id": experiment_id, "job_ids": job_ids, "cutoffs": config.backtest.cutoffs},
        )
        return record

    def _row_to_record(self, row: sqlite3.Row) -> JobRecord:
        keys = row.keys()
        return JobRecord(
            job_id=row["job_id"],
            status=JobStatus(row["status"]),
            config=TrainingConfig.model_validate_json(row["config_json"]),
            config_path=row["config_path"],
            experiment_id=row["experiment_id"] if "experiment_id" in keys else None,
            cutoff=row["cutoff"] if "cutoff" in keys else None,
            prediction_length=row["prediction_length"] if "prediction_length" in keys else None,
            context_length=row["context_length"] if "context_length" in keys else None,
            latest_checkpoint=row["latest_checkpoint"],
            last_good_checkpoint=row["last_good_checkpoint"],
            global_step=row["global_step"],
            epoch=row["epoch"],
            worker_id=row["worker_id"],
            resume_count=row["resume_count"],
            error=row["error"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def _row_to_experiment(self, row: sqlite3.Row) -> ExperimentRecord:
        return ExperimentRecord(
            experiment_id=row["experiment_id"],
            name=row["name"],
            config=ExperimentConfig.model_validate_json(row["config_json"]),
            job_ids=json.loads(row["job_ids_json"]),
            status=ExperimentStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._row_to_record(row) if row else None

    def get_experiment(self, experiment_id: str) -> ExperimentRecord | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)).fetchone()
        if row is None:
            return None
        exp = self._row_to_experiment(row)
        return self.refresh_experiment_status(exp)

    def refresh_experiment_status(self, experiment: ExperimentRecord) -> ExperimentRecord:
        statuses = []
        for job_id in experiment.job_ids:
            job = self.get_job(job_id)
            if job:
                statuses.append(job.status)

        if not statuses:
            new_status = experiment.status
        elif all(s == JobStatus.COMPLETED for s in statuses):
            new_status = ExperimentStatus.COMPLETED
        elif any(s == JobStatus.RUNNING for s in statuses):
            new_status = ExperimentStatus.RUNNING
        elif all(s in (JobStatus.QUEUED, JobStatus.RESUMING) for s in statuses):
            new_status = ExperimentStatus.QUEUED
        elif any(s == JobStatus.COMPLETED for s in statuses):
            new_status = ExperimentStatus.PARTIAL
        elif any(s == JobStatus.FAILED for s in statuses):
            new_status = ExperimentStatus.FAILED
        else:
            new_status = ExperimentStatus.QUEUED

        if new_status != experiment.status:
            with self._conn() as conn:
                conn.execute(
                    "UPDATE experiments SET status = ?, updated_at = ? WHERE experiment_id = ?",
                    (new_status.value, utcnow().isoformat(), experiment.experiment_id),
                )
            experiment = experiment.model_copy(update={"status": new_status, "updated_at": utcnow()})
        return experiment

    def update_job(self, job_id: str, **fields) -> JobRecord:
        allowed = {
            "status",
            "latest_checkpoint",
            "last_good_checkpoint",
            "global_step",
            "epoch",
            "worker_id",
            "resume_count",
            "error",
            "pending_failure",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if "status" in updates and isinstance(updates["status"], JobStatus):
            updates["status"] = updates["status"].value
        updates["updated_at"] = utcnow().isoformat()
        cols = ", ".join(f"{k} = ?" for k in updates)
        vals = list(updates.values()) + [job_id]
        with self._conn() as conn:
            conn.execute(f"UPDATE jobs SET {cols} WHERE job_id = ?", vals)
        record = self.get_job(job_id)
        if record is None:
            raise KeyError(job_id)
        if record.experiment_id:
            exp = self.get_experiment(record.experiment_id)
            if exp:
                self.refresh_experiment_status(exp)
        return record

    def append_event(
        self,
        job_id: str,
        event: str,
        *,
        worker_id: str | None = None,
        payload: dict | None = None,
    ) -> JobEvent:
        ev = JobEvent(
            t=utcnow(),
            job_id=job_id,
            event=event,
            worker_id=worker_id,
            payload=payload or {},
        )
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO events (job_id, t, event, worker_id, payload_json) VALUES (?, ?, ?, ?, ?)",
                (job_id, ev.t.isoformat(), event, worker_id, json.dumps(ev.payload)),
            )
        log_path = job_root(self.data_dir, job_id) / "logs" / "events.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as fh:
            fh.write(ev.model_dump_json() + "\n")
        return ev

    def list_events(self, job_id: str, limit: int = 100) -> list[JobEvent]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE job_id = ? ORDER BY id DESC LIMIT ?",
                (job_id, limit),
            ).fetchall()
        events = [
            JobEvent(
                t=datetime.fromisoformat(r["t"]),
                job_id=r["job_id"],
                event=r["event"],
                worker_id=r["worker_id"],
                payload=json.loads(r["payload_json"]),
            )
            for r in reversed(rows)
        ]
        return events

    def dequeue(self) -> str | None:
        with self._conn() as conn:
            row = conn.execute("SELECT id, job_id FROM queue ORDER BY id LIMIT 1").fetchone()
            if not row:
                return None
            conn.execute("DELETE FROM queue WHERE id = ?", (row["id"],))
            return row["job_id"]

    def enqueue(self, job_id: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO queue (job_id, enqueued_at) VALUES (?, ?)",
                (job_id, utcnow().isoformat()),
            )

    def set_pending_failure(self, job_id: str) -> None:
        self.update_job(job_id, pending_failure=1)
        self.append_event(job_id, "failure_scheduled", payload={"reason": "simulate-failure"})

    def pending_failure(self, job_id: str) -> bool:
        with self._conn() as conn:
            row = conn.execute("SELECT pending_failure FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return bool(row and row["pending_failure"])

    def clear_pending_failure(self, job_id: str) -> None:
        self.update_job(job_id, pending_failure=0)
