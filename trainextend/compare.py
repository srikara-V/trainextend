"""Aggregate horizon metrics across experiment jobs."""

from __future__ import annotations

import json
from pathlib import Path

from trainextend.paths import metrics_path
from trainextend.schemas import CompareResponse, CompareRow, ExperimentRecord, JobStatus
from trainextend.state import JobStore


def _load_job_metrics(store: JobStore, job_id: str) -> dict | None:
    path = metrics_path(store.data_dir, job_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def compare_experiment(store: JobStore, experiment: ExperimentRecord) -> CompareResponse:
    cfg = experiment.config
    rows: list[CompareRow] = []

    for job_id in experiment.job_ids:
        record = store.get_job(job_id)
        if record is None or record.status != JobStatus.COMPLETED:
            continue
        metrics = _load_job_metrics(store, job_id)
        if metrics is None:
            continue
        rows.append(
            CompareRow(
                model=metrics.get("model", cfg.model.name),
                dataset=metrics.get("dataset", cfg.dataset.name),
                horizon=int(metrics.get("horizon", cfg.model.prediction_length)),
                cutoffs=1,
                mse=metrics.get("mse"),
                mae=metrics.get("mae"),
                smape=metrics.get("smape"),
            )
        )

    if not rows:
        return CompareResponse(
            experiment_id=experiment.experiment_id,
            experiment_name=experiment.name,
            rows=[],
        )

    # Aggregate across cutoffs (mean per metric)
    n = len(rows)
    agg = CompareRow(
        model=rows[0].model,
        dataset=rows[0].dataset,
        horizon=rows[0].horizon,
        cutoffs=n,
        mse=sum(r.mse or 0 for r in rows) / n if rows[0].mse is not None else None,
        mae=sum(r.mae or 0 for r in rows) / n if rows[0].mae is not None else None,
        smape=sum(r.smape or 0 for r in rows) / n if rows[0].smape is not None else None,
    )
    return CompareResponse(
        experiment_id=experiment.experiment_id,
        experiment_name=experiment.name,
        rows=[agg],
    )
