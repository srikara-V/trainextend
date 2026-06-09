from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from pydantic_settings import BaseSettings, SettingsConfigDict

from trainextend import __version__
from trainextend.checkpointing import list_checkpoints
from trainextend.compare import compare_experiment
from trainextend.paths import checkpoints_dir
from trainextend.schemas import (
    CompareResponse,
    ExperimentConfig,
    ExperimentRecord,
    ExperimentSubmitResponse,
    JobCreate,
    JobEvent,
    JobRecord,
    JobStatus,
    JobSubmitResponse,
    TrainingConfig,
)
from trainextend.state import JobStore
from trainextend.worker_local import LocalWorkerPool

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRAINEXTEND_")

    data_dir: Path = Path("./trainextend_data")
    backend: str = "local"  # local | modal
    modal_app: str = "trainextend"
    max_workers: int = 0  # 0 = auto (min(4, cpu_count))


settings = Settings()
store = JobStore(settings.data_dir)
pool = LocalWorkerPool(
    store,
    max_workers=settings.max_workers or None,
)


def _use_modal() -> bool:
    return settings.backend == "modal"


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    if _use_modal():
        os.environ.setdefault("TRAINEXTEND_MODAL_APP", settings.modal_app)
        try:
            from trainextend import modal_client

            modal_client.check_modal_deployed()
            logger.info("Modal backend ready (app=%s)", settings.modal_app)
        except Exception as exc:
            logger.warning(
                "Modal backend selected but app may not be deployed: %s. "
                "Run: modal deploy trainextend/modal_app.py",
                exc,
            )
    else:
        pool.start()
        logger.info("Local worker pool started (max_workers=%s)", pool.max_workers)
    yield
    if not _use_modal():
        pool.stop()


app = FastAPI(title="TrainExtend", version=__version__, lifespan=lifespan)


def _load_training_config(path: Path) -> TrainingConfig:
    data = yaml.safe_load(path.read_text())
    return TrainingConfig.model_validate(data)


def _load_experiment_config(path: Path) -> ExperimentConfig:
    data = yaml.safe_load(path.read_text())
    return ExperimentConfig.model_validate(data)


def _not_found(resource: str, ident: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"{resource} not found: {ident}")


@app.post("/experiments", response_model=ExperimentSubmitResponse)
def create_experiment(body: ExperimentConfig) -> ExperimentSubmitResponse:
    if _use_modal():
        from trainextend import modal_client

        record = modal_client.create_experiment(body)
    else:
        record = store.create_experiment(body)

    return ExperimentSubmitResponse(
        experiment_id=record.experiment_id,
        experiment_name=record.name,
        job_ids=record.job_ids,
        status=record.status,
    )


@app.get("/experiments/{experiment_id}", response_model=ExperimentRecord)
def get_experiment(experiment_id: str) -> ExperimentRecord:
    if _use_modal():
        from trainextend import modal_client

        record = modal_client.get_experiment(experiment_id)
    else:
        record = store.get_experiment(experiment_id)

    if record is None:
        raise _not_found("Experiment", experiment_id)
    return record


@app.get("/experiments/{experiment_id}/compare", response_model=CompareResponse)
def compare_experiment_endpoint(experiment_id: str) -> CompareResponse:
    if _use_modal():
        from trainextend import modal_client

        return modal_client.compare_experiment(experiment_id)

    record = store.get_experiment(experiment_id)
    if record is None:
        raise _not_found("Experiment", experiment_id)
    return compare_experiment(store, record)


@app.post("/jobs", response_model=JobSubmitResponse)
def create_job(body: JobCreate) -> JobSubmitResponse:
    config = body.config
    if body.config_path:
        config = _load_training_config(Path(body.config_path))

    if _use_modal():
        from trainextend import modal_client

        record = modal_client.create_job(config)
    else:
        record = store.create_job(config)

    return JobSubmitResponse(job_id=record.job_id, status=record.status, config_path=record.config_path)


@app.get("/jobs/{job_id}", response_model=JobRecord)
def get_job(job_id: str) -> JobRecord:
    if _use_modal():
        from trainextend import modal_client

        record = modal_client.get_job(job_id)
    else:
        record = store.get_job(job_id)

    if record is None:
        raise _not_found("Job", job_id)
    return record


@app.post("/jobs/{job_id}/resume", response_model=JobRecord)
def resume_job(job_id: str) -> JobRecord:
    if _use_modal():
        from trainextend import modal_client

        record = modal_client.get_job(job_id)
        if record is None:
            raise _not_found("Job", job_id)
        if record.status == JobStatus.COMPLETED:
            raise HTTPException(status_code=400, detail="Job already completed")
        if record.status == JobStatus.CANCELLED:
            raise HTTPException(status_code=400, detail="Job cancelled")
        try:
            return modal_client.resume_job(job_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    record = store.get_job(job_id)
    if record is None:
        raise _not_found("Job", job_id)
    if record.status == JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job already completed")
    if record.status == JobStatus.CANCELLED:
        raise HTTPException(status_code=400, detail="Job cancelled")
    store.update_job(job_id, status=JobStatus.QUEUED, error=None)
    store.append_event(job_id, "resume_requested")
    pool.mark_resume(job_id)
    record = store.get_job(job_id)
    assert record is not None
    return record


@app.post("/jobs/{job_id}/cancel", response_model=JobRecord)
def cancel_job(job_id: str) -> JobRecord:
    if _use_modal():
        from trainextend import modal_client

        record = modal_client.get_job(job_id)
        if record is None:
            raise _not_found("Job", job_id)
        return modal_client.cancel_job(job_id)

    record = store.get_job(job_id)
    if record is None:
        raise _not_found("Job", job_id)
    store.update_job(job_id, status=JobStatus.CANCELLED)
    store.append_event(job_id, "cancelled")
    record = store.get_job(job_id)
    assert record is not None
    return record


@app.post("/jobs/{job_id}/simulate-failure", response_model=JobRecord)
def simulate_failure(job_id: str) -> JobRecord:
    if _use_modal():
        from trainextend import modal_client

        record = modal_client.get_job(job_id)
        if record is None:
            raise _not_found("Job", job_id)
        if record.status != JobStatus.RUNNING:
            raise HTTPException(status_code=400, detail="Job is not running")
        try:
            return modal_client.simulate_failure(job_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    record = store.get_job(job_id)
    if record is None:
        raise _not_found("Job", job_id)
    if record.status != JobStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Job is not running")
    store.set_pending_failure(job_id)
    record = store.get_job(job_id)
    assert record is not None
    return record


@app.get("/jobs/{job_id}/checkpoints")
def get_checkpoints(job_id: str):
    if _use_modal():
        from trainextend import modal_client

        if modal_client.get_job(job_id) is None:
            raise _not_found("Job", job_id)
        return modal_client.list_checkpoints(job_id)

    record = store.get_job(job_id)
    if record is None:
        raise _not_found("Job", job_id)
    return list_checkpoints(checkpoints_dir(settings.data_dir, job_id))


@app.get("/jobs/{job_id}/events", response_model=list[JobEvent])
def get_events(job_id: str, limit: int = 100) -> list[JobEvent]:
    if _use_modal():
        from trainextend import modal_client

        if modal_client.get_job(job_id) is None:
            raise _not_found("Job", job_id)
        return modal_client.list_events(job_id, limit=limit)

    record = store.get_job(job_id)
    if record is None:
        raise _not_found("Job", job_id)
    return store.list_events(job_id, limit=limit)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "name": "TrainExtend",
        "version": __version__,
        "backend": settings.backend,
        "modal_app": settings.modal_app if _use_modal() else None,
        "max_workers": pool.max_workers if not _use_modal() else None,
    }


def main():
    import uvicorn

    uvicorn.run("trainextend.api:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), reload=False)
