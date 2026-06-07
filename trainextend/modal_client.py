"""Local client for the deployed TrainExtend Modal app."""

from __future__ import annotations

import os

from trainextend.schemas import (
    CompareResponse,
    ExperimentConfig,
    ExperimentRecord,
    JobEvent,
    JobRecord,
    TrainingConfig,
)

APP_NAME = os.environ.get("TRAINEXTEND_MODAL_APP", "trainextend")


def _require_modal():
    try:
        import modal  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Modal backend requires: pip install -e '.[modal]'") from exc


def _fn(name: str):
    _require_modal()
    import modal

    return modal.Function.from_name(APP_NAME, name)


def check_modal_deployed() -> None:
    _fn("get_job_record")


def create_job(config: TrainingConfig) -> JobRecord:
    data = _fn("create_and_start_job").remote(config.model_dump())
    return JobRecord.model_validate(data)


def create_experiment(config: ExperimentConfig) -> ExperimentRecord:
    data = _fn("create_and_start_experiment").remote(config.model_dump())
    return ExperimentRecord.model_validate(data)


def get_experiment(experiment_id: str) -> ExperimentRecord | None:
    data = _fn("get_experiment_record").remote(experiment_id)
    if data is None:
        return None
    return ExperimentRecord.model_validate(data)


def compare_experiment(experiment_id: str) -> CompareResponse:
    data = _fn("compare_experiment_remote").remote(experiment_id)
    return CompareResponse.model_validate(data)


def get_job(job_id: str) -> JobRecord | None:
    data = _fn("get_job_record").remote(job_id)
    if data is None:
        return None
    return JobRecord.model_validate(data)


def resume_job(job_id: str) -> JobRecord:
    data = _fn("resume_job").remote(job_id)
    return JobRecord.model_validate(data)


def cancel_job(job_id: str) -> JobRecord:
    data = _fn("cancel_job").remote(job_id)
    return JobRecord.model_validate(data)


def simulate_failure(job_id: str) -> JobRecord:
    data = _fn("simulate_failure").remote(job_id)
    return JobRecord.model_validate(data)


def list_events(job_id: str, limit: int = 100) -> list[JobEvent]:
    rows = _fn("list_job_events").remote(job_id, limit)
    return [JobEvent.model_validate(row) for row in rows]


def list_checkpoints(job_id: str) -> list[dict]:
    return _fn("list_job_checkpoints").remote(job_id)
