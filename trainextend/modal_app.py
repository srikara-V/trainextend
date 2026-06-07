#!/usr/bin/env python3
"""Modal GPU workers and volume-backed job store for TrainExtend."""

from __future__ import annotations

import os
from pathlib import Path

import modal

APP_NAME = "trainextend"
VOLUME_NAME = os.environ.get("TRAINEXTEND_VOLUME", "trainextend-vol")
DATA_MOUNT = "/vol/trainextend"
DEFAULT_GPU = os.environ.get("TRAINEXTEND_MODAL_GPU", "T4")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

_pkg_root = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install("pyyaml", "pydantic>=2.9", "pydantic-settings>=2.6", "numpy")
    .add_local_dir(
        str(_pkg_root),
        remote_path="/root/app",
        ignore=["**/.venv", "**/__pycache__", "**/trainextend_data", "**/.pytest_cache", "**/tests"],
    )
)

volume_mount = {DATA_MOUNT: volume}


def _bootstrap() -> None:
    import sys

    if "/root/app" not in sys.path:
        sys.path.insert(0, "/root/app")


def _store():
    _bootstrap()
    from trainextend.state import JobStore

    return JobStore(Path(DATA_MOUNT))


def _spawn_train(job_id: str, resume: bool = False) -> None:
    train_job.spawn(job_id, resume=resume)


@app.function(image=image, volumes=volume_mount)
def create_and_start_job(config_dict: dict) -> dict:
    _bootstrap()
    from trainextend.schemas import TrainingConfig

    store = _store()
    config = TrainingConfig.model_validate(config_dict)
    record = store.create_job(config, enqueue=False)
    volume.commit()
    _spawn_train(record.job_id, resume=False)
    return record.model_dump(mode="json")


@app.function(image=image, volumes=volume_mount)
def create_and_start_experiment(config_dict: dict) -> dict:
    _bootstrap()
    from trainextend.schemas import ExperimentConfig

    store = _store()
    config = ExperimentConfig.model_validate(config_dict)
    record = store.create_experiment(config, enqueue=False)
    volume.commit()
    for job_id in record.job_ids:
        _spawn_train(job_id, resume=False)
    return record.model_dump(mode="json")


@app.function(
    image=image,
    volumes=volume_mount,
    gpu=DEFAULT_GPU,
    timeout=60 * 60,
    retries=0,
)
def train_job(job_id: str, resume: bool = False) -> dict:
    _bootstrap()
    from trainextend.training.dispatch import run_job

    store = _store()
    worker_id = modal.current_function_call_id()

    def commit_state() -> None:
        volume.commit()

    def should_fail() -> bool:
        volume.reload()
        return store.pending_failure(job_id)

    run_job(
        store,
        job_id,
        worker_id=worker_id,
        resume=resume,
        should_fail=should_fail,
        on_state_change=commit_state,
    )
    volume.commit()
    record = store.get_job(job_id)
    return record.model_dump(mode="json") if record else {"job_id": job_id}


@app.function(image=image, volumes=volume_mount)
def get_job_record(job_id: str) -> dict | None:
    volume.reload()
    record = _store().get_job(job_id)
    return record.model_dump(mode="json") if record else None


@app.function(image=image, volumes=volume_mount)
def get_experiment_record(experiment_id: str) -> dict | None:
    volume.reload()
    record = _store().get_experiment(experiment_id)
    return record.model_dump(mode="json") if record else None


@app.function(image=image, volumes=volume_mount)
def compare_experiment_remote(experiment_id: str) -> dict:
    volume.reload()
    from trainextend.compare import compare_experiment

    store = _store()
    record = store.get_experiment(experiment_id)
    if record is None:
        raise KeyError(experiment_id)
    return compare_experiment(store, record).model_dump(mode="json")


@app.function(image=image, volumes=volume_mount)
def list_job_events(job_id: str, limit: int = 100) -> list[dict]:
    volume.reload()
    events = _store().list_events(job_id, limit=limit)
    return [ev.model_dump(mode="json") for ev in events]


@app.function(image=image, volumes=volume_mount)
def list_job_checkpoints(job_id: str) -> list[dict]:
    volume.reload()
    from trainextend.checkpointing import list_checkpoints
    from trainextend.paths import checkpoints_dir

    return list_checkpoints(checkpoints_dir(Path(DATA_MOUNT), job_id))


@app.function(image=image, volumes=volume_mount)
def resume_job(job_id: str) -> dict:
    _bootstrap()
    from trainextend.schemas import JobStatus

    store = _store()
    record = store.get_job(job_id)
    if record is None:
        raise KeyError(job_id)
    if record.status == JobStatus.COMPLETED:
        raise ValueError("Job already completed")
    if record.status == JobStatus.CANCELLED:
        raise ValueError("Job cancelled")

    store.update_job(job_id, status=JobStatus.QUEUED, error=None)
    store.append_event(job_id, "resume_requested")
    volume.commit()
    _spawn_train(job_id, resume=True)
    record = store.get_job(job_id)
    assert record is not None
    return record.model_dump(mode="json")


@app.function(image=image, volumes=volume_mount)
def cancel_job(job_id: str) -> dict:
    _bootstrap()
    from trainextend.schemas import JobStatus

    store = _store()
    record = store.get_job(job_id)
    if record is None:
        raise KeyError(job_id)
    store.update_job(job_id, status=JobStatus.CANCELLED)
    store.append_event(job_id, "cancelled")
    volume.commit()
    record = store.get_job(job_id)
    assert record is not None
    return record.model_dump(mode="json")


@app.function(image=image, volumes=volume_mount)
def simulate_failure(job_id: str) -> dict:
    _bootstrap()
    from trainextend.schemas import JobStatus

    store = _store()
    record = store.get_job(job_id)
    if record is None:
        raise KeyError(job_id)
    if record.status != JobStatus.RUNNING:
        raise ValueError("Job is not running")
    store.set_pending_failure(job_id)
    volume.commit()
    record = store.get_job(job_id)
    assert record is not None
    return record.model_dump(mode="json")


@app.local_entrypoint()
def main(job_id: str = "", resume: bool = False):
    if not job_id:
        print("Usage: modal run trainextend/modal_app.py --job-id job_xxx [--resume]")
        print("Deploy first: modal deploy trainextend/modal_app.py")
        return
    print(train_job.remote(job_id, resume=resume))

