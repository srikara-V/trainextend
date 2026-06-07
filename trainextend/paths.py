from __future__ import annotations

from pathlib import Path


def volume_root(data_dir: Path) -> Path:
    return data_dir


def job_root(data_dir: Path, job_id: str) -> Path:
    return volume_root(data_dir) / "jobs" / job_id


def job_config_path(data_dir: Path, job_id: str) -> Path:
    return job_root(data_dir, job_id) / "config.yaml"


def checkpoints_dir(data_dir: Path, job_id: str) -> Path:
    return job_root(data_dir, job_id) / "checkpoints"


def logs_dir(data_dir: Path, job_id: str) -> Path:
    return job_root(data_dir, job_id) / "logs"


def artifacts_dir(data_dir: Path, job_id: str) -> Path:
    return job_root(data_dir, job_id) / "artifacts"


def events_log_path(data_dir: Path, job_id: str) -> Path:
    return logs_dir(data_dir, job_id) / "events.jsonl"


def latest_pointer_path(data_dir: Path, job_id: str) -> Path:
    return checkpoints_dir(data_dir, job_id) / "latest.json"


def last_good_pointer_path(data_dir: Path, job_id: str) -> Path:
    return checkpoints_dir(data_dir, job_id) / "last_good.json"


def metrics_path(data_dir: Path, job_id: str) -> Path:
    return artifacts_dir(data_dir, job_id) / "metrics.json"


def predictions_path(data_dir: Path, job_id: str) -> Path:
    return artifacts_dir(data_dir, job_id) / "predictions.csv"


def experiment_root(data_dir: Path, experiment_id: str) -> Path:
    return volume_root(data_dir) / "experiments" / experiment_id


def experiment_config_path(data_dir: Path, experiment_id: str) -> Path:
    return experiment_root(data_dir, experiment_id) / "config.yaml"
