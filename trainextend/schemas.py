from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    CHECKPOINTING = "checkpointing"
    PAUSED = "paused"
    RESUMING = "resuming"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExperimentStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class DatasetSpec(BaseModel):
    name: str = "synthetic"
    path: str | None = None
    target: str = "OT"
    freq: str = "H"
    time_col: str = "date"


class ModelSpec(BaseModel):
    name: str = "MLP"
    context_length: int = 96
    prediction_length: int = 96


class BacktestSpec(BaseModel):
    cutoffs: list[str]
    stride: int | None = None


class RuntimeSpec(BaseModel):
    gpu: str = "T4"
    use_gpu: bool = True
    max_steps: int = 500
    max_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    checkpoint_every_steps: int = 50
    resume: bool = True
    simulate_failure_at_step: int | None = None


class ExperimentConfig(BaseModel):
    experiment_name: str
    dataset: DatasetSpec
    model: ModelSpec
    backtest: BacktestSpec
    metrics: list[str] = Field(default_factory=lambda: ["mse", "mae", "smape"])
    runtime: RuntimeSpec = Field(default_factory=RuntimeSpec)


class TrainingConfig(BaseModel):
    """Per-job config. task=forecast for rolling backtest jobs; task=demo for legacy CNN demo."""

    task: Literal["demo", "forecast"] = "demo"
    name: str = "demo"

    # Legacy demo (MNIST / synthetic CNN)
    dataset: str = "mnist"

    # Forecast job metadata
    experiment_id: str | None = None
    experiment_name: str | None = None
    cutoff: str | None = None
    dataset_name: str | None = None
    dataset_path: str | None = None
    target: str = "OT"
    time_col: str = "date"
    model_name: str = "MLP"
    context_length: int = 96
    prediction_length: int = 24
    metric_names: list[str] = Field(default_factory=lambda: ["mse", "mae", "smape"])

    # Shared runtime
    gpu_type: str = "T4"
    max_steps: int = 1000
    max_epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.001
    checkpoint_every_steps: int = 100
    resume: bool = True
    simulate_failure_at_step: int | None = None
    use_gpu: bool = True


class JobCreate(BaseModel):
    config: TrainingConfig
    config_path: str | None = None


class JobRecord(BaseModel):
    job_id: str
    status: JobStatus
    config: TrainingConfig
    config_path: str
    experiment_id: str | None = None
    cutoff: str | None = None
    prediction_length: int | None = None
    context_length: int | None = None
    latest_checkpoint: str | None = None
    last_good_checkpoint: str | None = None
    global_step: int = 0
    epoch: int = 0
    worker_id: str | None = None
    resume_count: int = 0
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class JobEvent(BaseModel):
    t: datetime
    job_id: str
    event: str
    worker_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class CheckpointInfo(BaseModel):
    step: int
    path: str
    is_latest: bool = False
    is_last_good: bool = False


class JobSubmitResponse(BaseModel):
    job_id: str
    status: JobStatus
    config_path: str


class ExperimentSubmitResponse(BaseModel):
    experiment_id: str
    experiment_name: str
    job_ids: list[str]
    status: ExperimentStatus


class ExperimentRecord(BaseModel):
    experiment_id: str
    name: str
    config: ExperimentConfig
    job_ids: list[str]
    status: ExperimentStatus
    created_at: datetime
    updated_at: datetime


class CompareRow(BaseModel):
    model: str
    dataset: str
    horizon: int
    cutoffs: int
    mse: float | None = None
    mae: float | None = None
    smape: float | None = None


class CompareResponse(BaseModel):
    experiment_id: str
    experiment_name: str
    rows: list[CompareRow]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
