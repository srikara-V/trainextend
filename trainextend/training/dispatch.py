"""Route jobs to demo or forecast training loops."""

from __future__ import annotations

from collections.abc import Callable

from trainextend.forecasting.loop import run_forecast_job
from trainextend.state import JobStore
from trainextend.training.train_loop import run_training_job


def run_job(
    store: JobStore,
    job_id: str,
    *,
    worker_id: str,
    resume: bool = False,
    should_fail: Callable[[], bool] | None = None,
    on_state_change: Callable[[], None] | None = None,
) -> None:
    record = store.get_job(job_id)
    if record is None:
        raise KeyError(job_id)
    if record.config.task == "forecast":
        run_forecast_job(
            store,
            job_id,
            worker_id=worker_id,
            resume=resume,
            should_fail=should_fail,
            on_state_change=on_state_change,
        )
    else:
        run_training_job(
            store,
            job_id,
            worker_id=worker_id,
            resume=resume,
            should_fail=should_fail,
            on_state_change=on_state_change,
        )
