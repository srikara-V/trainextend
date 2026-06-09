from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import patch

from trainextend.schemas import TrainingConfig
from trainextend.state import JobStore
from trainextend.worker_local import LocalWorkerPool


def test_local_worker_pool_runs_jobs_in_parallel(tmp_path: Path):
    store = JobStore(tmp_path)
    barrier = threading.Barrier(2, timeout=10)
    lock = threading.Lock()
    active = 0
    peak = 0

    def fake_run_job(store, job_id, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        barrier.wait()
        with lock:
            active -= 1

    for i in range(2):
        store.create_job(
            TrainingConfig(
                task="demo",
                name=f"job_{i}",
                dataset="synthetic",
                use_gpu=False,
                gpu_type="cpu",
                max_steps=10,
            )
        )

    pool = LocalWorkerPool(store, max_workers=2, poll_sec=0.05)
    with patch("trainextend.worker_local.run_job", side_effect=fake_run_job):
        pool.start()
        deadline = time.time() + 10
        while time.time() < deadline and peak < 2:
            time.sleep(0.05)
        pool.stop()

    assert peak >= 2
