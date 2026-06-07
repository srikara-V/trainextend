from __future__ import annotations

import os
import threading
import time
import uuid

from trainextend.state import JobStore
from trainextend.training.dispatch import run_job


class LocalWorkerPool:
    """Background thread pool that drains the SQLite job queue."""

    def __init__(self, store: JobStore, poll_sec: float = 1.0) -> None:
        self.store = store
        self.poll_sec = poll_sec
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._resume_flags: dict[str, bool] = {}

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="trainextend-worker")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def mark_resume(self, job_id: str) -> None:
        self._resume_flags[job_id] = True
        self.store.enqueue(job_id)

    def _loop(self) -> None:
        while not self._stop.is_set():
            job_id = self.store.dequeue()
            if not job_id:
                time.sleep(self.poll_sec)
                continue
            resume = self._resume_flags.pop(job_id, False)
            worker_id = f"local_{uuid.uuid4().hex[:8]}"
            try:
                run_job(
                    self.store,
                    job_id,
                    worker_id=worker_id,
                    resume=resume,
                    should_fail=lambda: self.store.pending_failure(job_id),
                )
                if self.store.pending_failure(job_id):
                    self.store.clear_pending_failure(job_id)
            except Exception:
                pass
            time.sleep(0.1)

