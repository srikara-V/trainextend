from __future__ import annotations

import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from trainextend.state import JobStore
from trainextend.training.dispatch import run_job


def default_max_workers() -> int:
    cpus = os.cpu_count() or 1
    return max(1, min(4, cpus))


class LocalWorkerPool:
    """Background thread pool that drains the SQLite job queue."""

    def __init__(
        self,
        store: JobStore,
        *,
        max_workers: int | None = None,
        poll_sec: float = 1.0,
    ) -> None:
        self.store = store
        self.max_workers = max_workers if max_workers is not None else default_max_workers()
        self.poll_sec = poll_sec
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._resume_flags: dict[str, bool] = {}
        self._resume_lock = threading.Lock()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="trainextend-worker",
        )
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="trainextend-dispatcher",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self._executor:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None

    def mark_resume(self, job_id: str) -> None:
        with self._resume_lock:
            self._resume_flags[job_id] = True
        self.store.enqueue(job_id)

    def _run_job(self, job_id: str) -> None:
        with self._resume_lock:
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

    def _loop(self) -> None:
        assert self._executor is not None
        while not self._stop.is_set():
            job_id = self.store.dequeue()
            if not job_id:
                time.sleep(self.poll_sec)
                continue
            self._executor.submit(self._run_job, job_id)
