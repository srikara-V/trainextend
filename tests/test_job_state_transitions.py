from __future__ import annotations

from trainextend.schemas import JobStatus, TrainingConfig
from trainextend.state import JobStore


def test_job_state_transitions(tmp_path):
    store = JobStore(tmp_path)
    cfg = TrainingConfig(dataset="synthetic", max_steps=10, use_gpu=False)
    record = store.create_job(cfg)
    assert record.status == JobStatus.QUEUED

    store.update_job(record.job_id, status=JobStatus.RUNNING, worker_id="w1")
    assert store.get_job(record.job_id).status == JobStatus.RUNNING

    store.append_event(record.job_id, "checkpoint_saved", worker_id="w1", payload={"step": 100})
    events = store.list_events(record.job_id)
    assert events[-1].event == "checkpoint_saved"

    store.update_job(record.job_id, status=JobStatus.FAILED, error="crash")
    assert store.get_job(record.job_id).status == JobStatus.FAILED

    store.enqueue(record.job_id)
    assert store.dequeue() == record.job_id
