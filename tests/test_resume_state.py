from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from trainextend.schemas import JobStatus, TrainingConfig
from trainextend.state import JobStore


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TRAINEXTEND_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TRAINEXTEND_BACKEND", "local")

    import trainextend.api as api_mod

    api_mod.settings = api_mod.Settings()
    api_mod.store = JobStore(api_mod.settings.data_dir)
    api_mod.pool = api_mod.LocalWorkerPool(
        api_mod.store,
        max_workers=api_mod.settings.max_workers or None,
    )
    api_mod.pool.start()

    yield TestClient(api_mod.app)

    api_mod.pool.stop()


def test_job_lifecycle_and_resume(client: TestClient, tmp_path: Path):
    cfg = TrainingConfig(
        task="demo",
        dataset="synthetic",
        use_gpu=False,
        gpu_type="cpu",
        max_steps=250,
        max_epochs=50,
        checkpoint_every_steps=100,
        simulate_failure_at_step=150,
    )
    resp = client.post("/jobs", json={"config": cfg.model_dump()})
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    deadline = time.time() + 60
    status = JobStatus.QUEUED
    while time.time() < deadline:
        status = JobStatus(client.get(f"/jobs/{job_id}").json()["status"])
        if status in (JobStatus.FAILED, JobStatus.COMPLETED):
            break
        time.sleep(0.5)

    assert status == JobStatus.FAILED

    ckpts = client.get(f"/jobs/{job_id}/checkpoints").json()
    assert any(c["step"] == 100 for c in ckpts)

    resp = client.post(f"/jobs/{job_id}/resume")
    assert resp.status_code == 200

    deadline = time.time() + 90
    while time.time() < deadline:
        status = JobStatus(client.get(f"/jobs/{job_id}").json()["status"])
        if status == JobStatus.COMPLETED:
            break
        time.sleep(0.5)

    assert status == JobStatus.COMPLETED
    events = [e["event"] for e in client.get(f"/jobs/{job_id}/events").json()]
    assert "checkpoint_saved" in events
    assert "resumed_from_checkpoint" in events
    assert "completed" in events
