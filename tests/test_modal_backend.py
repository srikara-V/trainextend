from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from trainextend.schemas import JobStatus, TrainingConfig
from trainextend.state import JobStore


@pytest.fixture()
def modal_api_client(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINEXTEND_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TRAINEXTEND_BACKEND", "modal")

    store = JobStore(tmp_path)

    import trainextend.modal_client as modal_client_mod

    def mock_create(config: TrainingConfig):
        return store.create_job(config, enqueue=False)

    def mock_get(job_id: str):
        return store.get_job(job_id)

    def mock_resume(job_id: str):
        store.update_job(job_id, status=JobStatus.QUEUED, error=None)
        store.append_event(job_id, "resume_requested")
        record = store.get_job(job_id)
        assert record is not None
        return record

    def mock_cancel(job_id: str):
        store.update_job(job_id, status=JobStatus.CANCELLED)
        store.append_event(job_id, "cancelled")
        record = store.get_job(job_id)
        assert record is not None
        return record

    def mock_simulate(job_id: str):
        store.set_pending_failure(job_id)
        record = store.get_job(job_id)
        assert record is not None
        return record

    def mock_events(job_id: str, limit: int = 100):
        return store.list_events(job_id, limit=limit)

    def mock_checkpoints(job_id: str):
        from trainextend.checkpointing import list_checkpoints
        from trainextend.paths import checkpoints_dir

        return list_checkpoints(checkpoints_dir(tmp_path, job_id))

    with patch.object(modal_client_mod, "check_modal_deployed"), patch.object(
        modal_client_mod, "create_job", side_effect=mock_create
    ), patch.object(modal_client_mod, "get_job", side_effect=mock_get), patch.object(
        modal_client_mod, "resume_job", side_effect=mock_resume
    ), patch.object(
        modal_client_mod, "cancel_job", side_effect=mock_cancel
    ), patch.object(
        modal_client_mod, "simulate_failure", side_effect=mock_simulate
    ), patch.object(
        modal_client_mod, "list_events", side_effect=mock_events
    ), patch.object(
        modal_client_mod, "list_checkpoints", side_effect=mock_checkpoints
    ):
        import trainextend.api as api_mod

        api_mod.settings = api_mod.Settings()
        yield TestClient(api_mod.app)


def test_modal_backend_create_job(modal_api_client: TestClient):
    cfg = TrainingConfig(dataset="synthetic", use_gpu=True, gpu_type="T4", max_steps=100)
    resp = modal_api_client.post("/jobs", json={"config": cfg.model_dump()})
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    assert job_id.startswith("job_")

    status = modal_api_client.get(f"/jobs/{job_id}")
    assert status.status_code == 200
    assert status.json()["status"] == JobStatus.QUEUED.value


def test_modal_backend_health(modal_api_client: TestClient):
    resp = modal_api_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["backend"] == "modal"
