from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from trainextend.schemas import ExperimentStatus, JobStatus


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TRAINEXTEND_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TRAINEXTEND_BACKEND", "local")

    import trainextend.api as api_mod

    api_mod.settings = api_mod.Settings()
    api_mod.store = api_mod.JobStore(api_mod.settings.data_dir)
    api_mod.pool = api_mod.LocalWorkerPool(
        api_mod.store,
        max_workers=api_mod.settings.max_workers or None,
    )
    api_mod.pool.start()

    yield TestClient(api_mod.app)

    api_mod.pool.stop()


def test_experiment_submit_compare(client: TestClient):
    payload = {
        "experiment_name": "test_exp",
        "dataset": {"name": "synthetic"},
        "model": {"name": "MLP", "context_length": 96, "prediction_length": 24},
        "backtest": {"cutoffs": ["4500"]},
        "metrics": ["mse", "mae", "smape"],
        "runtime": {
            "gpu": "cpu",
            "use_gpu": False,
            "max_steps": 80,
            "max_epochs": 20,
            "batch_size": 32,
            "checkpoint_every_steps": 40,
            "resume": True,
        },
    }
    resp = client.post("/experiments", json=payload)
    assert resp.status_code == 200
    exp_id = resp.json()["experiment_id"]
    job_ids = resp.json()["job_ids"]
    assert len(job_ids) == 1

    deadline = time.time() + 90
    while time.time() < deadline:
        exp = client.get(f"/experiments/{exp_id}").json()
        if exp["status"] in (ExperimentStatus.COMPLETED.value, ExperimentStatus.FAILED.value):
            break
        time.sleep(0.5)

    assert exp["status"] == ExperimentStatus.COMPLETED.value
    job = client.get(f"/jobs/{job_ids[0]}").json()
    assert job["status"] == JobStatus.COMPLETED.value

    compare = client.get(f"/experiments/{exp_id}/compare").json()
    assert compare["rows"]
    assert compare["rows"][0]["mse"] is not None
    assert compare["rows"][0]["cutoffs"] == 1
