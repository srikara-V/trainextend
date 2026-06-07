from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import typer
import yaml

app = typer.Typer(no_args_is_help=True, help="TrainExtend — forecasting experiment runner with resumable GPU workers")


def _api_base() -> str:
    return os.environ.get("TRAINEXTEND_API_URL", "http://127.0.0.1:8000").rstrip("/")


def _client() -> httpx.Client:
    return httpx.Client(base_url=_api_base(), timeout=300.0)


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
    data_dir: Path = typer.Option(Path("./trainextend_data"), help="Volume / data root"),
    backend: str = typer.Option("local", help="local or modal"),
):
    """Start the FastAPI control plane."""
    os.environ["TRAINEXTEND_DATA_DIR"] = str(data_dir)
    os.environ["TRAINEXTEND_BACKEND"] = backend
    import uvicorn

    uvicorn.run("trainextend.api:app", host=host, port=port, reload=False)


@app.command()
def submit(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True, help="Experiment or job YAML"),
):
    """Submit a forecasting experiment or legacy single training job."""
    cfg = _load_yaml(config)
    with _client() as client:
        if "experiment_name" in cfg and "backtest" in cfg:
            resp = client.post("/experiments", json=cfg)
        else:
            resp = client.post("/jobs", json={"config": cfg})
        resp.raise_for_status()
        data = resp.json()
    typer.echo(json.dumps(data, indent=2))
    if "experiment_id" in data:
        typer.echo(f"experiment_id={data['experiment_id']}")
        for job_id in data.get("job_ids", []):
            typer.echo(f"job_id={job_id}")
    else:
        typer.echo(f"job_id={data['job_id']}")


@app.command()
def compare(experiment_id: str):
    """Compare horizon metrics across cutoffs for an experiment."""
    with _client() as client:
        resp = client.get(f"/experiments/{experiment_id}/compare")
        resp.raise_for_status()
        data = resp.json()

    typer.echo(f"Experiment: {data['experiment_name']} ({data['experiment_id']})")
    typer.echo("")
    header = f"{'Model':<10} {'Dataset':<12} {'Horizon':<8} {'Cutoffs':<8} {'MSE':<10} {'MAE':<10} {'sMAPE':<10}"
    typer.echo(header)
    typer.echo("-" * len(header))
    for row in data["rows"]:
        typer.echo(
            f"{row['model']:<10} {row['dataset']:<12} {row['horizon']:<8} {row['cutoffs']:<8} "
            f"{row.get('mse', '-'):!s:<10} {row.get('mae', '-'):!s:<10} {row.get('smape', '-'):!s:<10}"
        )


@app.command("exp-status")
def exp_status(experiment_id: str):
    """Get experiment status and child jobs."""
    with _client() as client:
        resp = client.get(f"/experiments/{experiment_id}")
        resp.raise_for_status()
        typer.echo(json.dumps(resp.json(), indent=2, default=str))


@app.command()
def status(job_id: str):
    """Get job status."""
    with _client() as client:
        resp = client.get(f"/jobs/{job_id}")
        resp.raise_for_status()
        typer.echo(json.dumps(resp.json(), indent=2, default=str))


@app.command("checkpoints")
def checkpoints_cmd(job_id: str):
    """List checkpoints for a job."""
    with _client() as client:
        resp = client.get(f"/jobs/{job_id}/checkpoints")
        resp.raise_for_status()
        typer.echo(json.dumps(resp.json(), indent=2))


@app.command()
def resume(job_id: str):
    """Resume a failed or interrupted job from last good checkpoint."""
    with _client() as client:
        resp = client.post(f"/jobs/{job_id}/resume")
        resp.raise_for_status()
        typer.echo(json.dumps(resp.json(), indent=2, default=str))


@app.command()
def cancel(job_id: str):
    """Cancel a job."""
    with _client() as client:
        resp = client.post(f"/jobs/{job_id}/cancel")
        resp.raise_for_status()
        typer.echo(json.dumps(resp.json(), indent=2, default=str))


@app.command()
def logs(job_id: str, limit: int = typer.Option(50, help="Max events")):
    """Show job event log."""
    with _client() as client:
        resp = client.get(f"/jobs/{job_id}/events", params={"limit": limit})
        resp.raise_for_status()
        for ev in resp.json():
            typer.echo(f"{ev['t']}  {ev['event']}  {json.dumps(ev.get('payload') or {})}")


@app.command("simulate-failure")
def simulate_failure(job_id: str):
    """Simulate worker crash on next training step (running jobs only)."""
    with _client() as client:
        resp = client.post(f"/jobs/{job_id}/simulate-failure")
        resp.raise_for_status()
        typer.echo(json.dumps(resp.json(), indent=2, default=str))


modal_app = typer.Typer(help="Modal deployment helpers")
app.add_typer(modal_app, name="modal")


@modal_app.command("deploy")
def modal_deploy():
    """Deploy TrainExtend GPU workers to Modal."""
    import subprocess

    app_path = Path(__file__).resolve().parent / "modal_app.py"
    subprocess.run(["modal", "deploy", str(app_path)], check=True)


def main():
    app()


if __name__ == "__main__":
    main()
