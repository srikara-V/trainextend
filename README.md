# TrainExtend

A resumable experiment runner for rolling-window time-series forecasting backtests.

TrainExtend is an **experiment control plane**. You describe a backtest in YAML; it expands that into one training job per cutoff, runs a PyTorch MLP forecaster on each job, checkpoints progress, evaluates holdout forecasts, and aggregates horizon metrics across cutoffs.

## Functionality

1. **Accepts an experiment config** with dataset, model, backtest cutoffs, metrics, and runtime settings.
2. **Expands the experiment into jobs** — one job per backtest cutoff.
3. **Trains an MLP forecaster** on sliding windows of history up to each cutoff.
4. **Checkpoints training** at a fixed step interval using atomic writes (`step_XXXXXX.pt`, `latest.json`, `last_good.json`).
5. **Evaluates a holdout forecast** after training and writes `predictions.csv` and `metrics.json` per job.
6. **Resumes failed or interrupted jobs** from the last good checkpoint.
7. **Aggregates results** across cutoffs with `compare` (mean MSE / MAE / sMAPE).

Datasets: built-in synthetic series, or a CSV path with `target` and `time_col` columns.

Workers: a local background thread pool, or Modal GPU workers (T4 by default) with a shared Modal Volume.

## Architecture

```text
CLI (submit / status / resume / compare)
        |
        v
FastAPI control plane  +  SQLite job store (metadata.db)
        |
        |  expand experiment -> N cutoff jobs
        v
Job queue
        |
        v
Local worker thread          Modal GPU worker (optional)
        |
        |  train MLP -> checkpoint -> holdout forecast -> metrics
        v
./trainextend_data/          Modal Volume (trainextend-vol)
  checkpoints/
  artifacts/
  logs/
```

## Install

Requires Python 3.11+.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For Modal GPU workers:

```bash
pip install -e ".[modal]"
modal setup
```

## Quick start (local CPU)

Terminal 1 — start the control plane:

```bash
trainextend serve --backend local
```

Terminal 2 — submit a 3-cutoff rolling backtest on synthetic data:

```bash
trainextend submit --config configs/synthetic_backtest.yaml
trainextend exp-status <experiment_id>
trainextend compare <experiment_id>
```

Example `compare` output (metrics averaged across cutoffs):

```text
Experiment: synthetic_mlp_demo (exp_xxxxxxxxxx)

Model      Dataset      Horizon  Cutoffs  MSE        MAE        sMAPE
MLP        synthetic    24       3        0.042      0.158      8.2
```

## Experiment config

```yaml
experiment_name: synthetic_mlp_demo

dataset:
  name: synthetic          # built-in series; or any name with dataset.path set
  path: null               # required for CSV datasets
  target: value            # CSV column to forecast
  time_col: date           # CSV timestamp column (used for cutoff resolution)

model:
  name: MLP
  context_length: 96       # history window length
  prediction_length: 24    # forecast horizon

backtest:
  cutoffs:
    - "4000"               # index into series, or timestamp for CSV data
    - "4200"
    - "4400"

metrics: [mse, mae, smape]

runtime:
  gpu: cpu                 # cpu, T4, etc.
  use_gpu: false
  max_steps: 200
  max_epochs: 20
  batch_size: 32
  learning_rate: 0.001
  checkpoint_every_steps: 50
  resume: true
  simulate_failure_at_step: null   # optional: crash at this step for demos
```

Each cutoff becomes its own job. Checkpoints store forecast metadata (`cutoff`, `horizon`, `context_length`, `experiment_id`).

## Failure recovery

Config-driven crash (see `configs/failure_forecast_demo.yaml`):

```bash
trainextend submit --config configs/failure_forecast_demo.yaml
trainextend status <job_id>    # failed at step 250; last good checkpoint at 200
trainextend resume <job_id>
trainextend compare <experiment_id>
```

Or trigger a crash on a running job:

```bash
trainextend simulate-failure <job_id>
trainextend resume <job_id>
```

Resume loads from `last_good.json` (falls back to `latest.json`), restores model and optimizer state, and continues training from that step.

## Modal GPU

```bash
trainextend modal deploy

trainextend serve --backend modal
trainextend submit --config configs/synthetic_backtest_gpu.yaml
```

Modal workers read and write the same artifact layout under `/vol/trainextend` on the `trainextend-vol` volume.

## CLI

| Command | Description |
|---------|-------------|
| `trainextend serve` | Start the FastAPI control plane |
| `trainextend submit -c <config.yaml>` | Submit an experiment or legacy single job |
| `trainextend exp-status <exp_id>` | Experiment status and child job IDs |
| `trainextend compare <exp_id>` | Aggregate metrics across completed cutoffs |
| `trainextend status <job_id>` | Job status, step, checkpoints, error |
| `trainextend resume <job_id>` | Re-queue a failed job from last good checkpoint |
| `trainextend checkpoints <job_id>` | List saved checkpoints |
| `trainextend logs <job_id>` | Job event log |
| `trainextend cancel <job_id>` | Cancel a job |
| `trainextend simulate-failure <job_id>` | Crash a running job on the next step |
| `trainextend modal deploy` | Deploy Modal GPU workers |

Environment variables:

- `TRAINEXTEND_API_URL` — CLI target (default `http://127.0.0.1:8000`)
- `TRAINEXTEND_DATA_DIR` — artifact root (default `./trainextend_data`)
- `TRAINEXTEND_BACKEND` — `local` or `modal`

## Legacy demo jobs

Configs without `experiment_name` / `backtest` submit a single MNIST or synthetic CNN training job (`configs/demo_cpu.yaml`). Useful for testing checkpoint/resume mechanics without forecasting.

## Artifacts

```text
trainextend_data/
  metadata.db
  experiments/<exp_id>/config.yaml
  jobs/<job_id>/
    config.yaml
    checkpoints/
      step_000050.pt
      latest.json
      last_good.json
    artifacts/
      metrics.json       # MSE, MAE, sMAPE + job metadata
      predictions.csv    # actual vs forecast per horizon step
    logs/
      events.jsonl
```

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

## Inspiration

This was done as an effort to create a generalizable lightweight training infra, where I originally used a similar approach for evaluating [HNMD-based loss function for time-series forecasting models](https://github.com/srikara-V/Hierarchical-NURBS-Inspired-Multi-Domain-Loss-for-Training-Deep-Time-Series-Forecasting-Models) and running evals for the Yale Graph and Geometric Learning Lab.
