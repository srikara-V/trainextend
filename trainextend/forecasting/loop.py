from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from trainextend.checkpointing import load_checkpoint, mark_last_good, save_checkpoint
from trainextend.forecasting.data import (
    build_window_dataset,
    denormalize,
    holdout_context_and_target,
    load_series,
    normalize_series,
    resolve_cutoff_index,
)
from trainextend.forecasting.metrics import compute_metrics
from trainextend.forecasting.model import MLPForecaster
from trainextend.paths import checkpoints_dir, metrics_path, predictions_path
from trainextend.schemas import JobStatus, TrainingConfig
from trainextend.state import JobStore


def _device(use_gpu: bool) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if use_gpu and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _checkpoint_metadata(config: TrainingConfig) -> dict:
    return {
        "experiment_id": config.experiment_id,
        "experiment_name": config.experiment_name,
        "dataset": config.dataset_name,
        "model": config.model_name,
        "cutoff": config.cutoff,
        "context_length": config.context_length,
        "prediction_length": config.prediction_length,
    }


def _save_predictions(path: Path, y_true: np.ndarray, y_pred: np.ndarray, cutoff: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["cutoff", "horizon_step", "actual", "forecast"])
        for i, (actual, forecast) in enumerate(zip(y_true, y_pred, strict=True)):
            writer.writerow([cutoff, i + 1, f"{actual:.6f}", f"{forecast:.6f}"])


def run_forecast_job(
    store: JobStore,
    job_id: str,
    *,
    worker_id: str,
    resume: bool = False,
    should_fail: Callable[[], bool] | None = None,
    on_state_change: Callable[[], None] | None = None,
) -> None:
    def _notify() -> None:
        if on_state_change is not None:
            on_state_change()

    record = store.get_job(job_id)
    if record is None:
        raise KeyError(job_id)

    config = record.config
    if config.cutoff is None or config.dataset_name is None:
        raise ValueError("Forecast job requires cutoff and dataset_name")

    ckpt_root = checkpoints_dir(store.data_dir, job_id)
    dev = _device(config.use_gpu and config.gpu_type != "cpu")

    series = load_series(
        dataset_name=config.dataset_name,
        path=config.dataset_path,
        target=config.target,
        time_col=config.time_col,
    )
    cutoff_index = resolve_cutoff_index(series, config.cutoff)
    full_norm, norm_mean, norm_std = normalize_series(series.values)
    norm_values = full_norm[:cutoff_index]

    model = MLPForecaster(config.context_length, config.prediction_length).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    global_step = 0
    epoch = 0
    start_step = 0

    if resume and config.resume:
        loaded = load_checkpoint(ckpt_root, prefer_last_good=True)
        if loaded:
            state, path = loaded
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            global_step = int(state.get("global_step", 0))
            epoch = int(state.get("epoch", 0))
            start_step = global_step
            store.update_job(
                job_id,
                status=JobStatus.RESUMING,
                global_step=global_step,
                epoch=epoch,
                worker_id=worker_id,
                latest_checkpoint=str(path),
                last_good_checkpoint=str(path),
                resume_count=record.resume_count + 1,
            )
            store.append_event(
                job_id,
                "resumed_from_checkpoint",
                worker_id=worker_id,
                payload={"step": global_step, "checkpoint": str(path), "cutoff": config.cutoff},
            )
            _notify()

    store.update_job(job_id, status=JobStatus.RUNNING, worker_id=worker_id)
    store.append_event(
        job_id,
        "worker_started",
        worker_id=worker_id,
        payload={"device": str(dev), "cutoff": config.cutoff, "task": "forecast"},
    )
    _notify()

    x_np, y_np = build_window_dataset(
        norm_values,
        context_length=config.context_length,
        prediction_length=config.prediction_length,
        end_index=len(norm_values),
    )
    x_t = torch.from_numpy(x_np).float()
    y_t = torch.from_numpy(y_np).float()
    loader = DataLoader(TensorDataset(x_t, y_t), batch_size=config.batch_size, shuffle=True)
    model.train()
    loss_value = 0.0
    meta = _checkpoint_metadata(config)

    try:
        while global_step < config.max_steps and epoch < config.max_epochs:
            for batch_x, batch_y in loader:
                if global_step >= config.max_steps:
                    break

                batch_x, batch_y = batch_x.to(dev), batch_y.to(dev)
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                global_step += 1
                loss_value = float(loss.item())

                if config.simulate_failure_at_step and global_step == config.simulate_failure_at_step:
                    rec = store.get_job(job_id)
                    if rec and rec.resume_count == 0:
                        store.append_event(
                            job_id,
                            "worker_failed",
                            worker_id=worker_id,
                            payload={"reason": "simulate_failure_at_step", "step": global_step},
                        )
                        raise RuntimeError(f"Simulated worker crash at step {global_step}")

                if should_fail and should_fail():
                    store.append_event(
                        job_id,
                        "worker_failed",
                        worker_id=worker_id,
                        payload={"reason": "pending_failure", "step": global_step},
                    )
                    raise RuntimeError(f"Simulated worker crash at step {global_step}")

                if global_step % config.checkpoint_every_steps == 0:
                    store.update_job(job_id, status=JobStatus.CHECKPOINTING, global_step=global_step, epoch=epoch)
                    state = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "global_step": global_step,
                        "epoch": epoch,
                        "metadata": meta,
                    }
                    path = save_checkpoint(ckpt_root, step=global_step, state=state)
                    mark_last_good(ckpt_root, path)
                    store.update_job(
                        job_id,
                        status=JobStatus.RUNNING,
                        latest_checkpoint=str(path),
                        last_good_checkpoint=str(path),
                        global_step=global_step,
                    )
                    store.append_event(
                        job_id,
                        "checkpoint_saved",
                        worker_id=worker_id,
                        payload={"step": global_step, "path": str(path), **meta},
                    )
                    _notify()

            epoch += 1

        # Evaluate on holdout window after cutoff
        model.eval()
        ctx_norm, tgt_norm = holdout_context_and_target(
            full_norm,
            cutoff_index=cutoff_index,
            context_length=config.context_length,
            prediction_length=config.prediction_length,
        )
        with torch.no_grad():
            pred_norm = model(torch.from_numpy(ctx_norm).float().unsqueeze(0).to(dev)).cpu().numpy()[0]

        y_true = denormalize(tgt_norm, norm_mean, norm_std)
        y_pred = denormalize(pred_norm, norm_mean, norm_std)
        horizon_metrics = compute_metrics(y_true, y_pred, config.metric_names)
        horizon_metrics.update(
            {
                "job_id": job_id,
                "experiment_id": config.experiment_id,
                "cutoff": config.cutoff,
                "horizon": config.prediction_length,
                "context_length": config.context_length,
                "dataset": config.dataset_name,
                "model": config.model_name,
                "final_loss": loss_value,
                "global_step": global_step,
                "resumed_from_step": start_step,
            }
        )

        pred_out = predictions_path(store.data_dir, job_id)
        _save_predictions(pred_out, y_true, y_pred, config.cutoff or "")

        out = metrics_path(store.data_dir, job_id)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(horizon_metrics, indent=2))

        store.append_event(
            job_id,
            "forecast_evaluated",
            worker_id=worker_id,
            payload=horizon_metrics,
        )
        store.update_job(job_id, status=JobStatus.COMPLETED, global_step=global_step, epoch=epoch, error=None)
        store.append_event(job_id, "completed", worker_id=worker_id, payload=horizon_metrics)
        _notify()

    except Exception as exc:
        store.update_job(job_id, status=JobStatus.FAILED, error=str(exc), global_step=global_step, epoch=epoch)
        store.append_event(
            job_id,
            "failed",
            worker_id=worker_id,
            payload={"error": str(exc), "step": global_step, "cutoff": config.cutoff},
        )
        _notify()
        raise
