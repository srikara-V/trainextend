from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from trainextend.checkpointing import load_checkpoint, mark_last_good, save_checkpoint
from trainextend.paths import artifacts_dir, checkpoints_dir, metrics_path
from trainextend.schemas import JobStatus, TrainingConfig
from trainextend.state import JobStore
from trainextend.training.demo_model import TinyCNN


def _device(use_gpu: bool) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if use_gpu and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_dataloader(config: TrainingConfig) -> DataLoader:
    if config.dataset == "synthetic":
        g = torch.Generator().manual_seed(42)
        x = torch.randn(512, 1, 28, 28, generator=g)
        y = torch.randint(0, 10, (512,), generator=g)
        ds = TensorDataset(x, y)
        return DataLoader(ds, batch_size=config.batch_size, shuffle=True)

    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    ds = datasets.MNIST(root=str(Path.home() / ".trainextend" / "mnist"), train=True, download=True, transform=transform)
    return DataLoader(ds, batch_size=config.batch_size, shuffle=True, num_workers=0)


def run_training_job(
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
    ckpt_root = checkpoints_dir(store.data_dir, job_id)
    dev = _device(config.use_gpu and config.gpu_type != "cpu")

    model = TinyCNN().to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

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
                payload={"step": global_step, "checkpoint": str(path)},
            )
            _notify()

    store.update_job(job_id, status=JobStatus.RUNNING, worker_id=worker_id)
    store.append_event(job_id, "worker_started", worker_id=worker_id, payload={"device": str(dev)})
    _notify()

    loader = _build_dataloader(config)
    model.train()
    loss_value = 0.0

    try:
        # Loop epochs until max_steps is reached; max_epochs is a safety cap only.
        while global_step < config.max_steps and epoch < config.max_epochs:
            for batch_x, batch_y in loader:
                if global_step >= config.max_steps:
                    break

                batch_x, batch_y = batch_x.to(dev), batch_y.to(dev)
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
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
                        payload={"step": global_step, "path": str(path)},
                    )
                    _notify()

            epoch += 1

        # Save a final checkpoint if we trained past the last periodic save.
        if global_step > 0 and global_step % config.checkpoint_every_steps != 0:
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
                "epoch": epoch,
            }
            path = save_checkpoint(ckpt_root, step=global_step, state=state)
            mark_last_good(ckpt_root, path)
            store.update_job(
                job_id,
                latest_checkpoint=str(path),
                last_good_checkpoint=str(path),
                global_step=global_step,
            )
            store.append_event(
                job_id,
                "checkpoint_saved",
                worker_id=worker_id,
                payload={"step": global_step, "path": str(path), "final": True},
            )
            _notify()

        metrics = {
            "final_loss": loss_value,
            "global_step": global_step,
            "epochs_run": epoch,
            "resumed_from_step": start_step,
        }
        out = metrics_path(store.data_dir, job_id)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2))

        store.update_job(job_id, status=JobStatus.COMPLETED, global_step=global_step, epoch=epoch, error=None)
        store.append_event(job_id, "completed", worker_id=worker_id, payload=metrics)
        _notify()

    except Exception as exc:
        store.update_job(job_id, status=JobStatus.FAILED, error=str(exc), global_step=global_step, epoch=epoch)
        store.append_event(
            job_id,
            "failed",
            worker_id=worker_id,
            payload={"error": str(exc), "step": global_step},
        )
        _notify()
        raise
