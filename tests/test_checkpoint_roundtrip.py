from __future__ import annotations

from pathlib import Path

import pytest
import torch

from trainextend.checkpointing import (
    load_checkpoint,
    load_pointer,
    mark_last_good,
    save_checkpoint,
)


def test_checkpoint_roundtrip(tmp_path: Path):
    ckpt_dir = tmp_path / "checkpoints"
    state = {
        "model": {"w": torch.tensor([1.0, 2.0])},
        "optimizer": {},
        "global_step": 100,
        "epoch": 1,
    }
    path = save_checkpoint(ckpt_dir, step=100, state=state)
    assert path.is_file()
    loaded, loaded_path = load_checkpoint(ckpt_dir)
    assert loaded is not None
    assert loaded["global_step"] == 100
    assert loaded_path == path


def test_latest_pointer(tmp_path: Path):
    ckpt_dir = tmp_path / "checkpoints"
    save_checkpoint(ckpt_dir, step=100, state={"global_step": 100})
    save_checkpoint(ckpt_dir, step=200, state={"global_step": 200})
    latest = load_pointer(ckpt_dir, "latest.json")
    assert latest is not None
    assert latest["step"] == 200


def test_last_good_updated_after_load(tmp_path: Path):
    ckpt_dir = tmp_path / "checkpoints"
    p = save_checkpoint(ckpt_dir, step=300, state={"global_step": 300})
    mark_last_good(ckpt_dir, p)
    good = load_pointer(ckpt_dir, "last_good.json")
    assert good["step"] == 300
