from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch


def checkpoint_file_for_step(checkpoints_root: Path, step: int) -> Path:
    return checkpoints_root / f"step_{step:06d}.pt"


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _atomic_write_json(target: Path, payload: dict[str, Any]) -> None:
    _atomic_write_bytes(target, json.dumps(payload, indent=2).encode())


def save_checkpoint(
    checkpoints_root: Path,
    *,
    step: int,
    state: dict[str, Any],
) -> Path:
    """Write step_XXXXXX.pt via tmp rename, then update latest.json."""
    final_path = checkpoint_file_for_step(checkpoints_root, step)
    tmp_path = final_path.with_suffix(".pt.tmp")

    checkpoints_root.mkdir(parents=True, exist_ok=True)
    torch.save(state, tmp_path)
    os.replace(tmp_path, final_path)

    _atomic_write_json(
        checkpoints_root / "latest.json",
        {"step": step, "path": str(final_path.name), "absolute_path": str(final_path)},
    )
    return final_path


def mark_last_good(checkpoints_root: Path, checkpoint_path: Path) -> None:
    step = _step_from_path(checkpoint_path)
    _atomic_write_json(
        checkpoints_root / "last_good.json",
        {
            "step": step,
            "path": checkpoint_path.name,
            "absolute_path": str(checkpoint_path),
        },
    )


def _step_from_path(path: Path) -> int:
    stem = path.stem  # step_000300
    return int(stem.split("_", 1)[1])


def load_pointer(checkpoints_root: Path, name: str) -> dict[str, Any] | None:
    ptr = checkpoints_root / name
    if not ptr.is_file():
        return None
    return json.loads(ptr.read_text())


def resolve_checkpoint_path(checkpoints_root: Path, pointer: dict[str, Any] | None) -> Path | None:
    if not pointer:
        return None
    if pointer.get("absolute_path"):
        p = Path(pointer["absolute_path"])
        if p.is_file():
            return p
    rel = pointer.get("path")
    if rel:
        p = checkpoints_root / rel
        if p.is_file():
            return p
    step = pointer.get("step")
    if step is not None:
        p = checkpoint_file_for_step(checkpoints_root, int(step))
        if p.is_file():
            return p
    return None


def load_checkpoint(checkpoints_root: Path, prefer_last_good: bool = True) -> tuple[dict[str, Any], Path] | None:
    ptr_name = "last_good.json" if prefer_last_good else "latest.json"
    ptr = load_pointer(checkpoints_root, ptr_name)
    path = resolve_checkpoint_path(checkpoints_root, ptr)
    if path is None and prefer_last_good:
        ptr = load_pointer(checkpoints_root, "latest.json")
        path = resolve_checkpoint_path(checkpoints_root, ptr)
    if path is None:
        return None
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    mark_last_good(checkpoints_root, path)
    return state, path


def list_checkpoints(checkpoints_root: Path) -> list[dict[str, Any]]:
    if not checkpoints_root.is_dir():
        return []
    latest = load_pointer(checkpoints_root, "latest.json")
    last_good = load_pointer(checkpoints_root, "last_good.json")
    latest_step = latest.get("step") if latest else None
    last_good_step = last_good.get("step") if last_good else None

    items: list[dict[str, Any]] = []
    for path in sorted(checkpoints_root.glob("step_*.pt")):
        step = _step_from_path(path)
        items.append(
            {
                "step": step,
                "path": str(path),
                "is_latest": step == latest_step,
                "is_last_good": step == last_good_step,
            }
        )
    return items
