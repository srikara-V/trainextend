from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SeriesData:
    values: np.ndarray
    timestamps: list[str] | None = None


def generate_synthetic_series(length: int = 6000, seed: int = 42) -> SeriesData:
    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=np.float64)
    values = np.sin(t / 50.0) + 0.15 * np.sin(t / 12.0) + 0.05 * rng.standard_normal(length)
    timestamps = [f"t_{i}" for i in range(length)]
    return SeriesData(values=values, timestamps=timestamps)


def load_series(
    *,
    dataset_name: str,
    path: str | None,
    target: str,
    time_col: str,
) -> SeriesData:
    if dataset_name == "synthetic":
        return generate_synthetic_series()

    if not path:
        raise ValueError(f"dataset.path required for dataset '{dataset_name}'")

    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    values: list[float] = []
    timestamps: list[str] = []
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {csv_path}")
        if target not in reader.fieldnames:
            raise ValueError(f"Target column '{target}' not in {reader.fieldnames}")
        for row in reader:
            values.append(float(row[target]))
            if time_col in row:
                timestamps.append(str(row[time_col]))
    return SeriesData(values=np.asarray(values, dtype=np.float64), timestamps=timestamps or None)


def resolve_cutoff_index(series: SeriesData, cutoff: str) -> int:
    """Map cutoff string to an index in the series (integer index or timestamp match)."""
    if cutoff.isdigit():
        idx = int(cutoff)
        if idx <= 0 or idx >= len(series.values):
            raise ValueError(f"Cutoff index {idx} out of range for series length {len(series.values)}")
        return idx

    if series.timestamps:
        try:
            return series.timestamps.index(cutoff)
        except ValueError as exc:
            raise ValueError(f"Cutoff timestamp '{cutoff}' not found in series") from exc

    raise ValueError(f"Cannot resolve cutoff '{cutoff}' without timestamps or integer index")


def normalize_series(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(values.mean())
    std = float(values.std()) or 1.0
    return (values - mean) / std, mean, std


def denormalize(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return values * std + mean


def build_window_dataset(
    values: np.ndarray,
    *,
    context_length: int,
    prediction_length: int,
    end_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows using series[:end_index] for training."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    max_start = end_index - context_length - prediction_length
    if max_start < 1:
        raise ValueError(
            f"Not enough history before cutoff index {end_index} "
            f"(need {context_length + prediction_length}, have {end_index})"
        )
    for start in range(0, max_start):
        ctx_end = start + context_length
        tgt_end = ctx_end + prediction_length
        xs.append(values[start:ctx_end])
        ys.append(values[ctx_end:tgt_end])
    return np.stack(xs), np.stack(ys)


def holdout_context_and_target(
    values: np.ndarray,
    *,
    cutoff_index: int,
    context_length: int,
    prediction_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    ctx_start = cutoff_index - context_length
    if ctx_start < 0:
        raise ValueError(f"Cutoff {cutoff_index} too early for context_length {context_length}")
    tgt_end = cutoff_index + prediction_length
    if tgt_end > len(values):
        raise ValueError(f"Not enough future values after cutoff {cutoff_index}")
    context = values[ctx_start:cutoff_index]
    target = values[cutoff_index:tgt_end]
    return context, target
