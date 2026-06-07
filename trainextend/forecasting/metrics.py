from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, 1.0, denom)
    return float(100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    names: list[str],
) -> dict[str, float]:
    fns = {"mae": mae, "mse": mse, "smape": smape}
    out: dict[str, float] = {}
    for name in names:
        fn = fns.get(name.lower())
        if fn is not None:
            out[name.lower()] = fn(y_true, y_pred)
    return out
