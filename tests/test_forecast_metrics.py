from __future__ import annotations

import numpy as np
import pytest

from trainextend.forecasting.metrics import compute_metrics, mae, mse, smape


def test_mae_mse_smape():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.0, 2.5])
    assert mae(y_true, y_pred) == pytest.approx(1 / 3)
    assert mse(y_true, y_pred) == pytest.approx(1 / 6)
    assert smape(y_true, y_pred) > 0


def test_compute_metrics_subset():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.1, 2.9])
    out = compute_metrics(y_true, y_pred, ["mse", "mae"])
    assert "mse" in out and "mae" in out
    assert "smape" not in out
