"""Tests for point and probabilistic evaluation metrics."""

import numpy as np
import pytest

from flexa.evaluation.metrics import (
    evaluate_forecast,
    mean_absolute_error,
    pinball_loss,
    root_mean_squared_error,
    weighted_absolute_percentage_error,
)


def test_standard_metrics_calculation():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 33.0, 37.0])
    # Errors: [2, 2, 3, 3]

    assert mean_absolute_error(y_true, y_pred) == 2.5
    assert pytest.approx(root_mean_squared_error(y_true, y_pred), rel=1e-4) == np.sqrt(
        (4 + 4 + 9 + 9) / 4
    )
    assert weighted_absolute_percentage_error(y_true, y_pred) == (10.0 / 100.0) * 100.0  # 10%


def test_pinball_loss_property():
    y_true = np.array([10.0, 10.0])
    # Case 1: Under-prediction (y > pred)
    y_pred_under = np.array([8.0, 8.0])
    # diff = 2.0. For q=0.9, loss = 0.9 * 2.0 = 1.8
    assert pytest.approx(pinball_loss(y_true, y_pred_under, q=0.9)) == 1.8

    # Case 2: Over-prediction (y < pred)
    y_pred_over = np.array([12.0, 12.0])
    # diff = -2.0. For q=0.9, loss = (0.9 - 1) * (-2.0) = 0.2
    assert pytest.approx(pinball_loss(y_true, y_pred_over, q=0.9)) == 0.2


def test_evaluate_forecast_wrapper():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([11.0, 19.0, 31.0])
    q_dict = {
        0.1: np.array([8.0, 16.0, 27.0]),
        0.9: np.array([13.0, 23.0, 34.0]),
    }

    metrics = evaluate_forecast(y_true, y_pred, quantiles_dict=q_dict)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "wape" in metrics
    assert "pinball_q_10" in metrics
    assert "pinball_q_90" in metrics
