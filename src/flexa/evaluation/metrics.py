"""Statistical and probabilistic evaluation metrics for time series forecasts."""

from __future__ import annotations

import numpy as np


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error (MAE)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error (RMSE)."""
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def mean_absolute_percentage_error(
    y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8
) -> float:
    """Mean Absolute Percentage Error (MAPE)."""
    denom = np.maximum(np.abs(y_true), epsilon)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def symmetric_mean_absolute_percentage_error(
    y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8
) -> float:
    """Symmetric Mean Absolute Percentage Error (sMAPE)."""
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, epsilon)
    return float(np.mean(np.abs(y_pred - y_true) / denom) * 100.0)


def weighted_absolute_percentage_error(
    y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8
) -> float:
    """Weighted Absolute Percentage Error (WAPE)."""
    total_abs_error = np.sum(np.abs(y_true - y_pred))
    total_actual = np.maximum(np.sum(np.abs(y_true)), epsilon)
    return float((total_abs_error / total_actual) * 100.0)


def pinball_loss(y_true: np.ndarray, y_pred_q: np.ndarray, q: float) -> float:
    """Quantile (Pinball) Loss for probabilistic forecasts.

    Args:
        y_true: Actual ground truth observations.
        y_pred_q: Predicted values for quantile q.
        q: Target quantile in (0, 1).
    """
    diff = y_true - y_pred_q
    loss = np.maximum(q * diff, (q - 1.0) * diff)
    return float(np.mean(loss))


def evaluate_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantiles_dict: dict[float, np.ndarray] | None = None,
) -> dict[str, float]:
    """Compute comprehensive point and quantile forecasting metrics.

    Args:
        y_true: Actual ground truth observations.
        y_pred: Point forecasts.
        quantiles_dict: Optional mapping of quantile levels to predicted values.

    Returns:
        Dictionary mapping metric names to computed numerical values.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    metrics = {
        "mae": mean_absolute_error(yt, yp),
        "rmse": root_mean_squared_error(yt, yp),
        "mape": mean_absolute_percentage_error(yt, yp),
        "smape": symmetric_mean_absolute_percentage_error(yt, yp),
        "wape": weighted_absolute_percentage_error(yt, yp),
    }

    if quantiles_dict:
        for q, q_pred in quantiles_dict.items():
            metrics[f"pinball_q_{int(q * 100):02d}"] = pinball_loss(yt, np.asarray(q_pred), q)

    return metrics
