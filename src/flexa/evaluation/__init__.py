"""Evaluation metrics and backtesting engine for time series forecasting."""

from flexa.evaluation.backtest import Backtester, BacktestFoldResult, BacktestSummary
from flexa.evaluation.metrics import (
    evaluate_forecast,
    mean_absolute_error,
    mean_absolute_percentage_error,
    pinball_loss,
    root_mean_squared_error,
    symmetric_mean_absolute_percentage_error,
    weighted_absolute_percentage_error,
)

__all__ = [
    "BacktestFoldResult",
    "BacktestSummary",
    "Backtester",
    "evaluate_forecast",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "pinball_loss",
    "root_mean_squared_error",
    "symmetric_mean_absolute_percentage_error",
    "weighted_absolute_percentage_error",
]
