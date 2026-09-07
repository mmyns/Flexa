"""Pytest fixtures for time series testing."""

from __future__ import annotations

from unittest.mock import MagicMock

import polars as pl
import pytest
import torch

from flexa.config import PipelineConfig
from flexa.data.loader import generate_synthetic_timeseries


@pytest.fixture
def synthetic_df() -> pl.DataFrame:
    """Generate a 2-series synthetic Polars DataFrame for testing."""
    return generate_synthetic_timeseries(n_series=2, length=90, seed=42)


@pytest.fixture
def single_series_df() -> pl.DataFrame:
    """Generate a single-series synthetic Polars DataFrame."""
    return generate_synthetic_timeseries(n_series=1, length=60, seed=123)


@pytest.fixture
def sample_config() -> PipelineConfig:
    """Provide a lightweight testing configuration."""
    config = PipelineConfig()
    config.features.lags = [1, 2, 7]
    config.features.rolling_windows = [7]
    config.features.rolling_metrics = ["mean", "std"]
    config.chronos.prediction_length = 7
    config.chronos.context_length = 30
    config.chronos.num_samples = 5
    config.backtest.n_splits = 2
    config.backtest.test_size = 7
    config.backtest.stride = 7
    return config


@pytest.fixture
def mock_chronos_pipeline():
    """Mock Chronos pipeline producing realistic probabilistic forecast tensors."""
    mock_pipe = MagicMock()

    def mock_predict_quantiles(inputs, prediction_length, quantile_levels, **kwargs):
        batch_size = len(inputs) if isinstance(inputs, list) else 1
        num_q = len(quantile_levels)
        # mean shape: (batch_size, prediction_length)
        mean_tensor = torch.full((batch_size, prediction_length), 50.0)
        # quantiles shape: (batch_size, prediction_length, num_quantiles)
        quantiles_tensor = torch.zeros((batch_size, prediction_length, num_q))
        for q_idx, q in enumerate(quantile_levels):
            quantiles_tensor[:, :, q_idx] = 50.0 + (q - 0.5) * 10.0
        return quantiles_tensor, mean_tensor

    def mock_predict(inputs, prediction_length, **kwargs):
        batch_size = len(inputs) if isinstance(inputs, list) else 1
        num_samples = 10
        return torch.randn((batch_size, num_samples, prediction_length)) + 50.0

    mock_pipe.predict_quantiles.side_effect = mock_predict_quantiles
    mock_pipe.predict.side_effect = mock_predict
    return mock_pipe
