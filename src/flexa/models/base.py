"""Abstract base forecaster and forecast result representations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import polars as pl


@dataclass
class ForecastResult:
    """Encapsulates deterministic point forecasts and probabilistic quantiles."""

    model_name: str
    prediction_length: int
    forecast_df: pl.DataFrame
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_polars(self) -> pl.DataFrame:
        """Return the complete forecast table with point predictions and quantiles."""
        return self.forecast_df


class BaseForecaster(ABC):
    """Abstract interface for all time series forecasting models."""

    def __init__(
        self,
        time_column: str = "timestamp",
        target_column: str = "target",
        id_column: str | None = "series_id",
    ) -> None:
        self.time_column = time_column
        self.target_column = target_column
        self.id_column = id_column

    @abstractmethod
    def fit(self, df: pl.DataFrame) -> BaseForecaster:
        """Fit the model to historical time series observations."""
        pass

    @abstractmethod
    def predict(
        self,
        context_df: pl.DataFrame,
        prediction_length: int,
        quantiles: list[float] | None = None,
    ) -> ForecastResult:
        """Generate forecasts for the specified horizon given historical context."""
        pass
