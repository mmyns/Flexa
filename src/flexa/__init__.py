"""Flexa: Production-ready time series forecasting with Polars, scikit-learn, and Chronos."""

from typing import TYPE_CHECKING, Any

from flexa.config import PipelineConfig
from flexa.data.loader import (
    generate_synthetic_timeseries,
    load_electricity_prices,
    load_timeseries,
)
from flexa.evaluation.metrics import evaluate_forecast
from flexa.models.base import BaseForecaster, ForecastResult
from flexa.models.baseline import BaselineForecaster
from flexa.optimization.battery import (
    Battery,
    BatteryAsset,
    BatteryConfig,
    BatteryModel,
)
from flexa.optimization.objectives import (
    BaseMarketObjective,
    CapacityReservationObjective,
    DayAheadArbitrageObjective,
    DegradationPenaltyObjective,
)
from flexa.optimization.optimizer import BatteryOptimizer, DayByDayResult, OptimizationResult
from flexa.pipeline import ForecastingPipeline

if TYPE_CHECKING:
    from flexa.models.chronos import ChronosForecaster


def __getattr__(name: str) -> Any:
    """Import ChronosForecaster lazily so torch stays an optional dependency."""
    if name == "ChronosForecaster":
        from flexa.models import ChronosForecaster

        return ChronosForecaster
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "0.1.0"

__all__ = [
    "BaseForecaster",
    "BaseMarketObjective",
    "BaselineForecaster",
    "Battery",
    "BatteryAsset",
    "BatteryConfig",
    "BatteryModel",
    "BatteryOptimizer",
    "CapacityReservationObjective",
    "ChronosForecaster",
    "DayAheadArbitrageObjective",
    "DayByDayResult",
    "DegradationPenaltyObjective",
    "ForecastResult",
    "ForecastingPipeline",
    "OptimizationResult",
    "PipelineConfig",
    "evaluate_forecast",
    "generate_synthetic_timeseries",
    "load_electricity_prices",
    "load_timeseries",
]
