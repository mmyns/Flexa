"""Forecasting models including scikit-learn tabular baselines and Chronos-2 foundation models."""

from typing import TYPE_CHECKING, Any

from flexa.models.base import BaseForecaster, ForecastResult
from flexa.models.baseline import BaselineForecaster, RidgeForecaster
from flexa.models.naive import NaiveForecaster, TMinus1Forecaster

if TYPE_CHECKING:
    from flexa.models.chronos import ChronosForecaster


def __getattr__(name: str) -> Any:
    """Import ChronosForecaster lazily so torch stays an optional dependency."""
    if name == "ChronosForecaster":
        try:
            from flexa.models.chronos import ChronosForecaster
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ImportError(
                "ChronosForecaster requires the optional 'chronos' extra "
                "(torch + chronos-forecasting). Install it with: "
                "uv sync --extra chronos"
            ) from exc
        return ChronosForecaster
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseForecaster",
    "BaselineForecaster",
    "ChronosForecaster",
    "ForecastResult",
    "NaiveForecaster",
    "RidgeForecaster",
    "TMinus1Forecaster",
]
