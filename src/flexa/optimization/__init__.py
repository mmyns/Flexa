"""Convex optimization models and battery arbitrage engine powered by CVXPY."""

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
from flexa.optimization.optimizer import (
    BatteryOptimizer,
    DayByDayResult,
    OptimizationResult,
)

__all__ = [
    "BaseMarketObjective",
    "Battery",
    "BatteryAsset",
    "BatteryConfig",
    "BatteryModel",
    "BatteryOptimizer",
    "CapacityReservationObjective",
    "DayAheadArbitrageObjective",
    "DayByDayResult",
    "DegradationPenaltyObjective",
    "OptimizationResult",
]
