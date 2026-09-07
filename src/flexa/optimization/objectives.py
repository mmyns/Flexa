"""Market objective abstractions and concrete implementations for battery optimization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import cvxpy as cp
import numpy as np

if TYPE_CHECKING:
    from flexa.optimization.battery import BatteryAsset


class BaseMarketObjective(ABC):
    """Abstract base class for electricity market objectives in convex optimization.

    Encapsulates market economics, revenue/cost mathematical formulations in CVXPY,
    and post-solve financial accounting across single or multiple battery assets.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def build_expression(
        self,
        batteries: list[BatteryAsset],
        dt_hours: float,
    ) -> cp.Expression:
        """Construct the CVXPY mathematical expression to be MAXIMIZED.

        Args:
            batteries: List of active BatteryAsset instances with initialized CVXPY variables.
            dt_hours: Duration of each interval in hours.

        Returns:
            A linear or concave CVXPY expression representing net profit (rewards minus costs).
        """
        ...

    @abstractmethod
    def calculate_breakdown(
        self,
        batteries: list[BatteryAsset],
        dt_hours: float,
    ) -> dict[str, float]:
        """Calculate numerical financial values for this objective after solver completion.

        Args:
            batteries: List of active BatteryAsset instances with solved decision variable values.
            dt_hours: Duration of each interval in hours.

        Returns:
            A dictionary mapping asset identifiers or summary keys to amounts in EUR.
        """
        ...


class DayAheadArbitrageObjective(BaseMarketObjective):
    """Spot / Day-Ahead energy market price arbitrage objective.

    Revenue from discharging into the grid minus energy purchase cost from charging:
        Profit = sum_t [ price[t] * sum_b (p_dis[b, t] - p_ch[b, t]) * dt ]
    """

    def __init__(
        self,
        prices: np.ndarray | list[float],
        name: str = "DayAheadArbitrage",
    ) -> None:
        super().__init__(name=name)
        self.prices = np.asarray(prices, dtype=np.float64)

    def build_expression(
        self,
        batteries: list[BatteryAsset],
        dt_hours: float,
    ) -> cp.Expression:
        """Build CVXPY expression for net spot power arbitrage profit."""
        total_expr: cp.Expression = cp.Constant(0.0)
        for b in batteries:
            assert b.p_charge is not None and b.p_discharge is not None
            net_power = b.p_discharge - b.p_charge
            total_expr += cp.sum(cp.multiply(self.prices, net_power) * dt_hours)
        return total_expr

    def calculate_breakdown(
        self,
        batteries: list[BatteryAsset],
        dt_hours: float,
    ) -> dict[str, float]:
        """Calculate net arbitrage earnings per battery."""
        breakdown: dict[str, float] = {}
        total = 0.0
        for b in batteries:
            assert b.p_charge is not None and b.p_charge.value is not None
            assert b.p_discharge is not None and b.p_discharge.value is not None
            ch = np.maximum(b.p_charge.value, 0.0)
            dis = np.maximum(b.p_discharge.value, 0.0)
            net_power = dis - ch
            profit = float(np.sum(net_power * self.prices * dt_hours))
            breakdown[b.name] = profit
            total += profit
        breakdown["total"] = total
        return breakdown


class DegradationPenaltyObjective(BaseMarketObjective):
    """Battery wear and cycle aging degradation cost penalty.

    Penalizes total energy throughput across the battery cells:
        Cost = - sum_b [ c_deg[b] * sum_t (p_ch[b, t] + p_dis[b, t]) * dt ]
    """

    def __init__(self, name: str = "DegradationPenalty") -> None:
        super().__init__(name=name)

    def build_expression(
        self,
        batteries: list[BatteryAsset],
        dt_hours: float,
    ) -> cp.Expression:
        """Build CVXPY linear penalty expression for cell throughput degradation."""
        penalty_expr: cp.Expression = cp.Constant(0.0)
        for b in batteries:
            c_deg = b.config.degradation_cost_per_mwh
            if c_deg > 0:
                assert b.p_charge is not None and b.p_discharge is not None
                throughput = cp.sum(b.p_charge + b.p_discharge) * dt_hours
                penalty_expr -= c_deg * throughput
        return penalty_expr

    def calculate_breakdown(
        self,
        batteries: list[BatteryAsset],
        dt_hours: float,
    ) -> dict[str, float]:
        """Calculate degradation expense per battery."""
        breakdown: dict[str, float] = {}
        total = 0.0
        for b in batteries:
            c_deg = b.config.degradation_cost_per_mwh
            if c_deg > 0:
                assert b.p_charge is not None and b.p_charge.value is not None
                assert b.p_discharge is not None and b.p_discharge.value is not None
                ch = np.maximum(b.p_charge.value, 0.0)
                dis = np.maximum(b.p_discharge.value, 0.0)
                cost = float(c_deg * np.sum(ch + dis) * dt_hours)
            else:
                cost = 0.0
            breakdown[b.name] = -cost
            total -= cost
        breakdown["total"] = total
        return breakdown


class CapacityReservationObjective(BaseMarketObjective):
    """Ancillary / capacity market reserve commitment objective.

    Rewards reserving symmetric or asymmetric power capacity (e.g. FCR, aFRR):
        Revenue = sum_t [ capacity_price[t] * reserved_power_mw * dt ]
    """

    def __init__(
        self,
        reserve_prices: np.ndarray | list[float],
        reserved_power_mw: float,
        name: str = "CapacityReservation",
    ) -> None:
        super().__init__(name=name)
        self.reserve_prices = np.asarray(reserve_prices, dtype=np.float64)
        self.reserved_power_mw = float(reserved_power_mw)

    def build_expression(
        self,
        batteries: list[BatteryAsset],
        dt_hours: float,
    ) -> cp.Expression:
        """Capacity revenue (constant reward added to the objective)."""
        rev = float(np.sum(self.reserve_prices * self.reserved_power_mw * dt_hours))
        return cp.Constant(rev)

    def calculate_breakdown(
        self,
        batteries: list[BatteryAsset],
        dt_hours: float,
    ) -> dict[str, float]:
        rev = float(np.sum(self.reserve_prices * self.reserved_power_mw * dt_hours))
        return {"total": rev}
