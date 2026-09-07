"""Unit tests for CVXPY battery arbitrage convex optimizer."""

import numpy as np
import pytest

from flexa.data.loader import load_electricity_prices
from flexa.optimization.battery import BatteryAsset, BatteryConfig
from flexa.optimization.objectives import (
    CapacityReservationObjective,
    DayAheadArbitrageObjective,
    DegradationPenaltyObjective,
)
from flexa.optimization.optimizer import BatteryOptimizer


@pytest.fixture
def synthetic_24h_prices() -> np.ndarray:
    """Generate 24h prices with clear off-peak valley and evening peak."""
    # Hours 0-5: low (20 EUR/MWh)
    # Hours 6-16: medium (60 EUR/MWh)
    # Hours 17-21: high evening peak (180 EUR/MWh)
    # Hours 22-23: low (30 EUR/MWh)
    prices = [20.0] * 6 + [60.0] * 11 + [180.0] * 5 + [30.0] * 2
    return np.array(prices, dtype=np.float64)


def test_optimizer_basic_arbitrage(synthetic_24h_prices: np.ndarray) -> None:
    """Test basic arbitrage on 1 MW / 1 MWh battery with empty start and end."""
    b = BatteryConfig(
        capacity_mwh=1.0,
        max_charge_power_mw=1.0,
        max_discharge_power_mw=1.0,
        charging_efficiency=0.95,
        discharging_efficiency=0.95,
        min_soc_mwh=0.0,
    )
    opt = BatteryOptimizer(battery=b)
    res = opt.optimize_horizon(
        prices=synthetic_24h_prices,
        empty_at_end=True,
    )

    assert res.is_optimal
    assert res.total_profit_eur > 0
    assert pytest.approx(res.terminal_soc_mwh, abs=1e-5) == 0.0

    # Verify battery charges during low prices (0-5) and discharges during peak (17-21)
    ch = res.schedule["charge_mw"].to_numpy()
    dis = res.schedule["discharge_mw"].to_numpy()
    assert np.sum(ch[:6]) > 0.0
    assert np.sum(dis[17:22]) > 0.0


def test_empty_at_end_of_day_constraint(synthetic_24h_prices: np.ndarray) -> None:
    """Verify empty-at-end-of-day condition holds strictly."""
    b = BatteryConfig(
        capacity_mwh=2.0,
        max_charge_power_mw=1.0,
        max_discharge_power_mw=1.0,
        min_soc_mwh=0.0,
    )
    opt = BatteryOptimizer(battery=b)
    res = opt.optimize_horizon(
        prices=synthetic_24h_prices,
        initial_soc_mwh=0.0,
        empty_at_end=True,
    )

    assert res.is_optimal
    assert pytest.approx(res.terminal_soc_mwh, abs=1e-5) == 0.0


def test_hard_physical_constraints(synthetic_24h_prices: np.ndarray) -> None:
    """Verify power limits, SOC limits, and energy conservation dynamics."""
    cap = 1.5
    p_max = 0.75
    eta_ch = 0.92
    eta_dis = 0.90
    b = BatteryConfig(
        capacity_mwh=cap,
        max_charge_power_mw=p_max,
        max_discharge_power_mw=p_max,
        charging_efficiency=eta_ch,
        discharging_efficiency=eta_dis,
        min_soc_mwh=0.0,
    )
    opt = BatteryOptimizer(battery=b)
    res = opt.optimize_horizon(prices=synthetic_24h_prices, empty_at_end=True)

    ch = res.schedule["charge_mw"].to_numpy()
    dis = res.schedule["discharge_mw"].to_numpy()
    soc = res.schedule["soc_mwh"].to_numpy()
    term_soc = res.terminal_soc_mwh

    # Power bounds
    assert np.all(ch >= -1e-6)
    assert np.all(ch <= p_max + 1e-6)
    assert np.all(dis >= -1e-6)
    assert np.all(dis <= p_max + 1e-6)

    # SOC bounds
    assert np.all(soc >= -1e-6)
    assert np.all(soc <= cap + 1e-6)
    assert term_soc >= -1e-6
    assert term_soc <= cap + 1e-6

    # Conservation of energy dynamics across steps
    all_soc = np.append(soc, term_soc)
    for t in range(len(ch)):
        expected_next = all_soc[t] + (eta_ch * ch[t] - (1.0 / eta_dis) * dis[t])
        assert pytest.approx(all_soc[t + 1], abs=1e-5) == expected_next


def test_cycling_constraint() -> None:
    """Test optional daily cycling limit: total(ch + dis) / (2 * cap) <= max_cycles."""
    # Create volatile price pattern with multiple swings
    volatile_prices = np.array(
        [10, 150, 10, 150, 10, 150, 10, 150, 10, 150, 10, 150] * 2,
        dtype=np.float64,
    )
    b = BatteryConfig(
        capacity_mwh=1.0,
        max_charge_power_mw=1.0,
        max_discharge_power_mw=1.0,
        charging_efficiency=0.98,
        discharging_efficiency=0.98,
        min_soc_mwh=0.0,
    )
    opt = BatteryOptimizer(battery=b)

    # Unconstrained
    res_unconstrained = opt.optimize_horizon(volatile_prices, max_cycles=None, empty_at_end=True)

    # Constrained to 1.0 cycle per day
    res_constrained_1 = opt.optimize_horizon(volatile_prices, max_cycles=1.0, empty_at_end=True)

    # Constrained to 0.5 cycle per day
    res_constrained_half = opt.optimize_horizon(volatile_prices, max_cycles=0.5, empty_at_end=True)

    assert res_unconstrained.cycles_used > 1.0
    assert res_constrained_1.cycles_used <= 1.0 + 1e-5
    assert res_constrained_half.cycles_used <= 0.5 + 1e-5

    # Monotonicity of profit with respect to cycle constraints
    assert res_unconstrained.total_profit_eur >= res_constrained_1.total_profit_eur - 1e-5
    assert res_constrained_1.total_profit_eur >= res_constrained_half.total_profit_eur - 1e-5


def test_compare_1mw_1mwh_vs_1mw_2mwh(synthetic_24h_prices: np.ndarray) -> None:
    """Compare 1 MW / 1 MWh vs 1 MW / 2 MWh battery."""
    b_1mwh = BatteryConfig(
        capacity_mwh=1.0,
        max_charge_power_mw=1.0,
        max_discharge_power_mw=1.0,
    )
    b_2mwh = BatteryConfig(
        capacity_mwh=2.0,
        max_charge_power_mw=1.0,
        max_discharge_power_mw=1.0,
    )

    opt_1mwh = BatteryOptimizer(battery=b_1mwh)
    opt_2mwh = BatteryOptimizer(battery=b_2mwh)

    res_1 = opt_1mwh.optimize_horizon(synthetic_24h_prices, empty_at_end=True)
    res_2 = opt_2mwh.optimize_horizon(synthetic_24h_prices, empty_at_end=True)

    assert res_1.is_optimal and res_2.is_optimal
    # 2 MWh battery can store 2 hours of energy, capturing more peak spread
    assert res_2.total_profit_eur >= res_1.total_profit_eur
    assert res_2.total_discharge_mwh >= res_1.total_discharge_mwh


def test_optimize_day_by_day_real_prices() -> None:
    """Test sequential day-by-day optimization on real market prices."""
    # Load 1 week of electricity prices
    prices_df = load_electricity_prices(
        start_date="2022-01-01 00:00:00",
        end_date="2022-01-07 23:00:00",
    )

    b = BatteryConfig(
        capacity_mwh=1.0,
        max_charge_power_mw=1.0,
        max_discharge_power_mw=1.0,
    )
    opt = BatteryOptimizer(battery=b)

    res = opt.optimize_day_by_day(
        prices_df=prices_df,
        empty_at_end_of_day=True,
        max_cycles_per_day=1.0,
    )

    assert res.n_days == 7
    assert res.daily_summaries.height == 7
    assert res.hourly_schedule.height == 168
    assert res.total_profit_eur > 0
    assert res.avg_cycles_per_day <= 1.0 + 1e-5
    # Verify each day has optimal status
    assert all(s in ("optimal", "optimal_inaccurate") for s in res.daily_summaries["status"])


def test_battery_asset_build_constraints() -> None:
    """Test that BatteryAsset builds self-contained constraints directly."""
    b = BatteryAsset(
        name="AssetTest",
        capacity_mwh=2.0,
        max_charge_power_mw=1.0,
        max_discharge_power_mw=1.0,
        min_soc_mwh=0.1,
    )
    b.init_variables(horizon=24)
    constraints = b.build_constraints(
        dt_hours=1.0,
        initial_soc_mwh=0.5,
        empty_at_end=True,
        max_cycles=1.5,
    )

    # Power bounds (2) + SOC bounds (2) + initial SOC (1) + dynamics (24) + terminal (1) + cycling (1) = 31 constraints
    assert len(constraints) == 31


def test_multi_battery_shared_grid_limit(synthetic_24h_prices: np.ndarray) -> None:
    """Test co-optimizing two batteries sharing an aggregate grid export limit."""
    b1 = BatteryAsset(
        name="Bat_A",
        capacity_mwh=1.0,
        max_charge_power_mw=1.0,
        max_discharge_power_mw=1.0,
    )
    b2 = BatteryAsset(
        name="Bat_B",
        capacity_mwh=1.0,
        max_charge_power_mw=1.0,
        max_discharge_power_mw=1.0,
    )

    # Shared grid bottleneck of 1.2 MW total export (individual sum is 2.0 MW)
    opt = BatteryOptimizer(
        batteries=[b1, b2],
        max_grid_export_mw=1.2,
    )

    res = opt.optimize_horizon(prices=synthetic_24h_prices, empty_at_end=True)
    assert res.is_optimal

    # Check that individual battery schedules exist
    assert res.battery_schedules is not None
    assert "Bat_A" in res.battery_schedules
    assert "Bat_B" in res.battery_schedules

    sched_a = res.battery_schedules["Bat_A"]
    sched_b = res.battery_schedules["Bat_B"]

    # Check shared grid limit is strictly respected at every hour
    combined_discharge = sched_a["discharge_mw"] + sched_b["discharge_mw"]
    assert np.all(combined_discharge.to_numpy() <= 1.2 + 1e-5)


def test_modular_market_objectives(synthetic_24h_prices: np.ndarray) -> None:
    """Test explicit composition of modular market objectives."""
    b = BatteryAsset(
        name="Bat_Modular",
        capacity_mwh=1.0,
        max_charge_power_mw=1.0,
        max_discharge_power_mw=1.0,
        degradation_cost_per_mwh=5.0,
    )

    # Compose 3 separate market objectives: Day-Ahead, Degradation, and Capacity Reservation
    reserve_prices = np.full(24, 10.0)  # 10 EUR/MW-h reserve availability payment
    objectives = [
        DayAheadArbitrageObjective(prices=synthetic_24h_prices),
        DegradationPenaltyObjective(),
        CapacityReservationObjective(reserve_prices=reserve_prices, reserved_power_mw=0.2),
    ]

    opt = BatteryOptimizer(battery=b, objectives=objectives)
    res = opt.optimize_horizon(empty_at_end=True)

    assert res.is_optimal
    assert res.objective_breakdown is not None
    assert "DayAheadArbitrage" in res.objective_breakdown
    assert "DegradationPenalty" in res.objective_breakdown
    assert "CapacityReservation" in res.objective_breakdown

    # Capacity reservation revenue: 24h * 10 EUR/MW-h * 0.2 MW = 48 EUR
    assert pytest.approx(res.objective_breakdown["CapacityReservation"]["total"]) == 48.0
    # Degradation penalty should be negative or zero
    assert res.objective_breakdown["DegradationPenalty"]["total"] <= 0.0
    # Net total matches the sum of objectives
    expected_sum = (
        res.objective_breakdown["DayAheadArbitrage"]["total"]
        + res.objective_breakdown["DegradationPenalty"]["total"]
        + res.objective_breakdown["CapacityReservation"]["total"]
    )
    assert pytest.approx(res.total_profit_eur) == expected_sum
