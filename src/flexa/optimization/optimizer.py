"""Battery arbitrage convex optimizer powered by CVXPY with multi-battery and multi-market OOP support."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import cvxpy as cp
import numpy as np
import polars as pl

from flexa.optimization.battery import BatteryAsset, BatteryConfig
from flexa.optimization.objectives import (
    BaseMarketObjective,
    DayAheadArbitrageObjective,
    DegradationPenaltyObjective,
)


@dataclass
class OptimizationResult:
    """Result of a single-horizon battery convex optimization run."""

    schedule: pl.DataFrame
    total_profit_eur: float
    total_charge_mwh: float
    total_discharge_mwh: float
    cycles_used: float
    terminal_soc_mwh: float
    solver_status: str
    solve_time_ms: float
    battery_config: BatteryConfig
    battery_schedules: dict[str, pl.DataFrame] | None = None
    objective_breakdown: dict[str, dict[str, float]] | None = None

    @property
    def is_optimal(self) -> bool:
        """Whether the solver achieved an optimal status."""
        return self.solver_status in ("optimal", "optimal_inaccurate")


@dataclass
class DayByDayResult:
    """Aggregated results from day-by-day battery optimization."""

    daily_summaries: pl.DataFrame
    hourly_schedule: pl.DataFrame
    total_profit_eur: float
    total_charge_mwh: float
    total_discharge_mwh: float
    total_cycles: float
    avg_cycles_per_day: float
    n_days: int
    battery_config: BatteryConfig
    battery_schedules: dict[str, pl.DataFrame] | None = None


class BatteryOptimizer:
    """Object-oriented battery optimizer supporting multiple battery assets and decoupled market objectives."""

    def __init__(
        self,
        battery: BatteryAsset | BatteryConfig | None = None,
        batteries: list[BatteryAsset | BatteryConfig] | None = None,
        objectives: list[BaseMarketObjective] | None = None,
        max_grid_export_mw: float | None = None,
        max_grid_import_mw: float | None = None,
        solver: str | None = None,
    ) -> None:
        """Initialize optimizer with battery asset(s) and market objectives.

        Args:
            battery: A single battery asset or configuration.
            batteries: A list of battery assets or configurations for portfolio dispatch.
            objectives: Optional list of market objectives to optimize.
            max_grid_export_mw: Optional aggregate grid export interconnection power limit in MW.
            max_grid_import_mw: Optional aggregate grid import interconnection power limit in MW.
            solver: Preferred CVXPY solver (e.g. 'CLARABEL', 'HIGHS', 'OSQP').
        """
        raw_batteries: list[BatteryAsset | BatteryConfig] = []
        if batteries is not None:
            raw_batteries.extend(batteries)
        elif battery is not None:
            raw_batteries.append(battery)
        else:
            raise ValueError("Must provide either 'battery' or 'batteries'.")

        self.batteries: list[BatteryAsset] = [
            b if isinstance(b, BatteryAsset) else b.to_asset() for b in raw_batteries
        ]
        # Reference to primary/first battery for single-battery ergonomics
        self.battery = self.batteries[0]
        self.objectives = objectives
        self.max_grid_export_mw = max_grid_export_mw
        self.max_grid_import_mw = max_grid_import_mw
        self.solver = solver or self._get_best_solver()

    @staticmethod
    def _get_best_solver() -> str:
        """Select the best available LP solver."""
        installed = cp.installed_solvers()
        for candidate in ["CLARABEL", "HIGHS", "OSQP", "SCS", "SCIPY"]:
            if candidate in installed:
                return candidate
        return installed[0] if installed else "CLARABEL"

    def optimize_horizon(
        self,
        prices: np.ndarray | list[float] | None = None,
        timestamps: list[datetime] | None = None,
        horizon: int | None = None,
        dt_hours: float = 1.0,
        initial_soc_mwh: float = 0.0,
        empty_at_end: bool = True,
        target_final_soc_mwh: float | None = None,
        max_cycles: float | None = None,
        objectives: list[BaseMarketObjective] | None = None,
    ) -> OptimizationResult:
        """Optimize battery dispatch over a single horizon.

        Assembles constraints from each battery asset (power limits, energy balance,
        terminal SOC, cycling limits) and maximizes the sum of all active market objectives.

        Args:
            prices: Optional spot electricity prices. If provided and objectives is None,
                    automatically instantiates DayAheadArbitrageObjective and DegradationPenaltyObjective.
            timestamps: Optional list of timestamps.
            horizon: Optional explicit number of intervals in the optimization horizon.
            dt_hours: Interval length in hours (default 1.0).
            initial_soc_mwh: Starting State of Charge in MWh (float, default 0.0).
            empty_at_end: If True, enforces empty batteries at horizon end (e[T] == 0).
            target_final_soc_mwh: Explicit terminal SOC target in MWh (float or None).
            max_cycles: Optional cycling limit (float or None).
            objectives: Explicit market objectives overriding self.objectives.

        Returns:
            An OptimizationResult containing the dispatch schedule, metrics, and breakdowns.
        """
        # Assemble active market objectives
        active_objectives: list[BaseMarketObjective] = []
        if objectives is not None:
            active_objectives.extend(objectives)
        elif self.objectives is not None:
            active_objectives.extend(self.objectives)
        elif prices is not None:
            active_objectives.append(DayAheadArbitrageObjective(prices=prices))
            active_objectives.append(DegradationPenaltyObjective())
        else:
            raise ValueError("No market objectives specified and no prices provided.")

        # Determine horizon length
        if horizon is not None:
            n_steps = horizon
        elif prices is not None:
            n_steps = len(prices)
        elif timestamps is not None:
            n_steps = len(timestamps)
        else:
            # Infer horizon from active objectives
            inferred = None
            for obj in active_objectives:
                if hasattr(obj, "prices") and obj.prices is not None:
                    inferred = len(obj.prices)
                    break
                elif hasattr(obj, "reserve_prices") and obj.reserve_prices is not None:
                    inferred = len(obj.reserve_prices)
                    break
            if inferred is not None:
                n_steps = inferred
            else:
                raise ValueError(
                    "Must provide 'prices', 'timestamps', or 'horizon' to define horizon length."
                )

        if n_steps == 0:
            raise ValueError("Horizon length cannot be 0.")

        t_start = time.perf_counter()

        # 1. Initialize CVXPY variables on every battery
        for b in self.batteries:
            b.init_variables(horizon=n_steps)

        # 2. Collect internal physical and operational constraints from each battery
        constraints: list[cp.Constraint] = []
        for b in self.batteries:
            b_constraints = b.build_constraints(
                dt_hours=dt_hours,
                initial_soc_mwh=initial_soc_mwh,
                empty_at_end=empty_at_end,
                target_final_soc_mwh=target_final_soc_mwh,
                max_cycles=max_cycles,
            )
            constraints.extend(b_constraints)

        # 3. Add portfolio-level interconnection power constraints (if configured)
        if self.max_grid_export_mw is not None:
            dis_vars = [b.p_discharge for b in self.batteries if b.p_discharge is not None]
            if dis_vars:
                total_dis = cp.sum(dis_vars, axis=0)
                constraints.append(total_dis <= self.max_grid_export_mw)

        if self.max_grid_import_mw is not None:
            ch_vars = [b.p_charge for b in self.batteries if b.p_charge is not None]
            if ch_vars:
                total_ch = cp.sum(ch_vars, axis=0)
                constraints.append(total_ch <= self.max_grid_import_mw)

        # 4. Compile objective expression across all active market objectives
        total_expression: cp.Expression = cp.Constant(0.0)
        for obj in active_objectives:
            total_expression += obj.build_expression(batteries=self.batteries, dt_hours=dt_hours)

        problem = cp.Problem(cp.Maximize(total_expression), constraints)
        problem.solve(solver=self.solver)

        solve_time_ms = (time.perf_counter() - t_start) * 1000.0

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(
                f"Optimization failed with status '{problem.status}'. Check constraints."
            )

        # 5. Extract schedules and calculate objective breakdown
        battery_schedules: dict[str, pl.DataFrame] = {}
        total_ch_mwh = 0.0
        total_dis_mwh = 0.0
        agg_ch = np.zeros(n_steps, dtype=np.float64)
        agg_dis = np.zeros(n_steps, dtype=np.float64)
        agg_soc = np.zeros(n_steps, dtype=np.float64)

        for b in self.batteries:
            b_df = b.extract_schedule(dt_hours=dt_hours, timestamps=timestamps)
            battery_schedules[b.name] = b_df
            ch = b_df["charge_mw"].to_numpy()
            dis = b_df["discharge_mw"].to_numpy()
            soc = b_df["soc_mwh"].to_numpy()
            agg_ch += ch
            agg_dis += dis
            agg_soc += soc
            total_ch_mwh += float(np.sum(ch) * dt_hours)
            total_dis_mwh += float(np.sum(dis) * dt_hours)

        # Objective financial breakdowns
        breakdowns: dict[str, dict[str, float]] = {}
        total_profit = 0.0
        for obj in active_objectives:
            bd = obj.calculate_breakdown(batteries=self.batteries, dt_hours=dt_hours)
            breakdowns[obj.name] = bd
            total_profit += bd.get("total", 0.0)

        # Build master schedule DataFrame
        net_power = agg_dis - agg_ch
        portfolio_cap = sum(b.capacity_mwh for b in self.batteries)
        cycles = (total_ch_mwh + total_dis_mwh) / (2.0 * portfolio_cap)

        schedule_dict: dict[str, Any] = {}
        if timestamps is not None and len(timestamps) == n_steps:
            schedule_dict["timestamp"] = timestamps
        else:
            schedule_dict["step"] = list(range(n_steps))

        display_prices = prices
        if display_prices is None:
            for obj in active_objectives:
                if isinstance(obj, DayAheadArbitrageObjective):
                    display_prices = obj.prices
                    break

        if display_prices is not None:
            price_arr = np.asarray(display_prices, dtype=np.float64)
            schedule_dict["price_eur_per_mwh"] = price_arr
            schedule_dict["cashflow_eur"] = net_power * price_arr * dt_hours
        else:
            schedule_dict["cashflow_eur"] = [total_profit / n_steps] * n_steps

        schedule_dict.update(
            {
                "charge_mw": agg_ch,
                "discharge_mw": agg_dis,
                "net_power_mw": net_power,
                "soc_mwh": agg_soc,
                "soc_ratio": agg_soc / portfolio_cap,
            }
        )
        master_schedule = pl.DataFrame(schedule_dict)

        # Terminal SOC of primary battery or total terminal SOC
        term_soc = sum(
            float(b.soc.value[-1])
            for b in self.batteries
            if b.soc is not None and b.soc.value is not None
        )

        return OptimizationResult(
            schedule=master_schedule,
            total_profit_eur=total_profit,
            total_charge_mwh=total_ch_mwh,
            total_discharge_mwh=total_dis_mwh,
            cycles_used=cycles,
            terminal_soc_mwh=term_soc,
            solver_status=problem.status,
            solve_time_ms=solve_time_ms,
            battery_config=self.battery.config,
            battery_schedules=battery_schedules,
            objective_breakdown=breakdowns,
        )

    def optimize_day_by_day(
        self,
        prices_df: pl.DataFrame,
        time_column: str = "timestamp",
        price_column: str = "price_eur_per_mwh",
        initial_soc_mwh: float = 0.0,
        empty_at_end_of_day: bool = True,
        max_cycles_per_day: float | None = None,
    ) -> DayByDayResult:
        """Perform sequential day-by-day battery optimization across calendar days.

        Args:
            prices_df: Polars DataFrame containing [time_column, price_column].
            time_column: Name of timestamp column.
            price_column: Name of price column.
            initial_soc_mwh: Starting SOC at start of each day.
            empty_at_end_of_day: If True, enforces empty battery at 23:59 end of each day.
            max_cycles_per_day: Optional maximum daily full equivalent cycles constraint.

        Returns:
            A DayByDayResult containing daily summaries and complete hourly schedules.
        """
        df = prices_df.with_columns(pl.col(time_column).dt.date().alias("_date")).sort(time_column)

        unique_dates: list[date] = df["_date"].unique().sort().to_list()

        daily_rows: list[dict[str, Any]] = []
        all_schedules: list[pl.DataFrame] = []
        battery_all_schedules: dict[str, list[pl.DataFrame]] = {b.name: [] for b in self.batteries}

        total_profit = 0.0
        total_charge = 0.0
        total_discharge = 0.0
        total_cycles = 0.0

        for current_date in unique_dates:
            day_df = df.filter(pl.col("_date") == current_date)
            day_prices = day_df[price_column].to_numpy()
            day_timestamps = day_df[time_column].to_list()

            res = self.optimize_horizon(
                prices=day_prices,
                timestamps=day_timestamps,
                dt_hours=1.0,
                initial_soc_mwh=initial_soc_mwh,
                empty_at_end=empty_at_end_of_day,
                max_cycles=max_cycles_per_day,
            )

            p_min = float(np.min(day_prices))
            p_max = float(np.max(day_prices))
            p_avg = float(np.mean(day_prices))

            daily_rows.append(
                {
                    "date": current_date,
                    "profit_eur": res.total_profit_eur,
                    "charge_mwh": res.total_charge_mwh,
                    "discharge_mwh": res.total_discharge_mwh,
                    "cycles": res.cycles_used,
                    "avg_price_eur_per_mwh": p_avg,
                    "price_spread_eur_per_mwh": p_max - p_min,
                    "status": res.solver_status,
                }
            )

            all_schedules.append(res.schedule)
            if res.battery_schedules is not None:
                for b_name, b_sched in res.battery_schedules.items():
                    battery_all_schedules[b_name].append(b_sched)

            total_profit += res.total_profit_eur
            total_charge += res.total_charge_mwh
            total_discharge += res.total_discharge_mwh
            total_cycles += res.cycles_used

        summary_df = pl.DataFrame(daily_rows)
        combined_schedule = pl.concat(all_schedules)
        combined_b_schedules = {
            b_name: pl.concat(schedules)
            for b_name, schedules in battery_all_schedules.items()
            if len(schedules) > 0
        }

        n_days = len(unique_dates)
        avg_cycles = total_cycles / n_days if n_days > 0 else 0.0

        return DayByDayResult(
            daily_summaries=summary_df,
            hourly_schedule=combined_schedule,
            total_profit_eur=total_profit,
            total_charge_mwh=total_charge,
            total_discharge_mwh=total_discharge,
            total_cycles=total_cycles,
            avg_cycles_per_day=avg_cycles,
            n_days=n_days,
            battery_config=self.battery.config,
            battery_schedules=combined_b_schedules if combined_b_schedules else None,
        )
