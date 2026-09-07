"""Pydantic data models for battery storage systems."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import cvxpy as cp
import numpy as np
import polars as pl
from pydantic import BaseModel, Field, model_validator


class BatteryConfig(BaseModel):
    """Data model representing a Battery Energy Storage System (BESS) configuration.

    This model defines the physical, operational, and financial parameters
    of an electrical battery for use in mathematical modeling and convex optimization.
    """

    name: str = Field(default="Battery", description="Name or identifier of the battery asset.")
    capacity_mwh: float = Field(
        gt=0.0,
        description="Total nominal energy storage capacity in megawatt-hours (MWh).",
    )
    max_charge_power_mw: float = Field(
        gt=0.0,
        description="Maximum charging power rate in megawatts (MW).",
    )
    max_discharge_power_mw: float = Field(
        gt=0.0,
        description="Maximum discharging power rate in megawatts (MW).",
    )
    charging_efficiency: float = Field(
        default=0.95,
        gt=0.0,
        le=1.0,
        description="One-way charging efficiency η_ch in (0, 1].",
    )
    discharging_efficiency: float = Field(
        default=0.95,
        gt=0.0,
        le=1.0,
        description="One-way discharging efficiency η_dis in (0, 1].",
    )
    min_soc_mwh: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum allowable State of Charge (SOC) in MWh to prevent deep discharge.",
    )
    max_soc_mwh: float | None = Field(
        default=None,
        ge=0.0,
        description="Maximum allowable State of Charge (SOC) in MWh. Defaults to capacity_mwh if None.",
    )
    initial_soc_mwh: float | None = Field(
        default=None,
        ge=0.0,
        description="Initial State of Charge (SOC) in MWh at t=0. Defaults to 50% capacity if None.",
    )
    target_final_soc_mwh: float | None = Field(
        default=None,
        ge=0.0,
        description="Optional minimum/target State of Charge in MWh at the end of the optimization horizon.",
    )
    degradation_cost_per_mwh: float = Field(
        default=0.0,
        ge=0.0,
        description="Marginal degradation or cycle aging cost per MWh of throughput in EUR/MWh.",
    )
    max_cycles_per_day: float | None = Field(
        default=None,
        gt=0.0,
        description="Optional maximum daily equivalent full cycles (total throughput / (2 * capacity)).",
    )

    @model_validator(mode="after")
    def _validate_and_set_defaults(self) -> BatteryConfig:
        """Validate energy capacity limits and set sensible default values."""
        # 1. Default max_soc_mwh to total capacity
        if self.max_soc_mwh is None:
            self.max_soc_mwh = self.capacity_mwh

        # 2. Validate SOC boundaries
        if self.min_soc_mwh > self.capacity_mwh:
            raise ValueError(
                f"min_soc_mwh ({self.min_soc_mwh:.3f} MWh) cannot exceed capacity_mwh ({self.capacity_mwh:.3f} MWh)."
            )
        if self.max_soc_mwh > self.capacity_mwh:
            raise ValueError(
                f"max_soc_mwh ({self.max_soc_mwh:.3f} MWh) cannot exceed capacity_mwh ({self.capacity_mwh:.3f} MWh)."
            )
        if self.min_soc_mwh > self.max_soc_mwh:
            raise ValueError(
                f"min_soc_mwh ({self.min_soc_mwh:.3f} MWh) cannot exceed max_soc_mwh ({self.max_soc_mwh:.3f} MWh)."
            )

        # 3. Default initial_soc_mwh to 50% capacity (clamped between min and max SOC)
        if self.initial_soc_mwh is None:
            default_initial = 0.5 * self.capacity_mwh
            self.initial_soc_mwh = min(max(default_initial, self.min_soc_mwh), self.max_soc_mwh)
        else:
            if not (self.min_soc_mwh <= self.initial_soc_mwh <= self.max_soc_mwh):
                raise ValueError(
                    f"initial_soc_mwh ({self.initial_soc_mwh:.3f} MWh) must be between "
                    f"min_soc_mwh ({self.min_soc_mwh:.3f} MWh) and max_soc_mwh ({self.max_soc_mwh:.3f} MWh)."
                )

        # 4. Validate target final SOC if provided
        if self.target_final_soc_mwh is not None:
            if not (self.min_soc_mwh <= self.target_final_soc_mwh <= self.max_soc_mwh):
                raise ValueError(
                    f"target_final_soc_mwh ({self.target_final_soc_mwh:.3f} MWh) must be between "
                    f"min_soc_mwh ({self.min_soc_mwh:.3f} MWh) and max_soc_mwh ({self.max_soc_mwh:.3f} MWh)."
                )

        return self

    @property
    def round_trip_efficiency(self) -> float:
        """Round-trip AC-to-AC or DC-to-DC efficiency (η_ch * η_dis)."""
        return self.charging_efficiency * self.discharging_efficiency

    @property
    def duration_hours(self) -> float:
        """Duration rating in hours: capacity divided by maximum discharge power."""
        return self.capacity_mwh / self.max_discharge_power_mw

    @property
    def c_rate_charge(self) -> float:
        """C-rate for charging (1/hours): max charging power divided by capacity."""
        return self.max_charge_power_mw / self.capacity_mwh

    @property
    def c_rate_discharge(self) -> float:
        """C-rate for discharging (1/hours): max discharging power divided by capacity."""
        return self.max_discharge_power_mw / self.capacity_mwh

    @property
    def usable_capacity_mwh(self) -> float:
        """Usable energy capacity within allowable SOC window (max_soc_mwh - min_soc_mwh)."""
        assert self.max_soc_mwh is not None
        return self.max_soc_mwh - self.min_soc_mwh

    @property
    def initial_soc_ratio(self) -> float:
        """Initial state of charge as a fraction of total capacity [0, 1]."""
        assert self.initial_soc_mwh is not None
        return self.initial_soc_mwh / self.capacity_mwh

    @property
    def min_soc_ratio(self) -> float:
        """Minimum state of charge as a fraction of total capacity [0, 1]."""
        return self.min_soc_mwh / self.capacity_mwh

    @property
    def max_soc_ratio(self) -> float:
        """Maximum state of charge as a fraction of total capacity [0, 1]."""
        assert self.max_soc_mwh is not None
        return self.max_soc_mwh / self.capacity_mwh

    @classmethod
    def from_ratios(
        cls,
        capacity_mwh: float,
        max_power_mw: float | None = None,
        max_charge_power_mw: float | None = None,
        max_discharge_power_mw: float | None = None,
        charging_efficiency: float = 0.95,
        discharging_efficiency: float = 0.95,
        min_soc_ratio: float = 0.0,
        max_soc_ratio: float = 1.0,
        initial_soc_ratio: float = 0.5,
        target_final_soc_ratio: float | None = None,
        degradation_cost_per_mwh: float = 0.0,
        max_cycles_per_day: float | None = None,
        name: str = "Battery",
    ) -> BatteryConfig:
        """Factory constructor specifying state-of-charge limits as ratios [0, 1].

        Args:
            capacity_mwh: Total nominal storage capacity in MWh.
            max_power_mw: Symmetrical max charge/discharge power in MW (if separate powers are omitted).
            max_charge_power_mw: Max charge power in MW (defaults to max_power_mw).
            max_discharge_power_mw: Max discharge power in MW (defaults to max_power_mw).
            charging_efficiency: One-way charge efficiency.
            discharging_efficiency: One-way discharge efficiency.
            min_soc_ratio: Minimum SOC fraction (e.g. 0.05 for 5%).
            max_soc_ratio: Maximum SOC fraction (e.g. 0.95 for 95%).
            initial_soc_ratio: Initial SOC fraction at t=0 (e.g. 0.5 for 50%).
            target_final_soc_ratio: Target final SOC fraction at end of horizon.
            degradation_cost_per_mwh: Degradation cost in EUR/MWh.
            max_cycles_per_day: Optional maximum daily full equivalent cycles.
            name: Battery asset name.

        Returns:
            An initialized BatteryConfig instance.
        """
        charge_power = max_charge_power_mw if max_charge_power_mw is not None else max_power_mw
        discharge_power = (
            max_discharge_power_mw if max_discharge_power_mw is not None else max_power_mw
        )

        if charge_power is None or discharge_power is None:
            raise ValueError(
                "Must provide either 'max_power_mw' or both 'max_charge_power_mw' and 'max_discharge_power_mw'."
            )

        target_final_soc_mwh = (
            target_final_soc_ratio * capacity_mwh if target_final_soc_ratio is not None else None
        )

        return cls(
            name=name,
            capacity_mwh=capacity_mwh,
            max_charge_power_mw=charge_power,
            max_discharge_power_mw=discharge_power,
            charging_efficiency=charging_efficiency,
            discharging_efficiency=discharging_efficiency,
            min_soc_mwh=min_soc_ratio * capacity_mwh,
            max_soc_mwh=max_soc_ratio * capacity_mwh,
            initial_soc_mwh=initial_soc_ratio * capacity_mwh,
            target_final_soc_mwh=target_final_soc_mwh,
            degradation_cost_per_mwh=degradation_cost_per_mwh,
            max_cycles_per_day=max_cycles_per_day,
        )

    def to_asset(self) -> BatteryAsset:
        """Convert this configuration into an active object-oriented BatteryAsset instance."""
        return BatteryAsset(config=self)


class BatteryAsset:
    """Object-oriented physical battery asset.

    Encapsulates CVXPY decision variables, physical state-of-charge dynamics,
    power rating bounds, terminal state conditions, and cycling limits.
    """

    def __init__(self, config: BatteryConfig | None = None, **kwargs: Any) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = BatteryConfig(**kwargs)

        # Optimization decision variables
        self.p_charge: cp.Variable | None = None
        self.p_discharge: cp.Variable | None = None
        self.soc: cp.Variable | None = None

    @property
    def name(self) -> str:
        """Battery asset name or identifier."""
        return self.config.name

    @property
    def capacity_mwh(self) -> float:
        """Total energy storage capacity in MWh."""
        return self.config.capacity_mwh

    @property
    def max_charge_power_mw(self) -> float:
        """Maximum charging power in MW."""
        return self.config.max_charge_power_mw

    @property
    def max_discharge_power_mw(self) -> float:
        """Maximum discharging power in MW."""
        return self.config.max_discharge_power_mw

    @property
    def charging_efficiency(self) -> float:
        """One-way charge efficiency."""
        return self.config.charging_efficiency

    @property
    def discharging_efficiency(self) -> float:
        """One-way discharge efficiency."""
        return self.config.discharging_efficiency

    @property
    def min_soc_mwh(self) -> float:
        """Minimum allowable state of charge in MWh."""
        return self.config.min_soc_mwh

    @property
    def max_soc_mwh(self) -> float:
        """Maximum allowable state of charge in MWh."""
        return self.config.max_soc_mwh or self.config.capacity_mwh

    def init_variables(self, horizon: int) -> None:
        """Initialize CVXPY decision variables for a specific horizon length.

        Args:
            horizon: Number of time intervals in the optimization horizon.
        """
        self.p_charge = cp.Variable(horizon, nonneg=True, name=f"{self.name}_p_charge")
        self.p_discharge = cp.Variable(horizon, nonneg=True, name=f"{self.name}_p_discharge")
        self.soc = cp.Variable(horizon + 1, name=f"{self.name}_soc")

    def build_constraints(
        self,
        dt_hours: float = 1.0,
        initial_soc_mwh: float = 0.0,
        empty_at_end: bool = True,
        target_final_soc_mwh: float | None = None,
        max_cycles: float | None = None,
    ) -> list[cp.Constraint]:
        """Build all physical and operational hard constraints for this battery.

        Args:
            dt_hours: Interval length in hours (default: 1.0).
            initial_soc_mwh: Initial state of charge at t=0.
            empty_at_end: If True, enforces empty battery at end of horizon.
            target_final_soc_mwh: Explicit terminal SOC target (overrides empty_at_end).
            max_cycles: Optional maximum full equivalent cycles limit.

        Returns:
            A list of CVXPY constraints governing this battery asset.
        """
        if self.p_charge is None or self.p_discharge is None or self.soc is None:
            raise RuntimeError("Variables not initialized. Call init_variables(horizon) first.")

        n_steps = self.p_charge.shape[0]
        cap = self.capacity_mwh
        e_min = self.min_soc_mwh
        e_max = self.max_soc_mwh
        eta_ch = self.charging_efficiency
        eta_dis = self.discharging_efficiency

        constraints: list[cp.Constraint] = [
            self.p_charge <= self.max_charge_power_mw,
            self.p_discharge <= self.max_discharge_power_mw,
            self.soc >= e_min,
            self.soc <= e_max,
        ]

        # Initial SOC
        init_soc = max(min(initial_soc_mwh, e_max), e_min)
        constraints.append(self.soc[0] == init_soc)

        # Energy dynamics (conservation of energy)
        for t in range(n_steps):
            constraints.append(
                self.soc[t + 1]
                == self.soc[t]
                + (eta_ch * self.p_charge[t] - (1.0 / eta_dis) * self.p_discharge[t]) * dt_hours
            )

        # Terminal SOC (empty at end of day)
        if target_final_soc_mwh is not None:
            final_target = max(min(target_final_soc_mwh, e_max), e_min)
            constraints.append(self.soc[n_steps] == final_target)
        elif empty_at_end:
            constraints.append(self.soc[n_steps] == e_min)

        # Optional cycling constraint
        eff_max_cycles = max_cycles if max_cycles is not None else self.config.max_cycles_per_day
        if eff_max_cycles is not None and eff_max_cycles > 0:
            total_throughput = cp.sum(self.p_charge + self.p_discharge) * dt_hours
            constraints.append(total_throughput <= 2.0 * cap * eff_max_cycles)

        return constraints

    def extract_schedule(
        self,
        dt_hours: float = 1.0,
        timestamps: list[datetime] | None = None,
    ) -> pl.DataFrame:
        """Extract optimal dispatch schedule as a Polars DataFrame.

        Args:
            dt_hours: Time step in hours.
            timestamps: Optional list of timestamps.

        Returns:
            A Polars DataFrame containing the hourly schedule for this battery asset.
        """
        if (
            self.p_charge is None
            or self.p_charge.value is None
            or self.p_discharge is None
            or self.p_discharge.value is None
            or self.soc is None
            or self.soc.value is None
        ):
            raise RuntimeError(
                "Cannot extract schedule: optimization problem has not been solved yet."
            )

        n_steps = len(self.p_charge.value)
        ch = np.maximum(self.p_charge.value, 0.0)
        dis = np.maximum(self.p_discharge.value, 0.0)
        soc_vals = np.clip(self.soc.value[:-1], self.min_soc_mwh, self.max_soc_mwh)
        net_power = dis - ch

        data: dict[str, Any] = {
            "battery_name": [self.name] * n_steps,
            "charge_mw": ch,
            "discharge_mw": dis,
            "net_power_mw": net_power,
            "soc_mwh": soc_vals,
            "soc_ratio": soc_vals / self.capacity_mwh,
        }
        if timestamps is not None and len(timestamps) == n_steps:
            data["timestamp"] = timestamps
        else:
            data["step"] = list(range(n_steps))

        return pl.DataFrame(data)


# Convenient aliases for battery model
BatteryModel = BatteryConfig
Battery = BatteryConfig
