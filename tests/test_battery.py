"""Unit tests for the BatteryConfig Pydantic model."""

import pytest
from pydantic import ValidationError

from flexa.optimization.battery import Battery, BatteryConfig, BatteryModel


def test_battery_config_defaults() -> None:
    """Test default values and property calculations for BatteryConfig."""
    b = BatteryConfig(
        capacity_mwh=2.0,
        max_charge_power_mw=1.0,
        max_discharge_power_mw=1.0,
    )

    assert b.name == "Battery"
    assert b.capacity_mwh == 2.0
    assert b.max_charge_power_mw == 1.0
    assert b.max_discharge_power_mw == 1.0
    assert b.charging_efficiency == 0.95
    assert b.discharging_efficiency == 0.95
    assert b.min_soc_mwh == 0.0
    assert b.max_soc_mwh == 2.0
    assert b.initial_soc_mwh == 1.0  # 50% of capacity
    assert b.target_final_soc_mwh is None
    assert b.degradation_cost_per_mwh == 0.0

    # Computed properties
    assert pytest.approx(b.round_trip_efficiency) == 0.95 * 0.95
    assert b.duration_hours == 2.0
    assert b.c_rate_charge == 0.5
    assert b.c_rate_discharge == 0.5
    assert b.usable_capacity_mwh == 2.0
    assert b.initial_soc_ratio == 0.5
    assert b.min_soc_ratio == 0.0
    assert b.max_soc_ratio == 1.0


def test_battery_model_aliases() -> None:
    """Test Battery and BatteryModel aliases."""
    assert Battery is BatteryConfig
    assert BatteryModel is BatteryConfig


def test_battery_from_ratios() -> None:
    """Test alternate constructor from_ratios."""
    b = BatteryConfig.from_ratios(
        name="Tesla_Megapack",
        capacity_mwh=3.9,
        max_power_mw=1.95,
        charging_efficiency=0.96,
        discharging_efficiency=0.96,
        min_soc_ratio=0.05,
        max_soc_ratio=0.95,
        initial_soc_ratio=0.20,
        target_final_soc_ratio=0.50,
        degradation_cost_per_mwh=5.0,
    )

    assert b.name == "Tesla_Megapack"
    assert b.capacity_mwh == 3.9
    assert b.max_charge_power_mw == 1.95
    assert b.max_discharge_power_mw == 1.95
    assert pytest.approx(b.min_soc_mwh) == 0.05 * 3.9
    assert pytest.approx(b.max_soc_mwh) == 0.95 * 3.9
    assert pytest.approx(b.initial_soc_mwh) == 0.20 * 3.9
    assert pytest.approx(b.target_final_soc_mwh) == 0.50 * 3.9
    assert b.degradation_cost_per_mwh == 5.0
    assert pytest.approx(b.usable_capacity_mwh) == 0.90 * 3.9


def test_battery_validation_capacity_positive() -> None:
    """Test that zero or negative capacity raises validation error."""
    with pytest.raises(ValidationError):
        BatteryConfig(
            capacity_mwh=0.0,
            max_charge_power_mw=1.0,
            max_discharge_power_mw=1.0,
        )

    with pytest.raises(ValidationError):
        BatteryConfig(
            capacity_mwh=-5.0,
            max_charge_power_mw=1.0,
            max_discharge_power_mw=1.0,
        )


def test_battery_validation_power_positive() -> None:
    """Test that zero or negative power limits raise validation error."""
    with pytest.raises(ValidationError):
        BatteryConfig(
            capacity_mwh=2.0,
            max_charge_power_mw=0.0,
            max_discharge_power_mw=1.0,
        )

    with pytest.raises(ValidationError):
        BatteryConfig(
            capacity_mwh=2.0,
            max_charge_power_mw=1.0,
            max_discharge_power_mw=-1.0,
        )


def test_battery_validation_efficiency_bounds() -> None:
    """Test that efficiencies outside (0, 1] raise validation error."""
    with pytest.raises(ValidationError):
        BatteryConfig(
            capacity_mwh=2.0,
            max_charge_power_mw=1.0,
            max_discharge_power_mw=1.0,
            charging_efficiency=1.05,
        )

    with pytest.raises(ValidationError):
        BatteryConfig(
            capacity_mwh=2.0,
            max_charge_power_mw=1.0,
            max_discharge_power_mw=1.0,
            discharging_efficiency=0.0,
        )


def test_battery_validation_soc_bounds() -> None:
    """Test that invalid SOC limits raise ValueError."""
    # min_soc > capacity
    with pytest.raises(ValueError, match="min_soc_mwh .* cannot exceed capacity_mwh"):
        BatteryConfig(
            capacity_mwh=2.0,
            max_charge_power_mw=1.0,
            max_discharge_power_mw=1.0,
            min_soc_mwh=2.5,
        )

    # min_soc > max_soc
    with pytest.raises(ValueError, match="min_soc_mwh .* cannot exceed max_soc_mwh"):
        BatteryConfig(
            capacity_mwh=2.0,
            max_charge_power_mw=1.0,
            max_discharge_power_mw=1.0,
            min_soc_mwh=1.5,
            max_soc_mwh=1.0,
        )

    # initial_soc out of bounds
    with pytest.raises(ValueError, match="initial_soc_mwh .* must be between"):
        BatteryConfig(
            capacity_mwh=2.0,
            max_charge_power_mw=1.0,
            max_discharge_power_mw=1.0,
            min_soc_mwh=0.5,
            max_soc_mwh=1.5,
            initial_soc_mwh=0.2,
        )

    # target_final_soc out of bounds
    with pytest.raises(ValueError, match="target_final_soc_mwh .* must be between"):
        BatteryConfig(
            capacity_mwh=2.0,
            max_charge_power_mw=1.0,
            max_discharge_power_mw=1.0,
            min_soc_mwh=0.5,
            max_soc_mwh=1.5,
            target_final_soc_mwh=1.8,
        )
