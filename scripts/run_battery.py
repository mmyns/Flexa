"""Day-by-day battery convex optimization script comparing 1h and 2h duration assets.

Runs the convex optimizer twice:
1. 1h Duration Battery: 1 MW / 1 MWh
2. 2h Duration Battery: 1 MW / 2 MWh

Returns:
- Profits for both battery configurations (total EUR and uplift)
- Hourly charging, discharging, and SOC schedules for both configurations
- Exports schedules to CSV and Parquet in data/processed/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from flexa.data.loader import load_electricity_prices
from flexa.optimization.battery import BatteryConfig
from flexa.optimization.optimizer import BatteryOptimizer


def run_battery_comparison(
    n_days: int | None = None,
    max_cycles: float | None = None,
    empty_at_end_of_day: bool = True,
    data_path: Path | None = None,
    output_dir: Path | str = "data/processed",
    save_schedules: bool = True,
    verbose: bool = True,
) -> tuple[dict[str, float], dict[str, pl.DataFrame]]:
    """Run optimizer for 1h (1 MW / 1 MWh) and 2h (1 MW / 2 MWh) batteries.

    Args:
        n_days: Number of days to optimize over (default None = all available days).
        max_cycles: Optional daily cycling cap (e.g. 1.0 cycle/day for warranty limits).
        empty_at_end_of_day: Whether the battery must return to 0 SOC at 23:59 (default True).
        data_path: Optional custom path to spot electricity prices.
        output_dir: Directory where schedule CSV and Parquet files are saved.
        save_schedules: Whether to save schedules to disk.
        verbose: Whether to print rich console summaries and tables.

    Returns:
        tuple[dict[str, float], dict[str, pl.DataFrame]]:
            - profits: dictionary of total profits and uplift percentage.
            - schedules: dictionary mapping '1h' and '2h' to their hourly Polars schedule DataFrames.
    """
    console = Console()
    output_path = Path(output_dir)
    if save_schedules:
        output_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        horizon_desc = f"{n_days} days" if n_days else "Full dataset"
        cap_desc = f"Max {max_cycles:.1f} cyc/d" if max_cycles else "Unconstrained"
        console.print(
            Panel.fit(
                "[bold cyan]⚡ Flexa Battery Convex Optimizer Benchmark (1h vs. 2h)[/bold cyan]\n"
                f"[dim]Horizon: Day-by-Day ({horizon_desc}) | Terminal SOC: Empty (0 MWh) | Warranty: {cap_desc}[/dim]",
                border_style="cyan",
            )
        )

    # 1. Load spot electricity prices
    prices_df = load_electricity_prices(path=data_path)
    if n_days is not None:
        unique_dates = prices_df["timestamp"].dt.date().unique().sort().head(n_days).to_list()
        filtered_prices = prices_df.filter(prices_df["timestamp"].dt.date().is_in(unique_dates))
    else:
        unique_dates = prices_df["timestamp"].dt.date().unique().sort().to_list()
        filtered_prices = prices_df

    if verbose:
        console.print(
            f"[green]✓[/green] Loaded [bold]{filtered_prices.height:,}[/bold] hourly spot price intervals "
            f"across [bold]{len(unique_dates)}[/bold] calendar days."
        )

    # 2. Define Battery Assets
    b_1h = BatteryConfig(
        name="1h Duration (1 MW / 1 MWh)",
        capacity_mwh=1.0,
        max_charge_power_mw=1.0,
        max_discharge_power_mw=1.0,
        charging_efficiency=0.95,
        discharging_efficiency=0.95,
        min_soc_mwh=0.0,
    )

    b_2h = BatteryConfig(
        name="2h Duration (1 MW / 2 MWh)",
        capacity_mwh=2.0,
        max_charge_power_mw=1.0,
        max_discharge_power_mw=1.0,
        charging_efficiency=0.95,
        discharging_efficiency=0.95,
        min_soc_mwh=0.0,
    )

    opt_1h = BatteryOptimizer(battery=b_1h)
    opt_2h = BatteryOptimizer(battery=b_2h)

    # 3. Execute Optimization 1: 1h Battery (1 MW / 1 MWh)
    if verbose:
        console.print("\n[bold yellow]1/2 Solving 1h Duration Battery (1 MW / 1 MWh)...[/bold yellow]")
    res_1h = opt_1h.optimize_day_by_day(
        prices_df=filtered_prices,
        empty_at_end_of_day=empty_at_end_of_day,
        max_cycles_per_day=max_cycles,
    )

    # 4. Execute Optimization 2: 2h Battery (1 MW / 2 MWh)
    if verbose:
        console.print("[bold yellow]2/2 Solving 2h Duration Battery (1 MW / 2 MWh)...[/bold yellow]")
    res_2h = opt_2h.optimize_day_by_day(
        prices_df=filtered_prices,
        empty_at_end_of_day=empty_at_end_of_day,
        max_cycles_per_day=max_cycles,
    )

    # 5. Extract Financial Metrics
    p1 = res_1h.total_profit_eur
    p2 = res_2h.total_profit_eur
    uplift_eur = p2 - p1
    uplift_pct = (uplift_eur / p1) * 100.0 if p1 > 0 else 0.0

    profits: dict[str, float] = {
        "1h_profit_eur": p1,
        "2h_profit_eur": p2,
        "duration_uplift_eur": uplift_eur,
        "duration_uplift_pct": uplift_pct,
        "1h_avg_cycles_per_day": res_1h.avg_cycles_per_day,
        "2h_avg_cycles_per_day": res_2h.avg_cycles_per_day,
        "n_days": float(len(unique_dates)),
    }

    schedules: dict[str, pl.DataFrame] = {
        "1h": res_1h.hourly_schedule,
        "2h": res_2h.hourly_schedule,
    }

    # 6. Save Schedules to Disk
    if save_schedules:
        path_1h_csv = output_path / "schedule_1h_battery.csv"
        path_2h_csv = output_path / "schedule_2h_battery.csv"
        path_1h_parquet = output_path / "schedule_1h_battery.parquet"
        path_2h_parquet = output_path / "schedule_2h_battery.parquet"

        res_1h.hourly_schedule.write_csv(path_1h_csv)
        res_2h.hourly_schedule.write_csv(path_2h_csv)
        res_1h.hourly_schedule.write_parquet(path_1h_parquet)
        res_2h.hourly_schedule.write_parquet(path_2h_parquet)

    # 7. Print Rich Console Tables
    if verbose:
        # Comparison Table
        table = Table(
            title=f"\n📊 Battery Arbitrage Comparison ({len(unique_dates)} Days)",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Asset Configuration", style="bold white", justify="left")
        table.add_column("Duration", justify="center")
        table.add_column("Total Profit (€)", justify="right", style="green")
        table.add_column("Profit / MWh-cap (€)", justify="right")
        table.add_column("Total Cycled (MWh)", justify="right")
        table.add_column("Avg Cycles / Day", justify="right")
        table.add_column("Duration Uplift", justify="right", style="cyan")

        table.add_row(
            "1 MW / 1 MWh",
            "1-hour",
            f"€{p1:,.2f}",
            f"€{(p1 / 1.0):,.2f}",
            f"{(res_1h.total_charge_mwh + res_1h.total_discharge_mwh):,.1f} MWh",
            f"{res_1h.avg_cycles_per_day:.2f}",
            "-",
        )
        table.add_row(
            "1 MW / 2 MWh",
            "2-hour",
            f"€{p2:,.2f}",
            f"€{(p2 / 2.0):,.2f}",
            f"{(res_2h.total_charge_mwh + res_2h.total_discharge_mwh):,.1f} MWh",
            f"{res_2h.avg_cycles_per_day:.2f}",
            f"+{uplift_pct:.1f}% (+€{uplift_eur:,.2f})",
        )
        console.print(table)

        # Preview Table of Charging Schedules (First 24 hours side-by-side)
        preview_table = Table(
            title="\n⏱️ Charging & Discharging Schedule Preview (First 24 Hours)",
            show_header=True,
            header_style="bold blue",
        )
        preview_table.add_column("Timestamp", style="dim", justify="left")
        preview_table.add_column("Spot Price (€)", justify="right")
        preview_table.add_column("1h Chg (MW)", justify="right", style="blue")
        preview_table.add_column("1h Dis (MW)", justify="right", style="magenta")
        preview_table.add_column("1h SOC (MWh)", justify="right")
        preview_table.add_column("2h Chg (MW)", justify="right", style="cyan")
        preview_table.add_column("2h Dis (MW)", justify="right", style="red")
        preview_table.add_column("2h SOC (MWh)", justify="right")

        sched_1h_24 = res_1h.hourly_schedule.head(24)
        sched_2h_24 = res_2h.hourly_schedule.head(24)

        for i in range(min(24, sched_1h_24.height)):
            ts_str = str(sched_1h_24["timestamp"][i])[:16]
            price = sched_1h_24["price_eur_per_mwh"][i]
            ch1 = sched_1h_24["charge_mw"][i]
            dis1 = sched_1h_24["discharge_mw"][i]
            soc1 = sched_1h_24["soc_mwh"][i]
            ch2 = sched_2h_24["charge_mw"][i]
            dis2 = sched_2h_24["discharge_mw"][i]
            soc2 = sched_2h_24["soc_mwh"][i]

            preview_table.add_row(
                ts_str,
                f"€{price:.1f}",
                f"{ch1:.2f}" if ch1 > 0.01 else "-",
                f"{dis1:.2f}" if dis1 > 0.01 else "-",
                f"{soc1:.2f}",
                f"{ch2:.2f}" if ch2 > 0.01 else "-",
                f"{dis2:.2f}" if dis2 > 0.01 else "-",
                f"{soc2:.2f}",
            )

        console.print(preview_table)

        # Export paths panel
        if save_schedules:
            console.print(
                Panel.fit(
                    f"[bold green]✓ Schedules successfully exported to disk:[/bold green]\n"
                    f"• 1h Battery Schedule CSV: [cyan]{output_path / 'schedule_1h_battery.csv'}[/cyan]\n"
                    f"• 2h Battery Schedule CSV: [cyan]{output_path / 'schedule_2h_battery.csv'}[/cyan]\n"
                    f"• 1h Battery Schedule Parquet: [dim]{output_path / 'schedule_1h_battery.parquet'}[/dim]\n"
                    f"• 2h Battery Schedule Parquet: [dim]{output_path / 'schedule_2h_battery.parquet'}[/dim]\n"
                    f"• Columns: [dim]timestamp, price_eur_per_mwh, charge_mw, discharge_mw, net_power_mw, soc_mwh, profit_eur[/dim]",
                    border_style="green",
                )
            )

    return profits, schedules


# Backward compatibility alias
run_benchmark = run_battery_comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run day-by-day battery arbitrage optimization for 1h and 2h batteries."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Number of days to optimize over (default: all available days in dataset).",
    )
    parser.add_argument(
        "--max-cycles",
        type=float,
        default=None,
        help="Optional daily cycling limit constraint (e.g. 1.0 for warranty limit).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Custom path to electricity prices file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to export schedule CSV/Parquet files (default: data/processed).",
    )

    args = parser.parse_args()
    data_p = Path(args.data_path) if args.data_path else None
    run_battery_comparison(
        n_days=args.days,
        max_cycles=args.max_cycles,
        data_path=data_p,
        output_dir=args.output_dir,
    )
