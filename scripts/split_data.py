"""Split raw energy measurements and weather time series into train and test sets.

Splits:
- Training set: September 2021 to December 2022 (inclusive)
- Test set: January 2023 to May 2023 (inclusive)

Outputs saved to data/processed/ in both Parquet and CSV formats.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import polars as pl
from rich.console import Console
from rich.table import Table

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
console = Console()


def split_datasets(
    raw_dir: Path | str = "data/raw",
    processed_dir: Path | str = "data/processed",
    cutoff_date: str = "2023-01-01 00:00:00",
) -> None:
    """Split raw measurement and weather datasets by date cutoff."""
    raw_path = Path(raw_dir)
    proc_path = Path(processed_dir)
    proc_path.mkdir(parents=True, exist_ok=True)

    cutoff_dt = datetime.fromisoformat(cutoff_date)

    console.print("[bold cyan]====================================================[/bold cyan]")
    console.print("[bold cyan]  Flexa Dataset Split: Train / Test Generation      [/bold cyan]")
    console.print(f"[bold cyan]  Cutoff Date: {cutoff_date}                      [/bold cyan]")
    console.print("[bold cyan]====================================================[/bold cyan]\n")

    # 1. Process 1_measurements.csv
    meas_file = raw_path / "1_measurements.csv"
    if not meas_file.exists():
        raise FileNotFoundError(f"Missing raw measurements file: {meas_file}")

    console.print("[yellow]Loading and splitting 1_measurements.csv...[/yellow]")
    df_meas = (
        pl.read_csv(meas_file, try_parse_dates=False)
        .drop("")
        .with_columns(pl.col("datetime_utc").str.to_datetime("%Y-%m-%d %H:%M:%S"))
    )

    meas_train = df_meas.filter(pl.col("datetime_utc") < cutoff_dt).sort(["pool", "datetime_utc"])
    meas_test = df_meas.filter(pl.col("datetime_utc") >= cutoff_dt).sort(["pool", "datetime_utc"])

    # Save measurements
    meas_train.write_parquet(proc_path / "train_measurements.parquet")
    meas_train.write_csv(proc_path / "train_measurements.csv")
    meas_test.write_parquet(proc_path / "test_measurements.parquet")
    meas_test.write_csv(proc_path / "test_measurements.csv")

    # 2. Process 1_weather.csv
    weath_file = raw_path / "1_weather.csv"
    if not weath_file.exists():
        raise FileNotFoundError(f"Missing raw weather file: {weath_file}")

    console.print("[yellow]Loading and splitting 1_weather.csv...[/yellow]")
    df_weath = pl.read_csv(weath_file, try_parse_dates=False).with_columns(
        pl.col("forecast_datetime_utc").str.to_datetime("%Y-%m-%d %H:%M:%S")
    )

    weath_train = df_weath.filter(pl.col("forecast_datetime_utc") < cutoff_dt).sort(
        "forecast_datetime_utc"
    )
    weath_test = df_weath.filter(pl.col("forecast_datetime_utc") >= cutoff_dt).sort(
        "forecast_datetime_utc"
    )

    # Save weather
    weath_train.write_parquet(proc_path / "train_weather.parquet")
    weath_train.write_csv(proc_path / "train_weather.csv")
    weath_test.write_parquet(proc_path / "test_weather.parquet")
    weath_test.write_csv(proc_path / "test_weather.csv")

    # 3. Summary Table
    table = Table(title="Generated Dataset Splits Summary")
    table.add_column("Dataset", style="cyan", no_wrap=True)
    table.add_column("Split", style="bold")
    table.add_column("Rows", justify="right")
    table.add_column("Start Date", justify="center")
    table.add_column("End Date", justify="center")
    table.add_column("Unique Timestamps", justify="right")

    table.add_row(
        "Measurements",
        "[green]Train[/green]",
        f"{len(meas_train):,}",
        meas_train["datetime_utc"].min().strftime("%Y-%m-%d %H:%M"),
        meas_train["datetime_utc"].max().strftime("%Y-%m-%d %H:%M"),
        f"{meas_train['datetime_utc'].n_unique():,}",
    )
    table.add_row(
        "Measurements",
        "[magenta]Test[/magenta]",
        f"{len(meas_test):,}",
        meas_test["datetime_utc"].min().strftime("%Y-%m-%d %H:%M"),
        meas_test["datetime_utc"].max().strftime("%Y-%m-%d %H:%M"),
        f"{meas_test['datetime_utc'].n_unique():,}",
    )
    table.add_section()
    table.add_row(
        "Weather",
        "[green]Train[/green]",
        f"{len(weath_train):,}",
        weath_train["forecast_datetime_utc"].min().strftime("%Y-%m-%d %H:%M"),
        weath_train["forecast_datetime_utc"].max().strftime("%Y-%m-%d %H:%M"),
        f"{weath_train['forecast_datetime_utc'].n_unique():,}",
    )
    table.add_row(
        "Weather",
        "[magenta]Test[/magenta]",
        f"{len(weath_test):,}",
        weath_test["forecast_datetime_utc"].min().strftime("%Y-%m-%d %H:%M"),
        weath_test["forecast_datetime_utc"].max().strftime("%Y-%m-%d %H:%M"),
        f"{weath_test['forecast_datetime_utc'].n_unique():,}",
    )

    console.print(table)
    console.print(
        f"\n[bold green]✓ Successfully generated files in: {proc_path.resolve()}[/bold green]\n"
    )


if __name__ == "__main__":
    split_datasets()
