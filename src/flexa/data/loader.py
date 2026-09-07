"""High-performance time series data loaders powered by Polars."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl


def load_timeseries(
    path: str | Path,
    time_column: str = "timestamp",
    target_column: str = "target",
    id_column: str | None = None,
) -> pl.DataFrame:
    """Load a time series dataset from a CSV or Parquet file using Polars.

    Args:
        path: Path to the CSV or Parquet file.
        time_column: Name of the timestamp column.
        target_column: Name of the numerical target column.
        id_column: Name of the series identifier column (optional).

    Returns:
        A Polars DataFrame sorted by timestamp (and id_column if provided).
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix.lower() == ".parquet":
        df = pl.read_parquet(file_path)
    elif file_path.suffix.lower() in [".csv", ".txt"]:
        df = pl.read_csv(file_path, try_parse_dates=True)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Use .csv or .parquet.")

    if time_column not in df.columns:
        raise ValueError(f"Missing time column: '{time_column}' in dataset.")
    if target_column not in df.columns:
        raise ValueError(f"Missing target column: '{target_column}' in dataset.")

    # Ensure timestamp column is proper Datetime type
    if df[time_column].dtype not in (pl.Datetime, pl.Date):
        df = df.with_columns(pl.col(time_column).str.to_datetime())

    # Sort dataset
    sort_cols = [id_column, time_column] if id_column and id_column in df.columns else [time_column]
    df = df.sort(sort_cols)

    return df


def load_electricity_prices(
    path: str | Path | None = None,
    time_column: str = "timestamp",
    price_column: str = "price_eur_per_mwh",
    start_date: str | datetime | None = None,
    end_date: str | datetime | None = None,
) -> pl.DataFrame:
    """Load electricity spot/day-ahead market prices from the data directory.

    By default, resolves 'data/raw/2_electricity_prices.csv' from the repository.
    Parses timestamps and returns a standardized, sorted Polars DataFrame.

    Args:
        path: Path to the electricity prices CSV or Parquet file.
              If None, auto-discovers 'data/raw/2_electricity_prices.csv'.
        time_column: Desired name for the timestamp column in output DataFrame.
        price_column: Desired name for the price column in output DataFrame.
        start_date: Optional inclusive start date/datetime filter (str or datetime).
        end_date: Optional inclusive end date/datetime filter (str or datetime).

    Returns:
        A Polars DataFrame containing [time_column, price_column] sorted by timestamp.
    """
    if path is None:
        candidates = [
            Path("data/raw/2_electricity_prices.csv"),
            Path(__file__).resolve().parents[3] / "data" / "raw" / "2_electricity_prices.csv",
            Path(__file__).resolve().parents[2] / "data" / "raw" / "2_electricity_prices.csv",
        ]
        file_path = next((p for p in candidates if p.exists()), candidates[0])
    else:
        file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Electricity prices file not found: {file_path}")

    if file_path.suffix.lower() == ".parquet":
        df = pl.read_parquet(file_path)
    elif file_path.suffix.lower() in [".csv", ".txt"]:
        df = pl.read_csv(file_path, try_parse_dates=False)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Use .csv or .parquet.")

    # Identify source time column
    time_candidates = [
        "forecast_date_EET/EEST",
        "datetime_utc",
        "timestamp",
        "datetime",
        "date",
        time_column,
    ]
    source_time = next((c for c in time_candidates if c in df.columns), df.columns[0])

    # Identify source price column
    price_candidates = [
        "euros_per_mwh",
        "price_eur_per_mwh",
        "price",
        "target",
        price_column,
    ]
    source_price = next(
        (c for c in price_candidates if c in df.columns and c != source_time),
        df.columns[1] if len(df.columns) > 1 else df.columns[0],
    )

    # Parse and format datetime column
    if df[source_time].dtype not in (pl.Datetime, pl.Date):
        try:
            df = df.with_columns(
                pl.col(source_time).str.to_datetime("%Y-%m-%d %H:%M:%S").alias(time_column)
            )
        except Exception:
            df = df.with_columns(pl.col(source_time).str.to_datetime().alias(time_column))
    elif source_time != time_column:
        df = df.with_columns(pl.col(source_time).alias(time_column))

    # Format price column
    df = df.with_columns(pl.col(source_price).cast(pl.Float64).alias(price_column))

    # Select standardized columns and sort
    df = df.select([time_column, price_column]).sort(time_column)

    # Apply date filters if provided
    if start_date is not None:
        start_dt = (
            datetime.fromisoformat(str(start_date)) if isinstance(start_date, str) else start_date
        )
        df = df.filter(pl.col(time_column) >= start_dt)

    if end_date is not None:
        end_dt = datetime.fromisoformat(str(end_date)) if isinstance(end_date, str) else end_date
        df = df.filter(pl.col(time_column) <= end_dt)

    return df


def generate_synthetic_timeseries(
    n_series: int = 2,
    length: int = 180,
    freq_days: int = 1,
    start_date: datetime | None = None,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate realistic synthetic time series with trend, weekly/monthly seasonality, and noise.

    Args:
        n_series: Number of unique series to generate.
        length: Number of time steps per series.
        freq_days: Frequency in days between timestamps.
        start_date: Start timestamp (defaults to 2024-01-01).
        seed: Random seed for reproducibility.

    Returns:
        A Polars DataFrame containing [timestamp, series_id, target].
    """
    rng = np.random.default_rng(seed)
    if start_date is None:
        start_date = datetime(2024, 1, 1, 0, 0, 0)

    timestamps = [start_date + timedelta(days=i * freq_days) for i in range(length)]
    t = np.arange(length)

    series_dfs = []
    for s_idx in range(n_series):
        series_id = f"series_{s_idx + 1:02d}"

        # Trend
        slope = rng.uniform(0.05, 0.2)
        intercept = rng.uniform(20.0, 50.0)
        trend = intercept + slope * t

        # Seasonality: Weekly (period 7) + Monthly (period 30)
        weekly = 5.0 * np.sin(2 * np.pi * t / 7)
        monthly = 8.0 * np.cos(2 * np.pi * t / 30)

        # Noise
        noise = rng.normal(0, 1.5, size=length)

        # Combine
        target = np.maximum(trend + weekly + monthly + noise, 0.0)

        s_df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "series_id": [series_id] * length,
                "target": target,
            }
        )
        series_dfs.append(s_df)

    return pl.concat(series_dfs).sort(["series_id", "timestamp"])
