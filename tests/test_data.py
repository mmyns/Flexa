"""Tests for data ingestion, temporal validation, and preprocessing."""

from datetime import datetime

import polars as pl
import pytest

from flexa.data.loader import (
    generate_synthetic_timeseries,
    load_electricity_prices,
    load_timeseries,
)
from flexa.data.preprocessor import TimeSeriesPreprocessor
from flexa.data.validation import validate_timeseries


def test_synthetic_timeseries_generation():
    n_series = 3
    length = 50
    df = generate_synthetic_timeseries(n_series=n_series, length=length, seed=42)

    assert isinstance(df, pl.DataFrame)
    assert df.height == n_series * length
    assert set(df.columns) == {"timestamp", "series_id", "target"}
    assert df["series_id"].n_unique() == n_series
    assert df["target"].null_count() == 0


def test_load_timeseries(tmp_path):
    df_orig = generate_synthetic_timeseries(n_series=2, length=30, seed=1)

    # Test CSV
    csv_file = tmp_path / "test_data.csv"
    df_orig.write_csv(csv_file)
    loaded_csv = load_timeseries(csv_file, id_column="series_id")
    assert loaded_csv.height == df_orig.height
    assert loaded_csv.columns == df_orig.columns

    # Test Parquet
    parquet_file = tmp_path / "test_data.parquet"
    df_orig.write_parquet(parquet_file)
    loaded_pq = load_timeseries(parquet_file, id_column="series_id")
    assert loaded_pq.height == df_orig.height


def test_validation_report_valid(synthetic_df):
    report = validate_timeseries(synthetic_df, id_column="series_id")
    assert report.is_valid is True
    assert report.has_nulls is False
    assert report.is_monotonic is True
    assert report.has_duplicates is False
    assert len(report.errors) == 0


def test_validation_report_duplicates():
    # Duplicate timestamps in same series
    df = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "series_id": ["s1", "s1"],
            "target": [10.0, 20.0],
        }
    )
    report = validate_timeseries(df, id_column="series_id")
    assert report.is_valid is False
    assert report.has_duplicates is True


def test_preprocessor_temporal_split(synthetic_df):
    preprocessor = TimeSeriesPreprocessor(id_column="series_id")
    test_size = 14
    split = preprocessor.temporal_train_test_split(synthetic_df, test_size=test_size)

    # 2 series, each test split has 14 rows -> total 28 rows
    assert split.test.height == 2 * test_size
    assert split.train.height == synthetic_df.height - (2 * test_size)

    # Verify no temporal overlap per series
    for s_id in synthetic_df["series_id"].unique():
        s_train = split.train.filter(pl.col("series_id") == s_id)
        s_test = split.test.filter(pl.col("series_id") == s_id)
        assert s_train["timestamp"].max() < s_test["timestamp"].min()


def test_preprocessor_fill_missing():
    preprocessor = TimeSeriesPreprocessor(id_column="series_id")
    df = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "series_id": ["s1", "s1", "s1"],
            "target": [10.0, None, 30.0],
        }
    )
    filled = preprocessor.fill_missing(df, strategy="forward")
    assert filled["target"].null_count() == 0
    assert filled["target"][1] == 10.0


def test_preprocessor_split_by_date(synthetic_df):
    preprocessor = TimeSeriesPreprocessor(id_column="series_id")
    cutoff = datetime(2024, 3, 1)
    split = preprocessor.split_by_date(synthetic_df, cutoff_date=cutoff)

    assert split.train["timestamp"].max() < cutoff
    assert split.test["timestamp"].min() >= cutoff
    assert split.train.height + split.test.height == synthetic_df.height


def test_preprocessor_clean_outliers():
    preprocessor = TimeSeriesPreprocessor(id_column=None, time_column="timestamp")
    # Series with normal values ~30, one massive glitch at index 3 (335.0)
    # and a sunny day daylight generation spike at index 6 with high solar radiation
    dates = [datetime(2023, 1, 1, h) for h in range(8)]
    df = pl.DataFrame(
        {
            "timestamp": dates,
            "is_consumption": [1, 1, 1, 1, 0, 0, 0, 0],
            "target": [32.0, 31.0, 33.0, 335.0, 2.0, 1.0, 500.0, 480.0],
            "radiation": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 600.0, 550.0],
        }
    )

    res = preprocessor.clean_outliers(
        df,
        threshold=4.0,
        min_residual=30.0,
        radiation_column="radiation",
        is_consumption_column="is_consumption",
    )

    # Spike at index 3 should be detected as outlier
    assert res.n_outliers == 1
    assert res.df["is_outlier"][3] is True
    # Cleaned target at index 3 should be reasonable (~32-33)
    assert res.df["target"][3] < 50.0
    assert res.df["target_raw"][3] == 335.0

    # Sunny day generation at index 6 & 7 should NOT be flagged as outliers
    assert res.df["is_outlier"][6] is False
    assert res.df["is_outlier"][7] is False


def test_load_electricity_prices_default():
    """Test loading electricity prices with default path."""
    df = load_electricity_prices()
    assert isinstance(df, pl.DataFrame)
    assert df.height == 15286
    assert df.columns == ["timestamp", "price_eur_per_mwh"]
    assert df["timestamp"].dtype in (pl.Datetime("us"), pl.Datetime("ms"), pl.Datetime("ns"))
    assert df["price_eur_per_mwh"].dtype == pl.Float64
    assert df["timestamp"].null_count() == 0
    assert df["price_eur_per_mwh"].null_count() == 0
    # Verify monotonic increasing timestamps
    assert df["timestamp"].is_sorted()


def test_load_electricity_prices_filtered():
    """Test electricity prices filtering with start and end dates."""
    df = load_electricity_prices(
        start_date="2022-01-01 00:00:00",
        end_date="2022-01-07 23:00:00",
    )
    assert df.height == 7 * 24  # 168 hours
    assert df["timestamp"].min() == datetime(2022, 1, 1, 0, 0, 0)
    assert df["timestamp"].max() == datetime(2022, 1, 7, 23, 0, 0)


def test_load_electricity_prices_custom_names(tmp_path):
    """Test loading with custom column names and file."""
    custom_csv = tmp_path / "prices.csv"
    custom_df = pl.DataFrame(
        {
            "dt": ["2023-01-01 00:00:00", "2023-01-01 01:00:00"],
            "eur": [55.2, 48.9],
        }
    )
    custom_df.write_csv(custom_csv)

    loaded = load_electricity_prices(
        path=custom_csv,
        time_column="time",
        price_column="price",
    )
    assert loaded.columns == ["time", "price"]
    assert loaded.height == 2
    assert loaded["price"][0] == 55.2


def test_load_electricity_prices_not_found():
    """Test error when electricity price file is missing."""
    with pytest.raises(FileNotFoundError):
        load_electricity_prices(path="non_existent_file.csv")
