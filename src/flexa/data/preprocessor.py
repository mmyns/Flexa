"""Data preprocessing, temporal splitting, and scaling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import polars as pl


@dataclass
class TrainTestSplit:
    """Dataclass holding train and test Polars DataFrames."""

    train: pl.DataFrame
    test: pl.DataFrame


@dataclass
class OutlierResult:
    """Dataclass holding outlier cleaning results and metadata."""

    df: pl.DataFrame
    n_outliers: int
    outlier_records: list[dict]


class TimeSeriesPreprocessor:
    """Preprocesses time series data with missing value imputation and temporal splitting."""

    def __init__(
        self,
        time_column: str = "timestamp",
        target_column: str = "target",
        id_column: str | None = "series_id",
    ) -> None:
        self.time_column = time_column
        self.target_column = target_column
        self.id_column = id_column

    def fill_missing(
        self,
        df: pl.DataFrame,
        strategy: str = "forward",
    ) -> pl.DataFrame:
        """Fill missing values in the target column.

        Args:
            df: Input Polars DataFrame.
            strategy: 'forward' (forward fill then backward fill) or 'zero'.

        Returns:
            DataFrame with nulls imputed.
        """
        if self.id_column and self.id_column in df.columns:
            if strategy == "forward":
                return df.with_columns(
                    pl.col(self.target_column).forward_fill().backward_fill().over(self.id_column)
                )
            elif strategy == "zero":
                return df.with_columns(pl.col(self.target_column).fill_null(0.0))
            else:
                raise ValueError(f"Unknown fill strategy: {strategy}")
        else:
            if strategy == "forward":
                return df.with_columns(pl.col(self.target_column).forward_fill().backward_fill())
            elif strategy == "zero":
                return df.with_columns(pl.col(self.target_column).fill_null(0.0))
            else:
                raise ValueError(f"Unknown fill strategy: {strategy}")

    def temporal_train_test_split(
        self,
        df: pl.DataFrame,
        test_size: int,
    ) -> TrainTestSplit:
        """Split time series temporally into train and test sets for each series.

        Args:
            df: Input Polars DataFrame.
            test_size: Number of final observations per series reserved for testing.

        Returns:
            TrainTestSplit with train and test DataFrames.
        """
        if df.is_empty():
            raise ValueError("Cannot split empty DataFrame.")

        sort_cols = (
            [self.id_column, self.time_column]
            if self.id_column and self.id_column in df.columns
            else [self.time_column]
        )
        sorted_df = df.sort(sort_cols)

        if self.id_column and self.id_column in sorted_df.columns:
            # Number of rows per series
            counts = sorted_df.group_by(self.id_column).len()
            min_len_val = counts["len"].min()
            if isinstance(min_len_val, (int, float)):
                min_len = int(min_len_val)
                if min_len <= test_size:
                    raise ValueError(
                        f"test_size ({test_size}) must be strictly smaller than the shortest series length ({min_len})."
                    )

            # Add reverse index per series
            indexed = sorted_df.with_columns(
                rev_idx=pl.int_range(pl.len(), 0, -1).over(self.id_column)
            )
            train = indexed.filter(pl.col("rev_idx") > test_size).drop("rev_idx")
            test = indexed.filter(pl.col("rev_idx") <= test_size).drop("rev_idx")
        else:
            if sorted_df.height <= test_size:
                raise ValueError(
                    f"test_size ({test_size}) must be strictly smaller than total rows ({sorted_df.height})."
                )
            split_idx = sorted_df.height - test_size
            train = sorted_df.slice(0, split_idx)
            test = sorted_df.slice(split_idx, test_size)

        return TrainTestSplit(train=train, test=test)

    def split_by_date(
        self,
        df: pl.DataFrame,
        cutoff_date: str | datetime,
    ) -> TrainTestSplit:
        """Split time series temporally into train and test sets using an explicit cutoff date.

        Observations strictly before cutoff_date become the training set.
        Observations at or after cutoff_date become the test set.

        Args:
            df: Input Polars DataFrame.
            cutoff_date: Cutoff timestamp (ISO string e.g. '2023-01-01' or datetime).

        Returns:
            TrainTestSplit with train and test DataFrames.
        """
        if df.is_empty():
            raise ValueError("Cannot split empty DataFrame.")

        if isinstance(cutoff_date, str):
            cutoff_dt = datetime.fromisoformat(cutoff_date)
        else:
            cutoff_dt = cutoff_date

        sort_cols = (
            [self.id_column, self.time_column]
            if self.id_column and self.id_column in df.columns
            else [self.time_column]
        )
        sorted_df = df.sort(sort_cols)

        # Cast to datetime if not already
        if sorted_df[self.time_column].dtype not in (pl.Datetime, pl.Date):
            sorted_df = sorted_df.with_columns(pl.col(self.time_column).str.to_datetime())

        train = sorted_df.filter(pl.col(self.time_column) < cutoff_dt)
        test = sorted_df.filter(pl.col(self.time_column) >= cutoff_dt)

        if train.is_empty():
            raise ValueError(f"Training set is empty for cutoff_date: {cutoff_date}")
        if test.is_empty():
            raise ValueError(f"Test set is empty for cutoff_date: {cutoff_date}")

        return TrainTestSplit(train=train, test=test)

    def clean_outliers(
        self,
        df: pl.DataFrame,
        window_size: int = 5,
        threshold: float = 5.0,
        min_residual: float = 40.0,
        radiation_column: str | None = None,
        is_consumption_column: str | None = "is_consumption",
    ) -> OutlierResult:
        """Detect and remove point outliers using a weather-guarded robust moving average filter.

        Guards against false positives on sudden sunny days:
        - Injection surges accompanied by high solar radiation are preserved.
        - Consumption dips accompanied by high solar radiation (self-consumption) are preserved.
        - Isolated point spikes (e.g. 10x sensor glitch) or nighttime solar injection are cleaned.

        Args:
            df: Input Polars DataFrame.
            window_size: Centered rolling window size for local median (default 5).
            threshold: Z-score/MAD threshold for anomaly flagging (default 5.0).
            min_residual: Minimum absolute deviation to qualify as an outlier (default 40.0).
            radiation_column: Name of solar radiation column to guard sunny days.
            is_consumption_column: Name of injection/consumption indicator column.

        Returns:
            OutlierResult containing cleaned DataFrame, outlier count, and outlier metadata.
        """
        if df.is_empty():
            raise ValueError("Cannot clean outliers on empty DataFrame.")

        sort_cols = (
            [self.id_column, self.time_column]
            if self.id_column and self.id_column in df.columns
            else [self.time_column]
        )
        sorted_df = df.sort(sort_cols)

        partition = (
            [self.id_column] if (self.id_column and self.id_column in sorted_df.columns) else []
        )
        if is_consumption_column and is_consumption_column in sorted_df.columns:
            partition.append(is_consumption_column)

        # 1. Local median baseline
        med_expr = pl.col(self.target_column).rolling_median(
            window_size=window_size, min_samples=1, center=True
        )
        if partition:
            med_expr = med_expr.over(partition)

        result_df = sorted_df.with_columns(med_expr.alias("_local_med"))

        # 2. Residual and rolling MAD
        diff_expr = (pl.col(self.target_column) - pl.col("_local_med")).abs()
        mad_expr = (
            diff_expr.rolling_median(window_size=window_size * 5, min_samples=1, center=True)
            * 1.4826
        )
        if partition:
            mad_expr = mad_expr.over(partition)

        result_df = result_df.with_columns(
            diff_expr.alias("_resid"),
            pl.when(mad_expr.is_null() | (mad_expr < 1.0))
            .then(1.0)
            .otherwise(mad_expr)
            .alias("_mad"),
        ).with_columns((pl.col("_resid") / pl.col("_mad")).fill_null(0.0).alias("_z_score"))

        # 3. Weather Guard condition
        guard_condition = pl.lit(False)
        if (
            radiation_column
            and radiation_column in result_df.columns
            and is_consumption_column
            and is_consumption_column in result_df.columns
        ):
            guard_solar = (pl.col(is_consumption_column) == 0) & (pl.col(radiation_column) > 20.0)
            guard_cons = (
                (pl.col(is_consumption_column) == 1)
                & (pl.col(radiation_column) > 50.0)
                & (pl.col(self.target_column) < pl.col("_local_med"))
            )
            guard_condition = guard_solar | guard_cons

        # Nighttime injection glitch check (e.g. 100 kW injection at night)
        night_glitch = pl.lit(False)
        if (
            radiation_column
            and radiation_column in result_df.columns
            and is_consumption_column
            and is_consumption_column in result_df.columns
        ):
            night_glitch = (
                (pl.col(is_consumption_column) == 0)
                & (pl.col(radiation_column) <= 0.1)
                & (pl.col(self.target_column) > 15.0)
            )

        # 4. Outlier flag
        stat_outlier = (
            (pl.col("_z_score") > threshold) & (pl.col("_resid") > min_residual) & ~guard_condition
        )
        outlier_condition = stat_outlier | night_glitch

        result_df = result_df.with_columns(
            outlier_condition.alias("is_outlier"),
            pl.col(self.target_column).alias("target_raw"),
        ).with_columns(
            pl.when(pl.col("is_outlier"))
            .then(pl.col("_local_med"))
            .otherwise(pl.col(self.target_column))
            .alias(self.target_column)
        )

        outlier_rows = result_df.filter(pl.col("is_outlier"))
        meta_cols = [
            c
            for c in [
                self.time_column,
                self.id_column,
                is_consumption_column,
                "target_raw",
                self.target_column,
                radiation_column,
                "_z_score",
            ]
            if c and c in outlier_rows.columns
        ]
        records = outlier_rows.select(meta_cols).to_dicts()

        cleaned_df = result_df.drop(["_local_med", "_resid", "_mad", "_z_score"])

        return OutlierResult(
            df=cleaned_df,
            n_outliers=len(records),
            outlier_records=records,
        )
