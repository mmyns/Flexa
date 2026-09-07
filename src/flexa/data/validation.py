"""Temporal validation and schema checking for time series datasets."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass
class ValidationReport:
    """Summary of time series validation checks."""

    is_valid: bool
    num_rows: int
    num_series: int
    has_nulls: bool
    null_counts: dict[str, int]
    is_monotonic: bool
    has_duplicates: bool
    min_timestamp: str
    max_timestamp: str
    errors: list[str]


def validate_timeseries(
    df: pl.DataFrame,
    time_column: str = "timestamp",
    target_column: str = "target",
    id_column: str | None = "series_id",
    raise_on_error: bool = False,
) -> ValidationReport:
    """Validate time series consistency, monotonicity, missing values, and duplicates.

    Args:
        df: Input Polars DataFrame.
        time_column: Name of timestamp column.
        target_column: Name of target column.
        id_column: Optional series identifier column.
        raise_on_error: If True, raises ValueError when validation fails.

    Returns:
        ValidationReport dataclass summarizing results.
    """
    errors: list[str] = []

    if df.is_empty():
        errors.append("DataFrame is empty.")
        report = ValidationReport(
            is_valid=False,
            num_rows=0,
            num_series=0,
            has_nulls=False,
            null_counts={},
            is_monotonic=False,
            has_duplicates=False,
            min_timestamp="",
            max_timestamp="",
            errors=errors,
        )
        if raise_on_error:
            raise ValueError("; ".join(errors))
        return report

    # Required columns check
    for col in [time_column, target_column]:
        if col not in df.columns:
            errors.append(f"Required column '{col}' is missing.")

    if id_column and id_column not in df.columns:
        errors.append(f"Specified id_column '{id_column}' not found.")

    if errors:
        if raise_on_error:
            raise ValueError("; ".join(errors))
        return ValidationReport(
            is_valid=False,
            num_rows=df.height,
            num_series=0,
            has_nulls=False,
            null_counts={},
            is_monotonic=False,
            has_duplicates=False,
            min_timestamp="",
            max_timestamp="",
            errors=errors,
        )

    # Check nulls
    null_counts = {col: df[col].null_count() for col in df.columns}
    has_nulls = sum(null_counts.values()) > 0
    if null_counts.get(target_column, 0) > 0:
        errors.append(
            f"Target column '{target_column}' contains {null_counts[target_column]} null values."
        )

    # Check duplicates and monotonicity
    partition_cols = [id_column] if id_column and id_column in df.columns else []

    if id_column and id_column in df.columns:
        # Check duplicates within each series
        duplicates_df = df.group_by([id_column, time_column]).len().filter(pl.col("len") > 1)
        has_duplicates = duplicates_df.height > 0
        num_series = df[id_column].n_unique()
    else:
        duplicates_df = df.group_by([time_column]).len().filter(pl.col("len") > 1)
        has_duplicates = duplicates_df.height > 0
        num_series = 1

    if has_duplicates:
        errors.append(
            f"Detected duplicate timestamps in dataset ({duplicates_df.height} collisions)."
        )

    # Check monotonicity
    if partition_cols:
        monotonic_check = (
            df.sort([id_column, time_column])
            .with_columns(
                is_increasing=(
                    pl.col(time_column) > pl.col(time_column).shift(1).over(id_column)
                ).fill_null(True)
            )
            .select(pl.col("is_increasing").all())
            .item()
        )
    else:
        monotonic_check = (
            df.sort(time_column)
            .with_columns(
                is_increasing=(pl.col(time_column) > pl.col(time_column).shift(1)).fill_null(True)
            )
            .select(pl.col("is_increasing").all())
            .item()
        )

    is_monotonic = bool(monotonic_check)
    if not is_monotonic:
        errors.append("Timestamps are not strictly monotonically increasing.")

    min_ts = str(df[time_column].min())
    max_ts = str(df[time_column].max())

    is_valid = len(errors) == 0
    if not is_valid and raise_on_error:
        raise ValueError("; ".join(errors))

    return ValidationReport(
        is_valid=is_valid,
        num_rows=df.height,
        num_series=num_series,
        has_nulls=has_nulls,
        null_counts=null_counts,
        is_monotonic=is_monotonic,
        has_duplicates=has_duplicates,
        min_timestamp=min_ts,
        max_timestamp=max_ts,
        errors=errors,
    )
