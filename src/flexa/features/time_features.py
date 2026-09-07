"""Vectorized time series feature engineering powered by Polars."""

from __future__ import annotations

import numpy as np
import polars as pl

from flexa.config import FeatureConfig


def add_paired_telemetry(
    df: pl.DataFrame,
    context_df: pl.DataFrame | None = None,
    target_col: str = "target",
    time_col: str = "datetime_utc",
    pool_col: str = "pool",
    is_consumption_col: str = "is_consumption",
    peak_col: str = "solar_peak_to_t",
) -> pl.DataFrame:
    """Pair simultaneous consumption and injection telemetry columns for every pool.

    Also computes the expanding maximum solar injection peak up until point T
    (shifted by 1 time step to strictly avoid lookahead target leakage) per pool.

    Args:
        df: Measurements DataFrame with pool, timestamp, is_consumption, and target.
        context_df: Optional historical context DataFrame preceding df.
        target_col: Name of target power column.
        time_col: Timestamp column name.
        pool_col: Pool identifier column.
        is_consumption_col: Binary flag column (1=consumption, 0=injection).
        peak_col: Name for the historical solar peak column up until point T.

    Returns:
        DataFrame augmented with 'injection', 'consumption', and 'solar_peak_to_t' columns.
    """
    if is_consumption_col not in df.columns or pool_col not in df.columns:
        return df

    cols_to_drop = [c for c in ["injection", "consumption", peak_col] if c in df.columns]
    base_df = df.drop(cols_to_drop) if cols_to_drop else df

    if context_df is not None and not context_df.is_empty():
        req = [pool_col, time_col, is_consumption_col, target_col]
        ctx_sub = context_df.select([c for c in req if c in context_df.columns])
        df_sub = df.select([c for c in req if c in df.columns])
        combined = (
            pl.concat([ctx_sub, df_sub])
            .unique(subset=[pool_col, time_col, is_consumption_col])
            .sort([pool_col, time_col])
        )
    else:
        combined = df.sort([pool_col, time_col])

    inj = combined.filter(pl.col(is_consumption_col) == 0).select(
        [pool_col, time_col, pl.col(target_col).alias("injection")]
    )
    cons = combined.filter(pl.col(is_consumption_col) == 1).select(
        [pool_col, time_col, pl.col(target_col).alias("consumption")]
    )
    paired = inj.join(cons, on=[pool_col, time_col], how="inner").sort([pool_col, time_col])

    # Highest solar peak up until point T (shift 1 to avoid leaking current hour)
    paired = paired.with_columns(
        pl.col("injection").shift(1).cum_max().over(pool_col).fill_null(0.0).alias(peak_col)
    )

    return base_df.join(paired, on=[pool_col, time_col], how="left")


class TimeSeriesFeatureEngineer:
    """Extracts calendar, cyclical, lag, and rolling features using Polars expressions."""

    def __init__(
        self,
        config: FeatureConfig | None = None,
        time_column: str = "timestamp",
        target_column: str = "target",
        id_column: str | None = "series_id",
        autoregressive_columns: list[str] | None = None,
    ) -> None:
        self.config = config or FeatureConfig()
        self.time_column = time_column
        self.target_column = target_column
        self.id_column = id_column
        self.autoregressive_columns = autoregressive_columns
        self.feature_names: list[str] = []

    def create_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Engineer all configured features on the input DataFrame.

        To prevent data leakage, rolling window statistics are strictly computed on
        lag-1 target values, ensuring future values are never visible during prediction.

        Args:
            df: Input Polars DataFrame containing time, target, and optional id columns.

        Returns:
            Polars DataFrame enriched with feature columns.
        """
        exprs: list[pl.Expr] = []
        new_feature_names: list[str] = []

        # 1. Calendar Features
        if self.config.include_calendar:
            cal_feats = self.config.calendar_features or ["day_of_week", "day_of_month", "month", "hour"]
            if "day_of_week" in cal_feats:
                exprs.append(pl.col(self.time_column).dt.weekday().alias("day_of_week"))
                new_feature_names.append("day_of_week")
            if "day_of_month" in cal_feats:
                exprs.append(pl.col(self.time_column).dt.day().alias("day_of_month"))
                new_feature_names.append("day_of_month")
            if "month" in cal_feats:
                exprs.append(pl.col(self.time_column).dt.month().alias("month"))
                new_feature_names.append("month")
            if "hour" in cal_feats:
                exprs.append(pl.col(self.time_column).dt.hour().alias("hour"))
                new_feature_names.append("hour")

        # 2. Cyclical Encodings (sin / cos)
        if self.config.include_cyclical:
            # Hour of day: 0-23 (continuous diurnal cycle: 23:00 wraps smoothly to 00:00)
            hour_norm = pl.col(self.time_column).dt.hour() / 24.0 * 2.0 * np.pi
            exprs.append(hour_norm.sin().alias("sin_hour"))
            exprs.append(hour_norm.cos().alias("cos_hour"))

            # Month of year: 1-12 (seasonal cycle)
            month_norm = (pl.col(self.time_column).dt.month() - 1) / 12.0 * 2.0 * np.pi
            exprs.append(month_norm.sin().alias("sin_month"))
            exprs.append(month_norm.cos().alias("cos_month"))

            new_feature_names.extend(["sin_hour", "cos_hour", "sin_month", "cos_month"])

        # 3. Autoregressive Lag Features
        partition = [self.id_column] if self.id_column and self.id_column in df.columns else []

        ar_cols = (
            [c for c in self.autoregressive_columns if c in df.columns]
            if self.autoregressive_columns
            else [self.target_column]
        )

        for col_name in ar_cols:
            for lag in self.config.lags:
                lag_feat_name = (
                    f"{col_name}_lag_{lag}"
                    if len(ar_cols) > 1 or col_name != self.target_column
                    else f"lag_{lag}"
                )
                if partition:
                    expr = pl.col(col_name).shift(lag).over(partition).alias(lag_feat_name)
                else:
                    expr = pl.col(col_name).shift(lag).alias(lag_feat_name)
                exprs.append(expr)
                new_feature_names.append(lag_feat_name)

        # 4. Delta Lag Features (e.g. difference between lag 1 and lag 2: y_{t-1} - y_{t-2})
        if self.config.delta_lags:
            for col_name in ar_cols:
                for lag_a, lag_b in self.config.delta_lags:
                    delta_name = (
                        f"{col_name}_delta_{lag_a}_{lag_b}"
                        if len(ar_cols) > 1 or col_name != self.target_column
                        else f"delta_{lag_a}_{lag_b}"
                    )
                    shift_a = pl.col(col_name).shift(lag_a)
                    shift_b = pl.col(col_name).shift(lag_b)
                    if partition:
                        d_expr = (shift_a - shift_b).over(partition).alias(delta_name)
                    else:
                        d_expr = (shift_a - shift_b).alias(delta_name)
                    exprs.append(d_expr)
                    new_feature_names.append(delta_name)

        # Apply base calendar, lag, & delta expressions first
        df_feat = df.with_columns(exprs)

        # 5. Rolling Window Statistics (computed on lag_1 to prevent data leakage)
        # If lag_1 is not in lags, create a temporary shifted series
        rolling_exprs: list[pl.Expr] = []
        base_series = pl.col(self.target_column).shift(1)
        if partition:
            base_series = base_series.over(partition)

        for w in self.config.rolling_windows:
            for metric in self.config.rolling_metrics:
                roll_name = f"roll_{metric}_{w}"
                if metric == "mean":
                    r_expr = base_series.rolling_mean(window_size=w)
                elif metric == "std":
                    r_expr = base_series.rolling_std(window_size=w)
                elif metric == "min":
                    r_expr = base_series.rolling_min(window_size=w)
                elif metric == "max":
                    r_expr = base_series.rolling_max(window_size=w)
                else:
                    continue

                if partition:
                    r_expr = r_expr.over(partition)

                rolling_exprs.append(r_expr.alias(roll_name))
                new_feature_names.append(roll_name)

        if rolling_exprs:
            df_feat = df_feat.with_columns(rolling_exprs)

        if self.config.exclude_features:
            new_feature_names = [
                f for f in new_feature_names if f not in self.config.exclude_features
            ]

        self.feature_names = new_feature_names
        return df_feat
