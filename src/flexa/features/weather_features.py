"""Vectorized weather feature engineering for energy forecasting powered by Polars."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from pydantic import BaseModel, Field


class WeatherFeatureConfig(BaseModel):
    """Configuration for domain-specific weather feature engineering."""

    base_temp_c: float = 18.0
    extreme_cold_temp_c: float = 10.0
    daylight_radiation_threshold: float = 10.0
    solar_rolling_windows: list[int] = Field(default_factory=lambda: [3, 6])
    temp_rolling_windows: list[int] = Field(default_factory=lambda: [6, 24])
    include_lags: bool = True
    lags: list[int] = Field(default_factory=lambda: [1, 24])


class WeatherFeatureEngineer:
    """Extracts meteorological and physical proxy features from weather forecasts."""

    def __init__(
        self,
        config: WeatherFeatureConfig | None = None,
        time_column: str = "forecast_datetime_utc",
    ) -> None:
        self.config = config or WeatherFeatureConfig()
        self.time_column = time_column
        self.feature_names: list[str] = []

    def enrich_weather(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate solar, thermal, and atmospheric features from hourly weather forecasts.

        Args:
            df: Raw weather forecast Polars DataFrame with standard hourly ECMWF / NWP columns.

        Returns:
            Polars DataFrame enriched with engineered domain weather features.
        """
        sorted_df = df.sort(self.time_column)
        exprs: list[pl.Expr] = []
        new_features: list[str] = []

        cols = set(sorted_df.columns)

        # 1. Wind speed from 10m u and v components
        if "10_metre_u_wind_component" in cols and "10_metre_v_wind_component" in cols:
            exprs.append(
                (
                    pl.col("10_metre_u_wind_component").pow(2)
                    + pl.col("10_metre_v_wind_component").pow(2)
                )
                .sqrt()
                .alias("wind_speed_10m")
            )
            new_features.append("wind_speed_10m")

        # 2. Dewpoint depression (humidity proxy)
        if "temperature" in cols and "dewpoint" in cols:
            exprs.append((pl.col("temperature") - pl.col("dewpoint")).alias("dewpoint_depression"))
            new_features.append("dewpoint_depression")

        # 3. Precipitation & Snow indicators
        if "total_precipitation" in cols:
            exprs.append(
                (pl.col("total_precipitation") > 0.0).cast(pl.Float64).alias("is_precipitation")
            )
            new_features.append("is_precipitation")
        if "snowfall" in cols:
            exprs.append((pl.col("snowfall") > 0.0).cast(pl.Float64).alias("is_snow"))
            new_features.append("is_snow")

        # 3b. Cloud Cover Change / Delta (T-1 - T-2)
        if "cloudcover_total" in cols:
            cloud_delta = (
                pl.col("cloudcover_total").shift(1)
                - pl.col("cloudcover_total").shift(2)
            ).fill_null(0.0)
            exprs.append(cloud_delta.alias("cloudcover_delta_1_2"))
            exprs.append(cloud_delta.alias("cloudcover_total_delta_1_2"))
            new_features.extend(["cloudcover_delta_1_2", "cloudcover_total_delta_1_2"])

        # 4. Solar Radiation Proxies (key for solar injection / PV generation)
        has_direct = "direct_solar_radiation" in cols
        has_surface = "surface_solar_radiation_downwards" in cols

        if has_surface:
            # Daylight indicator
            exprs.append(
                (
                    pl.col("surface_solar_radiation_downwards")
                    > self.config.daylight_radiation_threshold
                )
                .cast(pl.Float64)
                .alias("is_daylight")
            )
            new_features.append("is_daylight")

            # Diffuse solar radiation
            if has_direct:
                exprs.append(
                    (pl.col("surface_solar_radiation_downwards") - pl.col("direct_solar_radiation"))
                    .clip(lower_bound=0.0)
                    .alias("diffuse_solar_radiation")
                )
                new_features.append("diffuse_solar_radiation")

            # Solar ramp rate (1-hour difference)
            exprs.append(
                pl.col("surface_solar_radiation_downwards")
                .diff(1)
                .fill_null(0.0)
                .alias("solar_ramp_1h")
            )
            new_features.append("solar_ramp_1h")

            # Estimated PV cell temperature (cell temp increases with radiation, reducing efficiency)
            if "temperature" in cols:
                exprs.append(
                    (
                        pl.col("temperature") + 0.03 * pl.col("surface_solar_radiation_downwards")
                    ).alias("pv_cell_temp_est")
                )
                new_features.append("pv_cell_temp_est")

            # Solar rolling averages (persistent sunshine / clear-sky trend)
            for w in self.config.solar_rolling_windows:
                feat_name = f"solar_roll_mean_{w}h"
                exprs.append(
                    pl.col("surface_solar_radiation_downwards")
                    .rolling_mean(window_size=w, min_samples=1)
                    .alias(feat_name)
                )
                new_features.append(feat_name)

        # 5. Temperature & Degree Hours (key for heating & cooling electricity consumption)
        if "temperature" in cols:
            # Heating degree hours (below base temp)
            exprs.append(
                (self.config.base_temp_c - pl.col("temperature"))
                .clip(lower_bound=0.0)
                .alias("hdd_18")
            )
            new_features.append("hdd_18")

            # Cooling degree hours (above base temp)
            exprs.append(
                (pl.col("temperature") - self.config.base_temp_c)
                .clip(lower_bound=0.0)
                .alias("cdd_18")
            )
            new_features.append("cdd_18")

            # Extreme cold proxy
            exprs.append(
                (self.config.extreme_cold_temp_c - pl.col("temperature"))
                .clip(lower_bound=0.0)
                .alias("hdd_10")
            )
            new_features.append("hdd_10")

            # Temperature deltas (rate of cooling / warming)
            exprs.append(pl.col("temperature").diff(1).fill_null(0.0).alias("temp_diff_1h"))
            new_features.append("temp_diff_1h")

            exprs.append(pl.col("temperature").diff(24).fill_null(0.0).alias("temp_diff_24h"))
            new_features.append("temp_diff_24h")

            # Temperature rolling averages (thermal inertia of buildings)
            for w in self.config.temp_rolling_windows:
                feat_name = f"temp_roll_mean_{w}h"
                exprs.append(
                    pl.col("temperature")
                    .rolling_mean(window_size=w, min_samples=1)
                    .alias(feat_name)
                )
                new_features.append(feat_name)

        # Apply computed base expressions
        enriched = sorted_df.with_columns(exprs)

        # 6. Optional Lags of Weather Features
        if self.config.include_lags:
            lag_exprs: list[pl.Expr] = []
            lag_targets = [
                f
                for f in ["temperature", "surface_solar_radiation_downwards", "wind_speed_10m"]
                if f in enriched.columns
            ]
            for lt in lag_targets:
                for lag in self.config.lags:
                    lag_feat = f"{lt}_lag_{lag}h"
                    lag_exprs.append(
                        pl.col(lt)
                        .shift(lag)
                        .fill_null(strategy="forward")
                        .fill_null(strategy="backward")
                        .alias(lag_feat)
                    )
                    new_features.append(lag_feat)

            if lag_exprs:
                enriched = enriched.with_columns(lag_exprs)

        # Record all features (including relevant raw columns)
        base_weather_cols = [
            c
            for c in [
                "temperature",
                "dewpoint",
                "cloudcover_total",
                "cloudcover_low",
                "cloudcover_mid",
                "cloudcover_high",
                "surface_solar_radiation_downwards",
                "direct_solar_radiation",
                "total_precipitation",
                "snowfall",
            ]
            if c in sorted_df.columns
        ]

        self.feature_names = base_weather_cols + new_features
        return enriched

    def join_weather(
        self,
        measurements_df: pl.DataFrame,
        weather_df: pl.DataFrame,
        time_col_meas: str = "datetime_utc",
        time_col_weather: str = "forecast_datetime_utc",
    ) -> pl.DataFrame:
        """Perform an inner join between measurements and weather forecasts on hourly timestamps.

        Args:
            measurements_df: Measurements Polars DataFrame.
            weather_df: Weather (or enriched weather) Polars DataFrame.
            time_col_meas: Timestamp column name in measurements.
            time_col_weather: Timestamp column name in weather.

        Returns:
            Joined Polars DataFrame.
        """
        # Ensure timestamp alignment
        renamed_weather = weather_df.rename({time_col_weather: time_col_meas})
        return measurements_df.join(renamed_weather, on=time_col_meas, how="inner")

    def to_chronos_covariates(
        self,
        weather_df: pl.DataFrame,
        feature_names: list[str] | None = None,
        time_column: str | None = None,
        start_time: Any = None,
        end_time: Any = None,
    ) -> dict[str, np.ndarray]:
        """Extract weather feature columns as a dictionary of float32 1D numpy arrays for Chronos-2.

        Args:
            weather_df: Enriched weather Polars DataFrame.
            feature_names: List of weather feature column names to extract.
            time_column: Timestamp column to filter if time bounds are provided.
            start_time: Optional start timestamp (inclusive).
            end_time: Optional end timestamp (inclusive).

        Returns:
            Dictionary mapping feature_name -> 1D np.ndarray (float32).
        """
        sub_df = weather_df
        t_col = time_column or self.time_column

        if start_time is not None and t_col in sub_df.columns:
            sub_df = sub_df.filter(pl.col(t_col) >= start_time)
        if end_time is not None and t_col in sub_df.columns:
            sub_df = sub_df.filter(pl.col(t_col) <= end_time)

        sub_df = sub_df.sort(t_col)
        selected_features = feature_names or self.feature_names

        covariates: dict[str, np.ndarray] = {}
        for feat in selected_features:
            if feat in sub_df.columns:
                covariates[feat] = sub_df[feat].to_numpy().astype(np.float32)

        return covariates
