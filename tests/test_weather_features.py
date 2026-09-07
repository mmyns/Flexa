"""Unit tests for weather feature engineering and model integration."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from flexa.config import BaselineModelConfig, FeatureConfig
from flexa.features.weather_features import WeatherFeatureConfig, WeatherFeatureEngineer
from flexa.models.baseline import BaselineForecaster
from flexa.models.chronos import ChronosForecaster


@pytest.fixture
def sample_weather_df() -> pl.DataFrame:
    """Create a small deterministic hourly weather DataFrame."""
    base = datetime(2022, 6, 1, 0, 0, 0)
    timestamps = [base + timedelta(hours=i) for i in range(48)]
    return pl.DataFrame(
        {
            "forecast_datetime_utc": timestamps,
            "temperature": [15.0 + 5.0 * np.sin(i / 24.0 * 2 * np.pi) for i in range(48)],
            "dewpoint": [10.0 for _ in range(48)],
            "cloudcover_high": [0.2 for _ in range(48)],
            "cloudcover_low": [0.1 for _ in range(48)],
            "cloudcover_mid": [0.1 for _ in range(48)],
            "cloudcover_total": [0.3 for _ in range(48)],
            "10_metre_u_wind_component": [3.0 for _ in range(48)],
            "10_metre_v_wind_component": [4.0 for _ in range(48)],
            "direct_solar_radiation": [
                max(0.0, 400.0 * np.sin((i % 24) / 24.0 * np.pi)) for i in range(48)
            ],
            "surface_solar_radiation_downwards": [
                max(0.0, 500.0 * np.sin((i % 24) / 24.0 * np.pi)) for i in range(48)
            ],
            "snowfall": [0.0 for _ in range(48)],
            "total_precipitation": [0.5 if i == 10 else 0.0 for i in range(48)],
        }
    )


def test_weather_feature_calculations(sample_weather_df):
    cfg = WeatherFeatureConfig(base_temp_c=18.0)
    engineer = WeatherFeatureEngineer(config=cfg)
    enriched = engineer.enrich_weather(sample_weather_df)

    # 1. Wind speed: sqrt(3^2 + 4^2) = 5.0
    wind_speeds = enriched["wind_speed_10m"].to_list()
    assert np.isclose(wind_speeds[0], 5.0)

    # 2. Dewpoint depression: temp - 10.0
    assert "dewpoint_depression" in enriched.columns
    assert enriched["dewpoint_depression"][0] == enriched["temperature"][0] - 10.0

    # 3. Precipitation & Cloud indicators
    assert enriched["is_precipitation"][10] == 1.0
    assert enriched["is_precipitation"][0] == 0.0
    assert enriched["is_snow"][0] == 0.0
    assert "cloudcover_delta_1_2" in enriched.columns
    assert "cloudcover_total_delta_1_2" in enriched.columns

    # 4. Solar proxies
    assert "is_daylight" in enriched.columns
    assert "diffuse_solar_radiation" in enriched.columns
    assert "pv_cell_temp_est" in enriched.columns
    assert "solar_roll_mean_3h" in enriched.columns
    assert "solar_ramp_1h" in enriched.columns

    # 5. Degree hours
    assert "hdd_18" in enriched.columns
    assert "cdd_18" in enriched.columns
    assert "hdd_10" in enriched.columns
    # Check degree hours logic
    for row in enriched.iter_rows(named=True):
        t = row["temperature"]
        assert row["hdd_18"] == max(18.0 - t, 0.0)
        assert row["cdd_18"] == max(t - 18.0, 0.0)

    # No nulls should remain
    assert enriched.null_count().sum_horizontal()[0] == 0


def test_join_weather(sample_weather_df):
    engineer = WeatherFeatureEngineer()
    enriched = engineer.enrich_weather(sample_weather_df)

    # Mock measurements matching timestamps
    meas_df = pl.DataFrame(
        {
            "pool": [1] * 48,
            "datetime_utc": sample_weather_df["forecast_datetime_utc"],
            "is_consumption": [1] * 48,
            "target": [100.0] * 48,
        }
    )

    joined = engineer.join_weather(meas_df, enriched)
    assert joined.height == 48
    assert "wind_speed_10m" in joined.columns
    assert "target" in joined.columns


def test_to_chronos_covariates(sample_weather_df):
    engineer = WeatherFeatureEngineer()
    enriched = engineer.enrich_weather(sample_weather_df)

    features = ["temperature", "wind_speed_10m"]
    covs = engineer.to_chronos_covariates(enriched, feature_names=features)
    assert "temperature" in covs
    assert "wind_speed_10m" in covs
    assert len(covs["temperature"]) == 48
    assert covs["temperature"].dtype == np.float32


def test_baseline_forecaster_with_exogenous():
    # Synthetic time series with an exogenous feature
    timestamps = [datetime(2022, 1, 1, 0, 0) + timedelta(hours=i) for i in range(100)]
    weather_val = [20.0 + 5.0 * np.sin(i / 12.0) for i in range(100)]
    target_val = [w * 2.0 + np.random.normal(0, 0.5) for w in weather_val]

    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "target": target_val,
            "series_id": ["s1"] * 100,
            "weather_temp": weather_val,
        }
    )

    train_df = df.slice(0, 80)
    future_df = df.slice(80, 20)

    forecaster = BaselineForecaster(
        config=BaselineModelConfig(model_type="hist_gradient_boosting", max_iter=50),
        feature_config=FeatureConfig(lags=[1, 2], rolling_windows=[3], rolling_metrics=["mean"]),
        exogenous_columns=["weather_temp"],
    )

    forecaster.fit(train_df)
    assert "weather_temp" in forecaster.feature_names

    # Predict with future exogenous values
    res = forecaster.predict(context_df=train_df, prediction_length=20, future_df=future_df)
    assert res.prediction_length == 20
    assert res.forecast_df.height == 20
    assert "forecast" in res.forecast_df.columns

    # Check feature importances
    importances = forecaster.get_feature_importances(train_df)
    assert "weather_temp" in importances
    assert importances["weather_temp"] >= 0.0


def test_chronos_forecaster_with_covariates(mock_chronos_pipeline):
    forecaster = ChronosForecaster()
    forecaster.pipeline = mock_chronos_pipeline
    forecaster.is_loaded = True

    timestamps = [datetime(2022, 1, 1, 0, 0) + timedelta(hours=i) for i in range(40)]
    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "target": [50.0] * 40,
            "series_id": ["s1"] * 40,
        }
    )

    past_covs = {"temp": np.ones(40, dtype=np.float32)}
    future_covs = {"temp": np.ones(10, dtype=np.float32)}

    res = forecaster.predict(
        df,
        prediction_length=10,
        past_covariates=past_covs,
        future_covariates=future_covs,
    )

    assert res.prediction_length == 10
    assert res.forecast_df.height == 10
    assert "forecast" in res.forecast_df.columns


def test_baseline_predict_one_step_ahead():
    timestamps = [datetime(2022, 1, 1, 0, 0) + timedelta(hours=i) for i in range(50)]
    weather_val = [10.0 + i * 0.1 for i in range(50)]
    target_val = [w * 1.5 + np.random.normal(0, 0.1) for w in weather_val]

    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "target": target_val,
            "series_id": ["s1"] * 50,
            "weather_temp": weather_val,
        }
    )

    train_df = df.slice(0, 35)
    test_df = df.slice(35, 15)

    forecaster = BaselineForecaster(
        config=BaselineModelConfig(model_type="hist_gradient_boosting", max_iter=30),
        feature_config=FeatureConfig(lags=[1, 2], rolling_windows=[3], rolling_metrics=["mean"]),
        exogenous_columns=["weather_temp"],
    )
    forecaster.fit(train_df)

    res = forecaster.predict_one_step_ahead(test_df=test_df, context_df=train_df)
    assert res.prediction_length == 15
    assert res.forecast_df.height == 15
    assert "forecast" in res.forecast_df.columns
    assert "q_10" in res.forecast_df.columns
    assert "q_90" in res.forecast_df.columns


def test_chronos_predict_one_step_ahead(mock_chronos_pipeline):
    forecaster = ChronosForecaster()
    forecaster.pipeline = mock_chronos_pipeline
    forecaster.is_loaded = True

    timestamps = [datetime(2022, 1, 1, 0, 0) + timedelta(hours=i) for i in range(50)]
    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "target": [20.0] * 50,
            "series_id": ["s1"] * 50,
        }
    )

    train_df = df.slice(0, 35)
    test_df = df.slice(35, 15)

    res = forecaster.predict_one_step_ahead(test_df=test_df, context_df=train_df)
    assert res.prediction_length == 15
    assert res.forecast_df.height == 15
    assert "forecast" in res.forecast_df.columns
