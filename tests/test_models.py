"""Tests for Baseline and Chronos forecasting models."""

import numpy as np
import polars as pl
import pytest

from flexa.config import BaselineModelConfig, ChronosModelConfig, FeatureConfig
from flexa.models.baseline import BaselineForecaster
from flexa.models.chronos import ChronosForecaster, resolve_device


def test_baseline_forecaster_workflow(synthetic_df):
    model_cfg = BaselineModelConfig(model_type="hist_gradient_boosting", max_iter=20)
    feat_cfg = FeatureConfig(lags=[1, 2, 7], rolling_windows=[7])

    forecaster = BaselineForecaster(
        config=model_cfg,
        feature_config=feat_cfg,
        id_column="series_id",
    )

    # Train on first 70 rows per series
    train_df = synthetic_df.filter(
        pl.col("timestamp") < pl.col("timestamp").max() - pl.duration(days=14)
    )
    forecaster.fit(train_df)
    assert forecaster.is_fitted is True

    # Predict 14 steps
    pred_len = 14
    result = forecaster.predict(train_df, prediction_length=pred_len, quantiles=[0.1, 0.5, 0.9])

    assert result.prediction_length == pred_len
    assert result.forecast_df.height == 2 * pred_len  # 2 series
    assert "forecast" in result.forecast_df.columns
    assert "q_10" in result.forecast_df.columns
    assert "q_90" in result.forecast_df.columns

    # Check quantile ordering: q_10 <= q_90
    q10 = result.forecast_df["q_10"].to_numpy()
    q90 = result.forecast_df["q_90"].to_numpy()
    assert np.all(q10 <= q90)


def test_baseline_unfitted_raises(synthetic_df):
    forecaster = BaselineForecaster()
    with pytest.raises(RuntimeError, match="must be fit before"):
        forecaster.predict(synthetic_df, prediction_length=5)


def test_chronos_forecaster_with_mock(synthetic_df, mock_chronos_pipeline):
    chronos_cfg = ChronosModelConfig(
        model_id="mock/chronos-test",
        prediction_length=7,
        num_samples=10,
    )
    forecaster = ChronosForecaster(config=chronos_cfg, id_column="series_id")
    forecaster.pipeline = mock_chronos_pipeline
    forecaster.is_loaded = True

    result = forecaster.predict(synthetic_df, prediction_length=7, quantiles=[0.1, 0.5, 0.9])

    assert result.prediction_length == 7
    assert result.forecast_df.height == 2 * 7  # 2 series * 7 steps
    assert "forecast" in result.forecast_df.columns
    assert "q_10" in result.forecast_df.columns
    assert "q_50" in result.forecast_df.columns
    assert "q_90" in result.forecast_df.columns

    # Quantile monotonicity check: q10 <= q50 <= q90
    q10 = result.forecast_df["q_10"].to_numpy()
    q50 = result.forecast_df["q_50"].to_numpy()
    q90 = result.forecast_df["q_90"].to_numpy()
    assert np.all(q10 <= q50 + 1e-5)
    assert np.all(q50 <= q90 + 1e-5)


def test_device_resolution():
    assert resolve_device("cpu") == "cpu"
    auto_dev = resolve_device("auto")
    assert auto_dev in ["cpu", "mps", "cuda"]


def test_t_minus_1_forecaster(synthetic_df):
    from flexa.models.naive import TMinus1Forecaster

    forecaster = TMinus1Forecaster(
        time_column="timestamp", target_column="target", id_column="series_id"
    )
    # Train on first 70 rows
    train_df = synthetic_df.filter(
        pl.col("timestamp") < pl.col("timestamp").max() - pl.duration(days=14)
    )
    test_df = synthetic_df.filter(
        pl.col("timestamp") >= pl.col("timestamp").max() - pl.duration(days=14)
    )

    forecaster.fit(train_df)
    assert forecaster.is_fitted is True

    # Multi-step predict
    pred_res = forecaster.predict(train_df, prediction_length=7)
    assert pred_res.forecast_df.height == 2 * 7

    # 1-step-ahead predict
    one_step_res = forecaster.predict_one_step_ahead(test_df, context_df=train_df)
    assert one_step_res.forecast_df.height == test_df.height
    assert "forecast" in one_step_res.forecast_df.columns
    assert "q_10" in one_step_res.forecast_df.columns
    assert "q_90" in one_step_res.forecast_df.columns

    # For a specific series, verify forecast at index 1 matches true target at index 0 of test_df
    s01_test = test_df.filter(pl.col("series_id") == "series_01").sort("timestamp")
    s01_pred = one_step_res.forecast_df.filter(pl.col("series_id") == "series_01").sort("timestamp")
    assert np.isclose(s01_pred["forecast"][1], s01_test["target"][0])


def test_ridge_forecaster(synthetic_df):
    from flexa.models.baseline import RidgeForecaster

    feat_cfg = FeatureConfig(lags=[1, 2], rolling_windows=[])
    forecaster = RidgeForecaster(
        alpha=1.0,
        feature_config=feat_cfg,
        time_column="timestamp",
        target_column="target",
        id_column="series_id",
    )

    train_df = synthetic_df.filter(
        pl.col("timestamp") < pl.col("timestamp").max() - pl.duration(days=14)
    )
    test_df = synthetic_df.filter(
        pl.col("timestamp") >= pl.col("timestamp").max() - pl.duration(days=14)
    )

    forecaster.fit(train_df)
    assert forecaster.is_fitted is True

    res = forecaster.predict_one_step_ahead(test_df, context_df=train_df)
    assert res.forecast_df.height == test_df.height
    # Non-negative predictions
    assert np.all(res.forecast_df["forecast"].to_numpy() >= 0.0)


def test_add_paired_telemetry():
    from datetime import datetime

    from flexa.features.time_features import add_paired_telemetry

    data = pl.DataFrame(
        {
            "pool": [0, 0, 0, 0],
            "datetime_utc": [
                datetime(2023, 1, 1, 0),
                datetime(2023, 1, 1, 0),
                datetime(2023, 1, 1, 1),
                datetime(2023, 1, 1, 1),
            ],
            "is_consumption": [0, 1, 0, 1],
            "target": [5.0, 100.0, 10.0, 120.0],
        }
    )

    paired = add_paired_telemetry(data)
    assert "injection" in paired.columns
    assert "consumption" in paired.columns
    assert "solar_peak_to_t" in paired.columns
    assert paired.height == 4

    row0 = paired.filter(
        (pl.col("datetime_utc") == datetime(2023, 1, 1, 0)) & (pl.col("is_consumption") == 0)
    )
    assert row0["injection"][0] == 5.0
    assert row0["consumption"][0] == 100.0
    assert row0["solar_peak_to_t"][0] == 0.0

    row1 = paired.filter(
        (pl.col("datetime_utc") == datetime(2023, 1, 1, 1)) & (pl.col("is_consumption") == 0)
    )
    assert row1["solar_peak_to_t"][0] == 5.0

    # Test with context_df
    data_next = pl.DataFrame(
        {
            "pool": [0, 0],
            "datetime_utc": [datetime(2023, 1, 1, 2), datetime(2023, 1, 1, 2)],
            "is_consumption": [0, 1],
            "target": [8.0, 130.0],
        }
    )
    paired_with_ctx = add_paired_telemetry(data_next, context_df=data)
    assert paired_with_ctx["solar_peak_to_t"].to_list() == [10.0, 10.0]


def test_baseline_rolling_walk_forward(synthetic_df):
    from flexa.config import BaselineModelConfig, FeatureConfig
    from flexa.models.baseline import BaselineForecaster

    feat_cfg = FeatureConfig(lags=[1, 2], rolling_windows=[])
    forecaster = BaselineForecaster(
        config=BaselineModelConfig(model_type="ridge"),
        feature_config=feat_cfg,
        time_column="timestamp",
        target_column="target",
        id_column="series_id",
    )

    # Train context: first 60 days, Test: last 30 days
    split_ts = synthetic_df["timestamp"].min() + pl.duration(days=60)
    train_df = synthetic_df.filter(pl.col("timestamp") < split_ts)
    test_df = synthetic_df.filter(pl.col("timestamp") >= split_ts)

    res = forecaster.predict_rolling_walk_forward(
        test_df=test_df,
        context_df=train_df,
        window_days=30,
        quantiles=[0.1, 0.5, 0.9],
    )

    assert res.prediction_length == test_df.height
    assert res.forecast_df.height == test_df.height
    assert "forecast" in res.forecast_df.columns
    assert "q_10" in res.forecast_df.columns
    assert "q_90" in res.forecast_df.columns
    assert np.all(res.forecast_df["forecast"].to_numpy() >= 0.0)


def test_baseline_predict_difference(synthetic_df):
    from flexa.config import BaselineModelConfig, FeatureConfig
    from flexa.models.baseline import BaselineForecaster

    feat_cfg = FeatureConfig(lags=[1, 2], rolling_windows=[])
    forecaster = BaselineForecaster(
        config=BaselineModelConfig(model_type="hist_gradient_boosting", max_iter=20),
        feature_config=feat_cfg,
        time_column="timestamp",
        target_column="target",
        id_column="series_id",
        predict_difference=True,
    )

    split_ts = synthetic_df["timestamp"].min() + pl.duration(days=60)
    train_df = synthetic_df.filter(pl.col("timestamp") < split_ts)
    test_df = synthetic_df.filter(pl.col("timestamp") >= split_ts)

    forecaster.fit(train_df)
    assert forecaster.is_fitted is True

    # Test 1-step ahead
    pred_1step = forecaster.predict_one_step_ahead(test_df, context_df=train_df)
    assert pred_1step.forecast_df.height == test_df.height
    assert np.all(pred_1step.forecast_df["forecast"].to_numpy() >= 0.0)

    # Test rolling walk-forward with diff
    res_rolling = forecaster.predict_rolling_walk_forward(
        test_df=test_df,
        context_df=train_df,
        window_days=30,
        predict_difference=True,
    )
    assert res_rolling.forecast_df.height == test_df.height
    assert np.all(res_rolling.forecast_df["forecast"].to_numpy() >= 0.0)
