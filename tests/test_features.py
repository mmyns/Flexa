"""Tests for Polars-powered feature engineering."""

import numpy as np
import pytest

from flexa.config import FeatureConfig
from flexa.features.time_features import TimeSeriesFeatureEngineer


def test_calendar_and_cyclical_features(synthetic_df):
    config = FeatureConfig(
        lags=[1, 2],
        rolling_windows=[7],
        rolling_metrics=["mean"],
        include_calendar=True,
        include_cyclical=True,
    )
    fe = TimeSeriesFeatureEngineer(config=config, id_column="series_id")
    df_feat = fe.create_features(synthetic_df)

    expected_cols = [
        "day_of_week",
        "day_of_month",
        "month",
        "sin_hour",
        "cos_hour",
        "sin_month",
        "cos_month",
        "lag_1",
        "lag_2",
        "roll_mean_7",
    ]
    for col in expected_cols:
        assert col in df_feat.columns

    # Check bounds of cyclical features
    for col in ["sin_hour", "cos_hour", "sin_month", "cos_month"]:
        vals = df_feat[col].to_numpy()
        assert np.all(vals >= -1.0 - 1e-6)
        assert np.all(vals <= 1.0 + 1e-6)


def test_lag_values_correctness(single_series_df):
    config = FeatureConfig(
        lags=[1, 3], rolling_windows=[], include_calendar=False, include_cyclical=False
    )
    fe = TimeSeriesFeatureEngineer(config=config, id_column="series_id")
    df_feat = fe.create_features(single_series_df)

    targets = df_feat["target"].to_list()
    lag_1 = df_feat["lag_1"].to_list()
    lag_3 = df_feat["lag_3"].to_list()

    # Verify lag 1
    assert lag_1[0] is None
    assert lag_1[1] == targets[0]
    assert lag_1[5] == targets[4]

    # Verify lag 3
    assert lag_3[0] is None
    assert lag_3[1] is None
    assert lag_3[2] is None
    assert lag_3[3] == targets[0]


def test_rolling_window_no_data_leakage(single_series_df):
    config = FeatureConfig(
        lags=[1],
        rolling_windows=[3],
        rolling_metrics=["mean"],
        include_calendar=False,
        include_cyclical=False,
    )
    fe = TimeSeriesFeatureEngineer(config=config, id_column="series_id")
    df_feat = fe.create_features(single_series_df)

    # For index 3, the rolling mean of size 3 should be mean of targets[0], targets[1], targets[2]
    # NOT including target[3]
    targets = df_feat["target"].to_numpy()
    roll_mean = df_feat["roll_mean_3"].to_numpy()

    expected_mean_at_3 = np.mean(targets[0:3])
    assert pytest.approx(roll_mean[3], rel=1e-5) == expected_mean_at_3


def test_delta_lags_feature(single_series_df):
    config = FeatureConfig(
        lags=[1, 2],
        delta_lags=[(1, 2)],
        rolling_windows=[],
        include_calendar=False,
        include_cyclical=False,
    )
    fe = TimeSeriesFeatureEngineer(config=config, id_column="series_id")
    df_feat = fe.create_features(single_series_df)

    assert "delta_1_2" in df_feat.columns
    targets = df_feat["target"].to_list()
    deltas = df_feat["delta_1_2"].to_list()

    # row 0 & 1 must be None due to lag 2 requirement
    assert deltas[0] is None
    assert deltas[1] is None
    # row 2: targets[1] - targets[0]
    assert pytest.approx(deltas[2], rel=1e-5) == targets[1] - targets[0]
    # row 3: targets[2] - targets[1]
    assert pytest.approx(deltas[3], rel=1e-5) == targets[2] - targets[1]
