"""Integration tests for the complete ForecastingPipeline."""

import polars as pl

from flexa.pipeline import ForecastingPipeline


def test_pipeline_load_and_validate(sample_config):
    pipeline = ForecastingPipeline(config=sample_config)
    df = pipeline.load_data()

    assert isinstance(df, pl.DataFrame)
    assert df.height > 0
    assert "timestamp" in df.columns
    assert "target" in df.columns


def test_pipeline_baseline_forecast_run(sample_config, synthetic_df):
    pipeline = ForecastingPipeline(config=sample_config)
    forecaster = pipeline.get_forecaster("baseline")

    run_result = pipeline.run_forecast(forecaster, synthetic_df, test_size=7)

    assert run_result.model_name == "baseline_hist_gradient_boosting"
    assert run_result.evaluation_df.height == 2 * 7
    assert "mae" in run_result.metrics
    assert "rmse" in run_result.metrics
    assert run_result.metrics["mae"] >= 0.0


def test_pipeline_chronos_forecast_run_with_mock(
    sample_config, synthetic_df, mock_chronos_pipeline
):
    pipeline = ForecastingPipeline(config=sample_config)
    forecaster = pipeline.get_forecaster("chronos")
    forecaster.pipeline = mock_chronos_pipeline
    forecaster.is_loaded = True

    run_result = pipeline.run_forecast(forecaster, synthetic_df, test_size=7)

    assert "chronos" in run_result.model_name
    assert run_result.evaluation_df.height == 2 * 7
    assert "mae" in run_result.metrics
    assert "pinball_q_10" in run_result.metrics
    assert "pinball_q_90" in run_result.metrics


def test_pipeline_backtest_with_baseline(sample_config, synthetic_df):
    pipeline = ForecastingPipeline(config=sample_config)
    forecaster = pipeline.get_forecaster("baseline")

    summary = pipeline.run_backtest(forecaster, synthetic_df)

    assert summary.num_folds == sample_config.backtest.n_splits
    assert len(summary.fold_results) == sample_config.backtest.n_splits
    assert "mae" in summary.mean_metrics
    assert "rmse" in summary.mean_metrics
