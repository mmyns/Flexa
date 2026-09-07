"""End-to-end forecasting pipeline coordinator."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import polars as pl

from flexa.config import PipelineConfig
from flexa.data.loader import generate_synthetic_timeseries, load_timeseries
from flexa.data.preprocessor import TimeSeriesPreprocessor
from flexa.data.validation import validate_timeseries
from flexa.evaluation.backtest import Backtester, BacktestSummary
from flexa.evaluation.metrics import evaluate_forecast
from flexa.models.base import BaseForecaster, ForecastResult
from flexa.models.baseline import BaselineForecaster

logger = logging.getLogger(__name__)


@dataclass
class ForecastRunResult:
    """Output of a single model forecast and evaluation run."""

    model_name: str
    forecast_result: ForecastResult
    evaluation_df: pl.DataFrame
    metrics: dict[str, float]


@dataclass
class BenchmarkComparison:
    """Side-by-side benchmark comparison between baseline and Chronos foundation model."""

    baseline_metrics: dict[str, float]
    chronos_metrics: dict[str, float]
    comparison_df: pl.DataFrame


class ForecastingPipeline:
    """High-level pipeline for loading, training, forecasting, and benchmarking."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.preprocessor = TimeSeriesPreprocessor(
            time_column=self.config.data.time_column,
            target_column=self.config.data.target_column,
            id_column=self.config.data.id_column,
        )

    def load_data(self, data_path: str | None = None) -> pl.DataFrame:
        """Load data from file or generate synthetic benchmark data."""
        if data_path:
            logger.info(f"Loading time series from: {data_path}")
            df = load_timeseries(
                data_path,
                time_column=self.config.data.time_column,
                target_column=self.config.data.target_column,
                id_column=self.config.data.id_column,
            )
        else:
            logger.info("Generating synthetic time series benchmark dataset.")
            df = generate_synthetic_timeseries(
                n_series=2,
                length=180,
                seed=42,
            )

        # Validate
        validate_timeseries(
            df,
            time_column=self.config.data.time_column,
            target_column=self.config.data.target_column,
            id_column=self.config.data.id_column,
            raise_on_error=True,
        )
        return df

    def get_forecaster(self, model_type: str = "chronos") -> BaseForecaster:
        """Instantiate forecaster based on requested type ('baseline' or 'chronos')."""
        if model_type == "baseline":
            return BaselineForecaster(
                config=self.config.baseline,
                feature_config=self.config.features,
                time_column=self.config.data.time_column,
                target_column=self.config.data.target_column,
                id_column=self.config.data.id_column,
            )
        elif model_type == "chronos":
            # Imported here so that torch remains an optional dependency; only
            # this branch needs it.
            from flexa.models import ChronosForecaster

            return ChronosForecaster(
                config=self.config.chronos,
                time_column=self.config.data.time_column,
                target_column=self.config.data.target_column,
                id_column=self.config.data.id_column,
            )
        else:
            raise ValueError(f"Unknown model_type '{model_type}'. Use 'baseline' or 'chronos'.")

    def run_forecast(
        self,
        forecaster: BaseForecaster,
        df: pl.DataFrame,
        test_size: int | None = None,
    ) -> ForecastRunResult:
        """Execute temporal train/test split, forecast, and evaluation."""
        horizon = test_size or self.config.chronos.prediction_length

        split = self.preprocessor.temporal_train_test_split(df, test_size=horizon)
        train_df, test_df = split.train, split.test

        forecaster.fit(train_df)
        forecast_res = forecaster.predict(context_df=train_df, prediction_length=horizon)

        # Evaluate
        join_cols = [self.config.data.time_column]
        if self.config.data.id_column and self.config.data.id_column in df.columns:
            join_cols.append(self.config.data.id_column)

        eval_df = test_df.join(
            forecast_res.to_polars(),
            on=join_cols,
            how="inner",
        )

        y_true = eval_df[self.config.data.target_column].to_numpy()
        y_pred = eval_df["forecast"].to_numpy()

        quantiles_dict = {}
        for q in forecast_res.quantiles:
            q_col = f"q_{int(q * 100):02d}"
            if q_col in eval_df.columns:
                quantiles_dict[q] = eval_df[q_col].to_numpy()

        metrics = evaluate_forecast(y_true, y_pred, quantiles_dict)

        return ForecastRunResult(
            model_name=forecast_res.model_name,
            forecast_result=forecast_res,
            evaluation_df=eval_df,
            metrics=metrics,
        )

    def run_backtest(
        self,
        forecaster: BaseForecaster,
        df: pl.DataFrame,
    ) -> BacktestSummary:
        """Execute full temporal backtesting."""
        backtester = Backtester(
            config=self.config.backtest,
            time_column=self.config.data.time_column,
            target_column=self.config.data.target_column,
            id_column=self.config.data.id_column,
        )
        return backtester.run(forecaster, df)

    def compare_models(
        self,
        df: pl.DataFrame,
        test_size: int | None = None,
    ) -> BenchmarkComparison:
        """Run both scikit-learn baseline and Chronos foundation models side-by-side."""
        horizon = test_size or self.config.chronos.prediction_length

        baseline_forecaster = self.get_forecaster("baseline")
        chronos_forecaster = self.get_forecaster("chronos")

        baseline_run = self.run_forecast(baseline_forecaster, df, test_size=horizon)
        chronos_run = self.run_forecast(chronos_forecaster, df, test_size=horizon)

        # Build comparison Polars DataFrame
        metrics_list = ["mae", "rmse", "mape", "wape"]
        rows: list[dict[str, Any]] = []
        for m in metrics_list:
            b_val = baseline_run.metrics.get(m, float("nan"))
            c_val = chronos_run.metrics.get(m, float("nan"))
            diff_pct = ((c_val - b_val) / b_val * 100.0) if b_val != 0 else 0.0
            rows.append(
                {
                    "metric": m.upper(),
                    "baseline": round(b_val, 4),
                    "chronos": round(c_val, 4),
                    "diff_pct": round(diff_pct, 2),
                }
            )

        comparison_df = pl.DataFrame(rows)

        return BenchmarkComparison(
            baseline_metrics=baseline_run.metrics,
            chronos_metrics=chronos_run.metrics,
            comparison_df=comparison_df,
        )
