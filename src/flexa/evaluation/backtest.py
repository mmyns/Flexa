"""Time series backtesting engine supporting expanding and rolling temporal cross-validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from flexa.config import BacktestConfig
from flexa.evaluation.metrics import evaluate_forecast
from flexa.models.base import BaseForecaster, ForecastResult


@dataclass
class BacktestFoldResult:
    """Evaluation result for a single backtest fold."""

    fold_idx: int
    cutoff_timestamp: str
    metrics: dict[str, float]
    forecast_result: ForecastResult


@dataclass
class BacktestSummary:
    """Aggregated evaluation across all backtest folds."""

    model_name: str
    num_folds: int
    fold_results: list[BacktestFoldResult]
    mean_metrics: dict[str, float]
    std_metrics: dict[str, float]


class Backtester:
    """Executes expanding or rolling window temporal backtesting."""

    def __init__(
        self,
        config: BacktestConfig | None = None,
        time_column: str = "timestamp",
        target_column: str = "target",
        id_column: str | None = "series_id",
    ) -> None:
        self.config = config or BacktestConfig()
        self.time_column = time_column
        self.target_column = target_column
        self.id_column = id_column

    def run(
        self,
        forecaster: BaseForecaster,
        df: pl.DataFrame,
    ) -> BacktestSummary:
        """Run temporal cross-validation over the dataset.

        Args:
            forecaster: The forecasting model instance.
            df: Complete historical Polars DataFrame.

        Returns:
            BacktestSummary with fold-by-fold results and aggregate statistics.
        """
        sort_cols = (
            [self.id_column, self.time_column]
            if self.id_column and self.id_column in df.columns
            else [self.time_column]
        )
        sorted_df = df.sort(sort_cols)

        # Get unique timestamps to define cutoffs
        unique_timestamps = sorted_df[self.time_column].unique().sort()
        total_time_steps = unique_timestamps.len()

        test_size = self.config.test_size
        stride = self.config.stride
        n_splits = self.config.n_splits

        # Ensure enough data points
        required_steps = test_size + (n_splits - 1) * stride + 20
        if total_time_steps < required_steps:
            raise ValueError(
                f"Not enough timestamps ({total_time_steps}) for {n_splits} splits with "
                f"test_size={test_size} and stride={stride}. Minimum required is {required_steps}."
            )

        fold_results: list[BacktestFoldResult] = []

        for fold in range(n_splits):
            # Calculate end index for test set in this fold (working backwards)
            offset = (n_splits - 1 - fold) * stride
            test_end_idx = total_time_steps - offset
            test_start_idx = test_end_idx - test_size
            cutoff_ts = unique_timestamps[test_start_idx - 1]
            end_ts = unique_timestamps[test_end_idx - 1]

            # Split data
            if self.config.strategy == "expanding":
                train_df = sorted_df.filter(pl.col(self.time_column) <= cutoff_ts)
            else:  # rolling
                window_size = total_time_steps - (n_splits * stride) - test_size
                train_start_ts = unique_timestamps[max(0, test_start_idx - window_size)]
                train_df = sorted_df.filter(
                    (pl.col(self.time_column) >= train_start_ts)
                    & (pl.col(self.time_column) <= cutoff_ts)
                )

            test_df = sorted_df.filter(
                (pl.col(self.time_column) > cutoff_ts) & (pl.col(self.time_column) <= end_ts)
            )

            # Fit and predict
            forecaster.fit(train_df)
            forecast_res = forecaster.predict(
                context_df=train_df,
                prediction_length=test_size,
            )

            # Join actuals with predictions
            join_cols = [self.time_column]
            if self.id_column and self.id_column in sorted_df.columns:
                join_cols.append(self.id_column)

            eval_df = test_df.join(
                forecast_res.to_polars(),
                on=join_cols,
                how="inner",
            )

            if eval_df.is_empty():
                continue

            y_true = eval_df[self.target_column].to_numpy()
            y_pred = eval_df["forecast"].to_numpy()

            quantiles_dict: dict[float, np.ndarray] = {}
            for q in forecast_res.quantiles:
                q_col = f"q_{int(q * 100):02d}"
                if q_col in eval_df.columns:
                    quantiles_dict[q] = eval_df[q_col].to_numpy()

            metrics = evaluate_forecast(y_true, y_pred, quantiles_dict)

            fold_results.append(
                BacktestFoldResult(
                    fold_idx=fold + 1,
                    cutoff_timestamp=str(cutoff_ts),
                    metrics=metrics,
                    forecast_result=forecast_res,
                )
            )

        if not fold_results:
            raise RuntimeError("Backtesting generated no valid evaluated folds.")

        # Compute aggregate metrics
        metric_keys = fold_results[0].metrics.keys()
        mean_metrics = {
            k: float(np.mean([f.metrics[k] for f in fold_results])) for k in metric_keys
        }
        std_metrics = {k: float(np.std([f.metrics[k] for f in fold_results])) for k in metric_keys}

        return BacktestSummary(
            model_name=fold_results[0].forecast_result.model_name,
            num_folds=len(fold_results),
            fold_results=fold_results,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
        )
