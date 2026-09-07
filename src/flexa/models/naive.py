"""Naive persistence baselines including T-1 lag forecaster."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import norm

from flexa.models.base import BaseForecaster, ForecastResult


class TMinus1Forecaster(BaseForecaster):
    """T-1 persistence baseline forecaster.

    In 1-step-ahead prediction with telemetry available at T-1,
    the prediction at timestamp T is identically the observed target at T-1:
    y_hat_T = y_{T-1}.
    """

    def __init__(
        self,
        time_column: str = "timestamp",
        target_column: str = "target",
        id_column: str | None = "series_id",
    ) -> None:
        super().__init__(time_column, target_column, id_column)
        self.is_fitted = False
        self.residual_std: float = 1.0
        self.last_observed: dict[str, float] = {}

    def fit(self, df: pl.DataFrame) -> TMinus1Forecaster:
        """Fit empirical residual standard deviation for uncertainty estimation."""
        series_keys: list[str]
        if self.id_column and self.id_column in df.columns:
            series_keys = df[self.id_column].unique().to_list()
        else:
            series_keys = ["series_01"]

        all_diffs: list[float] = []
        for s_key in series_keys:
            s_df = (
                df.filter(pl.col(self.id_column) == s_key).sort(self.time_column)
                if self.id_column and self.id_column in df.columns
                else df.sort(self.time_column)
            )
            if s_df.height > 1:
                diffs = s_df[self.target_column].diff().drop_nulls().to_numpy()
                all_diffs.extend(diffs)
                self.last_observed[s_key] = float(s_df[self.target_column][-1])

        if all_diffs:
            self.residual_std = float(np.std(all_diffs)) or 1.0
        else:
            self.residual_std = 1.0

        self.is_fitted = True
        return self

    def predict(
        self,
        context_df: pl.DataFrame,
        prediction_length: int,
        quantiles: list[float] | None = None,
    ) -> ForecastResult:
        """Generate multi-step flat persistence forecast: y_hat_{T+h} = y_T."""
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]

        series_keys: list[str]
        if self.id_column and self.id_column in context_df.columns:
            series_keys = context_df[self.id_column].unique().to_list()
        else:
            series_keys = ["series_01"]

        results = []
        for s_key in series_keys:
            s_context = (
                context_df.filter(pl.col(self.id_column) == s_key).sort(self.time_column)
                if self.id_column and self.id_column in context_df.columns
                else context_df.sort(self.time_column)
            )

            last_val = float(s_context[self.target_column][-1]) if s_context.height > 0 else 0.0
            last_ts = s_context[self.time_column][-1]
            future_ts = [last_ts + timedelta(hours=i + 1) for i in range(prediction_length)]
            forecast_points = [last_val] * prediction_length

            series_res: dict[str, list[Any]] = {
                self.time_column: future_ts,
                "forecast": forecast_points,
            }
            if self.id_column and self.id_column in context_df.columns:
                series_res[self.id_column] = [s_key] * prediction_length

            for q in quantiles:
                z = norm.ppf(q)
                col_name = f"q_{int(q * 100):02d}"
                series_res[col_name] = [float(p + z * self.residual_std) for p in forecast_points]

            results.append(pl.DataFrame(series_res))

        combined_res = pl.concat(results)
        return ForecastResult(
            model_name="t_minus_1_persistence",
            prediction_length=prediction_length,
            forecast_df=combined_res,
            quantiles=quantiles,
            metadata={"residual_std": self.residual_std},
        )

    def predict_one_step_ahead(
        self,
        test_df: pl.DataFrame,
        context_df: pl.DataFrame | None = None,
        quantiles: list[float] | None = None,
    ) -> ForecastResult:
        """Generate 1-step-ahead forecast where prediction at T is true observation at T-1."""
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]

        series_keys: list[str]
        if self.id_column and self.id_column in test_df.columns:
            series_keys = test_df[self.id_column].unique().to_list()
        else:
            series_keys = ["series_01"]

        results = []
        for s_key in series_keys:
            s_test = (
                test_df.filter(pl.col(self.id_column) == s_key).sort(self.time_column)
                if self.id_column and self.id_column in test_df.columns
                else test_df.sort(self.time_column)
            )
            s_context = (
                context_df.filter(pl.col(self.id_column) == s_key).sort(self.time_column)
                if context_df is not None
                and self.id_column
                and self.id_column in context_df.columns
                else (context_df.sort(self.time_column) if context_df is not None else None)
            )

            # Combine trailing observation with test_df to calculate lag_1
            if s_context is not None and s_context.height > 0:
                combined = pl.concat([s_context.tail(1), s_test])
            else:
                combined = s_test

            shifted = combined.with_columns(
                pl.col(self.target_column).shift(1).alias("t_minus_1_pred")
            )

            test_start_ts = s_test[self.time_column].min()
            eval_df = shifted.filter(pl.col(self.time_column) >= test_start_ts).sort(
                self.time_column
            )

            preds = (
                eval_df["t_minus_1_pred"].fill_null(strategy="forward").fill_null(0.0).to_numpy()
            )
            # Clip at 0 for physical non-negative energy power
            preds = np.clip(preds, 0.0, None)

            series_res: dict[str, list[Any]] = {
                self.time_column: eval_df[self.time_column].to_list(),
                "forecast": [float(p) for p in preds],
            }
            if self.id_column and self.id_column in test_df.columns:
                series_res[self.id_column] = [s_key] * eval_df.height

            for q in quantiles:
                z = norm.ppf(q)
                col_name = f"q_{int(q * 100):02d}"
                series_res[col_name] = [float(max(0.0, p + z * self.residual_std)) for p in preds]

            results.append(pl.DataFrame(series_res))

        combined_res = pl.concat(results)
        return ForecastResult(
            model_name="t_minus_1_baseline",
            prediction_length=test_df.height,
            forecast_df=combined_res,
            quantiles=quantiles,
            metadata={"residual_std": self.residual_std, "mode": "1_step_ahead_oracle_t_minus_1"},
        )


# Convenient alias
NaiveForecaster = TMinus1Forecaster
