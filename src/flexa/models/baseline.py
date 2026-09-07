"""Classical machine learning baseline forecasters powered by scikit-learn."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from flexa.config import BaselineModelConfig, FeatureConfig
from flexa.features.time_features import TimeSeriesFeatureEngineer
from flexa.models.base import BaseForecaster, ForecastResult


class BaselineForecaster(BaseForecaster):
    """Autoregressive tabular time series forecaster using scikit-learn."""

    def __init__(
        self,
        config: BaselineModelConfig | None = None,
        feature_config: FeatureConfig | None = None,
        time_column: str = "timestamp",
        target_column: str = "target",
        id_column: str | None = "series_id",
        exogenous_columns: list[str] | None = None,
        autoregressive_columns: list[str] | None = None,
        categorical_columns: list[str] | None = None,
        predict_difference: bool = False,
    ) -> None:
        super().__init__(time_column, target_column, id_column)
        self.config = config or BaselineModelConfig()
        self.exogenous_columns = list(exogenous_columns) if exogenous_columns else []
        self.autoregressive_columns = (
            list(autoregressive_columns) if autoregressive_columns else None
        )
        self.categorical_columns = (
            list(categorical_columns) if categorical_columns is not None else ["day_of_week"]
        )
        self.predict_difference = predict_difference
        self.feature_engineer = TimeSeriesFeatureEngineer(
            config=feature_config,
            time_column=time_column,
            target_column=target_column,
            id_column=id_column,
            autoregressive_columns=self.autoregressive_columns,
        )

        self.model: Any = None
        self.is_fitted = False
        self.feature_names: list[str] = []
        self.residual_std: float = 1.0

    def _find_target_lag_col(self, columns: list[str]) -> str:
        """Find the lag-1 feature column associated with the target."""
        candidates = [
            f"{self.target_column}_lag_1",
            "target_lag_1",
            "lag_1",
        ]
        for c in candidates:
            if c in columns:
                return c
        for c in columns:
            if c.endswith("_lag_1"):
                return c
        raise ValueError(
            f"No lag-1 feature found for target '{self.target_column}' in columns: {columns}"
        )

    def _build_estimator(self, cat_indices: list[int]) -> Any:
        """Instantiate a fresh estimator based on model_type and categorical indices."""
        if self.config.model_type == "hist_gradient_boosting":
            return HistGradientBoostingRegressor(
                categorical_features=cat_indices if cat_indices else None,
                l2_regularization=self.config.l2_regularization,
                max_iter=self.config.max_iter,
                random_state=self.config.random_state,
            )
        elif self.config.model_type == "ridge":
            if cat_indices:
                from sklearn.compose import ColumnTransformer
                from sklearn.preprocessing import OneHotEncoder

                num_indices = [i for i in range(len(self.feature_names)) if i not in cat_indices]
                transformers: list[Any] = [
                    (
                        "cat",
                        OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                        cat_indices,
                    ),
                ]
                if num_indices:
                    transformers.append(("num", StandardScaler(), num_indices))

                return make_pipeline(
                    ColumnTransformer(transformers=transformers),
                    Ridge(
                        alpha=self.config.l2_regularization,
                        random_state=self.config.random_state,
                    ),
                )
            else:
                return make_pipeline(
                    StandardScaler(),
                    Ridge(
                        alpha=self.config.l2_regularization,
                        random_state=self.config.random_state,
                    ),
                )
        else:
            raise ValueError(f"Unsupported model_type: {self.config.model_type}")

    def fit(self, df: pl.DataFrame) -> BaselineForecaster:
        """Engineer features and fit the scikit-learn estimator."""
        df_feat = self.feature_engineer.create_features(df)
        available_exo = [c for c in self.exogenous_columns if c in df.columns]
        all_candidate_feats = self.feature_engineer.feature_names + available_exo
        if self.feature_engineer.config.exclude_features:
            self.feature_names = [
                f
                for f in all_candidate_feats
                if f not in self.feature_engineer.config.exclude_features
            ]
        else:
            self.feature_names = all_candidate_feats

        if self.predict_difference:
            lag_col = self._find_target_lag_col(df_feat.columns)
            df_feat = df_feat.with_columns(
                (pl.col(self.target_column) - pl.col(lag_col)).alias("_target_diff")
            )
            train_target_col = "_target_diff"
        else:
            train_target_col = self.target_column

        # Drop initial rows with nulls caused by lags/rolling windows
        valid_df = df_feat.drop_nulls(subset=[*self.feature_names, train_target_col])

        if valid_df.is_empty():
            raise ValueError("Not enough historical data to compute all lags and rolling features.")

        cat_cols = [c for c in self.categorical_columns if c in self.feature_names]
        cat_indices = [i for i, name in enumerate(self.feature_names) if name in cat_cols]

        self.model = self._build_estimator(cat_indices)

        X = valid_df.select(self.feature_names).to_numpy()
        y = valid_df[train_target_col].to_numpy()

        self.model.fit(X, y)
        t_min = valid_df[self.target_column].min()
        target_non_negative = isinstance(t_min, (int, float)) and t_min >= 0.0

        if self.predict_difference:
            y_diff_pred = self.model.predict(X)
            y_lag1 = valid_df[lag_col].to_numpy()
            y_pred = y_lag1 + y_diff_pred
            if target_non_negative:
                y_pred = np.clip(y_pred, 0.0, None)
            self.residual_std = (
                float(np.std(valid_df[self.target_column].to_numpy() - y_pred)) or 1.0
            )
        else:
            y_pred = self.model.predict(X)
            if target_non_negative:
                y_pred = np.clip(y_pred, 0.0, None)
            self.residual_std = float(np.std(y - y_pred)) or 1.0
        self.is_fitted = True

        return self

    def get_feature_importances(
        self,
        df: pl.DataFrame,
        n_repeats: int = 5,
        random_state: int = 42,
    ) -> dict[str, float]:
        """Compute permutation feature importances for all model features."""
        if not self.is_fitted:
            raise RuntimeError("Forecaster must be fit before computing feature importance.")

        from sklearn.inspection import permutation_importance

        df_feat = self.feature_engineer.create_features(df)
        valid_df = df_feat.drop_nulls(subset=[*self.feature_names, self.target_column])
        if valid_df.is_empty():
            return {}

        X = valid_df.select(self.feature_names).to_numpy()
        if self.predict_difference:
            lag_1_col = self._find_target_lag_col(df_feat.columns)
            if lag_1_col and lag_1_col in valid_df.columns:
                y = valid_df[self.target_column].to_numpy() - valid_df[lag_1_col].to_numpy()
            else:
                y = valid_df[self.target_column].to_numpy()
        else:
            y = valid_df[self.target_column].to_numpy()

        res = permutation_importance(
            self.model, X, y, n_repeats=n_repeats, random_state=random_state
        )
        return dict(zip(self.feature_names, [float(v) for v in res.importances_mean], strict=True))

    def predict(
        self,
        context_df: pl.DataFrame,
        prediction_length: int,
        quantiles: list[float] | None = None,
        future_df: pl.DataFrame | None = None,
        autoregressive: bool = True,
    ) -> ForecastResult:
        """Generate forecasts per series.

        If autoregressive=False and future_df contains the actual target column,
        it evaluates 1-step-ahead forecasts using the true T-1 observations.
        """
        if not self.is_fitted:
            raise RuntimeError("BaselineForecaster must be fit before calling predict.")

        if not autoregressive and future_df is not None and self.target_column in future_df.columns:
            target_slice = future_df.slice(0, prediction_length)
            return self.predict_one_step_ahead(
                test_df=target_slice,
                context_df=context_df,
                quantiles=quantiles,
            )

        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]

        series_keys: list[str]
        if self.id_column and self.id_column in context_df.columns:
            series_keys = context_df[self.id_column].unique().to_list()
        else:
            series_keys = ["series_01"]

        results = []

        for s_key in series_keys:
            if self.id_column and self.id_column in context_df.columns:
                s_context = context_df.filter(pl.col(self.id_column) == s_key).sort(
                    self.time_column
                )
            else:
                s_context = context_df.sort(self.time_column)

            last_timestamp = s_context[self.time_column].max()
            # Estimate step delta
            if s_context.height > 1:
                step_delta = s_context[self.time_column][-1] - s_context[self.time_column][-2]
            else:
                step_delta = timedelta(days=1)

            current_df = s_context
            forecast_points: list[float] = []
            forecast_timestamps: list[Any] = []

            for step in range(1, prediction_length + 1):
                next_ts = last_timestamp + (step * step_delta)
                forecast_timestamps.append(next_ts)

                # Append placeholder row for the next step to extract engineered features
                placeholder_row: dict[str, list[Any]] = {
                    self.time_column: [next_ts],
                    self.target_column: [0.0],
                }
                if self.id_column and self.id_column in current_df.columns:
                    placeholder_row[self.id_column] = [s_key]

                # Populate exogenous columns from future_df if provided
                if self.exogenous_columns:
                    if future_df is not None:
                        f_match = future_df.filter(pl.col(self.time_column) == next_ts)
                        if self.id_column and self.id_column in f_match.columns:
                            f_match = f_match.filter(pl.col(self.id_column) == s_key)
                        for col in self.exogenous_columns:
                            if col in f_match.columns and f_match.height > 0:
                                placeholder_row[col] = [f_match[col][0]]
                            elif col in current_df.columns:
                                placeholder_row[col] = [current_df[col][-1]]
                            else:
                                placeholder_row[col] = [0.0]
                    else:
                        for col in self.exogenous_columns:
                            if col in current_df.columns:
                                placeholder_row[col] = [current_df[col][-1]]
                            else:
                                placeholder_row[col] = [0.0]

                # Align columns with current_df
                missing_in_ph = [c for c in current_df.columns if c not in placeholder_row]
                for c in missing_in_ph:
                    placeholder_row[c] = [current_df[c][-1]]

                extended_df = pl.concat(
                    [current_df, pl.DataFrame(placeholder_row).select(current_df.columns)]
                )
                feat_df = self.feature_engineer.create_features(extended_df)

                # Extract features for the final row
                latest_x = feat_df.select(self.feature_names)[-1].to_numpy()

                # Replace nulls in feature row if any
                latest_x = np.nan_to_num(latest_x, nan=0.0)

                pred_val = float(self.model.predict(latest_x)[0])
                forecast_points.append(pred_val)

                # Update current_df with predicted value for recursive continuation
                update_row: dict[str, list[Any]] = {
                    self.time_column: [next_ts],
                    self.target_column: [pred_val],
                }
                if self.id_column and self.id_column in current_df.columns:
                    update_row[self.id_column] = [s_key]

                for col in self.exogenous_columns:
                    if col in placeholder_row:
                        update_row[col] = placeholder_row[col]

                for c in missing_in_ph:
                    if c not in update_row:
                        update_row[c] = [current_df[c][-1]]

                current_df = pl.concat(
                    [current_df, pl.DataFrame(update_row).select(current_df.columns)]
                )

            series_res = {
                "timestamp": forecast_timestamps,
                "forecast": forecast_points,
            }
            if self.id_column and self.id_column in context_df.columns:
                series_res[self.id_column] = [s_key] * prediction_length

            # Normal distribution quantile estimates based on residual error
            from scipy.stats import norm

            for q in quantiles:
                z = norm.ppf(q)
                col_name = f"q_{int(q * 100):02d}"
                series_res[col_name] = [float(p + z * self.residual_std) for p in forecast_points]

            results.append(pl.DataFrame(series_res))

        combined_df = pl.concat(results)
        return ForecastResult(
            model_name=f"baseline_{self.config.model_type}",
            prediction_length=prediction_length,
            forecast_df=combined_df,
            quantiles=quantiles,
            metadata={"residual_std": self.residual_std},
        )

    def predict_one_step_ahead(
        self,
        test_df: pl.DataFrame,
        context_df: pl.DataFrame | None = None,
        quantiles: list[float] | None = None,
    ) -> ForecastResult:
        """Generate 1-step-ahead predictions across the entire test_df using true observations up to T-1.

        This eliminates autoregressive error accumulation by strictly using actual observed history
        at T-1, T-2, ... for every timestamp T in test_df.
        """
        if not self.is_fitted:
            raise RuntimeError("BaselineForecaster must be fit before calling predict.")

        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]

        series_keys: list[str]
        if self.id_column and self.id_column in test_df.columns:
            series_keys = test_df[self.id_column].unique().to_list()
        else:
            series_keys = ["series_01"]

        results = []
        for s_key in series_keys:
            if self.id_column and self.id_column in test_df.columns:
                s_test = test_df.filter(pl.col(self.id_column) == s_key).sort(self.time_column)
                s_context = (
                    context_df.filter(pl.col(self.id_column) == s_key).sort(self.time_column)
                    if context_df is not None
                    else None
                )
            else:
                s_test = test_df.sort(self.time_column)
                s_context = context_df.sort(self.time_column) if context_df is not None else None

            # Concatenate trailing context (e.g. last 200 rows) with test_df to compute lags without leakage
            if s_context is not None and s_context.height > 0:
                max_history = min(s_context.height, 200)
                combined_df = pl.concat([s_context.tail(max_history), s_test])
            else:
                combined_df = s_test

            # Engineer features across the combined series
            feat_df = self.feature_engineer.create_features(combined_df)

            # Isolate the rows belonging to test_df
            test_start_ts = s_test[self.time_column].min()
            eval_feat = feat_df.filter(pl.col(self.time_column) >= test_start_ts).sort(
                self.time_column
            )

            X_test = eval_feat.select(self.feature_names).to_numpy()
            X_test = np.nan_to_num(X_test, nan=0.0)

            is_positive = False
            if self.target_column in eval_feat.columns:
                e_min = eval_feat[self.target_column].min()
                is_positive = isinstance(e_min, (int, float)) and e_min >= 0.0
            if self.predict_difference:
                lag_col = self._find_target_lag_col(eval_feat.columns)
                y_diff = self.model.predict(X_test)
                y_lag1 = eval_feat[lag_col].to_numpy()
                y_pred = y_lag1 + y_diff
                if is_positive:
                    y_pred = np.clip(y_pred, 0.0, None)
            else:
                y_pred = self.model.predict(X_test)
                if is_positive:
                    y_pred = np.clip(y_pred, 0.0, None)

            series_res: dict[str, list[Any]] = {
                self.time_column: eval_feat[self.time_column].to_list(),
                "forecast": [float(p) for p in y_pred],
            }
            if self.id_column and self.id_column in test_df.columns:
                series_res[self.id_column] = [s_key] * eval_feat.height

            from scipy.stats import norm

            for q in quantiles:
                z = norm.ppf(q)
                col_name = f"q_{int(q * 100):02d}"
                series_res[col_name] = [float(max(0.0, p + z * self.residual_std)) for p in y_pred]

            results.append(pl.DataFrame(series_res))

        combined_res = pl.concat(results)
        return ForecastResult(
            model_name=f"baseline_{self.config.model_type}_1step",
            prediction_length=test_df.height,
            forecast_df=combined_res,
            quantiles=quantiles,
            metadata={"residual_std": self.residual_std, "mode": "1_step_ahead_oracle_t_minus_1"},
        )

    def predict_rolling_walk_forward(
        self,
        test_df: pl.DataFrame,
        context_df: pl.DataFrame,
        window_days: int = 30,
        quantiles: list[float] | None = None,
        predict_difference: bool | None = None,
    ) -> ForecastResult:
        """Run rolling walk-forward evaluation retrained every day using a trailing sliding window.

        For each day in test_df, the model is retrained strictly on the trailing `window_days`
        preceding that day, and predicts 1-step-ahead forecasts for all hours in that day.
        Features that require more historical data than `window_days` are naturally dropped/infeasible,
        mitigating data drift while maintaining lean, high-signal short-term features.
        """
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]

        use_diff = self.predict_difference if predict_difference is None else predict_difference

        series_keys: list[str]
        if self.id_column and self.id_column in test_df.columns:
            series_keys = test_df[self.id_column].unique().to_list()
        else:
            series_keys = ["series_01"]

        results = []
        all_residuals: list[float] = []

        for s_key in series_keys:
            if self.id_column and self.id_column in test_df.columns:
                s_test = test_df.filter(pl.col(self.id_column) == s_key).sort(self.time_column)
                s_context = (
                    context_df.filter(pl.col(self.id_column) == s_key).sort(self.time_column)
                    if context_df is not None
                    else None
                )
            else:
                s_test = test_df.sort(self.time_column)
                s_context = context_df.sort(self.time_column) if context_df is not None else None

            if s_context is not None and s_context.height > 0:
                s_combined = pl.concat([s_context, s_test])
            else:
                s_combined = s_test

            feat_df = self.feature_engineer.create_features(s_combined)

            available_exo = [c for c in self.exogenous_columns if c in s_combined.columns]
            feature_names = self.feature_engineer.feature_names + available_exo
            self.feature_names = feature_names

            if use_diff:
                lag_col = self._find_target_lag_col(feat_df.columns)
                feat_df = feat_df.with_columns(
                    (pl.col(self.target_column) - pl.col(lag_col)).alias("_target_diff")
                )
                train_target_col = "_target_diff"
            else:
                train_target_col = self.target_column

            cat_cols = [c for c in self.categorical_columns if c in feature_names]
            cat_indices = [i for i, name in enumerate(feature_names) if name in cat_cols]

            test_dates = (
                s_test.select(pl.col(self.time_column).dt.date())
                .unique()
                .sort(self.time_column)[self.time_column]
                .to_list()
            )

            s_timestamps: list[Any] = []
            s_preds: list[float] = []

            for cur_date in test_dates:
                day_slice = feat_df.filter(pl.col(self.time_column).dt.date() == cur_date).sort(
                    self.time_column
                )
                if day_slice.is_empty():
                    continue

                train_start = cur_date - timedelta(days=window_days)
                train_slice = feat_df.filter(
                    (pl.col(self.time_column).dt.date() >= train_start)
                    & (pl.col(self.time_column).dt.date() < cur_date)
                )

                valid_train = train_slice.drop_nulls(subset=[*feature_names, train_target_col])
                if valid_train.is_empty():
                    valid_train = (
                        feat_df.filter(pl.col(self.time_column).dt.date() < cur_date)
                        .drop_nulls(subset=[*feature_names, train_target_col])
                        .tail(window_days * 24)
                    )

                if valid_train.is_empty():
                    valid_train = feat_df.filter(
                        pl.col(self.time_column).dt.date() <= cur_date
                    ).drop_nulls(subset=[*feature_names, train_target_col])

                X_train = valid_train.select(feature_names).to_numpy()
                y_train = valid_train[train_target_col].to_numpy()

                model = self._build_estimator(cat_indices)
                model.fit(X_train, y_train)
                self.model = model
                self.is_fitted = True

                X_day = day_slice.select(feature_names).to_numpy()
                X_day = np.nan_to_num(X_day, nan=0.0)
                p_day = model.predict(X_day)

                v_min = valid_train[self.target_column].min()
                is_positive = isinstance(v_min, (int, float)) and v_min >= 0.0
                if use_diff:
                    day_lag1 = day_slice[lag_col].to_numpy()
                    y_day_pred = day_lag1 + p_day
                    if is_positive:
                        y_day_pred = np.clip(y_day_pred, 0.0, None)
                else:
                    y_day_pred = p_day
                    if is_positive:
                        y_day_pred = np.clip(y_day_pred, 0.0, None)

                s_timestamps.extend(day_slice[self.time_column].to_list())
                s_preds.extend([float(p) for p in y_day_pred])

                if self.target_column in day_slice.columns:
                    y_day_true = day_slice[self.target_column].to_numpy()
                    all_residuals.extend((y_day_true - y_day_pred).tolist())

            series_res: dict[str, list[Any]] = {
                self.time_column: s_timestamps,
                "forecast": s_preds,
            }
            if self.id_column and self.id_column in test_df.columns:
                series_res[self.id_column] = [s_key] * len(s_timestamps)

            res_std = float(np.std(all_residuals)) if all_residuals else 1.0
            self.residual_std = res_std
            from scipy.stats import norm

            for q in quantiles:
                z = norm.ppf(q)
                col_name = f"q_{int(q * 100):02d}"
                series_res[col_name] = [float(p + z * res_std) for p in s_preds]

            results.append(pl.DataFrame(series_res))

        combined_res = pl.concat(results)
        return ForecastResult(
            model_name=f"baseline_{self.config.model_type}_rolling_{window_days}d",
            prediction_length=len(combined_res),
            forecast_df=combined_res,
            quantiles=quantiles,
            metadata={
                "residual_std": self.residual_std,
                "window_days": window_days,
                "mode": "rolling_walk_forward_daily_retrained",
                "predict_difference": use_diff,
            },
        )


class RidgeForecaster(BaselineForecaster):
    """Linear Ridge regression forecaster with feature standardization and L2 regularization."""

    def __init__(
        self,
        alpha: float = 1.0,
        feature_config: FeatureConfig | None = None,
        time_column: str = "timestamp",
        target_column: str = "target",
        id_column: str | None = "series_id",
        exogenous_columns: list[str] | None = None,
        autoregressive_columns: list[str] | None = None,
        categorical_columns: list[str] | None = None,
        random_state: int = 42,
        predict_difference: bool = False,
    ) -> None:
        cfg = BaselineModelConfig(
            model_type="ridge",
            l2_regularization=alpha,
            random_state=random_state,
        )
        super().__init__(
            config=cfg,
            feature_config=feature_config,
            time_column=time_column,
            target_column=target_column,
            id_column=id_column,
            exogenous_columns=exogenous_columns,
            autoregressive_columns=autoregressive_columns,
            categorical_columns=categorical_columns,
            predict_difference=predict_difference,
        )
