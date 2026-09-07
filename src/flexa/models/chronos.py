"""Amazon Chronos-2 foundation model forecaster with PyTorch backend."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

import numpy as np
import polars as pl
import torch

from flexa.config import ChronosModelConfig
from flexa.models.base import BaseForecaster, ForecastResult

logger = logging.getLogger(__name__)


def resolve_device(device_str: str) -> str:
    """Resolve compute device automatically based on available hardware."""
    if device_str != "auto":
        return device_str

    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_torch_dtype(dtype_str: str) -> torch.dtype:
    """Resolve torch dtype from configuration string."""
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping.get(dtype_str, torch.bfloat16)


class ChronosForecaster(BaseForecaster):
    """Zero-shot and probabilistic time series forecaster using Amazon Chronos foundation models."""

    def __init__(
        self,
        config: ChronosModelConfig | None = None,
        time_column: str = "timestamp",
        target_column: str = "target",
        id_column: str | None = "series_id",
    ) -> None:
        super().__init__(time_column, target_column, id_column)
        self.config = config or ChronosModelConfig()
        self.device = resolve_device(self.config.device)
        self.torch_dtype = resolve_torch_dtype(self.config.torch_dtype)

        # On MPS, bfloat16 may have limited support in some operators; fallback to float32 if needed
        if self.device == "mps" and self.torch_dtype == torch.bfloat16:
            logger.info("Using float32 on MPS for maximum operator compatibility.")
            self.torch_dtype = torch.float32

        self.pipeline: Any = None
        self.is_loaded = False

    def load_pipeline(self) -> ChronosForecaster:
        """Load pretrained Chronos pipeline from Hugging Face."""
        if self.is_loaded:
            return self

        from chronos import BaseChronosPipeline

        logger.info(
            f"Loading Chronos model '{self.config.model_id}' on device '{self.device}' with dtype '{self.torch_dtype}'."
        )
        self.pipeline = BaseChronosPipeline.from_pretrained(
            self.config.model_id,
            device_map=self.device,
            torch_dtype=self.torch_dtype,
        )
        self.is_loaded = True
        return self

    def fit(self, df: pl.DataFrame) -> ChronosForecaster:
        """Pre-load the zero-shot pipeline (Chronos foundation models require no task-specific fitting)."""
        return self.load_pipeline()

    def predict(
        self,
        context_df: pl.DataFrame,
        prediction_length: int | None = None,
        quantiles: list[float] | None = None,
        past_covariates: dict[str, np.ndarray] | None = None,
        future_covariates: dict[str, np.ndarray] | None = None,
    ) -> ForecastResult:
        """Generate probabilistic forecasts across all series in the context DataFrame.

        Args:
            context_df: Polars DataFrame containing historical observations.
            prediction_length: Horizon length (defaults to config.prediction_length).
            quantiles: Target quantiles to compute (defaults to [0.1, 0.5, 0.9]).
            past_covariates: Optional dictionary mapping covariate name -> 1D array of past values.
            future_covariates: Optional dictionary mapping covariate name -> 1D array of future values.

        Returns:
            ForecastResult containing point forecasts and quantile estimates.
        """
        if not self.is_loaded:
            self.load_pipeline()

        pred_len = prediction_length or self.config.prediction_length
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]

        series_keys: list[str]
        if self.id_column and self.id_column in context_df.columns:
            series_keys = context_df[self.id_column].unique().to_list()
        else:
            series_keys = ["series_01"]

        model_inputs: list[Any] = []
        series_metadata: list[dict[str, Any]] = []

        for s_key in series_keys:
            if self.id_column and self.id_column in context_df.columns:
                s_df = context_df.filter(pl.col(self.id_column) == s_key).sort(self.time_column)
            else:
                s_df = context_df.sort(self.time_column)

            # Restrict context window if context_length is specified
            if self.config.context_length and s_df.height > self.config.context_length:
                s_df = s_df.slice(
                    s_df.height - self.config.context_length, self.config.context_length
                )

            vals = s_df[self.target_column].to_numpy().astype(np.float32)

            supports_covariates = (
                self.pipeline is not None and "Chronos2" in self.pipeline.__class__.__name__
            )

            if (
                past_covariates is not None
                and future_covariates is not None
                and supports_covariates
            ):
                s_past = {
                    k: v[-len(vals) :] if len(v) >= len(vals) else v
                    for k, v in past_covariates.items()
                }
                s_future = {
                    k: v[:pred_len] if len(v) >= pred_len else v
                    for k, v in future_covariates.items()
                }
                model_inputs.append(
                    {
                        "target": vals,
                        "past_covariates": s_past,
                        "future_covariates": s_future,
                    }
                )
            else:
                if past_covariates is not None and not supports_covariates:
                    logger.warning(
                        "Current pipeline does not support past/future covariates. Falling back to univariate input."
                    )
                model_inputs.append(torch.tensor(vals, dtype=torch.float32))

            last_ts = s_df[self.time_column].max()
            if s_df.height > 1:
                step_delta = s_df[self.time_column][-1] - s_df[self.time_column][-2]
            else:
                step_delta = timedelta(days=1)

            future_timestamps = [last_ts + ((step + 1) * step_delta) for step in range(pred_len)]
            series_metadata.append(
                {
                    "series_id": s_key,
                    "timestamps": future_timestamps,
                }
            )

        # Generate forecasts with Chronos pipeline
        if hasattr(self.pipeline, "predict_quantiles"):
            quantiles_res, mean_res = self.pipeline.predict_quantiles(
                inputs=model_inputs,
                prediction_length=pred_len,
                quantile_levels=quantiles,
            )

            # Robustly convert list of tensors or single tensor to numpy
            if isinstance(quantiles_res, list):
                q_np = np.stack(
                    [
                        q.squeeze(0).detach().cpu().to(torch.float32).numpy()
                        if q.ndim == 3
                        else q.detach().cpu().to(torch.float32).numpy()
                        for q in quantiles_res
                    ]
                )
            else:
                q_np = quantiles_res.detach().cpu().to(torch.float32).numpy()

            if isinstance(mean_res, list):
                mean_np = np.stack(
                    [
                        m.squeeze(0).detach().cpu().to(torch.float32).numpy()
                        if m.ndim == 2
                        else m.detach().cpu().to(torch.float32).numpy()
                        for m in mean_res
                    ]
                )
            else:
                mean_np = mean_res.detach().cpu().to(torch.float32).numpy()

            results = []
            for idx, meta in enumerate(series_metadata):
                record: dict[str, Any] = {
                    "timestamp": meta["timestamps"],
                    "forecast": mean_np[idx].tolist(),
                }
                if self.id_column and self.id_column in context_df.columns:
                    record[self.id_column] = [meta["series_id"]] * pred_len

                for q_idx, q in enumerate(quantiles):
                    col_name = f"q_{int(q * 100):02d}"
                    record[col_name] = q_np[idx, :, q_idx].tolist()

                results.append(pl.DataFrame(record))
        else:
            # Fallback for models only implementing predict
            forecast_samples = self.pipeline.predict(
                inputs=model_inputs,
                prediction_length=pred_len,
                limit_prediction_length=False,
            )
            if isinstance(forecast_samples, torch.Tensor):
                samples_np = forecast_samples.detach().cpu().to(torch.float32).numpy()
            else:
                samples_np = np.asarray(forecast_samples, dtype=np.float32)

            results = []
            for idx, meta in enumerate(series_metadata):
                s_samples = samples_np[idx]
                point_forecast = np.median(s_samples, axis=0)

                record = {
                    "timestamp": meta["timestamps"],
                    "forecast": point_forecast.tolist(),
                }
                if self.id_column and self.id_column in context_df.columns:
                    record[self.id_column] = [meta["series_id"]] * pred_len

                for q in quantiles:
                    q_vals = np.quantile(s_samples, q=q, axis=0)
                    col_name = f"q_{int(q * 100):02d}"
                    record[col_name] = q_vals.tolist()

                results.append(pl.DataFrame(record))

        combined_df = pl.concat(results)
        return ForecastResult(
            model_name=f"chronos_{self.config.model_id.split('/')[-1]}",
            prediction_length=pred_len,
            forecast_df=combined_df,
            quantiles=quantiles,
            metadata={"device": self.device, "model_id": self.config.model_id},
        )

    def predict_one_step_ahead(
        self,
        test_df: pl.DataFrame,
        context_df: pl.DataFrame,
        quantiles: list[float] | None = None,
        covariates_df: pl.DataFrame | None = None,
        covariate_cols: list[str] | None = None,
        batch_size: int = 256,
    ) -> ForecastResult:
        """Generate 1-step-ahead forecasts across test_df using true observations up to T-1.

        For every timestamp T in test_df, the context is formed by trailing observations ending
        at T-1, removing multi-step autoregressive drift.
        """
        if not self.is_loaded:
            self.load_pipeline()

        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]

        c_len = self.config.context_length or 64
        supports_covariates = (
            self.pipeline is not None and "Chronos2" in self.pipeline.__class__.__name__
        )

        series_keys: list[str]
        if self.id_column and self.id_column in test_df.columns:
            series_keys = test_df[self.id_column].unique().to_list()
        else:
            series_keys = ["series_01"]

        results = []

        for s_key in series_keys:
            if self.id_column and self.id_column in test_df.columns:
                s_test = test_df.filter(pl.col(self.id_column) == s_key).sort(self.time_column)
                s_context = context_df.filter(pl.col(self.id_column) == s_key).sort(
                    self.time_column
                )
            else:
                s_test = test_df.sort(self.time_column)
                s_context = context_df.sort(self.time_column)

            n_steps = s_test.height
            timestamps = s_test[self.time_column].to_list()

            # Build full concatenated target array (trailing context + test)
            full_target = np.concatenate(
                [
                    s_context[self.target_column].to_numpy()[-c_len:],
                    s_test[self.target_column].to_numpy(),
                ]
            ).astype(np.float32)

            # Build full covariates dictionary if available
            cov_dict: dict[str, np.ndarray] = {}
            if covariates_df is not None and covariate_cols and supports_covariates:
                if self.id_column and self.id_column in covariates_df.columns:
                    s_cov = covariates_df.filter(pl.col(self.id_column) == s_key).sort(
                        self.time_column
                    )
                else:
                    s_cov = covariates_df.sort(self.time_column)

                req_start = s_context[self.time_column][-c_len]
                matched_cov = s_cov.filter(pl.col(self.time_column) >= req_start).sort(
                    self.time_column
                )
                for col in covariate_cols:
                    if col in matched_cov.columns:
                        cov_dict[col] = matched_cov[col].to_numpy().astype(np.float32)

            model_inputs: list[Any] = []
            has_valid_covs = bool(
                cov_dict and all(len(arr) >= n_steps + c_len for arr in cov_dict.values())
            )

            for i in range(n_steps):
                ctx_target = full_target[i : i + c_len]
                if has_valid_covs:
                    past_c = {k: v[i : i + c_len] for k, v in cov_dict.items()}
                    fut_c = {k: v[i + c_len : i + c_len + 1] for k, v in cov_dict.items()}
                    model_inputs.append(
                        {
                            "target": ctx_target,
                            "past_covariates": past_c,
                            "future_covariates": fut_c,
                        }
                    )
                else:
                    model_inputs.append(torch.tensor(ctx_target, dtype=torch.float32))

            if hasattr(self.pipeline, "predict_quantiles"):
                quantiles_res, mean_res = self.pipeline.predict_quantiles(
                    inputs=model_inputs,
                    prediction_length=1,
                    quantile_levels=quantiles,
                    batch_size=batch_size,
                )

                if isinstance(mean_res, list):
                    point_preds = [float(m.item()) for m in mean_res]
                else:
                    point_preds = [
                        float(p)
                        for p in mean_res.detach().cpu().to(torch.float32).numpy().flatten()
                    ]

                if isinstance(quantiles_res, list):
                    q_arrays = np.stack(
                        [
                            q.squeeze(0).squeeze(0).detach().cpu().to(torch.float32).numpy()
                            if q.ndim == 3
                            else q.squeeze(0).detach().cpu().to(torch.float32).numpy()
                            for q in quantiles_res
                        ]
                    )
                else:
                    q_arrays = quantiles_res.squeeze(1).detach().cpu().to(torch.float32).numpy()

                record: dict[str, Any] = {
                    self.time_column: timestamps,
                    "forecast": point_preds,
                }
                if self.id_column and self.id_column in test_df.columns:
                    record[self.id_column] = [s_key] * n_steps

                for q_idx, q in enumerate(quantiles):
                    col_name = f"q_{int(q * 100):02d}"
                    record[col_name] = q_arrays[:, q_idx].tolist()

                results.append(pl.DataFrame(record))
            else:
                samples = self.pipeline.predict(model_inputs, prediction_length=1)
                samples_np = (
                    samples.detach().cpu().to(torch.float32).numpy()
                    if isinstance(samples, torch.Tensor)
                    else np.asarray(samples, dtype=np.float32)
                )
                point_preds = np.median(samples_np.squeeze(-1), axis=1).tolist()
                record = {
                    self.time_column: timestamps,
                    "forecast": point_preds,
                }
                if self.id_column and self.id_column in test_df.columns:
                    record[self.id_column] = [s_key] * n_steps
                for q in quantiles:
                    q_vals = np.quantile(samples_np.squeeze(-1), q=q, axis=1).tolist()
                    record[f"q_{int(q * 100):02d}"] = q_vals
                results.append(pl.DataFrame(record))

        combined_res = pl.concat(results)
        return ForecastResult(
            model_name=f"chronos_{self.config.model_id.split('/')[-1]}_1step",
            prediction_length=test_df.height,
            forecast_df=combined_res,
            quantiles=quantiles,
            metadata={
                "device": self.device,
                "model_id": self.config.model_id,
                "mode": "1_step_ahead_oracle_t_minus_1",
            },
        )
