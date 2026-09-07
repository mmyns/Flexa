"""Configuration schemas for data, features, models, and pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataConfig(BaseModel):
    """Configuration for data ingestion and temporal validation."""

    time_column: str = "timestamp"
    target_column: str = "target"
    id_column: str | None = "series_id"
    freq: str = "1d"


class FeatureConfig(BaseModel):
    """Configuration for time series feature engineering with Polars."""

    lags: list[int] = Field(default_factory=lambda: [1, 2, 3, 7, 14, 28])
    delta_lags: list[tuple[int, int]] = Field(default_factory=list)
    rolling_windows: list[int] = Field(default_factory=lambda: [7, 14, 30])
    rolling_metrics: list[str] = Field(default_factory=lambda: ["mean", "std", "min", "max"])
    include_calendar: bool = True
    include_cyclical: bool = True
    calendar_features: list[str] = Field(
        default_factory=lambda: ["day_of_week", "day_of_month", "month", "hour"]
    )
    exclude_features: list[str] = Field(default_factory=list)


class BaselineModelConfig(BaseModel):
    """Configuration for classical scikit-learn forecaster."""

    model_type: Literal["hist_gradient_boosting", "ridge"] = "hist_gradient_boosting"
    l2_regularization: float = 0.1
    max_iter: int = 200
    random_state: int = 42


class ChronosModelConfig(BaseModel):
    """Configuration for Amazon Chronos foundation model."""

    model_id: str = "amazon/chronos-2"
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    torch_dtype: Literal["bfloat16", "float32", "float16"] = "bfloat16"
    prediction_length: int = 14
    context_length: int = 64
    num_samples: int = 20
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0


class BacktestConfig(BaseModel):
    """Configuration for time series backtesting."""

    strategy: Literal["expanding", "rolling"] = "expanding"
    n_splits: int = 3
    test_size: int = 14
    stride: int = 7


class PipelineConfig(BaseSettings):
    """Master configuration for time series forecasting pipeline."""

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    baseline: BaselineModelConfig = Field(default_factory=BaselineModelConfig)
    chronos: ChronosModelConfig = Field(default_factory=ChronosModelConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        """Load configuration from a YAML file."""
        yaml_path = Path(path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, encoding="utf-8") as f:
            raw_data: dict[str, Any] = yaml.safe_load(f) or {}

        return cls(**raw_data)
