# Flexa: Production-Ready Time Series Forecasting

A production-grade time series forecasting engine combining **Polars** for high-performance temporal feature engineering, **scikit-learn** for classical machine learning baselines, and **Amazon's `chronos-forecasting` (Chronos-2 / Chronos-Bolt)** with **PyTorch** for zero-shot probabilistic foundation model forecasting. Managed with **uv**.

---

## Key Highlights

- ⚡ **High-Throughput Feature Pipeline**: Vectorized calendar, cyclical trigonometric, autoregressive lag, and leakage-free rolling window features using Polars expressions.
- 🤖 **Chronos-2 Foundation Models**: Zero-shot and probabilistic time series forecasting with Amazon Chronos (`amazon/chronos-bolt-small`, `amazon/chronos-bolt-base`, and Chronos-2 checkpoints). Auto-detects Apple Silicon Metal (`mps`), CUDA (`cuda`), or CPU.
- 📈 **Classical Baselines**: Recursive multi-step forecasters with scikit-learn (`HistGradientBoostingRegressor` and `Ridge`) for rigorous benchmarking against foundation models.
- 📊 **Probabilistic & Deterministic Evaluation**: Computes MAE, RMSE, MAPE, sMAPE, WAPE, and Quantile (Pinball) loss with confidence intervals.
- 🔁 **Temporal Backtesting Engine**: Expanding and rolling window cross-validation avoiding lookahead bias.
- 📦 **Modern Tooling & Packaging**: Managed by `uv`, PEP 621 `pyproject.toml`, Typer CLI, Ruff linting/formatting, MyPy typing, and Pytest test suite with CI/CD.

---

## Repository Structure

```text
Flexa/
├── Dockerfile                     # Container image (uv + uv.lock, core deps only)
├── docker-compose.yml             # compose run --rm forecast | battery | report | all
├── docker/entrypoint.sh           # Task dispatcher inside the image
├── .github/workflows/ci.yml       # GitHub Actions CI workflow
├── .python-version                # Pinned to Python 3.11
├── .env.example                   # Environment variable template
├── Makefile                       # make forecast | battery | report | docker-*
├── pyproject.toml                 # Packaging metadata, dependencies, tool settings
├── uv.lock                        # Fully pinned, cross-platform lockfile
├── configs/
│   └── default.yaml               # Declarative experiment & model configuration
├── data/
│   ├── raw/                       # 1_measurements.csv, 1_weather.csv, 2_electricity_prices.csv
│   ├── processed/                 # Train/test splits, battery schedules, PDF reports
│   ├── forecasts/                 # Champion forecasts + evaluation summary
│   └── benchmarks/                # Model-selection ablation results (report input)
├── src/
│   └── flexa/                     # Core Python package
│       ├── __init__.py            # Public API exports
│       ├── config.py              # Pydantic configuration schemas
│       ├── data/                  # loader.py, preprocessor.py, validation.py
│       ├── features/              # time_features.py, weather_features.py
│       ├── models/                # base.py, baseline.py (champion), naive.py, chronos.py
│       ├── evaluation/            # metrics.py, backtest.py
│       ├── optimization/          # battery.py, objectives.py, optimizer.py (CVXPY)
│       ├── pipeline.py            # End-to-end forecasting pipeline
│       └── cli.py                 # Typer command-line interface
├── scripts/
│   ├── run_forecast.py            # ► Entry point 1 — forecasting
│   ├── run_battery.py             # ► Entry point 2 — battery steering
│   ├── split_data.py              # Raw CSV → train/test Parquet splits
│   ├── generate_forecast_report.py  # Forecasting section of the report
│   ├── generate_bess_report.py      # BESS section of the report
│   └── generate_full_report.py    # ► Combined 9-page PDF report
├── notebooks/                     # marimo notebooks (EDA, forecasting exploration)
└── tests/                         # Pytest test suite
```

---

## Quickstart

### 1. Run with Docker (no Python needed on the host)

The container is the reproducible path — it pins every dependency from `uv.lock`, so the
same versions resolve on macOS, Linux and Windows/WSL2.

```bash
docker build -t flexa:latest .

# Entry point 1 — forecasting
docker run --rm -v "$(pwd)/data:/app/data" flexa:latest forecast

# Entry point 2 — battery steering
docker run --rm -v "$(pwd)/data:/app/data" flexa:latest battery

# Combined PDF report, or the whole chain from raw CSV to final PDF
docker run --rm -v "$(pwd)/data:/app/data" flexa:latest report
docker run --rm -v "$(pwd)/data:/app/data" flexa:latest all
```

The `-v "$(pwd)/data:/app/data"` bind mount is what makes results land on the host: the
container reads `data/raw` and `data/benchmarks` from your checkout and writes its outputs
back into the same directory. `docker compose run --rm forecast|battery|report|all` are
equivalent shorthands. Any extra flags are forwarded verbatim to the underlying script:

```bash
docker run --rm -v "$(pwd)/data:/app/data" flexa:latest forecast --pools 0 4 11
docker run --rm -v "$(pwd)/data:/app/data" flexa:latest battery --days 90 --max-cycles 1.0
```

| Command    | Runs                              | Writes                                                        |
|------------|-----------------------------------|---------------------------------------------------------------|
| `forecast` | `scripts/run_forecast.py`         | `data/forecasts/forecasts_champion_model.{parquet,csv}` + summary JSON |
| `battery`  | `scripts/run_battery.py`          | `data/processed/schedule_{1h,2h}_battery.{csv,parquet}`       |
| `report`   | `scripts/generate_full_report.py` | `data/processed/flexa_full_report.pdf`                        |
| `split`    | `scripts/split_data.py`           | `data/processed/{train,test}_*.{parquet,csv}`                 |
| `all`      | all four, in order                | everything above                                              |
| `shell`    | `/bin/bash`                       | —                                                              |

### 2. Run natively with uv

Identical results — the container simply wraps these commands.

```bash
uv sync --frozen                                # create .venv from uv.lock

uv run python scripts/run_forecast.py           # Entry point 1 — forecasting
uv run python scripts/run_battery.py            # Entry point 2 — battery steering
uv run python scripts/generate_full_report.py   # Combined PDF report

make forecast | make battery | make report      # same three, via the Makefile
```

### 3. Optional: the Chronos foundation model

`torch`, `transformers` and `accelerate` are pulled in **only** by
`chronos-forecasting`, which is a benchmark comparator rather than part of either entry
point. They sit behind an extra and are absent from the runtime container:

```bash
uv sync --extra chronos     # adds torch + chronos-forecasting
```

Without the extra, `flexa.ChronosForecaster` raises an `ImportError` naming the command
above; everything else works unchanged.

### 4. Command Line Interface (CLI)

Flexa provides an intuitive CLI:

```bash
# Generate synthetic benchmark data
uv run flexa synthetic --n-series 3 --length 180 --output data/raw/synthetic.csv

# Run Chronos zero-shot forecast (needs the 'chronos' extra)
uv run flexa forecast --data data/raw/synthetic.csv --model chronos --horizon 14

# Run Scikit-Learn baseline forecast
uv run flexa forecast --data data/raw/synthetic.csv --model baseline --horizon 14

# Compare Baseline vs Chronos side-by-side (needs the 'chronos' extra)
uv run flexa benchmark --data data/raw/synthetic.csv --horizon 14
```

---

## Python API Usage

### Zero-Shot Forecasting with Chronos

> Requires the optional extra: `uv sync --extra chronos`.

```python
import polars as pl
from flexa.data.loader import generate_synthetic_timeseries
from flexa.models.chronos import ChronosForecaster
from flexa.config import ChronosModelConfig

# 1. Load or generate data with Polars
df = generate_synthetic_timeseries(n_series=2, length=120)

# 2. Configure Chronos model
config = ChronosModelConfig(
    model_id="amazon/chronos-bolt-small",  # or "amazon/chronos-bolt-base" / "amazon/chronos-2"
    device="auto",  # auto-selects mps, cuda, or cpu
    prediction_length=14,
    num_samples=20,
)

# 3. Forecast
forecaster = ChronosForecaster(config=config, id_column="series_id")
result = forecaster.predict(context_df=df, prediction_length=14, quantiles=[0.1, 0.5, 0.9])

# 4. View forecast table as Polars DataFrame
forecast_df = result.to_polars()
print(forecast_df.head(10))
```

### Classical Baseline Forecaster (Scikit-Learn)

```python
from flexa.config import BaselineModelConfig, FeatureConfig
from flexa.models.baseline import BaselineForecaster

feat_config = FeatureConfig(lags=[1, 2, 7, 14], rolling_windows=[7, 14])
baseline = BaselineForecaster(
    config=BaselineModelConfig(model_type="hist_gradient_boosting"),
    feature_config=feat_config,
    id_column="series_id",
)

baseline.fit(train_df)
result = baseline.predict(context_df=train_df, prediction_length=14)
print(result.to_polars())
```

### Temporal Backtesting

```python
from flexa.evaluation.backtest import Backtester
from flexa.config import BacktestConfig

backtester = Backtester(
    config=BacktestConfig(strategy="expanding", n_splits=3, test_size=14, stride=7),
    id_column="series_id",
)

summary = backtester.run(forecaster=baseline, df=df)
print("Mean MAE across folds:", summary.mean_metrics["mae"])
print("Mean WAPE across folds:", summary.mean_metrics["wape"])
```

---

## Development & Verification

Developer recipes are available via `make`:

```bash
make sync            # Sync frozen lockfile dependencies
make lint            # Run ruff and mypy
make format          # Format code with ruff
make test            # Run pytest test suite
make coverage        # Run pytest with code coverage report
make clean           # Remove caches and temporary files

make forecast        # Entry point 1 — forecasting
make battery         # Entry point 2 — battery steering
make report          # Build the combined PDF report

make docker-build    # Build the image
make docker-forecast # Run forecasting inside the container
make docker-battery  # Run battery steering inside the container
make docker-report   # Build the report inside the container
```

---

## License

MIT License.
