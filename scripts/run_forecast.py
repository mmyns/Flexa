"""Production execution and evaluation script for the champion forecasting model.

This script executes:
1. Loading processed telemetry and enriching exogenous weather features
   (surface_solar_radiation_downwards, temperature, snowfall, cloudcover_delta_1_2, solar_peak_to_t).
2. Training the champion Gradient Boosted Decision Tree (GBDT) on First Differences (Delta T)
   using autoregressive lags [1, 24, 168], delta lag (1, 2), and calendar features.
3. Generating 1-step ahead hourly forecasts for all 14 pools over the test period
   for Solar Injection, Electricity Consumption, and derived Net Load.
4. Exporting the granular forecast predictions to Parquet and CSV.
5. Reporting detailed MAE and WAPE metrics for each pool separately and all pools combined.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from flexa.config import BaselineModelConfig, FeatureConfig
from flexa.features.time_features import add_paired_telemetry
from flexa.features.weather_features import WeatherFeatureEngineer
from flexa.models.baseline import BaselineForecaster
from flexa.models.naive import TMinus1Forecaster

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("flexa.forecast")
console = Console()


def compute_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Weighted Absolute Percentage Error (WAPE) as a percentage."""
    denom = np.sum(np.abs(y_true))
    return float(np.sum(np.abs(y_true - y_pred)) / (denom if denom > 0 else 1.0) * 100.0)


def build_feature_configs():
    """Build standardized feature configurations for the champion Delta T model."""
    f_cfg = FeatureConfig(
        lags=[1, 24, 168],
        delta_lags=[(1, 2)],
        rolling_windows=[],
        rolling_metrics=[],
        include_calendar=True,
        include_cyclical=True,
        calendar_features=["day_of_week", "hour"],
        exclude_features=["month", "day_of_month", "wind_speed_10m"],
    )
    b_cfg = BaselineModelConfig(
        model_type="hist_gradient_boosting",
        l2_regularization=1.0,
        max_iter=150,
        random_state=42,
    )
    return f_cfg, b_cfg


def load_and_prepare_data(data_dir: Path):
    """Load train/test telemetry and join enriched weather features."""
    console.print("[cyan]Loading processed datasets...[/cyan]")
    train_raw = pl.read_parquet(data_dir / "train_measurements.parquet")
    test_raw = pl.read_parquet(data_dir / "test_measurements.parquet")

    # Add paired telemetry (both consumption and injection columns available at each row)
    df_train_meas = add_paired_telemetry(train_raw)
    df_test_meas = add_paired_telemetry(test_raw, context_df=train_raw)

    df_train_weather = pl.read_parquet(data_dir / "train_weather.parquet")
    df_test_weather = pl.read_parquet(data_dir / "test_weather.parquet")

    wf_engineer = WeatherFeatureEngineer()
    train_w_enriched = wf_engineer.enrich_weather(df_train_weather)
    test_w_enriched = wf_engineer.enrich_weather(df_test_weather)

    train_joined = wf_engineer.join_weather(df_train_meas, train_w_enriched)
    test_joined = wf_engineer.join_weather(df_test_meas, test_w_enriched)

    return train_joined, test_joined


def run_forecasting(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    target_series: int,
    series_name: str,
    pools: list[int],
    f_cfg: FeatureConfig,
    b_cfg: BaselineModelConfig,
    model_variant: str = "global",
) -> tuple[pl.DataFrame, list[dict], dict]:
    """Train champion model and evaluate per pool and combined."""
    if target_series == 0:
        raw_w_cols = [
            "surface_solar_radiation_downwards",
            "temperature",
            "snowfall",
            "cloudcover_delta_1_2",
        ]
    else:
        raw_w_cols = [
            "temperature",
            "snowfall",
            "cloudcover_delta_1_2",
        ]

    train_series = train_df.filter(pl.col("is_consumption") == target_series).sort(
        ["pool", "datetime_utc"]
    )
    test_series = test_df.filter(pl.col("is_consumption") == target_series).sort(
        ["pool", "datetime_utc"]
    )

    active_exo = [f for f in raw_w_cols if f in train_series.columns]
    if "solar_peak_to_t" in train_series.columns:
        active_exo.append("solar_peak_to_t")

    ar_cols = ["target", "consumption", "injection"]
    cat_cols = ["day_of_week"]

    forecast_dfs = []
    pool_metrics = []

    if model_variant == "global":
        console.print(
            f"[bold]Fitting Global Champion GBDT (Delta T) on all 14 pools for {series_name}...[/bold]"
        )
        exo_global = ["pool"] + active_exo
        forecaster = BaselineForecaster(
            config=b_cfg,
            feature_config=f_cfg,
            time_column="datetime_utc",
            target_column="target",
            id_column="pool",
            exogenous_columns=exo_global,
            autoregressive_columns=ar_cols,
            categorical_columns=["day_of_week", "pool"],
            predict_difference=True,
        )
        forecaster.fit(train_series)
        preds = forecaster.predict_one_step_ahead(test_series, context_df=train_series)
        df_forecasts = preds.forecast_df

        # Evaluate per pool
        for p_id in pools:
            p_actual = (
                test_series.filter(pl.col("pool") == p_id)
                .sort("datetime_utc")["target"]
                .to_numpy()
            )
            p_pred_df = df_forecasts.filter(pl.col("pool") == p_id).sort("datetime_utc")
            p_pred = p_pred_df["forecast"].to_numpy()

            # Naive baseline for comparison
            m_t1 = TMinus1Forecaster(
                time_column="datetime_utc", target_column="target", id_column=None
            )
            p_train = train_series.filter(pl.col("pool") == p_id)
            p_test = test_series.filter(pl.col("pool") == p_id)
            m_t1.fit(p_train)
            t1_pred = m_t1.predict_one_step_ahead(p_test, context_df=p_train)
            y_t1 = t1_pred.forecast_df["forecast"].to_numpy()
            mae_t1 = mean_absolute_error(p_actual, y_t1)

            mae = mean_absolute_error(p_actual, p_pred)
            rmse = root_mean_squared_error(p_actual, p_pred)
            wape_val = compute_wape(p_actual, p_pred)
            mean_kw = float(np.mean(p_actual))
            max_kw = float(np.max(p_actual))
            imp_pct = ((mae_t1 - mae) / mae_t1 * 100.0) if mae_t1 > 0 else 0.0

            pool_metrics.append(
                {
                    "pool": p_id,
                    "mean_kw": mean_kw,
                    "max_kw": max_kw,
                    "t1_mae": float(mae_t1),
                    "mae": float(mae),
                    "rmse": float(rmse),
                    "wape": float(wape_val),
                    "improvement_pct": float(imp_pct),
                }
            )

            p_eval_df = p_pred_df.with_columns(
                [
                    pl.Series("actual", p_actual),
                    pl.lit(series_name).alias("series_name"),
                    pl.lit(target_series).alias("is_consumption"),
                ]
            )
            forecast_dfs.append(p_eval_df)

    else:  # rolling_30d walk-forward single pool
        console.print(
            f"[bold]Fitting 30-Day Daily Walk-Forward GBDT (Delta T) for {series_name}...[/bold]"
        )
        for p_id in pools:
            p_train = train_series.filter(pl.col("pool") == p_id).sort("datetime_utc")
            p_test = test_series.filter(pl.col("pool") == p_id).sort("datetime_utc")
            p_actual = p_test["target"].to_numpy()

            forecaster = BaselineForecaster(
                config=b_cfg,
                feature_config=f_cfg,
                time_column="datetime_utc",
                target_column="target",
                id_column=None,
                exogenous_columns=active_exo,
                autoregressive_columns=ar_cols,
                categorical_columns=cat_cols,
                predict_difference=True,
            )
            forecaster.fit(p_train)
            pred_wf = forecaster.predict_rolling_walk_forward(
                test_df=p_test,
                context_df=p_train,
                window_days=30,
                predict_difference=True,
            )
            p_pred = pred_wf.forecast_df["forecast"].to_numpy()

            # Naive baseline
            m_t1 = TMinus1Forecaster(
                time_column="datetime_utc", target_column="target", id_column=None
            )
            m_t1.fit(p_train)
            t1_pred = m_t1.predict_one_step_ahead(p_test, context_df=p_train)
            y_t1 = t1_pred.forecast_df["forecast"].to_numpy()
            mae_t1 = mean_absolute_error(p_actual, y_t1)

            mae = mean_absolute_error(p_actual, p_pred)
            rmse = root_mean_squared_error(p_actual, p_pred)
            wape_val = compute_wape(p_actual, p_pred)
            imp_pct = ((mae_t1 - mae) / mae_t1 * 100.0) if mae_t1 > 0 else 0.0

            pool_metrics.append(
                {
                    "pool": p_id,
                    "mean_kw": float(np.mean(p_actual)),
                    "max_kw": float(np.max(p_actual)),
                    "t1_mae": float(mae_t1),
                    "mae": float(mae),
                    "rmse": float(rmse),
                    "wape": float(wape_val),
                    "improvement_pct": float(imp_pct),
                }
            )

            p_eval_df = pred_wf.forecast_df.with_columns(
                [
                    pl.lit(p_id).alias("pool"),
                    pl.Series("actual", p_actual),
                    pl.lit(series_name).alias("series_name"),
                    pl.lit(target_series).alias("is_consumption"),
                ]
            )
            forecast_dfs.append(p_eval_df)

    combined_forecast_df = pl.concat(forecast_dfs)

    # Combined aggregate metrics
    all_actuals = combined_forecast_df["actual"].to_numpy()
    all_preds = combined_forecast_df["forecast"].to_numpy()
    comb_mae = mean_absolute_error(all_actuals, all_preds)
    comb_rmse = root_mean_squared_error(all_actuals, all_preds)
    comb_wape = compute_wape(all_actuals, all_preds)
    mean_pool_mae = float(np.mean([m["mae"] for m in pool_metrics]))
    mean_pool_wape = float(np.mean([m["wape"] for m in pool_metrics]))
    mean_t1_mae = float(np.mean([m["t1_mae"] for m in pool_metrics]))
    comb_imp = ((mean_t1_mae - mean_pool_mae) / mean_t1_mae * 100.0) if mean_t1_mae > 0 else 0.0

    combined_metrics = {
        "mean_pool_mae": mean_pool_mae,
        "mean_pool_wape": mean_pool_wape,
        "mean_t1_mae": mean_t1_mae,
        "overall_portfolio_mae": float(comb_mae),
        "overall_portfolio_rmse": float(comb_rmse),
        "overall_portfolio_wape": float(comb_wape),
        "portfolio_improvement_pct": float(comb_imp),
    }

    return combined_forecast_df, pool_metrics, combined_metrics


def print_results_table(series_name: str, pool_metrics: list[dict], combined_metrics: dict) -> None:
    """Render high-contrast Rich table reporting separate pools and combined summary."""
    table = Table(
        title=f"Forecast Evaluation Results: {series_name} (GBDT Delta T)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Pool ID", style="cyan", justify="right")
    table.add_column("Mean (kW)", justify="right")
    table.add_column("Peak (kW)", justify="right")
    table.add_column("T-1 Naive (kW)", justify="right", style="dim")
    table.add_column("GBDT MAE (kW)", justify="right", style="bold green")
    table.add_column("RMSE (kW)", justify="right")
    table.add_column("WAPE (%)", justify="right", style="bold yellow")
    table.add_column("Gain vs T-1", justify="right", style="bold cyan")

    for m in pool_metrics:
        table.add_row(
            f"Pool {m['pool']:02d}",
            f"{m['mean_kw']:,.1f}",
            f"{m['max_kw']:,.1f}",
            f"{m['t1_mae']:,.1f}",
            f"{m['mae']:,.1f}",
            f"{m['rmse']:,.1f}",
            f"{m['wape']:.2f}%",
            f"-{m['improvement_pct']:.1f}%",
        )

    table.add_section()
    # Combined row
    table.add_row(
        "[bold]COMBINED (Mean)[/bold]",
        "-",
        "-",
        f"[bold]{combined_metrics['mean_t1_mae']:.2f}[/bold]",
        f"[bold green]{combined_metrics['mean_pool_mae']:.2f}[/bold green]",
        f"[bold]{combined_metrics['overall_portfolio_rmse']:.2f}[/bold]",
        f"[bold yellow]{combined_metrics['mean_pool_wape']:.2f}%[/bold yellow]",
        f"[bold cyan]-{combined_metrics['portfolio_improvement_pct']:.1f}%[/bold cyan]",
    )

    console.print(table)
    console.print()


def export_forecasts(
    df: pl.DataFrame,
    output_dir: Path,
    export_format: str,
) -> tuple[Path | None, Path | None]:
    """Export forecasts with computed residuals."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add error columns
    export_df = df.with_columns(
        [
            (pl.col("forecast") - pl.col("actual")).alias("residual_error"),
            (pl.col("forecast") - pl.col("actual")).abs().alias("absolute_error"),
            (
                (pl.col("forecast") - pl.col("actual")).abs()
                / (pl.when(pl.col("actual").abs() > 0).then(pl.col("actual").abs()).otherwise(1.0))
                * 100.0
            ).alias("absolute_percentage_error"),
        ]
    ).select(
        [
            "datetime_utc",
            "pool",
            "series_name",
            "is_consumption",
            "actual",
            "forecast",
            "residual_error",
            "absolute_error",
            "absolute_percentage_error",
        ]
    )

    parquet_path = None
    csv_path = None

    if export_format in ["parquet", "both"]:
        parquet_path = output_dir / "forecasts_champion_model.parquet"
        export_df.write_parquet(parquet_path)
        console.print(f"[green]✓ Exported Parquet forecasts to:[/green] {parquet_path}")

    if export_format in ["csv", "both"]:
        csv_path = output_dir / "forecasts_champion_model.csv"
        export_df.write_csv(csv_path)
        console.print(f"[green]✓ Exported CSV forecasts to:[/green] {csv_path}")

    return parquet_path, csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Run Flexa Champion Forecasting Model (GBDT Delta T) & Export Predictions."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing train/test parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/forecasts"),
        help="Directory to save exported forecasts and metrics.",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["parquet", "csv", "both"],
        default="both",
        help="Export format for predictions (parquet, csv, or both).",
    )
    parser.add_argument(
        "--model-variant",
        type=str,
        choices=["global", "rolling_30d"],
        default="global",
        help="Model variant: 'global' (unified cross-pool model) or 'rolling_30d' (daily retrained single pools).",
    )
    parser.add_argument(
        "--pools",
        type=int,
        nargs="+",
        default=None,
        help="Specific pool IDs to evaluate. Default: all available pools in test set.",
    )
    args = parser.parse_args()

    t_start = time.time()
    console.print(
        Panel.fit(
            "[bold white]Flexa Production Forecaster: Champion GBDT on Delta T[/bold white]\n"
            f"Strategy: [cyan]{args.model_variant}[/cyan] | Formats: [cyan]{args.format}[/cyan] | Output: [cyan]{args.output_dir}[/cyan]",
            border_style="green",
        )
    )

    train_df, test_df = load_and_prepare_data(args.data_dir)

    all_pools = sorted(test_df["pool"].unique().to_list())
    target_pools = [p for p in args.pools if p in all_pools] if args.pools else all_pools
    console.print(f"Target pools ({len(target_pools)}): {target_pools}\n")

    f_cfg, b_cfg = build_feature_configs()

    # 1. Solar Injection (series_val = 0)
    solar_df, solar_pool_metrics, solar_comb_metrics = run_forecasting(
        train_df=train_df,
        test_df=test_df,
        target_series=0,
        series_name="Solar Injection",
        pools=target_pools,
        f_cfg=f_cfg,
        b_cfg=b_cfg,
        model_variant=args.model_variant,
    )
    print_results_table("Solar Injection", solar_pool_metrics, solar_comb_metrics)

    # 2. Electricity Consumption (series_val = 1)
    cons_df, cons_pool_metrics, cons_comb_metrics = run_forecasting(
        train_df=train_df,
        test_df=test_df,
        target_series=1,
        series_name="Electricity Consumption",
        pools=target_pools,
        f_cfg=f_cfg,
        b_cfg=b_cfg,
        model_variant=args.model_variant,
    )
    print_results_table("Electricity Consumption", cons_pool_metrics, cons_comb_metrics)

    # Combine all predictions
    all_forecasts = pl.concat([solar_df, cons_df])

    # Export forecast datasets
    parquet_out, csv_out = export_forecasts(all_forecasts, args.output_dir, args.format)

    # Export summary metrics JSON
    metrics_summary = {
        "model": "GBDT (Delta T)",
        "model_variant": args.model_variant,
        "test_horizon_hours": test_df["datetime_utc"].n_unique(),
        "total_pools": len(target_pools),
        "solar_injection": {
            "per_pool": solar_pool_metrics,
            "combined": solar_comb_metrics,
        },
        "electricity_consumption": {
            "per_pool": cons_pool_metrics,
            "combined": cons_comb_metrics,
        },
        "elapsed_seconds": round(time.time() - t_start, 2),
    }

    json_path = args.output_dir / "forecast_evaluation_summary.json"
    with open(json_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    console.print(f"[green]✓ Exported metrics summary to:[/green] {json_path}")

    elapsed = time.time() - t_start
    console.print(
        Panel.fit(
            f"[bold green]✓ Forecasting & Evaluation Completed in {elapsed:.1f}s[/bold green]\n"
            f"Solar Mean MAE: [bold]{solar_comb_metrics['mean_pool_mae']:.2f} kW[/bold] (WAPE: [bold]{solar_comb_metrics['mean_pool_wape']:.2f}%[/bold])\n"
            f"Consumption Mean MAE: [bold]{cons_comb_metrics['mean_pool_mae']:.2f} kW[/bold] (WAPE: [bold]{cons_comb_metrics['mean_pool_wape']:.2f}%[/bold])",
            border_style="cyan",
        )
    )


if __name__ == "__main__":
    main()
