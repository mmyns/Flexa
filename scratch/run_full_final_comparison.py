"""Comprehensive Final Comparison Benchmark.

Runs Chronos-2, Ridge, GBDT (16m Full Lookback), and GBDT (30d Lookback with Daily Retraining)
on Delta T across all 14 pools separately and combined (global cross-pool).
"""

from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import polars as pl
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from flexa.config import BaselineModelConfig, ChronosModelConfig, FeatureConfig
from flexa.features.time_features import add_paired_telemetry
from flexa.features.weather_features import WeatherFeatureEngineer
from flexa.models.baseline import BaselineForecaster, RidgeForecaster
from flexa.models.chronos import ChronosForecaster
from flexa.models.naive import TMinus1Forecaster


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum(np.abs(y_true))
    return float(np.sum(np.abs(y_true - y_pred)) / (denom if denom > 0 else 1.0) * 100.0)


def run_benchmark():
    t_start = time.time()
    proc_dir = Path("data/processed")

    print("Loading data...")
    _train_raw = pl.read_parquet(proc_dir / "train_measurements.parquet")
    _test_raw = pl.read_parquet(proc_dir / "test_measurements.parquet")
    df_train_meas = add_paired_telemetry(_train_raw)
    df_test_meas = add_paired_telemetry(_test_raw, context_df=_train_raw)
    df_train_weather = pl.read_parquet(proc_dir / "train_weather.parquet")
    df_test_weather = pl.read_parquet(proc_dir / "test_weather.parquet")

    wf_engineer = WeatherFeatureEngineer()
    train_w_enriched = wf_engineer.enrich_weather(df_train_weather)
    test_w_enriched = wf_engineer.enrich_weather(df_test_weather)

    train_joined = wf_engineer.join_weather(df_train_meas, train_w_enriched)
    test_joined = wf_engineer.join_weather(df_test_meas, test_w_enriched)

    f_cfg = FeatureConfig(
        lags=[1, 24, 168],
        delta_lags=[(1, 2)],
        rolling_windows=[],
        rolling_metrics=[],
        include_calendar=True,
        include_cyclical=True,
        calendar_features=["day_of_week", "hour"],
        exclude_features=["month", "day_of_month"],
    )
    b_cfg = BaselineModelConfig(
        model_type="hist_gradient_boosting",
        l2_regularization=1.0,
        max_iter=150,
        random_state=42,
    )

    chronos_cfg = ChronosModelConfig(
        model_id="amazon/chronos-2",
        device="auto",
        prediction_length=1,
    )
    forecaster_chronos = ChronosForecaster(
        config=chronos_cfg,
        time_column="datetime_utc",
        target_column="target",
        id_column=None,
    )
    all_joined = pl.concat([train_joined, test_joined])

    pool_list = sorted(test_joined["pool"].unique().to_list())
    print(f"Target pools ({len(pool_list)}): {pool_list}")

    all_results = {}

    for series_val, s_name in [(0, "Solar Injection"), (1, "Electricity Consumption")]:
        print(f"\n=======================================================")
        print(f"RUNNING BENCHMARK FOR: {s_name} (series_val={series_val})")
        print(f"=======================================================")

        if series_val == 0:
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

        train_series = train_joined.filter(pl.col("is_consumption") == series_val).sort(["pool", "datetime_utc"])
        test_series = test_joined.filter(pl.col("is_consumption") == series_val).sort(["pool", "datetime_utc"])

        active_exo = [f for f in raw_w_cols if f in train_series.columns]
        if "solar_peak_to_t" in train_series.columns:
            active_exo.append("solar_peak_to_t")

        ar_cols = ["target", "consumption", "injection"]
        cat_cols = ["day_of_week"]

        print(f"Exogenous features: {active_exo}")

        # -----------------------------------------------------------------
        # 1. COMBINED / GLOBAL MODELS (Trained on all 14 pools pooled together)
        # -----------------------------------------------------------------
        print(f"\n--- Training Global Models on All 14 Pools Combined ---")
        exo_global = ["pool"] + active_exo

        # A. Global Ridge on Delta T
        print("Fitting Global Ridge (Delta T)...")
        m_ridge_global = RidgeForecaster(
            alpha=10.0,
            feature_config=f_cfg,
            time_column="datetime_utc",
            target_column="target",
            id_column="pool",
            exogenous_columns=exo_global,
            autoregressive_columns=ar_cols,
            categorical_columns=["day_of_week", "pool"],
            predict_difference=True,
        )
        m_ridge_global.fit(train_series)
        preds_ridge_global = m_ridge_global.predict_one_step_ahead(test_series, context_df=train_series)
        df_ridge_global = preds_ridge_global.forecast_df

        # B. Global GBDT (16m Full Lookback) on Delta T
        print("Fitting Global GBDT (16m Full Lookback, Delta T)...")
        m_gbdt_global = BaselineForecaster(
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
        m_gbdt_global.fit(train_series)
        preds_gbdt_global = m_gbdt_global.predict_one_step_ahead(test_series, context_df=train_series)
        df_gbdt_global = preds_gbdt_global.forecast_df

        # -----------------------------------------------------------------
        # 2. SEPARATE POOL-BY-POOL EVALUATIONS
        # -----------------------------------------------------------------
        pool_metrics = []

        for p_id in pool_list:
            print(f"Benchmarking Pool {p_id}...")
            p_train = train_series.filter(pl.col("pool") == p_id).sort("datetime_utc")
            p_test = test_series.filter(pl.col("pool") == p_id).sort("datetime_utc")
            y_true = p_test["target"].to_numpy()

            # 1. T-1 Naive Baseline
            m_t1 = TMinus1Forecaster(time_column="datetime_utc", target_column="target", id_column=None)
            m_t1.fit(p_train)
            pred_t1 = m_t1.predict_one_step_ahead(p_test, context_df=p_train)
            y_t1 = pred_t1.forecast_df["forecast"].to_numpy()
            mae_t1 = mean_absolute_error(y_true, y_t1)

            # 2. Chronos-2 (Foundation model, 1-step ahead)
            pred_c = forecaster_chronos.predict_one_step_ahead(
                test_df=p_test,
                context_df=p_train,
                covariates_df=all_joined,
                covariate_cols=active_exo[:2] if active_exo else None,
                batch_size=512,
            )
            y_chronos = pred_c.forecast_df["forecast"].to_numpy()
            mae_chronos = mean_absolute_error(y_true, y_chronos)
            wape_chronos = wape(y_true, y_chronos)

            # 3. Single Ridge (Delta T, 16m full lookback)
            m_ridge_single = RidgeForecaster(
                alpha=10.0,
                feature_config=f_cfg,
                time_column="datetime_utc",
                target_column="target",
                id_column=None,
                exogenous_columns=active_exo,
                autoregressive_columns=ar_cols,
                categorical_columns=cat_cols,
                predict_difference=True,
            )
            m_ridge_single.fit(p_train)
            pred_ridge_single = m_ridge_single.predict_one_step_ahead(p_test, context_df=p_train)
            y_ridge_single = pred_ridge_single.forecast_df["forecast"].to_numpy()
            mae_ridge_single = mean_absolute_error(y_true, y_ridge_single)

            # 4. Single GBDT (Delta T, 16m full lookback)
            m_gbdt_single_diff = BaselineForecaster(
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
            m_gbdt_single_diff.fit(p_train)
            pred_gbdt_single_diff = m_gbdt_single_diff.predict_one_step_ahead(p_test, context_df=p_train)
            y_gbdt_single_diff = pred_gbdt_single_diff.forecast_df["forecast"].to_numpy()
            mae_gbdt_single_diff = mean_absolute_error(y_true, y_gbdt_single_diff)

            # 5. Single GBDT (Delta T, 30d lookback with daily retraining)
            pred_gbdt_30d = m_gbdt_single_diff.predict_rolling_walk_forward(
                test_df=p_test,
                context_df=p_train,
                window_days=30,
                predict_difference=True,
            )
            y_gbdt_30d = pred_gbdt_30d.forecast_df["forecast"].to_numpy()
            mae_gbdt_30d = mean_absolute_error(y_true, y_gbdt_30d)

            # 6. Global Ridge (Delta T, Combined)
            y_ridge_global = df_ridge_global.filter(pl.col("pool") == p_id)["forecast"].to_numpy()
            mae_ridge_global = mean_absolute_error(y_true, y_ridge_global)

            # 7. Global GBDT (Delta T, 16m Combined)
            y_gbdt_global = df_gbdt_global.filter(pl.col("pool") == p_id)["forecast"].to_numpy()
            mae_gbdt_global = mean_absolute_error(y_true, y_gbdt_global)
            wape_gbdt_global = wape(y_true, y_gbdt_global)

            row = {
                "pool": p_id,
                "mean_kw": float(np.mean(y_true)),
                "max_kw": float(np.max(y_true)),
                "t1_mae": mae_t1,
                "chronos_mae": mae_chronos,
                "chronos_wape": wape_chronos,
                "ridge_single_mae": mae_ridge_single,
                "ridge_global_mae": mae_ridge_global,
                "gbdt_16m_single_mae": mae_gbdt_single_diff,
                "gbdt_30d_single_mae": mae_gbdt_30d,
                "gbdt_16m_global_mae": mae_gbdt_global,
                "gbdt_16m_global_wape": wape_gbdt_global,
            }
            pool_metrics.append(row)
            print(
                f"  Pool {p_id:02d}: T-1={mae_t1:.1f} | Chronos={mae_chronos:.1f} | Ridge-S={mae_ridge_single:.1f} | "
                f"Ridge-G={mae_ridge_global:.1f} | GBDT-16m-S={mae_gbdt_single_diff:.1f} | GBDT-30d-S={mae_gbdt_30d:.1f} | "
                f"GBDT-16m-G={mae_gbdt_global:.1f}"
            )

        all_results[s_name] = pool_metrics

    t_end = time.time()
    print(f"\nTotal benchmark run completed in {t_end - t_start:.1f}s")
    return all_results


if __name__ == "__main__":
    results = run_benchmark()
    import json
    with open("scratch/final_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results to scratch/final_benchmark_results.json")
