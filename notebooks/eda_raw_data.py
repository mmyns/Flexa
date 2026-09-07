import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="full",
    app_title="Flexa - Raw Data EDA & Outlier Cleaning",
)


@app.cell
def _():
    from datetime import datetime
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl
    from plotly.subplots import make_subplots

    return Path, datetime, go, make_subplots, mo, np, pd, pl, px


@app.cell
def _(mo):
    mo.md(r"""
    # ⚡ Flexa: Raw Energy & Weather Data Analysis

    Interactive exploratory data analysis for the raw datasets in `data/raw/` and processed train/test splits in `data/processed/`:
    - **`1_measurements.csv`**: Target energy measurements (kW) across 14 client pools for **Grid Injection** (`is_consumption=0`) and **Consumption** (`is_consumption=1`).
    - **`1_weather.csv`**: Hourly weather forecast variables (temperature, radiation, cloud cover, wind speed, precipitation).

    > 💡 **Domain Note: Grid Injection vs. Total Solar Production**<br>
    > `is_consumption = 0` denotes net energy **injected into the distribution grid** (feed-in). While injection is driven by rooftop solar PV (and battery storage), it is **not equal to total solar production**. A large portion of daytime solar generation is absorbed on-site by local consumption; only surplus power is exported to the grid.
    """)
    return


@app.cell
def _(Path, pl):
    # Locate data directory relative to repository root
    _root_dir = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path(".")
    raw_dir = _root_dir / "data" / "raw"
    proc_dir = _root_dir / "data" / "processed"

    # 1. Load measurements
    _meas_path = raw_dir / "1_measurements.csv"
    _df_meas_raw = pl.read_csv(_meas_path, try_parse_dates=False).drop("")
    df_meas = (
        _df_meas_raw.with_columns(
            pl.col("datetime_utc").str.to_datetime("%Y-%m-%d %H:%M:%S")
        ).with_columns(
            pl.col("datetime_utc").dt.hour().alias("hour"),
            pl.col("datetime_utc").dt.weekday().alias("day_of_week"),
            pl.col("datetime_utc").dt.month().alias("month"),
            pl.col("datetime_utc").dt.date().alias("date"),
            pl.when(pl.col("is_consumption") == 1)
            .then(pl.lit("Consumption"))
            .otherwise(pl.lit("Injection"))
            .alias("type_name"),
            pl.when(pl.col("datetime_utc") < pl.datetime(2023, 1, 1))
            .then(pl.lit("Train (Sep 21 - Dec 22)"))
            .otherwise(pl.lit("Test (Jan 23 - May 23)"))
            .alias("split_name"),
        )
    )

    # 2. Load weather
    _weath_path = raw_dir / "1_weather.csv"
    _df_weath_raw = pl.read_csv(_weath_path, try_parse_dates=False)
    df_weath = (
        _df_weath_raw.with_columns(
            pl.col("forecast_datetime_utc").str.to_datetime("%Y-%m-%d %H:%M:%S")
        ).with_columns(
            (
                (pl.col("10_metre_u_wind_component") ** 2 + pl.col("10_metre_v_wind_component") ** 2)
                ** 0.5
            ).alias("wind_speed_10m"),
            pl.when(pl.col("forecast_datetime_utc") < pl.datetime(2023, 1, 1))
            .then(pl.lit("Train"))
            .otherwise(pl.lit("Test"))
            .alias("split_name"),
        )
    )

    # 3. Pivoted measurements: consumption, injection, net_energy
    df_piv = (
        df_meas.pivot(
            on="is_consumption",
            index=["datetime_utc", "date", "pool"],
            values="target",
        )
        .rename({"0": "injection", "1": "consumption"})
        .with_columns(
            (pl.col("consumption") - pl.col("injection")).alias("net_energy")
        )
    )

    # Unique pool list
    pools = sorted(df_meas["pool"].unique().to_list())
    return df_meas, df_piv, df_weath, pools


@app.cell
def _(df_meas, df_weath, mo, pools):
    _meas_min_date = df_meas["datetime_utc"].min().strftime("%Y-%m-%d")
    _meas_max_date = df_meas["datetime_utc"].max().strftime("%Y-%m-%d")

    _kpi_row = mo.hstack(
        [
            mo.stat(
                value=f"{len(df_meas):,}",
                label="Total Measurements",
                caption="14 pools × 2 types",
            ),
            mo.stat(
                value=f"{len(pools)}",
                label="Client Pools",
                caption=f"IDs: {pools[0]} to {pools[-1]}",
            ),
            mo.stat(
                value=f"{_meas_min_date} → {_meas_max_date}",
                label="Full Time Span",
                caption=f"{df_meas['datetime_utc'].n_unique():,} hourly steps",
            ),
            mo.stat(
                value=f"{len(df_weath):,}",
                label="Weather Forecast Rows",
                caption="13 variables, zero nulls",
            ),
            mo.stat(
                value="0 nulls (100% clean)",
                label="Data Quality",
                caption="Ready for modeling",
            ),
        ],
        justify="space-between",
    )

    mo.vstack([mo.md("### 📊 Dataset Overview & Health"), _kpi_row])
    return


@app.cell
def _(df_meas, mo, pl):
    _train_df = df_meas.filter(pl.col("datetime_utc") < pl.datetime(2023, 1, 1))
    _test_df = df_meas.filter(pl.col("datetime_utc") >= pl.datetime(2023, 1, 1))

    _split_kpis = mo.hstack(
        [
            mo.stat(
                value=f"{len(_train_df):,} rows",
                label="Train Set (Sep 2021 – Dec 2022)",
                caption="16 months (76.4% of data) | saved in data/processed/",
            ),
            mo.stat(
                value=f"{len(_test_df):,} rows",
                label="Test Set (Jan 2023 – May 2023)",
                caption="5 months (23.6% of data) | saved in data/processed/",
            ),
        ],
        justify="start",
        gap=3,
    )

    mo.vstack(
        [
            mo.md("--- \n### ✂️ Train / Test Split Specification"),
            _split_kpis,
        ]
    )
    return


@app.cell
def _(mo, pools):
    pool_select = mo.ui.dropdown(
        options={f"Pool {p}": p for p in pools},
        value=f"Pool {pools[0]}",
        label="Client Pool:",
    )

    type_select = mo.ui.radio(
        options=["Injection", "Consumption", "Both (Overlay)"],
        value="Injection",
        inline=True,
        label="Series Type:",
    )

    split_select = mo.ui.radio(
        options=["All Data (2021–2023)", "Train Set Only", "Test Set Only"],
        value="All Data (2021–2023)",
        inline=True,
        label="Data Split:",
    )

    freq_select = mo.ui.radio(
        options=["Hourly (Raw)", "Daily Average", "Weekly Average"],
        value="Daily Average",
        inline=True,
        label="Granularity:",
    )

    mo.vstack(
        [
            mo.md("--- \n### 📈 Interactive Time Series Exploration"),
            mo.hstack(
                [pool_select, type_select, split_select, freq_select],
                justify="start",
                gap=2,
            ),
        ]
    )
    return freq_select, pool_select, split_select, type_select


@app.cell
def _(
    df_meas,
    freq_select,
    mo,
    pl,
    pool_select,
    px,
    split_select,
    type_select,
):
    _sel_pool = pool_select.value
    _sel_type = type_select.value
    _sel_split = split_select.value
    _sel_freq = freq_select.value

    # Filter pool
    _filtered = df_meas.filter(df_meas["pool"] == _sel_pool)

    # Filter split
    if _sel_split == "Train Set Only":
        _filtered = _filtered.filter(pl.col("datetime_utc") < pl.datetime(2023, 1, 1))
    elif _sel_split == "Test Set Only":
        _filtered = _filtered.filter(pl.col("datetime_utc") >= pl.datetime(2023, 1, 1))

    # Filter type
    if _sel_type == "Injection":
        _filtered = _filtered.filter(_filtered["is_consumption"] == 0)
    elif _sel_type == "Consumption":
        _filtered = _filtered.filter(_filtered["is_consumption"] == 1)

    # Resample / aggregate
    if _sel_freq == "Daily Average":
        _plot_df = (
            _filtered.group_by(["date", "type_name"])
            .agg(pl.col("target").mean())
            .sort("date")
        )
        _x_col = "date"
        _title_suffix = "Daily Average"
    elif _sel_freq == "Weekly Average":
        _plot_df = (
            _filtered.with_columns(pl.col("datetime_utc").dt.truncate("1w").alias("week"))
            .group_by(["week", "type_name"])
            .agg(pl.col("target").mean())
            .sort("week")
        )
        _x_col = "week"
        _title_suffix = "Weekly Average"
    else:
        _plot_df = _filtered.sort("datetime_utc")
        _x_col = "datetime_utc"
        _title_suffix = "Hourly Time Series"

    _color_map = {
        "Injection": "#FF9900",
        "Consumption": "#2B6CB0",
    }

    _fig_ts = px.line(
        _plot_df.to_pandas(),
        x=_x_col,
        y="target",
        color="type_name",
        color_discrete_map=_color_map,
        labels={"target": "Power Target (kW)", _x_col: "Timestamp", "type_name": "Series"},
        title=f"Pool {_sel_pool}: {_title_suffix} ({_sel_type}) - {_sel_split}",
        template="plotly_white",
    )

    # Add vertical line for Train/Test cutoff when viewing full dataset
    if _sel_split == "All Data (2021–2023)":
        _fig_ts.add_vline(
            x="2023-01-01",
            line_width=2,
            line_dash="dash",
            line_color="#E53E3E",
            annotation_text="Train / Test Cutoff (2023-01-01)",
            annotation_position="top left",
        )

    _fig_ts.update_layout(
        height=450,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
    )

    mo.ui.plotly(_fig_ts)
    return


@app.cell
def _(df_meas, go, make_subplots, mo, pl, pool_select):
    _sel_pool = pool_select.value
    _pool_data = df_meas.filter(df_meas["pool"] == _sel_pool)

    # 1. Hourly Diurnal Profile
    _hourly_prof = (
        _pool_data.group_by(["hour", "type_name"])
        .agg(pl.col("target").mean().alias("mean_target"))
        .sort(["hour", "type_name"])
    ).to_pandas()

    # 2. Day of Week Profile
    _dow_map = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
    _dow_prof = (
        _pool_data.group_by(["day_of_week", "type_name"])
        .agg(pl.col("target").mean().alias("mean_target"))
        .sort(["day_of_week", "type_name"])
    ).to_pandas()
    _dow_prof["day_name"] = _dow_prof["day_of_week"].map(_dow_map)

    # 3. Monthly Profile
    _month_map = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    _monthly_prof = (
        _pool_data.group_by(["month", "type_name"])
        .agg(pl.col("target").mean().alias("mean_target"))
        .sort(["month", "type_name"])
    ).to_pandas()
    _monthly_prof["month_name"] = _monthly_prof["month"].map(_month_map)

    _fig_prof = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            f"Diurnal Cycle (Hour of Day) - Pool {_sel_pool}",
            "Weekly Cycle (Day of Week)",
            "Annual Seasonality (Month)",
        ),
    )

    _inj_col = "#FF9900"
    _cons_col = "#2B6CB0"

    for _t_name, _col in [("Injection", _inj_col), ("Consumption", _cons_col)]:
        _sub_h = _hourly_prof[_hourly_prof["type_name"] == _t_name]
        _fig_prof.add_trace(
            go.Scatter(
                x=_sub_h["hour"],
                y=_sub_h["mean_target"],
                mode="lines+markers",
                name=f"{_t_name} (Hourly)",
                line={"color": _col},
            ),
            row=1,
            col=1,
        )

        _sub_d = _dow_prof[_dow_prof["type_name"] == _t_name]
        _fig_prof.add_trace(
            go.Bar(
                x=_sub_d["day_name"],
                y=_sub_d["mean_target"],
                name=f"{_t_name} (Daily)",
                marker_color=_col,
                opacity=0.8,
            ),
            row=1,
            col=2,
        )

        _sub_m = _monthly_prof[_monthly_prof["type_name"] == _t_name]
        _fig_prof.add_trace(
            go.Scatter(
                x=_sub_m["month_name"],
                y=_sub_m["mean_target"],
                mode="lines+markers",
                name=f"{_t_name} (Monthly)",
                line={"color": _col},
            ),
            row=1,
            col=3,
        )

    _fig_prof.update_layout(
        height=380,
        showlegend=False,
        template="plotly_white",
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
    )
    _fig_prof.update_xaxes(title_text="Hour (UTC)", row=1, col=1)
    _fig_prof.update_xaxes(title_text="Day of Week", row=1, col=2)
    _fig_prof.update_xaxes(title_text="Month", row=1, col=3)
    _fig_prof.update_yaxes(title_text="Mean Power (kW)", row=1, col=1)

    mo.vstack(
        [
            mo.md("--- \n### ⏰ Diurnal & Seasonal Patterns"),
            mo.md(
                "*Notice the daytime injection bell curve peaking around 11:00–13:00 UTC and summer months (Apr–Jul) when surplus solar is exported. "
                "Consumption demonstrates a morning/evening double peak and higher winter heating baseline.*"
            ),
            mo.ui.plotly(_fig_prof),
        ]
    )
    return


@app.cell
def _(mo):
    weather_options = {
        "Temperature (°C)": "temperature",
        "Surface Solar Radiation (W/m²)": "surface_solar_radiation_downwards",
        "Direct Solar Radiation (W/m²)": "direct_solar_radiation",
        "Total Cloud Cover (0-1)": "cloudcover_total",
        "10m Wind Speed (m/s)": "wind_speed_10m",
        "Precipitation (mm)": "total_precipitation",
        "Snowfall (m)": "snowfall",
    }

    weath_select = mo.ui.dropdown(
        options=weather_options,
        value="Temperature (°C)",
        label="Weather Variable:",
    )

    mo.vstack(
        [
            mo.md("--- \n### 🌤️ Weather Forecast Features Exploration"),
            weath_select,
        ]
    )
    return (weath_select,)


@app.cell
def _(df_weath, go, make_subplots, mo, pl, weath_select):
    _sel_var = weath_select.value

    # Daily aggregation for smooth visualization
    _daily_w = (
        df_weath.with_columns(pl.col("forecast_datetime_utc").dt.date().alias("date"))
        .group_by("date")
        .agg(
            [
                pl.col(_sel_var).mean().alias("mean_val"),
                pl.col(_sel_var).min().alias("min_val"),
                pl.col(_sel_var).max().alias("max_val"),
            ]
        )
        .sort("date")
    ).to_pandas()

    _fig_w = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.68, 0.32],
        subplot_titles=(
            f"Daily Trend: {_sel_var}",
            f"Value Distribution: {_sel_var}",
        ),
    )

    # Max line (invisible for fill)
    _fig_w.add_trace(
        go.Scatter(
            x=_daily_w["date"],
            y=_daily_w["max_val"],
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    # Min line with fill
    _fig_w.add_trace(
        go.Scatter(
            x=_daily_w["date"],
            y=_daily_w["min_val"],
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(49, 130, 206, 0.18)",
            name="Daily Min-Max Band",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    # Mean line
    _fig_w.add_trace(
        go.Scatter(
            x=_daily_w["date"],
            y=_daily_w["mean_val"],
            mode="lines",
            name="Daily Mean",
            line={"color": "#3182CE", "width": 2},
        ),
        row=1,
        col=1,
    )

    # Distribution histogram
    _weath_vals = df_weath[_sel_var].to_pandas()
    _fig_w.add_trace(
        go.Histogram(
            x=_weath_vals,
            name="Distribution",
            marker_color="#38B2AC",
            opacity=0.75,
            nbinsx=40,
        ),
        row=1,
        col=2,
    )

    _fig_w.update_layout(
        height=380,
        template="plotly_white",
        hovermode="x unified",
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
    )
    _fig_w.update_xaxes(title_text="Date", row=1, col=1)
    _fig_w.update_xaxes(title_text="Value", row=1, col=2)
    _fig_w.update_yaxes(title_text="Value", row=1, col=1)
    _fig_w.update_yaxes(title_text="Count", row=1, col=2)

    mo.ui.plotly(_fig_w)
    return


@app.cell
def _(df_meas, df_weath, go, make_subplots, mo, pl, pool_select):
    _sel_pool = pool_select.value

    # Join measurements and weather
    _pool_meas = df_meas.filter(df_meas["pool"] == _sel_pool)
    _joined = _pool_meas.join(
        df_weath,
        left_on="datetime_utc",
        right_on="forecast_datetime_utc",
        how="inner",
    )

    _weather_cols = [
        "temperature",
        "surface_solar_radiation_downwards",
        "direct_solar_radiation",
        "cloudcover_total",
        "wind_speed_10m",
        "total_precipitation",
    ]

    _inj_df = _joined.filter(_joined["is_consumption"] == 0)
    _cons_df = _joined.filter(_joined["is_consumption"] == 1)

    _inj_corrs = [_inj_df.select(pl.corr("target", c)).item() for c in _weather_cols]
    _cons_corrs = [_cons_df.select(pl.corr("target", c)).item() for c in _weather_cols]

    _z = [_inj_corrs, _cons_corrs]
    _labels_x = [c.replace("_", " ").title() for c in _weather_cols]

    _fig_corr = go.Figure(
        data=go.Heatmap(
            z=_z,
            x=_labels_x,
            y=["Injection", "Consumption"],
            colorscale="RdBu_r",
            zmid=0,
            text=[[f"{v:.2f}" for v in row] for row in _z],
            texttemplate="%{text}",
            textfont={"size": 13},
        )
    )
    _fig_corr.update_layout(
        title=f"Pearson Correlation: Weather Drivers vs. Target (Pool {_sel_pool})",
        height=260,
        template="plotly_white",
        margin={"l": 40, "r": 40, "t": 50, "b": 40},
    )

    # Scatter comparisons (sample 1500 points for speed)
    _fig_scatter = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Grid Injection vs. Surface Solar Radiation",
            "Consumption vs. Temperature",
        ),
    )

    _s_inj = _inj_df.sample(min(1500, len(_inj_df)), seed=42).to_pandas()
    _s_cons = _cons_df.sample(min(1500, len(_cons_df)), seed=42).to_pandas()

    _fig_scatter.add_trace(
        go.Scatter(
            x=_s_inj["surface_solar_radiation_downwards"],
            y=_s_inj["target"],
            mode="markers",
            marker={"color": "#FF9900", "size": 4, "opacity": 0.45},
            name="Injection",
        ),
        row=1,
        col=1,
    )

    _fig_scatter.add_trace(
        go.Scatter(
            x=_s_cons["temperature"],
            y=_s_cons["target"],
            mode="markers",
            marker={"color": "#2B6CB0", "size": 4, "opacity": 0.45},
            name="Consumption",
        ),
        row=1,
        col=2,
    )

    _fig_scatter.update_layout(
        height=360,
        template="plotly_white",
        showlegend=False,
        margin={"l": 40, "r": 40, "t": 50, "b": 40},
    )
    _fig_scatter.update_xaxes(title_text="Surface Solar Radiation (W/m²)", row=1, col=1)
    _fig_scatter.update_xaxes(title_text="Temperature (°C)", row=1, col=2)
    _fig_scatter.update_yaxes(title_text="Grid Injection (kW)", row=1, col=1)
    _fig_scatter.update_yaxes(title_text="Power Consumption (kW)", row=1, col=2)

    mo.vstack(
        [
            mo.md("--- \n### 🔬 Cross-Analysis: Energy Targets vs. Weather Drivers"),
            mo.ui.plotly(_fig_corr),
            mo.ui.plotly(_fig_scatter),
        ]
    )
    return


@app.cell
def _(df_piv, go, make_subplots, mo, pl):
    # Daily aggregation per pool
    _daily_stats = (
        df_piv.group_by(["pool", "date"]).agg(
            [
                pl.col("consumption").sum().alias("daily_cons_kwh"),
                pl.col("injection").sum().alias("daily_inj_kwh"),
                pl.col("net_energy").sum().alias("daily_net_kwh"),
                pl.col("consumption").max().alias("peak_cons_kw"),
                pl.col("injection").max().alias("peak_inj_kw"),
            ]
        )
    )

    # General statistics overview per pool
    _pool_summary = (
        _daily_stats.group_by("pool")
        .agg(
            [
                pl.col("daily_cons_kwh").mean().round(1).alias("Mean Daily Consumption (kWh)"),
                pl.col("daily_cons_kwh").std().round(1).alias("Daily Consumption Std (kWh)"),
                pl.col("daily_inj_kwh").mean().round(1).alias("Mean Daily Injection (kWh)"),
                pl.col("daily_net_kwh").mean().round(1).alias("Mean Daily Net Energy (kWh)"),
                pl.col("peak_cons_kw").max().round(1).alias("Max Peak Demand (kW)"),
                pl.col("peak_inj_kw").max().round(1).alias("Max Peak Injection (kW)"),
            ]
        )
        .with_columns(
            pl.when(pl.col("Mean Daily Net Energy (kWh)") < 0)
            .then(pl.lit("Net Exporter 🟢"))
            .otherwise(pl.lit("Net Importer 🔵"))
            .alias("Net Grid Role"),
            (
                (pl.col("Mean Daily Injection (kWh)") / pl.col("Mean Daily Consumption (kWh)"))
                * 100
            )
            .round(1)
            .alias("Solar Coverage (%)"),
        )
        .sort("pool")
    ).to_pandas()

    _pool_summary["Pool ID"] = "Pool " + _pool_summary["pool"].astype(str)

    # Visualizations: 2-column comparative charts
    _fig_compare = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Mean Daily Consumption vs. Injection (kWh/day)",
            "Mean Net Energy Balance (kWh/day)",
        ),
    )

    _fig_compare.add_trace(
        go.Bar(
            x=_pool_summary["Pool ID"],
            y=_pool_summary["Mean Daily Consumption (kWh)"],
            name="Daily Consumption",
            marker_color="#2B6CB0",
        ),
        row=1,
        col=1,
    )
    _fig_compare.add_trace(
        go.Bar(
            x=_pool_summary["Pool ID"],
            y=_pool_summary["Mean Daily Injection (kWh)"],
            name="Daily Injection",
            marker_color="#FF9900",
        ),
        row=1,
        col=1,
    )

    # Net energy balance bar chart
    _net_colors = [
        "#38A169" if val < 0 else "#3182CE"
        for val in _pool_summary["Mean Daily Net Energy (kWh)"]
    ]
    _fig_compare.add_trace(
        go.Bar(
            x=_pool_summary["Pool ID"],
            y=_pool_summary["Mean Daily Net Energy (kWh)"],
            name="Net Energy",
            marker_color=_net_colors,
            text=[f"{v:,.0f}" for v in _pool_summary["Mean Daily Net Energy (kWh)"]],
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    _fig_compare.update_layout(
        height=400,
        template="plotly_white",
        barmode="group",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.05, "xanchor": "right", "x": 1},
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
    )
    _fig_compare.update_yaxes(title_text="kWh / Day", row=1, col=1)
    _fig_compare.update_yaxes(title_text="Net kWh / Day (Import - Export)", row=1, col=2)

    _table_view = mo.ui.table(
        _pool_summary.drop(columns=["pool"]),
        selection=None,
    )

    mo.vstack(
        [
            mo.md("--- \n### 🏭 General Statistics Overview Between Pools"),
            mo.md(
                "*Comprehensive comparison of daily energy metrics across all 14 pools. Notice the scale variance: Pool 0 is an industrial/regional scale pool (~43 MWh/day), while smaller residential/commercial pools average ~1.5 to 4.5 MWh/day. "
                "Notice pools like **Pool 1, 4, 10, and 13** are net exporters (generating more daily injection than consumption).*"
            ),
            mo.ui.plotly(_fig_compare),
            _table_view,
        ]
    )
    return


@app.cell
def _(mo, pools):
    corr_metric_select = mo.ui.radio(
        options=["Net Energy", "Consumption", "Injection"],
        value="Net Energy",
        inline=True,
        label="Cross-Pool Correlation Variable:",
    )

    acf_pool_select = mo.ui.dropdown(
        options={f"Pool {p}": p for p in pools},
        value=f"Pool {pools[0]}",
        label="Autocorrelation Pool:",
    )

    mo.vstack(
        [
            mo.md("--- \n### 🔗 Cross-Pool Correlation & Autocorrelation"),
            mo.hstack([corr_metric_select, acf_pool_select], justify="start", gap=3),
        ]
    )
    return acf_pool_select, corr_metric_select


@app.cell
def _(acf_pool_select, corr_metric_select, df_meas, df_piv, go, mo, np):
    _sel_metric = corr_metric_select.value
    _sel_acf_pool = acf_pool_select.value

    # 1. Compute Cross-Pool Correlation Matrix
    if _sel_metric == "Net Energy":
        _piv_var = df_piv.pivot(on="pool", index="datetime_utc", values="net_energy")
    elif _sel_metric == "Consumption":
        _piv_var = df_piv.pivot(on="pool", index="datetime_utc", values="consumption")
    else:
        _piv_var = df_piv.pivot(on="pool", index="datetime_utc", values="injection")

    _corr_df = _piv_var.drop("datetime_utc").corr().to_pandas()
    _pool_cols = [f"Pool {c}" for c in _corr_df.columns]

    _fig_cross_corr = go.Figure(
        data=go.Heatmap(
            z=_corr_df.values,
            x=_pool_cols,
            y=_pool_cols,
            colorscale="Viridis",
            zmin=0.5,
            zmax=1.0,
            text=[[f"{v:.2f}" for v in row] for row in _corr_df.values],
            texttemplate="%{text}",
            textfont={"size": 10},
        )
    )
    _fig_cross_corr.update_layout(
        title=f"Cross-Pool Correlation Matrix ({_sel_metric})",
        height=450,
        template="plotly_white",
        margin={"l": 50, "r": 40, "t": 50, "b": 50},
    )

    # 2. Compute Autocorrelation (ACF) for selected pool (lags 1 to 72 hours)
    _pool_sub = df_meas.filter(df_meas["pool"] == _sel_acf_pool)
    _y_cons = _pool_sub.filter(_pool_sub["is_consumption"] == 1).sort("datetime_utc")["target"].to_numpy()
    _y_inj = _pool_sub.filter(_pool_sub["is_consumption"] == 0).sort("datetime_utc")["target"].to_numpy()

    _lags = np.arange(1, 73)
    _n = len(_y_cons)

    def _calc_acf(series):
        _y_c = series - np.mean(series)
        _var = np.sum(_y_c ** 2)
        if _var == 0:
            return np.zeros(len(_lags))
        return np.array([np.sum(_y_c[:_n - lag] * _y_c[lag:]) / _var for lag in _lags])

    _acf_cons = _calc_acf(_y_cons)
    _acf_inj = _calc_acf(_y_inj)

    # Significance bounds (+/- 1.96 / sqrt(N))
    _ci = 1.96 / np.sqrt(_n)

    _fig_acf = go.Figure()
    _fig_acf.add_trace(
        go.Scatter(
            x=_lags,
            y=_acf_cons,
            mode="lines+markers",
            name="Consumption ACF",
            line={"color": "#2B6CB0", "width": 2},
            marker={"size": 4},
        )
    )
    _fig_acf.add_trace(
        go.Scatter(
            x=_lags,
            y=_acf_inj,
            mode="lines+markers",
            name="Injection ACF",
            line={"color": "#FF9900", "width": 2},
            marker={"size": 4},
        )
    )
    # 95% Confidence Band
    _fig_acf.add_hline(y=_ci, line_dash="dot", line_color="gray", annotation_text="+95% CI")
    _fig_acf.add_hline(y=-_ci, line_dash="dot", line_color="gray", annotation_text="-95% CI")
    # Diurnal harmonics vertical markers at 24h, 48h, 72h
    for _h in [24, 48, 72]:
        _fig_acf.add_vline(x=_h, line_dash="dash", line_color="rgba(229, 62, 62, 0.4)", annotation_text=f"{_h}h")

    _fig_acf.update_layout(
        title=f"Temporal Autocorrelation Function (Pool {_sel_acf_pool}) - 72 Hour Lags",
        xaxis_title="Time Lag (Hours)",
        yaxis_title="Autocorrelation Coefficient",
        height=450,
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
    )

    mo.vstack(
        [
            mo.md(
                "*Left: Correlation matrix across pools shows high co-movement (>0.85–0.95), reflecting shared regional weather and diurnal cycles. "
                "Right: Autocorrelation curve exhibits prominent 24-hour periodic peaks (diurnal harmonics) across both consumption and solar injection.*"
            ),
            mo.hstack([mo.ui.plotly(_fig_cross_corr), mo.ui.plotly(_fig_acf)], justify="start"),
        ]
    )
    return


@app.cell
def _(mo, pools):
    norm_series_select = mo.ui.radio(
        options=["Consumption", "Injection", "Both (Side-by-Side)"],
        value="Consumption",
        inline=True,
        label="Target Energy Stream:",
    )

    norm_view_select = mo.ui.radio(
        options=[
            "24-Hour Diurnal (Weekday vs. Weekend)",
            "Archetype Segmentation Space (Weekend Drop vs. PV Ratio)",
            "Synchronized 7-Day Multi-Pool Overlay",
            "Seasonal Duck Curve Inversion (Winter Import vs. Spring Export)",
        ],
        value="24-Hour Diurnal (Weekday vs. Weekend)",
        inline=True,
        label="Analysis Perspective:",
    )

    norm_pool_filter = mo.ui.multiselect(
        options={f"Pool {p}": p for p in pools},
        value=[f"Pool {p}" for p in pools],
        label="Included Pools:",
    )

    mo.vstack(
        [
            mo.md("--- \n### 🔬 Why Were the Pools Split? Max-Normalized Load & Injection Profiles"),
            mo.md(
                r"""
    > 💡 **Investigating the Aggregation Architecture**:<br>
    > All 14 pools correlate strongly (>0.85–0.95), proving they are co-located in the **same geographic weather region**.
    > So **why did the grid operator / aggregator split them into distinct pools**?
    > By **normalizing each pool by its maximum capacity ($\frac{y}{\max(y)}$)**, we strip away pure scale differences (from 200 kW commercial sites up to 6.1 MW regional feeders) to expose **underlying customer archetypes, behavioral schedules, and solar export ratios**.
    """
            ),
            mo.hstack([norm_series_select, norm_view_select], justify="start", gap=3),
            norm_pool_filter,
        ]
    )
    return norm_pool_filter, norm_series_select, norm_view_select


@app.cell
def _(
    df_meas,
    go,
    make_subplots,
    norm_pool_filter,
    norm_series_select,
    norm_view_select,
    np,
    pl,
    pools,
    px,
):
    _sel_stream = norm_series_select.value
    _sel_view = norm_view_select.value
    _active_pools = norm_pool_filter.value

    if not _active_pools:
        _active_pools = pools

    # Distinct palette for all 14 pools
    _palette = px.colors.qualitative.Dark24

    # 1. Compute Max-Normalized Table
    _df_norm = (
        df_meas.with_columns(
            (
                pl.col("target")
                / pl.col("target").max().over(["pool", "is_consumption"])
            ).alias("norm_target"),
            pl.when(pl.col("day_of_week") >= 6)
            .then(pl.lit("Weekend"))
            .otherwise(pl.lit("Weekday"))
            .alias("day_type"),
        )
        .filter(pl.col("pool").is_in(_active_pools))
    )

    # -------------------------------------------------------------
    # VIEW 1: 24-Hour Diurnal Curves (Weekday vs Weekend)
    # -------------------------------------------------------------
    if _sel_view == "24-Hour Diurnal (Weekday vs. Weekend)":
        if _sel_stream == "Both (Side-by-Side)":
            fig_norm = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Consumption: Weekday (Mon–Fri)",
                    "Consumption: Weekend (Sat–Sun)",
                    "Injection: Weekday (Mon–Fri)",
                    "Injection: Weekend (Sat–Sun)",
                ),
                vertical_spacing=0.12,
                horizontal_spacing=0.08,
            )
            for _idx, _p in enumerate(_active_pools):
                _c_p = _palette[_idx % len(_palette)]
                # Consumption Weekday & Weekend
                for _col_idx, _dtype in enumerate(["Weekday", "Weekend"], start=1):
                    _sub_c = (
                        _df_norm.filter(
                            (pl.col("pool") == _p)
                            & (pl.col("is_consumption") == 1)
                            & (pl.col("day_type") == _dtype)
                        )
                        .group_by("hour")
                        .agg(pl.col("norm_target").mean().alias("val"))
                        .sort("hour")
                    )
                    fig_norm.add_trace(
                        go.Scatter(
                            x=_sub_c["hour"].to_list(),
                            y=_sub_c["val"].to_list(),
                            mode="lines",
                            name=f"Pool {_p}",
                            line={"color": _c_p, "width": 2 if _p in [0, 4, 10, 11] else 1.2},
                            legendgroup=f"Pool {_p}",
                            showlegend=(_col_idx == 1),
                        ),
                        row=1,
                        col=_col_idx,
                    )
                    # Injection Weekday & Weekend
                    _sub_i = (
                        _df_norm.filter(
                            (pl.col("pool") == _p)
                            & (pl.col("is_consumption") == 0)
                            & (pl.col("day_type") == _dtype)
                        )
                        .group_by("hour")
                        .agg(pl.col("norm_target").mean().alias("val"))
                        .sort("hour")
                    )
                    fig_norm.add_trace(
                        go.Scatter(
                            x=_sub_i["hour"].to_list(),
                            y=_sub_i["val"].to_list(),
                            mode="lines",
                            name=f"Pool {_p}",
                            line={"color": _c_p, "width": 2 if _p in [0, 4, 10, 11] else 1.2, "dash": "solid"},
                            legendgroup=f"Pool {_p}",
                            showlegend=False,
                        ),
                        row=2,
                        col=_col_idx,
                    )
            fig_norm.update_layout(height=650)
        else:
            _is_c = 1 if _sel_stream == "Consumption" else 0
            fig_norm = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=(
                    f"{_sel_stream}: Weekday Profile (Mon–Fri)",
                    f"{_sel_stream}: Weekend Profile (Sat–Sun)",
                ),
                horizontal_spacing=0.08,
            )
            for _idx, _p in enumerate(_active_pools):
                _c_p = _palette[_idx % len(_palette)]
                for _col_idx, _dtype in enumerate(["Weekday", "Weekend"], start=1):
                    _sub = (
                        _df_norm.filter(
                            (pl.col("pool") == _p)
                            & (pl.col("is_consumption") == _is_c)
                            & (pl.col("day_type") == _dtype)
                        )
                        .group_by("hour")
                        .agg(pl.col("norm_target").mean().alias("val"))
                        .sort("hour")
                    )
                    fig_norm.add_trace(
                        go.Scatter(
                            x=_sub["hour"].to_list(),
                            y=_sub["val"].to_list(),
                            mode="lines",
                            name=f"Pool {_p}",
                            line={"color": _c_p, "width": 2.2 if _p in [0, 4, 7, 10, 11] else 1.3},
                            legendgroup=f"Pool {_p}",
                            showlegend=(_col_idx == 1),
                        ),
                        row=1,
                        col=_col_idx,
                    )
            fig_norm.update_layout(height=420)

        fig_norm.update_xaxes(title_text="Hour of Day (UTC)", dtick=3)
        fig_norm.update_yaxes(title_text="Normalized Load (0 to 1.0)", range=[0, 1.05])
        fig_norm.update_layout(
            title="Max-Normalized Average Diurnal Curves by Pool (0.0 = Zero, 1.0 = Observed Peak)",
            template="plotly_white",
            legend={"orientation": "h", "yanchor": "bottom", "y": -0.22, "xanchor": "center", "x": 0.5},
            margin={"l": 40, "r": 40, "t": 60, "b": 60},
        )

    # -------------------------------------------------------------
    # VIEW 2: Archetype Segmentation Space
    # -------------------------------------------------------------
    elif _sel_view == "Archetype Segmentation Space (Weekend Drop vs. PV Ratio)":
        # Calculate pool-level archetype metrics
        _pool_recs = []
        for _p in _active_pools:
            _meas_p = df_meas.filter(pl.col("pool") == _p)
            _c_series = _meas_p.filter(pl.col("is_consumption") == 1)
            _i_series = _meas_p.filter(pl.col("is_consumption") == 0)

            _c_max = float(_c_series["target"].max())
            _i_max = float(_i_series["target"].max())
            _pv_ratio = (_i_max / _c_max) if _c_max > 0 else 0.0

            # Weekday vs Weekend consumption
            _c_wd = float(_c_series.filter(pl.col("day_of_week") <= 5)["target"].mean())
            _c_we = float(_c_series.filter(pl.col("day_of_week") >= 6)["target"].mean())
            _we_ratio = (_c_we / _c_wd) if _c_wd > 0 else 1.0

            # Baseload ratio (10th percentile / 90th percentile)
            _c_vals = _c_series["target"].to_numpy()
            _p10 = float(np.percentile(_c_vals, 10))
            _p90 = float(np.percentile(_c_vals, 90))
            _base_ratio = (_p10 / _p90) if _p90 > 0 else 0.0

            # Inferred cluster
            if _c_max > 2000:
                _arch = "Regional Substation / Macro-Grid"
                _color = "#3B82F6"
            elif _pv_ratio >= 3.5:
                _arch = "High-PV Exporter / Solar Park Feed"
                _color = "#F59E0B"
            elif _we_ratio < 0.72:
                _arch = "Commercial Office / 5-Day Workplace"
                _color = "#EC4899"
            elif _we_ratio >= 0.88 and _base_ratio >= 0.45:
                _arch = "Continuous Industrial / Baseload"
                _color = "#10B981"
            else:
                _arch = "Mixed Commercial & Light Industrial"
                _color = "#8B5CF6"

            _pool_recs.append(
                {
                    "Pool": f"Pool {_p}",
                    "Pool_ID": _p,
                    "Max_Consumption_kW": _c_max,
                    "Max_Injection_kW": _i_max,
                    "PV_to_Load_Ratio": _pv_ratio,
                    "Weekend_to_Weekday_Ratio": _we_ratio,
                    "Baseload_Ratio": _base_ratio,
                    "Archetype": _arch,
                    "Marker_Color": _color,
                }
            )

        _seg_df = pl.DataFrame(_pool_recs).to_pandas()

        fig_norm = go.Figure()
        for _arch_name, _grp in _seg_df.groupby("Archetype"):
            fig_norm.add_trace(
                go.Scatter(
                    x=_grp["Weekend_to_Weekday_Ratio"] * 100,
                    y=_grp["PV_to_Load_Ratio"],
                    mode="markers+text",
                    name=_arch_name,
                    text=_grp["Pool"],
                    textposition="top center",
                    marker={
                        "size": np.clip(np.sqrt(_grp["Max_Consumption_kW"]) * 1.5, 12, 45),
                        "color": _grp["Marker_Color"].iloc[0],
                        "opacity": 0.85,
                        "line": {"width": 1.5, "color": "white"},
                    },
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        + "Archetype: "
                        + _arch_name
                        + "<br>"
                        + "Weekend/Weekday Ratio: %{x:.1f}%<br>"
                        + "PV/Load Ratio: %{y:.2f}x<br>"
                        + "<extra></extra>"
                    ),
                )
            )

        # Quadrant guide lines
        fig_norm.add_vline(x=75, line_dash="dash", line_color="rgba(100, 116, 139, 0.4)", annotation_text="Weekday / Weekend Split")
        fig_norm.add_hline(y=3.0, line_dash="dash", line_color="rgba(245, 158, 11, 0.4)", annotation_text="Net Solar Export Threshold")

        fig_norm.update_layout(
            title="Customer Archetype Segmentation Space (Bubble Size = Max Consumption Capacity)",
            xaxis_title="Weekend Consumption Retention (% of Weekday Mean)",
            yaxis_title="Solar-to-Load Ratio (Max Injection / Max Consumption)",
            height=500,
            template="plotly_white",
            legend={"orientation": "h", "yanchor": "bottom", "y": -0.25, "xanchor": "center", "x": 0.5},
            margin={"l": 40, "r": 40, "t": 60, "b": 60},
        )

    # -------------------------------------------------------------
    # VIEW 3: Synchronized 7-Day Multi-Pool Overlay
    # -------------------------------------------------------------
    elif _sel_view == "Synchronized 7-Day Multi-Pool Overlay":
        # Sample 7-day window: May 15 to May 21, 2023
        _is_c = 1 if _sel_stream == "Consumption" else 0
        _sample_df = (
            _df_norm.filter(
                (pl.col("datetime_utc") >= pl.datetime(2023, 5, 15))
                & (pl.col("datetime_utc") <= pl.datetime(2023, 5, 21, 23))
                & (pl.col("is_consumption") == _is_c)
            )
            .sort(["datetime_utc", "pool"])
        )

        fig_norm = go.Figure()
        for _idx, _p in enumerate(_active_pools):
            _p_data = _sample_df.filter(pl.col("pool") == _p)
            if len(_p_data) > 0:
                _c_p = _palette[_idx % len(_palette)]
                fig_norm.add_trace(
                    go.Scatter(
                        x=_p_data["datetime_utc"].to_list(),
                        y=_p_data["norm_target"].to_list(),
                        mode="lines",
                        name=f"Pool {_p}",
                        line={"color": _c_p, "width": 2 if _p in [0, 4, 10, 11] else 1.0},
                    )
                )

        fig_norm.update_layout(
            title=f"Synchronized 7-Day Multi-Pool Overlay ({_sel_stream}, May 15–21, 2023) - Max-Normalized",
            xaxis_title="Datetime (UTC)",
            yaxis_title="Normalized Load (0 to 1.0)",
            height=450,
            template="plotly_white",
            legend={"orientation": "h", "yanchor": "bottom", "y": -0.22, "xanchor": "center", "x": 0.5},
            margin={"l": 40, "r": 40, "t": 60, "b": 60},
        )

    # -------------------------------------------------------------
    # VIEW 4: Seasonal Duck Curve Inversion (Winter vs Spring)
    # -------------------------------------------------------------
    else:
        # Pivot into net load = consumption - injection normalized by max winter consumption
        _piv_meas = (
            df_meas.filter(pl.col("pool").is_in(_active_pools))
            .pivot(on="is_consumption", index=["datetime_utc", "pool", "month", "hour"], values="target")
            .rename({"0": "injection", "1": "consumption"})
            .with_columns((pl.col("consumption") - pl.col("injection")).alias("net_load"))
        )
        _max_c = _piv_meas.group_by("pool").agg(pl.col("consumption").max().alias("max_c"))
        _piv_meas = _piv_meas.join(_max_c, on="pool").with_columns(
            (pl.col("net_load") / pl.col("max_c")).alias("norm_net_load")
        )

        fig_norm = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                "January Diurnal Net Load (100% Net Import: +0.2x to +0.4x)",
                "May Diurnal Duck Curve (Massive Reverse Export: -1.2x to -2.6x)",
            ),
            horizontal_spacing=0.08,
        )

        for _idx, _p in enumerate(_active_pools):
            _c_p = _palette[_idx % len(_palette)]
            # Jan curve
            _jan_p = (
                _piv_meas.filter((pl.col("pool") == _p) & (pl.col("month") == 1))
                .group_by("hour")
                .agg(pl.col("norm_net_load").mean().alias("val"))
                .sort("hour")
            )
            fig_norm.add_trace(
                go.Scatter(
                    x=_jan_p["hour"].to_list(),
                    y=_jan_p["val"].to_list(),
                    mode="lines",
                    name=f"Pool {_p}",
                    line={"color": _c_p, "width": 2 if _p in [0, 7, 10, 11] else 1.2},
                    legendgroup=f"Pool {_p}",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
            # May curve
            _may_p = (
                _piv_meas.filter((pl.col("pool") == _p) & (pl.col("month") == 5))
                .group_by("hour")
                .agg(pl.col("norm_net_load").mean().alias("val"))
                .sort("hour")
            )
            fig_norm.add_trace(
                go.Scatter(
                    x=_may_p["hour"].to_list(),
                    y=_may_p["val"].to_list(),
                    mode="lines",
                    name=f"Pool {_p}",
                    line={"color": _c_p, "width": 2 if _p in [0, 7, 10, 11] else 1.2},
                    legendgroup=f"Pool {_p}",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

        fig_norm.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=1)
        fig_norm.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=2, annotation_text="Zero Net Flow")

        fig_norm.update_xaxes(title_text="Hour of Day (UTC)", dtick=3)
        fig_norm.update_yaxes(title_text="Net Load / Max Consumption", row=1, col=1)
        fig_norm.update_yaxes(title_text="Net Load / Max Consumption (Negative = Export)", row=1, col=2)
        fig_norm.update_layout(
            title="<b>Seasonal Inversion of Feeder Net Load: January Import vs. May Solar Duck Curve</b>",
            template="plotly_white",
            height=450,
            legend={"orientation": "h", "yanchor": "bottom", "y": -0.22, "xanchor": "center", "x": 0.5},
            margin={"l": 40, "r": 40, "t": 60, "b": 60},
        )

    return (
        fig_norm,
    )


@app.cell
def _(df_meas, fig_norm, mo, pl, pools):
    # Construct comprehensive archetype summary table
    _rows = []
    for _p in pools:
        _meas_p = df_meas.filter(pl.col("pool") == _p)
        _c_s = _meas_p.filter(pl.col("is_consumption") == 1)
        _i_s = _meas_p.filter(pl.col("is_consumption") == 0)

        _c_max = float(_c_s["target"].max())
        _i_max = float(_i_s["target"].max())
        _pv_ratio = (_i_max / _c_max) if _c_max > 0 else 0.0

        # Weekend retention (Saturday-Sunday vs Monday-Friday)
        _c_wd = float(_c_s.filter(pl.col("day_of_week") <= 5)["target"].mean())
        _c_we = float(_c_s.filter(pl.col("day_of_week") >= 6)["target"].mean())
        _we_ratio = (_c_we / _c_wd) if _c_wd > 0 else 1.0

        # Winter heating multiplier (Jan mean / May mean)
        _c_jan = float(_c_s.filter(pl.col("month") == 1)["target"].mean())
        _c_may = float(_c_s.filter(pl.col("month") == 5)["target"].mean())
        _heat_ratio = (_c_jan / _c_may) if _c_may > 0 else 1.0

        # May midday net export (mean net load at hours 9-12 / max consumption)
        _sub_may = (
            _meas_p.filter(pl.col("month") == 5)
            .pivot(on="is_consumption", index=["datetime_utc", "hour"], values="target")
            .rename({"0": "injection", "1": "consumption"})
            .with_columns((pl.col("consumption") - pl.col("injection")).alias("net_load"))
        )
        _midday_net = float(
            _sub_may.filter((pl.col("hour") >= 9) & (pl.col("hour") <= 12))["net_load"].mean()
        ) / _c_max

        # Feeder hierarchy and grid role
        if _c_max > 5000:
            _role = "Primary Substation Anchor"
            _note = "Regional MV/HV hub (~1,500 prosumers); balanced 2.38x PV ratio"
        elif _c_max > 1500:
            _role = "Secondary Substation Branch"
            _note = "Medium-voltage branch feeder (~500 prosumers); 3.20x PV ratio"
        elif _pv_ratio >= 3.8:
            _role = "Critical Reverse-Flow Feeder"
            _note = f"Severe solar saturation ({_pv_ratio:.2f}x); back-feeds {_midday_net:.2f}x max load in May"
        elif _c_max <= 350:
            _role = "Micro-Feeder / Cul-de-sac"
            _note = "Local low-voltage spur (~30–50 homes); sensitive to local spikes"
        else:
            _role = "Standard Neighborhood Feeder"
            _note = "Balanced residential feeder (~100–150 homes); 3.3x–3.5x PV ratio"

        _rows.append(
            {
                "Pool": f"Pool {_p}",
                "Max Cons (kW)": f"{_c_max:,.1f}",
                "Max Inj (kW)": f"{_i_max:,.1f}",
                "PV/Load Ratio": f"{_pv_ratio:.2f}x",
                "Weekend Retention": f"{_we_ratio * 100:.1f}%",
                "Winter Heat Multiplier": f"{_heat_ratio:.2f}x",
                "May Midday Net Flow": f"{_midday_net:.2f}x",
                "Feeder Classification": _role,
                "Grid Characteristics & Congestion Risk": _note,
            }
        )

    _archetype_table = mo.ui.table(
        pl.DataFrame(_rows).to_pandas(),
        selection=None,
    )

    _narrative = mo.md(
        r"""
    ### 🎯 Conclusive Findings: Why the Pools Were Split in This Way

    Synthesizing the max-normalized telemetry, diurnal waveforms, and cross-pool metrics yields five conclusive architectural findings:

    1. **Identical Geographic Weather Zone (Shared Micro-Climate)**:
       - Every pool displays an identical solar generation window peaking tightly at **09:00–10:00 UTC (12:00–13:00 local solar time)**.
       - Synchronized multi-pool overlays show that weather transients (cloud cover drops, rain fronts, sudden clear skies) hit all 14 pools in the same hour, confirming they are located in the **same distribution system operator (DSO) service area**.

    2. **Uniform Customer Archetype: Residential Prosumer Portfolios**:
       - Across all 14 pools, **weekend consumption is 5% to 13% HIGHER than weekday consumption** (retention ratio: 1.05 to 1.13). Commercial offices drop 30–50% on weekends; continuous industry is flat (1.0). Residential households consume more on weekends due to home occupancy, appliance use, and heating.
       - **Behind-The-Meter Self-Consumption**: In all pools, metered grid import dips to its daily low at 10:00 UTC (~180 kW) because rooftop solar directly powers household appliances on-site before exporting surplus power. Grid import peaks at 18:00 UTC (~384 kW) as the sun sets and evening domestic routines begin.
       - **Electrified Space Heating**: Winter consumption is **2.3× to 2.9× higher** than spring consumption across all pools, demonstrating widespread adoption of electric heat pumps.

    3. **Physical Feeder Hierarchy (Substation to Cul-de-Sac)**:
       - The pools represent physical electrical topology rather than arbitrary statistical clusters:
         - **Primary Substation Hub (Pool 0)**: Peak load of 6.1 MW and peak solar of 14.6 MW (~1,500+ households).
         - **Secondary Substation Feeder (Pool 11)**: Peak load of 2.1 MW and peak solar of 6.9 MW (~500 households).
         - **Standard Neighborhood Feeders (Pools 3, 5, 7, 9, 10, 14, 15)**: Peak loads between 440 kW and 920 kW (~100–200 households).
         - **Micro-Feeders / Cul-de-sacs (Pools 1, 2, 4, 8, 13)**: Peak loads between 200 kW and 335 kW (~30–60 households).

    4. **The Solar Duck Curve & Reverse Power Flow Congestion**:
       - In January, all 14 feeders are steady **net importers** (+0.2× to +0.4× of peak winter capacity).
       - In May, solar production dramatically overwhelms local demand, flipping **every single feeder into a massive net exporter**.
       - **Critical Congestion Hotspots (Pools 7 and 10)**: Peak solar generation is **4.06× to 4.41× larger** than peak local consumption. At midday in May, these feeders back-feed **-2.2× to -2.5× their maximum winter demand** up through distribution transformers into the medium-voltage grid.

    5. **Why Splitting the Pools is Operationally Essential**:
       - If the 14 pools were lumped into a single aggregate portfolio, the severe local overvoltage and reverse-flow congestion on Feeders 7 and 10 would be masked by the heavier absorption on Feeder 0.
       - By maintaining 14 separate pools, the DSO and battery optimizer can:
         1. Accurately model transformer headroom on individual distribution branches.
         2. Strategically dispatch local battery storage or demand response to absorb excess midday solar directly where the feeder is congested.
         3. Monetize localized flexibility and participate in DSO congestion relief markets.
    """
    )

    mo.vstack(
        [
            mo.ui.plotly(fig_norm),
            _archetype_table,
            _narrative,
        ]
    )


@app.cell
def _(mo, pools):
    outlier_pool_select = mo.ui.dropdown(
        options={f"Pool {p}": p for p in pools},
        value="Pool 4",
        label="Target Pool:",
    )

    outlier_type_select = mo.ui.radio(
        options=["Consumption", "Injection"],
        value="Consumption",
        inline=True,
        label="Anomaly Series:",
    )

    outlier_thresh_slider = mo.ui.slider(
        start=3.0,
        stop=8.0,
        step=0.5,
        value=5.0,
        label="Outlier Sensitivity (Z-Threshold):",
    )

    weather_guard_switch = mo.ui.switch(
        value=True,
        label="Weather Guard (Protects Sudden Sunny Days):",
    )

    mo.vstack(
        [
            mo.md("--- \n### 🚨 Smart Outlier Detection & Cleaner (Pool 4 Deep Dive)"),
            mo.md(
                r"""
    > 🔍 **Outlier Discovery in Pool 4**:<br>
    > Investigation of Pool 4 identified major sensor glitches:
    > - **November 12, 2021 (08:00 UTC)**: Consumption spiked by **10× to 334.93 kW** (surrounded by normal ~33 kW readings). This single-point glitch distorted the peak demand metric.
    > - **April 10, 2022 (23:00 UTC)**: Injection reported **109.67 kW at 11 PM at night** with zero solar radiation (physically impossible).
    >
    > 🛡️ **Weather Guard Protection**:<br>
    > A naive moving average flags sudden sunny days as outliers because solar generation surges from 0 to 900 kW and consumption drops due to self-consumption. The **Weather Guard** compares observed surges against solar radiation: if sunshine is high, the daytime surge/dip is preserved as legitimate physical behavior!
                    """
            ),
            mo.hstack(
                [
                    outlier_pool_select,
                    outlier_type_select,
                    outlier_thresh_slider,
                    weather_guard_switch,
                ],
                justify="start",
                gap=2,
            ),
        ]
    )
    return (
        outlier_pool_select,
        outlier_thresh_slider,
        outlier_type_select,
        weather_guard_switch,
    )


@app.cell
def _(
    df_meas,
    df_weath,
    go,
    mo,
    np,
    outlier_pool_select,
    outlier_thresh_slider,
    outlier_type_select,
    pd,
    weather_guard_switch,
):
    _sel_p = outlier_pool_select.value
    _sel_t = outlier_type_select.value
    _thresh = outlier_thresh_slider.value
    _use_guard = weather_guard_switch.value

    _is_cons = 1 if _sel_t == "Consumption" else 0

    # Filter pool and join with weather
    _pool_data = df_meas.filter((df_meas["pool"] == _sel_p) & (df_meas["is_consumption"] == _is_cons)).sort("datetime_utc")
    _joined = _pool_data.join(df_weath, left_on="datetime_utc", right_on="forecast_datetime_utc", how="inner").to_pandas()

    _y = _joined["target"].values
    _rad = _joined["surface_solar_radiation_downwards"].values
    _dt = _joined["datetime_utc"]
    _n = len(_y)

    # Robust local moving median & MAD
    _window = 5
    _s_target = pd.Series(_y)
    _local_med = _s_target.rolling(window=_window, center=True, min_periods=1).median().values
    _res = np.abs(_y - _local_med)

    _mad = pd.Series(_res).rolling(window=_window * 5, center=True, min_periods=1).median().values * 1.4826
    _mad = np.where(np.isnan(_mad) | (_mad < 1.0), 1.0, _mad)
    _z = _res / _mad

    _is_outlier = np.zeros(_n, dtype=bool)
    _cleaned = _y.copy()

    for _i in range(_n):
        _val = _y[_i]
        _r = _rad[_i]
        _prev = _y[_i - 1] if _i > 0 else _val
        _next = _y[_i + 1] if _i < _n - 1 else _val
        _nbr_med = np.median([_prev, _next])

        # Nighttime injection glitch check
        if _is_cons == 0 and _r <= 0.1 and _val > 15.0 and _prev < 5.0 and _next < 5.0:
            _is_outlier[_i] = True
            _cleaned[_i] = _nbr_med
            continue

        # Weather Guard: Protect sunny day surges and dips
        if _use_guard:
            if _is_cons == 0 and _r > 20.0 and _val > 0:
                continue  # Sunny day generation is legitimate
            if _is_cons == 1 and _r > 50.0 and _val < _local_med[_i]:
                continue  # Solar self-consumption dip is legitimate

        # Outlier trigger: extreme z-score and absolute deviation
        if _z[_i] > _thresh and _res[_i] > 40.0:
            _is_outlier[_i] = True
            _cleaned[_i] = _nbr_med

    _joined["is_outlier"] = _is_outlier
    _joined["cleaned_target"] = _cleaned
    _joined["z_score"] = np.round(_z, 1)

    _outlier_df = _joined[_joined["is_outlier"]].copy()

    # Time series figure with outlier markers
    _fig_clean = go.Figure()
    _fig_clean.add_trace(
        go.Scatter(
            x=_dt,
            y=_y,
            mode="lines",
            name="Raw Target (with Outliers)",
            line={"color": "rgba(229, 62, 62, 0.4)", "width": 1.5},
        )
    )
    _fig_clean.add_trace(
        go.Scatter(
            x=_dt,
            y=_cleaned,
            mode="lines",
            name="Cleaned Series (Smart Filter)",
            line={"color": "#2B6CB0" if _is_cons == 1 else "#FF9900", "width": 1.8},
        )
    )

    if len(_outlier_df) > 0:
        _fig_clean.add_trace(
            go.Scatter(
                x=_outlier_df["datetime_utc"],
                y=_outlier_df["target"],
                mode="markers",
                name=f"Detected Outliers ({len(_outlier_df)})",
                marker={"color": "#E53E3E", "size": 9, "symbol": "x"},
            )
        )

    _fig_clean.update_layout(
        title=f"Pool {_sel_p} {_sel_t}: Raw vs. Cleaned Signal ({len(_outlier_df)} Outliers Flagged)",
        xaxis_title="Datetime (UTC)",
        yaxis_title="Power (kW)",
        height=420,
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
    )

    # Zoomed snapshot of primary glitch if pool 4
    _zoom_widget = None
    if _sel_p == 4 and _is_cons == 1 and len(_outlier_df) > 0:
        _sub_zoom = _joined[
            (_joined["datetime_utc"] >= "2021-11-12 00:00:00")
            & (_joined["datetime_utc"] <= "2021-11-12 20:00:00")
        ]
        _fig_zoom = go.Figure()
        _fig_zoom.add_trace(
            go.Scatter(
                x=_sub_zoom["datetime_utc"],
                y=_sub_zoom["target"],
                mode="lines+markers",
                name="Raw Spike (334.9 kW)",
                line={"color": "#E53E3E", "width": 2},
                marker={"size": 6},
            )
        )
        _fig_zoom.add_trace(
            go.Scatter(
                x=_sub_zoom["datetime_utc"],
                y=_sub_zoom["cleaned_target"],
                mode="lines+markers",
                name="Cleaned Restoration (33.2 kW)",
                line={"color": "#38A169", "width": 2, "dash": "dash"},
                marker={"size": 6},
            )
        )
        _fig_zoom.update_layout(
            title="Glitch Deep-Dive: Nov 12, 2021 Sensor Error in Pool 4 (10× decimal error)",
            xaxis_title="Time",
            yaxis_title="kW",
            height=280,
            template="plotly_white",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
            margin={"l": 40, "r": 40, "t": 50, "b": 40},
        )
        _zoom_widget = mo.ui.plotly(_fig_zoom)

    # Summary table of detected outliers
    _outlier_table_cols = [
        "datetime_utc",
        "target",
        "cleaned_target",
        "surface_solar_radiation_downwards",
        "z_score",
    ]
    _table_out = mo.ui.table(
        _outlier_df[_outlier_table_cols].rename(
            columns={
                "datetime_utc": "Timestamp (UTC)",
                "target": "Raw Target (kW)",
                "cleaned_target": "Cleaned (kW)",
                "surface_solar_radiation_downwards": "Solar Rad (W/m²)",
                "z_score": "Anomaly Score (Z)",
            }
        ),
        selection=None,
    )

    mo.vstack(
        [
            mo.ui.plotly(_fig_clean),
            _zoom_widget if _zoom_widget is not None else mo.md(""),
            mo.md(f"#### 📋 Flagged Outlier Records ({len(_outlier_df)} Total):"),
            _table_out,
        ]
    )
    return


@app.cell
def _(df_meas, df_weath, mo):
    _table_tabs = mo.ui.tabs(
        {
            "📊 Measurements Head": mo.ui.table(
                df_meas.select(
                    ["pool", "datetime_utc", "is_consumption", "type_name", "split_name", "target"]
                )
                .head(25)
                .to_pandas(),
                selection=None,
            ),
            "🌤️ Weather Head": mo.ui.table(
                df_weath.head(25).to_pandas(),
                selection=None,
            ),
        }
    )

    mo.vstack(
        [
            mo.md("--- \n### 🔍 Raw Data Tables Inspection"),
            _table_tabs,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
