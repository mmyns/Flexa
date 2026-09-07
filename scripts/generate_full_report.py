"""Build the single combined Flexa PDF report.

Structure (9 pages, continuously numbered):
  1   Codebase structure — repository layout, pipeline flow, design decisions
  2   How to run — Docker container, native uv fallback, inputs & outputs
  3-7 Forecasting model  (scripts/generate_forecast_report.py)
  8-9 Battery arbitrage  (scripts/generate_bess_report.py)

The two section modules expose `build_forecast_figures()` / `build_bess_figures()`
which return bare matplotlib figures; this script owns the page offsets, the
footer numbering and the single PdfPages handle.

Usage:
    uv run python scripts/generate_full_report.py [--output PATH] [--no-png]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_bess_report import N_BESS_PAGES, build_bess_figures  # noqa: E402
from generate_forecast_report import N_FORECAST_PAGES, build_forecast_figures  # noqa: E402

# Shared palette with the two section modules.
DARK_NAVY = "#0f172a"
SLATE_GRAY = "#475569"
LIGHT_BG = "#f8fafc"
CODE_BG = "#0f172a"
CODE_FG = "#e2e8f0"
ACCENT_BLUE = "#0284c7"
ACCENT_TEAL = "#0d9488"
ACCENT_AMBER = "#d97706"
ACCENT_CORAL = "#e11d48"
GREEN = "#16a34a"

N_COVER_PAGES = 2
TOTAL_PAGES = N_COVER_PAGES + N_FORECAST_PAGES + N_BESS_PAGES

MONO = ["DejaVu Sans Mono", "Menlo", "Consolas", "Courier New", "monospace"]


# =============================================================================
# Shared page furniture
# =============================================================================
def add_header_footer(fig: Figure, title: str, subtitle: str, page_num: int) -> None:
    """Draw the report-wide header banner and footer on a cover page."""
    fig.patch.set_facecolor("#ffffff")
    fig.text(
        0.07,
        0.962,
        "FLEXA ENERGY FORECASTING & BATTERY STEERING",
        fontsize=14,
        fontweight="bold",
        color=DARK_NAVY,
    )
    fig.text(0.07, 0.942, f"{title}: {subtitle}", fontsize=8.8, color=SLATE_GRAY)
    fig.text(
        0.07,
        0.925,
        "Combined Technical Report | Forecasting Pipeline + BESS Arbitrage Optimizer | Reproducible via Docker",
        fontsize=7.5,
        color="#64748b",
    )
    fig.add_artist(plt.Line2D([0.07, 0.93], [0.916, 0.916], color=DARK_NAVY, linewidth=1.5))

    fig.text(
        0.07,
        0.022,
        "Flexa | Polars feature engineering · scikit-learn GBDT · CVXPY convex optimization",
        fontsize=6.8,
        color="#94a3b8",
    )
    fig.text(
        0.93,
        0.022,
        f"Page {page_num} of {TOTAL_PAGES}",
        fontsize=6.8,
        color="#94a3b8",
        ha="right",
    )


def panel(
    fig: Figure,
    gs_cell,
    title: str,
    *,
    face: str = LIGHT_BG,
    edge: str = "#cbd5e1",
    title_color: str = DARK_NAVY,
):
    """Create a titled, bordered panel axes and return it."""
    ax = fig.add_subplot(gs_cell)
    ax.set_facecolor(face)
    for spine in ax.spines.values():
        spine.set_edgecolor(edge)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.text(
            0.022,
            0.945,
            title,
            transform=ax.transAxes,
            fontsize=8.0,
            fontweight="bold",
            color=title_color,
            va="top",
        )
    return ax


def code_block(
    ax,
    x: float,
    y: float,
    lines: str,
    *,
    fontsize: float = 6.1,
    width: float = 0.956,
    pad: float = 0.018,
) -> float:
    """Render a dark terminal-style code block; returns the y of its bottom edge."""
    n_lines = lines.count("\n") + 1
    # Axes-fraction height of the block: line height is derived from the figure
    # height so blocks stay proportional regardless of the panel's own size.
    fig_h_pt = ax.figure.get_size_inches()[1] * 72
    ax_h_pt = ax.get_position().height * fig_h_pt
    block_h = (n_lines * fontsize * 1.42 + 2 * pad * ax_h_pt) / ax_h_pt

    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y - block_h),
            width,
            block_h,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            transform=ax.transAxes,
            facecolor=CODE_BG,
            edgecolor="#1e293b",
            linewidth=0.6,
        )
    )
    ax.text(
        x + 0.014,
        y - pad,
        lines,
        transform=ax.transAxes,
        fontsize=fontsize,
        color=CODE_FG,
        family=MONO,
        va="top",
        linespacing=1.42,
    )
    return y - block_h


# =============================================================================
# PAGE 1: CODEBASE STRUCTURE
# =============================================================================
REPO_TREE = """Flexa/
├── Dockerfile                        Image definition — uv + uv.lock, core deps only (no torch)
├── docker-compose.yml                Compose services: forecast · battery · report · all
├── docker/entrypoint.sh              Task dispatcher inside the image; forwards CLI flags verbatim
├── Makefile                          make forecast | battery | report | docker-build | docker-*
├── pyproject.toml · uv.lock          PEP 621 metadata + fully pinned, cross-platform lockfile
│
├── src/flexa/                        The installable library. Scripts are thin wrappers over this.
│   ├── config.py                     Pydantic schemas: PipelineConfig, FeatureConfig, BaselineModelConfig
│   ├── data/                         loader.py (measurements · weather · spot prices)
│   │                                 preprocessor.py · validation.py
│   ├── features/                     time_features.py    lags, deltas, calendar, cyclical encodings
│   │                                 weather_features.py irradiance, cloudcover delta, solar_peak_to_t
│   ├── models/                       base.py      BaseForecaster ABC + ForecastResult
│   │                                 baseline.py  CHAMPION — GBDT on first differences (Delta T)
│   │                                 naive.py     T-1 persistence reference
│   │                                 chronos.py   Chronos-2 benchmark — optional `chronos` extra
│   ├── evaluation/                   metrics.py (MAE·RMSE·WAPE·pinball) · backtest.py (expanding CV)
│   ├── optimization/                 battery.py    BatteryConfig / BatteryAsset (power, energy, eta)
│   │                                 objectives.py day-ahead arbitrage, degradation, capacity
│   │                                 optimizer.py  CVXPY linear program, day-by-day rolling solve
│   ├── pipeline.py                   End-to-end orchestration used by the CLI
│   └── cli.py                        Typer CLI entry point (`flexa forecast`, `flexa synthetic`, ...)
│
├── scripts/                          Exactly two operational entry points, plus reporting:
│   ├── run_forecast.py               ►  ENTRY POINT 1 — forecasting
│   ├── run_battery.py                ►  ENTRY POINT 2 — battery steering
│   ├── split_data.py                 One-off: raw CSV → train/test Parquet splits
│   ├── generate_forecast_report.py   Section pages 3–7 of this document
│   ├── generate_bess_report.py       Section pages 8–9 of this document
│   └── generate_full_report.py       ►  Builds THIS document (cover pages + both sections)
│
├── notebooks/                        marimo notebooks: eda_raw_data.py (EDA + outlier cleaning),
│                                     forecasting.py (model exploration)
├── tests/                            pytest: data, features, models, metrics, pipeline, battery,
│                                     optimizer, weather features
└── data/
    ├── raw/                          1_measurements.csv · 1_weather.csv · 2_electricity_prices.csv (git-ignored)
    ├── processed/                    train/test splits · battery schedules · generated PDFs
    ├── forecasts/                    champion forecasts (Parquet + CSV) · evaluation summary JSON
    └── benchmarks/                   model-selection ablation results consumed by the report"""

FLOW_STAGES = [
    ("data/raw\nCSV", "#64748b"),
    ("split_data.py\ntrain / test", ACCENT_BLUE),
    ("features/\nlags · weather", ACCENT_BLUE),
    ("run_forecast.py\nGBDT Delta T", GREEN),
    ("data/forecasts/\nparquet · csv", ACCENT_TEAL),
]

FLOW_STAGES_2 = [
    ("data/raw\nspot prices", "#64748b"),
    ("optimization/\nCVXPY LP", ACCENT_AMBER),
    ("run_battery.py\n1h vs 2h", GREEN),
    ("data/processed/\nschedules", ACCENT_TEAL),
    ("generate_full_report.py\nthis PDF", ACCENT_CORAL),
]


def _draw_flow(ax, stages, y_center: float, label: str) -> None:
    """Draw one horizontal pipeline row of labelled boxes joined by arrows."""
    n = len(stages)
    box_w = 0.163
    gap = (1.0 - n * box_w) / (n - 1)
    box_h = 0.30

    ax.text(
        0.0,
        y_center + box_h / 2 + 0.10,
        label,
        fontsize=6.6,
        fontweight="bold",
        color=SLATE_GRAY,
        va="bottom",
    )

    for i, (text, color) in enumerate(stages):
        x = i * (box_w + gap)
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y_center - box_h / 2),
                box_w,
                box_h,
                boxstyle="round,pad=0.006,rounding_size=0.02",
                facecolor="#ffffff",
                edgecolor=color,
                linewidth=1.1,
            )
        )
        ax.add_patch(
            patches.Rectangle(
                (x, y_center + box_h / 2 - 0.035),
                box_w,
                0.035,
                facecolor=color,
                edgecolor="none",
            )
        )
        ax.text(
            x + box_w / 2,
            y_center - 0.02,
            text,
            ha="center",
            va="center",
            fontsize=5.9,
            color=DARK_NAVY,
            linespacing=1.35,
        )
        if i < n - 1:
            ax.annotate(
                "",
                xy=(x + box_w + gap - 0.008, y_center),
                xytext=(x + box_w + 0.008, y_center),
                arrowprops={"arrowstyle": "-|>", "color": SLATE_GRAY, "linewidth": 0.9},
            )


def build_codebase_page(page_num: int) -> Figure:
    """Page 1 — repository layout, pipeline flow and design decisions."""
    fig = plt.figure(figsize=(8.27, 11.69), dpi=300)
    add_header_footer(
        fig, "CODEBASE STRUCTURE", "Repository Layout, Pipeline Flow & Design Decisions", page_num
    )

    gs_tree = GridSpec(1, 1, figure=fig, left=0.07, right=0.93, top=0.895, bottom=0.440)
    gs_flow = GridSpec(1, 1, figure=fig, left=0.07, right=0.93, top=0.413, bottom=0.225)
    gs_notes = GridSpec(
        1, 2, figure=fig, left=0.07, right=0.93, top=0.198, bottom=0.048, wspace=0.05
    )

    # --- Repository tree ------------------------------------------------------
    ax_tree = panel(fig, gs_tree[0, 0], "REPOSITORY LAYOUT")
    ax_tree.text(
        0.014,
        0.905,
        REPO_TREE,
        transform=ax_tree.transAxes,
        fontsize=5.9,
        color="#1e293b",
        family=MONO,
        va="top",
        linespacing=1.34,
    )

    # --- Pipeline flow --------------------------------------------------------
    ax_flow = fig.add_subplot(gs_flow[0, 0])
    ax_flow.set_xlim(0, 1)
    ax_flow.set_ylim(0, 1)
    ax_flow.axis("off")
    _draw_flow(ax_flow, FLOW_STAGES, 0.72, "FORECASTING PATH")
    _draw_flow(ax_flow, FLOW_STAGES_2, 0.20, "BATTERY STEERING PATH")

    # --- Layer responsibilities ----------------------------------------------
    ax_resp = panel(fig, gs_notes[0, 0], "WHAT EACH LAYER OWNS")
    ax_resp.text(
        0.028,
        0.83,
        "• src/flexa/  — all logic worth testing. Pure functions over Polars\n"
        "  frames; no file paths or CLI concerns leak into the library.\n"
        "• scripts/    — orchestration only: parse flags, call the library,\n"
        "  print a Rich summary, write artefacts to data/.\n"
        "• tests/      — pytest against the library, not the scripts.\n"
        "• data/       — the sole mutable surface; bind-mounted into Docker\n"
        "  so container runs write straight back to the host.",
        transform=ax_resp.transAxes,
        fontsize=6.0,
        color="#1e293b",
        linespacing=1.32,
        va="top",
    )

    # --- Design decisions -----------------------------------------------------
    ax_dec = panel(fig, gs_notes[0, 1], "KEY DESIGN DECISIONS")
    ax_dec.text(
        0.028,
        0.83,
        "• Champion model is a GBDT trained on first differences (Delta T),\n"
        "  pooled globally across all 14 pools with pool as a categorical.\n"
        "• Strict temporal split (train ≤ Dec 2022, test ≥ Jan 2023) and\n"
        "  1-step-ahead evaluation — no lookahead leakage anywhere.\n"
        "• Battery dispatch is a convex LP solved per calendar day, so the\n"
        "  schedule is globally optimal given the price path.\n"
        "• uv.lock pins every transitive dependency: identical versions\n"
        "  resolve on macOS, Linux and inside the container. torch sits\n"
        "  behind the optional `chronos` extra, so it is absent at runtime.",
        transform=ax_dec.transAxes,
        fontsize=6.0,
        color="#1e293b",
        linespacing=1.32,
        va="top",
    )

    return fig


# =============================================================================
# PAGE 2: HOW TO RUN
# =============================================================================
def build_howto_page(page_num: int) -> Figure:
    """Page 2 — Docker workflow, native fallback, container commands, I/O contract."""
    fig = plt.figure(figsize=(8.27, 11.69), dpi=300)
    add_header_footer(
        fig, "RUNNING FLEXA", "Docker Container, Native Fallback & Data Contract", page_num
    )

    gs_docker = GridSpec(1, 1, figure=fig, left=0.07, right=0.93, top=0.895, bottom=0.592)
    gs_cmds = GridSpec(1, 1, figure=fig, left=0.07, right=0.93, top=0.567, bottom=0.372)
    gs_native = GridSpec(1, 1, figure=fig, left=0.07, right=0.93, top=0.347, bottom=0.196)
    gs_io = GridSpec(1, 1, figure=fig, left=0.07, right=0.93, top=0.171, bottom=0.048)

    # --- Docker ---------------------------------------------------------------
    ax_d = panel(
        fig,
        gs_docker[0, 0],
        "A · RUN WITH DOCKER  (works on any machine with Docker installed)",
        title_color=ACCENT_BLUE,
    )
    ax_d.text(
        0.022,
        0.885,
        "Requires only Docker — no Python, no uv, no compilers on the host. The image pins every dependency from uv.lock,\n"
        "so the numbers it produces are byte-for-byte reproducible across macOS, Linux and Windows/WSL2.",
        transform=ax_d.transAxes,
        fontsize=6.2,
        color="#1e293b",
        linespacing=1.3,
        va="top",
    )

    y = 0.795
    ax_d.text(
        0.022,
        y,
        "1. Clone and build the image (once, ~1–2 min — no torch in the runtime image):",
        transform=ax_d.transAxes,
        fontsize=6.5,
        fontweight="bold",
        color=DARK_NAVY,
        va="top",
    )
    y = code_block(
        ax_d,
        0.022,
        y - 0.035,
        "git clone <repo-url> && cd Flexa\ndocker build -t flexa:latest .",
    )

    y -= 0.045
    ax_d.text(
        0.022,
        y,
        "2. Run the forecasting pipeline (ENTRY POINT 1):",
        transform=ax_d.transAxes,
        fontsize=6.5,
        fontweight="bold",
        color=DARK_NAVY,
        va="top",
    )
    y = code_block(
        ax_d,
        0.022,
        y - 0.035,
        'docker run --rm -v "$(pwd)/data:/app/data" flexa:latest forecast\n'
        "#  optional flags, forwarded straight through to scripts/run_forecast.py:\n"
        "#     --model-variant global|rolling_30d   --pools 0 4 11   --format parquet|csv|both",
    )

    y -= 0.045
    ax_d.text(
        0.022,
        y,
        "3. Run the battery steering optimizer (ENTRY POINT 2):",
        transform=ax_d.transAxes,
        fontsize=6.5,
        fontweight="bold",
        color=DARK_NAVY,
        va="top",
    )
    y = code_block(
        ax_d,
        0.022,
        y - 0.035,
        'docker run --rm -v "$(pwd)/data:/app/data" flexa:latest battery\n'
        "#  optional flags, forwarded straight through to scripts/run_battery.py:\n"
        "#     --days 90        --max-cycles 1.0        --output-dir data/processed",
    )

    y -= 0.045
    ax_d.text(
        0.022,
        y,
        'The  -v "$(pwd)/data:/app/data"  bind mount is what makes results land on the host: the container reads\n'
        "data/raw and data/benchmarks from your checkout and writes its outputs back into the same directory.\n"
        "docker compose run --rm forecast  /  battery  /  report  are equivalent shorthands.",
        transform=ax_d.transAxes,
        fontsize=6.0,
        color=SLATE_GRAY,
        linespacing=1.3,
        va="top",
    )

    # --- Container commands ---------------------------------------------------
    ax_c = panel(fig, gs_cmds[0, 0], "B · CONTAINER COMMANDS", title_color=ACCENT_TEAL)
    rows = [
        (
            "forecast",
            "scripts/run_forecast.py",
            "Champion GBDT (Delta T), all 14 pools, solar + consumption",
            "data/forecasts/forecasts_champion_model.{parquet,csv} + evaluation summary JSON",
        ),
        (
            "battery",
            "scripts/run_battery.py",
            "Day-by-day convex arbitrage LP, 1h vs 2h battery",
            "data/processed/schedule_{1h,2h}_battery.{csv,parquet}",
        ),
        (
            "report",
            "scripts/generate_full_report.py",
            "Rebuild this 9-page combined PDF",
            "data/processed/flexa_full_report.pdf",
        ),
        (
            "split",
            "scripts/split_data.py",
            "Rebuild train/test splits from data/raw",
            "data/processed/{train,test}_{measurements,weather}.{parquet,csv}",
        ),
        (
            "all",
            "— all four, in order —",
            "split -> forecast -> battery -> report",
            "every artefact above, from raw CSV to final PDF",
        ),
        ("shell", "/bin/bash", "Interactive shell inside the image for debugging", "—"),
    ]
    ax_c.text(
        0.024,
        0.790,
        "COMMAND",
        transform=ax_c.transAxes,
        fontsize=6.0,
        fontweight="bold",
        color=SLATE_GRAY,
        family=MONO,
    )
    ax_c.text(
        0.135,
        0.790,
        "RUNS",
        transform=ax_c.transAxes,
        fontsize=6.0,
        fontweight="bold",
        color=SLATE_GRAY,
        family=MONO,
    )
    ax_c.text(
        0.375,
        0.790,
        "WHAT IT DOES  /  WHAT IT WRITES",
        transform=ax_c.transAxes,
        fontsize=6.0,
        fontweight="bold",
        color=SLATE_GRAY,
        family=MONO,
    )
    ax_c.add_artist(
        plt.Line2D(
            [0.024, 0.976], [0.752, 0.752], transform=ax_c.transAxes, color="#cbd5e1", linewidth=0.7
        )
    )

    row_y = 0.700
    for cmd, runs, does, writes in rows:
        ax_c.text(
            0.024,
            row_y,
            cmd,
            transform=ax_c.transAxes,
            fontsize=6.2,
            fontweight="bold",
            color=ACCENT_BLUE,
            family=MONO,
            va="top",
        )
        ax_c.text(
            0.135,
            row_y,
            runs,
            transform=ax_c.transAxes,
            fontsize=5.7,
            color="#1e293b",
            family=MONO,
            va="top",
        )
        ax_c.text(
            0.375, row_y, does, transform=ax_c.transAxes, fontsize=5.9, color="#1e293b", va="top"
        )
        ax_c.text(
            0.375,
            row_y - 0.052,
            writes,
            transform=ax_c.transAxes,
            fontsize=5.5,
            color="#64748b",
            family=MONO,
            va="top",
        )
        row_y -= 0.113

    # --- Native ---------------------------------------------------------------
    ax_n = panel(
        fig,
        gs_native[0, 0],
        "C · RUN WITHOUT DOCKER  (uv, Python 3.11/3.12)",
        title_color=ACCENT_AMBER,
    )
    ax_n.text(
        0.022,
        0.80,
        "Identical results — the container simply wraps these commands. uv installs the exact lockfile versions.",
        transform=ax_n.transAxes,
        fontsize=6.2,
        color="#1e293b",
        va="top",
    )
    code_block(
        ax_n,
        0.022,
        0.700,
        "curl -LsSf https://astral.sh/uv/install.sh | sh     # skip if uv is already installed\n"
        "uv sync --frozen                                    # create .venv from uv.lock\n"
        "\n"
        "uv run python scripts/run_forecast.py                # ENTRY POINT 1 — forecasting\n"
        "uv run python scripts/run_battery.py                 # ENTRY POINT 2 — battery steering\n"
        "uv run python scripts/generate_full_report.py        # rebuild this PDF\n"
        "\n"
        "make forecast | make battery | make report           # same three, via the Makefile",
        fontsize=6.0,
    )

    # --- I/O contract ---------------------------------------------------------
    ax_io = panel(
        fig, gs_io[0, 0], "D · DATA CONTRACT", face="#fffbeb", edge="#fcd34d", title_color="#92400e"
    )
    ax_io.add_patch(
        patches.Rectangle(
            (0, 0), 0.006, 1.0, transform=ax_io.transAxes, color=ACCENT_AMBER, clip_on=False
        )
    )
    ax_io.text(
        0.022,
        0.800,
        "REQUIRED INPUTS  (the container reads these through the bind mount)\n"
        "    data/raw/1_measurements.csv · data/raw/1_weather.csv · data/raw/2_electricity_prices.csv   — git-ignored, supply them locally\n"
        "    data/benchmarks/model_selection_benchmark.json   — tracked in git; ablation results the forecast section renders\n"
        "OUTPUTS  (written back to the host through the same mount)\n"
        "    data/processed/{train,test}_{measurements,weather}.{parquet,csv}  ·  schedule_{1h,2h}_battery.{csv,parquet}\n"
        "    data/forecasts/forecasts_champion_model.{parquet,csv}  ·  forecast_evaluation_summary.json\n"
        "    data/processed/flexa_full_report.pdf  (+ one PNG per page)\n"
        "COLD START  on a fresh clone, use  all  — the report section needs a completed  forecast  run to exist first.",
        transform=ax_io.transAxes,
        fontsize=5.9,
        color="#78350f",
        linespacing=1.34,
        va="top",
    )

    return fig


# =============================================================================
# Assembly
# =============================================================================
def generate_full_report(
    output_pdf_path: Path | str = "data/processed/flexa_full_report.pdf",
    write_pngs: bool = True,
) -> Path:
    """Assemble cover pages, the forecasting section and the BESS section into one PDF."""
    output_path = Path(output_pdf_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Building combined report ({TOTAL_PAGES} pages) -> {output_path}")

    print("Rendering Page 1: Codebase Structure...")
    print("Rendering Page 2: Running Flexa...")
    figs: list[Figure] = [build_codebase_page(1), build_howto_page(2)]

    print(
        f"Rendering Pages {N_COVER_PAGES + 1}-{N_COVER_PAGES + N_FORECAST_PAGES}: Forecasting section..."
    )
    figs += build_forecast_figures(page_offset=N_COVER_PAGES, total_pages=TOTAL_PAGES)

    bess_offset = N_COVER_PAGES + N_FORECAST_PAGES
    print(f"Rendering Pages {bess_offset + 1}-{TOTAL_PAGES}: Battery arbitrage section...")
    figs += build_bess_figures(page_offset=bess_offset, total_pages=TOTAL_PAGES)

    with PdfPages(output_path) as pdf:
        for i, fig in enumerate(figs, start=1):
            if write_pngs:
                fig.savefig(output_path.parent / f"flexa_full_report_page_{i}.png", dpi=300)
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Successfully generated {len(figs)}-page PDF: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the combined Flexa PDF report (codebase guide + forecasting + BESS)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/flexa_full_report.pdf"),
        help="Destination path for the combined PDF.",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Skip writing one PNG per page alongside the PDF.",
    )
    args = parser.parse_args()
    generate_full_report(output_pdf_path=args.output, write_pngs=not args.no_png)


if __name__ == "__main__":
    main()
