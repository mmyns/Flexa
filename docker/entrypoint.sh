#!/usr/bin/env bash
# Dispatch table for the Flexa image. The first argument selects a task; every
# remaining argument is forwarded verbatim to the underlying script, so the
# container accepts the same flags as a local `uv run python scripts/...` call.
set -euo pipefail

usage() {
    cat <<'EOF'
Flexa — energy forecasting & battery steering

Usage:
  docker run --rm -v "$(pwd)/data:/app/data" flexa:latest <command> [args...]

Commands:
  forecast [args]   Champion GBDT (Delta T) forecaster for all 14 pools.
                    Writes data/forecasts/{forecasts_champion_model.{parquet,csv},
                    forecast_evaluation_summary.json}.
                    Flags: --data-dir --output-dir --format {parquet,csv,both}
                           --model-variant {global,rolling_30d} --pools 0 1 2 ...

  battery [args]    Day-by-day convex arbitrage optimizer, 1h vs 2h battery.
                    Writes data/processed/schedule_{1h,2h}_battery.{csv,parquet}.
                    Flags: --days --max-cycles --data-path --output-dir

  report [args]     Build the combined PDF (codebase guide + forecast + BESS)
                    into data/processed/flexa_full_report.pdf.

  split [args]      Rebuild train/test splits from data/raw into data/processed.

  all               Run split -> forecast -> battery -> report in sequence.

  shell             Drop into an interactive bash shell.
  help              Show this message.

Anything else is executed as a raw command inside the image, e.g.
  docker run --rm flexa:latest python -c "import flexa; print(flexa.__version__)"
EOF
}

cmd="${1:-help}"
shift || true

case "$cmd" in
    forecast) exec python scripts/run_forecast.py "$@" ;;
    battery)  exec python scripts/run_battery.py "$@" ;;
    report)   exec python scripts/generate_full_report.py "$@" ;;
    split)    exec python scripts/split_data.py "$@" ;;
    all)
        python scripts/split_data.py
        python scripts/run_forecast.py
        python scripts/run_battery.py
        exec python scripts/generate_full_report.py
        ;;
    shell)    exec /bin/bash "$@" ;;
    help|-h|--help) usage ;;
    *)        exec "$cmd" "$@" ;;
esac
