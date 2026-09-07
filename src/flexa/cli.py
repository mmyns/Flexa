"""Command-line interface for Flexa time series forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from flexa.config import PipelineConfig
from flexa.data.loader import generate_synthetic_timeseries
from flexa.pipeline import ForecastingPipeline

app = typer.Typer(
    name="flexa",
    help="Flexa: Production-ready time series forecasting with Polars, scikit-learn, and Chronos.",
    add_completion=False,
)
console = Console()


@app.command("version")
def version() -> None:
    """Display Flexa version."""
    import flexa

    console.print(f"[bold green]Flexa[/bold green] version: [cyan]{flexa.__version__}[/cyan]")


@app.command("synthetic")
def synthetic(
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Path to save CSV/Parquet file")
    ] = Path("data/raw/synthetic.csv"),
    n_series: Annotated[
        int, typer.Option("--n-series", "-n", help="Number of distinct series")
    ] = 2,
    length: Annotated[
        int, typer.Option("--length", "-l", help="Number of time steps per series")
    ] = 180,
    seed: Annotated[int, typer.Option("--seed", "-s", help="Random seed")] = 42,
) -> None:
    """Generate synthetic time series benchmark data."""
    output.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_timeseries(n_series=n_series, length=length, seed=seed)

    if output.suffix == ".parquet":
        df.write_parquet(output)
    else:
        df.write_csv(output)

    console.print(
        f"[green]Successfully generated {n_series} series ({length} steps each) to [bold]{output}[/bold][/green]"
    )


@app.command("forecast")
def forecast(
    config_path: Annotated[
        Path | None, typer.Option("--config", "-c", help="Path to YAML config")
    ] = None,
    data_path: Annotated[
        Path | None, typer.Option("--data", "-d", help="Path to CSV/Parquet data file")
    ] = None,
    model: Annotated[
        str, typer.Option("--model", "-m", help="Model type: 'chronos' or 'baseline'")
    ] = "chronos",
    horizon: Annotated[int | None, typer.Option("--horizon", "-h", help="Forecast horizon")] = None,
) -> None:
    """Run forecast and evaluation using Chronos or Baseline model."""
    config = PipelineConfig.from_yaml(config_path) if config_path else PipelineConfig()
    pipeline = ForecastingPipeline(config=config)

    console.print("[bold blue]Loading data...[/bold blue]")
    df = pipeline.load_data(str(data_path) if data_path else None)

    forecaster = pipeline.get_forecaster(model)
    console.print(f"[bold blue]Running {model} forecaster...[/bold blue]")
    result = pipeline.run_forecast(forecaster, df, test_size=horizon)

    # Print results
    table = Table(title=f"Evaluation Metrics: {result.model_name}")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    for k, v in result.metrics.items():
        table.add_row(k.upper(), f"{v:.4f}")

    console.print(table)


@app.command("benchmark")
def benchmark(
    config_path: Annotated[
        Path | None, typer.Option("--config", "-c", help="Path to YAML config")
    ] = None,
    data_path: Annotated[
        Path | None, typer.Option("--data", "-d", help="Path to CSV/Parquet data file")
    ] = None,
    horizon: Annotated[int | None, typer.Option("--horizon", "-h", help="Forecast horizon")] = None,
) -> None:
    """Run side-by-side benchmark comparing Baseline (scikit-learn) vs Chronos."""
    config = PipelineConfig.from_yaml(config_path) if config_path else PipelineConfig()
    pipeline = ForecastingPipeline(config=config)

    console.print("[bold blue]Loading dataset...[/bold blue]")
    df = pipeline.load_data(str(data_path) if data_path else None)

    console.print("[bold blue]Running benchmark (Baseline vs Chronos)...[/bold blue]")
    comp = pipeline.compare_models(df, test_size=horizon)

    table = Table(title="Model Benchmark Comparison")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Baseline (scikit-learn)", style="yellow")
    table.add_column("Chronos Foundation Model", style="green")
    table.add_column("Relative Difference (%)", style="magenta")

    for row in comp.comparison_df.iter_rows(named=True):
        diff = row["diff_pct"]
        diff_str = f"{diff:+.2f}%"
        diff_styled = f"[green]{diff_str}[/green]" if diff < 0 else f"[yellow]{diff_str}[/yellow]"
        table.add_row(
            row["metric"],
            f"{row['baseline']:.4f}",
            f"{row['chronos']:.4f}",
            diff_styled,
        )

    console.print(table)


if __name__ == "__main__":
    app()
