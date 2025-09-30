#!/usr/bin/env python3
"""
Simple backtest runner for paper trading.
Loads cached data, computes KPIs, logs to DB, generates reports.
"""

import argparse
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.progress import track

from data.cache_io import get_cache
from data_tools.validate import validate_ohlcv
from reporting.narration import summarize_backtest
from storage.db import EquityPoint, Run, get_session
from telemetry.wandb_utils import get_wandb
from ui.discord_bot.cogs.reporting import create_equity_chart

console = Console()


def calculate_kpis(equity_curve: pd.Series) -> dict[str, Any]:
    """
    Calculate performance KPIs from equity curve.
    
    Args:
        equity_curve: Pandas series of equity values
        
    Returns:
        Dictionary of KPIs
    """
    returns = equity_curve.pct_change().dropna()

    # Total return
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1 if len(equity_curve) > 0 else 0

    # Sharpe ratio (annualized, assuming daily data)
    mean_return = returns.mean()
    std_return = returns.std()
    sharpe = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0

    # Sortino ratio (only downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std()
    sortino = (mean_return / downside_std * np.sqrt(252)) if downside_std > 0 and len(downside_returns) > 0 else 0

    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    # Win rate (simple threshold)
    win_rate = (returns > 0).mean()

    return {
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_dd": float(max_dd),
        "win_rate": float(win_rate),
        "total_trades": 0,  # Placeholder
        "num_bars": len(equity_curve)
    }


def run_backtest(
    start: datetime,
    end: datetime,
    universe: list[str],
    interval: str = "1d",
    initial_equity: float = 100000.0
) -> str:
    """
    Run simple backtest.
    
    Args:
        start: Start date
        end: End date
        universe: List of symbols
        interval: Time interval
        initial_equity: Starting equity
        
    Returns:
        Run ID
    """
    console.print("\n[bold cyan]Running Backtest[/bold cyan]")
    console.print(f"Period: {start.date()} to {end.date()}")
    console.print(f"Universe: {', '.join(universe)}")
    console.print(f"Interval: {interval}\n")

    # Create run
    run_id = str(uuid.uuid4())
    run = Run(
        id=run_id,
        started_at=datetime.utcnow(),
        mode="backtest",
        universe=",".join(universe),
        params={"interval": interval, "initial_equity": initial_equity}
    )

    # Load data from cache
    cache = get_cache()
    data_frames = {}

    for symbol in track(universe, description="Loading data..."):
        # Try equity first, then crypto
        df = cache.load_range("equity", symbol, interval, start, end)
        if df is None or df.empty:
            df = cache.load_range("crypto", symbol.lower(), interval, start, end)

        if df is not None and not df.empty:
            try:
                validate_ohlcv(df)
                data_frames[symbol] = df
            except Exception as e:
                logger.warning(f"Validation failed for {symbol}: {e}")

    if not data_frames:
        console.print("[red]❌ No data available for backtest[/red]")
        return run_id

    # Simple buy-and-hold simulation
    console.print("\n[cyan]Simulating buy-and-hold strategy...[/cyan]")

    # Combine all data into single equity curve (equal weight)
    all_dates = sorted(set(date for df in data_frames.values() for date in df["ts"]))

    equity_curve = []
    timestamps = []

    for date in all_dates:
        # Calculate portfolio value (equal weighted)
        total_value = 0
        count = 0

        for symbol, df in data_frames.items():
            # Find closest price
            df_date = df[df["ts"] <= date]
            if not df_date.empty:
                price = df_date.iloc[-1]["close"]
                total_value += price
                count += 1

        if count > 0:
            # Normalize to initial equity
            avg_price = total_value / count
            if not equity_curve:
                equity_value = initial_equity
                initial_avg_price = avg_price
            else:
                # Scale based on price changes
                equity_value = initial_equity * (avg_price / initial_avg_price)

            equity_curve.append(equity_value)
            timestamps.append(date)

    if not equity_curve:
        console.print("[red]❌ Failed to generate equity curve[/red]")
        return run_id

    # Calculate KPIs
    equity_series = pd.Series(equity_curve, index=timestamps)
    kpis = calculate_kpis(equity_series)

    console.print("\n[bold green]Backtest Complete[/bold green]")
    console.print(f"Total Return: {kpis['total_return']:.2%}")
    console.print(f"Sharpe Ratio: {kpis['sharpe']:.2f}")
    console.print(f"Sortino Ratio: {kpis['sortino']:.2f}")
    console.print(f"Max Drawdown: {kpis['max_dd']:.2%}")
    console.print(f"Win Rate: {kpis['win_rate']:.1%}\n")

    # Save to database
    with get_session() as session:
        # Update run with KPIs
        run.ended_at = datetime.utcnow()
        run.kpis = kpis
        session.add(run)

        # Save equity curve
        for ts, equity in zip(timestamps, equity_curve, strict=False):
            point = EquityPoint(run_id=run_id, ts=ts, equity=equity)
            session.add(point)

        session.commit()

    logger.info(f"Backtest results saved to database (run_id: {run_id})")

    # Generate chart
    chart_path = Path(f"/tmp/equity_{run_id}.png")
    chart_buf = create_equity_chart(timestamps, equity_curve, "Backtest Equity Curve")

    if chart_buf:
        with open(chart_path, "wb") as f:
            f.write(chart_buf.read())
        console.print(f"[green]Chart saved: {chart_path}[/green]")

    # W&B logging
    wandb = get_wandb()
    if wandb.enabled:
        wandb.init_run(project="quantbot", tags=["backtest"], config={
            "universe": universe,
            "interval": interval,
            "start": start.isoformat(),
            "end": end.isoformat()
        })
        wandb.log_summary(kpis)
        if chart_path.exists():
            wandb.log_artifact(str(chart_path), "equity_chart", "chart")
        wandb.finish()

    # Generate narration
    narration = summarize_backtest(
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
        universe=", ".join(universe),
        kpis=kpis
    )

    console.print("\n[bold]Summary:[/bold]")
    console.print(narration)
    console.print()

    return run_id


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--universe", required=True, help="Comma-separated symbols (e.g., SPY,AAPL,btc)")
    parser.add_argument("--interval", default="1d", help="Time interval (default: 1d)")
    parser.add_argument("--initial-equity", type=float, default=100000.0, help="Initial equity")

    args = parser.parse_args()

    # Parse dates
    try:
        start = datetime.strptime(args.start, "%Y-%m-%d")
        end = datetime.strptime(args.end, "%Y-%m-%d")
    except ValueError as e:
        console.print(f"[red]❌ Invalid date format: {e}[/red]")
        return 1

    # Parse universe
    universe = [s.strip() for s in args.universe.split(",")]

    # Run backtest
    try:
        run_id = run_backtest(start, end, universe, args.interval, args.initial_equity)
        console.print(f"[green]✅ Backtest complete. Run ID: {run_id}[/green]")
        return 0
    except Exception as e:
        console.print(f"[red]❌ Backtest failed: {e}[/red]")
        logger.exception("Backtest failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
