#!/usr/bin/env python3
"""
CLI tool to download equity OHLCV data and cache to Parquet.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from config.settings import settings
from data.cache_io import get_cache
from data_providers.tiingo_client import TiingoClient
from data_tools.validate import (
    tz_to_utc, drop_duplicates, assert_monotonic_ts,
    align_to_bar, forbid_forward_fill_leakage, validate_ohlcv
)

console = Console()


def download_equities(
    provider: str,
    symbols: list[str],
    start: datetime,
    end: datetime,
    interval: str = "1d"
) -> None:
    """
    Download equity data and save to cache.
    
    Args:
        provider: Data provider (tiingo, polygon, alphavantage)
        symbols: List of symbols to download
        start: Start date
        end: End date
        interval: Time interval
    """
    if provider == "tiingo":
        if not settings.TIINGO_API_KEY:
            console.print("[red]❌ Tiingo API key not set (TIINGO_API_KEY)[/red]")
            sys.exit(1)
        client = TiingoClient()
    else:
        console.print(f"[red]❌ Unsupported provider: {provider}[/red]")
        console.print("[yellow]Currently supported: tiingo[/yellow]")
        sys.exit(1)
    
    cache = get_cache()
    
    console.print(f"\n[bold cyan]Downloading Equity Data[/bold cyan]")
    console.print(f"Provider: {provider}")
    console.print(f"Symbols: {', '.join(symbols)}")
    console.print(f"Range: {start.date()} to {end.date()}")
    console.print(f"Interval: {interval}\n")
    
    # Create progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("Downloading...", total=len(symbols))
        
        for symbol in symbols:
            progress.update(task, description=f"Fetching {symbol}...")
            
            try:
                # Fetch data based on interval
                if interval == "1d":
                    df = client.fetch_daily(symbol=symbol, start=start, end=end)
                else:
                    # Intraday intervals
                    interval_map = {
                        "1m": "1min",
                        "5m": "5min",
                        "15m": "15min",
                        "30m": "30min",
                        "1h": "1hour",
                        "4h": "4hour"
                    }
                    
                    tiingo_interval = interval_map.get(interval, interval)
                    df = client.fetch_intraday(
                        symbol=symbol,
                        start=start,
                        end=end,
                        interval=tiingo_interval
                    )
                
                if df.empty:
                    console.print(f"[yellow]⚠️  No data for {symbol}[/yellow]")
                    progress.advance(task)
                    continue
                
                # Validate and clean
                df = tz_to_utc(df, 'ts')
                df = drop_duplicates(df, 'ts', keep='last')
                df = validate_ohlcv(df)
                df = align_to_bar(df, interval, 'ts', tolerance_seconds=300)
                df = forbid_forward_fill_leakage(df, max_gap_bars=5, interval=interval)
                assert_monotonic_ts(df, 'ts')
                
                # Save to cache
                cache.save_parquet(df, asset='equity', symbol=symbol, interval=interval)
                
                console.print(f"[green]✅ {symbol}: {len(df)} bars saved[/green]")
                
            except Exception as e:
                console.print(f"[red]❌ {symbol}: {e}[/red]")
                logger.exception(f"Failed to download {symbol}")
            
            progress.advance(task)
    
    # Print coverage report
    console.print("\n[bold cyan]Coverage Report:[/bold cyan]")
    
    coverage_table = Table(show_header=True, header_style="bold cyan")
    coverage_table.add_column("Symbol", style="dim")
    coverage_table.add_column("Status")
    coverage_table.add_column("Coverage")
    coverage_table.add_column("Bars")
    coverage_table.add_column("Expected")
    
    for symbol in symbols:
        has_cov, cov_ratio, actual, expected = cache.has_coverage(
            asset='equity',
            symbol=symbol,
            interval=interval,
            start=start,
            end=end,
            min_coverage=0.95
        )
        
        status = "✅" if has_cov else ("⚠️" if cov_ratio > 0 else "❌")
        coverage_table.add_row(
            symbol,
            status,
            f"{cov_ratio*100:.1f}%",
            str(actual),
            str(expected)
        )
    
    console.print(coverage_table)
    console.print()


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Download equity OHLCV data")
    parser.add_argument(
        "--provider",
        choices=["tiingo"],
        default="tiingo",
        help="Data provider"
    )
    parser.add_argument(
        "--symbols",
        required=True,
        help="Comma-separated list of symbols (e.g., SPY,AAPL,MSFT)"
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--interval",
        choices=["1d", "1h", "30m", "15m", "5m", "1m"],
        default="1d",
        help="Time interval"
    )
    
    args = parser.parse_args()
    
    # Parse dates
    try:
        start = datetime.strptime(args.start, "%Y-%m-%d")
        end = datetime.strptime(args.end, "%Y-%m-%d")
    except ValueError as e:
        console.print(f"[red]❌ Invalid date format: {e}[/red]")
        return 1
    
    # Parse symbols
    symbols = [s.strip().upper() for s in args.symbols.split(',')]
    
    # Download
    try:
        download_equities(
            provider=args.provider,
            symbols=symbols,
            start=start,
            end=end,
            interval=args.interval
        )
        return 0
    except Exception as e:
        console.print(f"[red]❌ Download failed: {e}[/red]")
        logger.exception("Download failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
