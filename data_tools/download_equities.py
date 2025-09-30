"""
CLI tool to download equity data.
Usage: python -m data_tools.download_equities --provider tiingo --symbols SPY,AAPL --start 2022-01-01 --end 2024-12-31 --interval 1d
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from config.settings import get_settings
from data_providers.tiingo_client import TiingoClient
from data.cache_io import save_parquet, has_coverage
from data_tools.validate import clean_and_validate


def main():
    """Main entry point for equity data download CLI."""
    parser = argparse.ArgumentParser(description='Download equity data')
    parser.add_argument('--provider', default='tiingo', choices=['tiingo'],
                        help='Data provider')
    parser.add_argument('--symbols', required=True, help='Comma-separated symbols (e.g., SPY,AAPL)')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--interval', default='1d', choices=['1d', '1h'],
                        help='Data interval')
    
    args = parser.parse_args()
    
    # Load settings
    settings = get_settings()
    
    # Parse dates
    start_date = datetime.strptime(args.start, '%Y-%m-%d')
    end_date = datetime.strptime(args.end, '%Y-%m-%d')
    
    # Parse symbols
    symbols = [s.strip().upper() for s in args.symbols.split(',')]
    
    print("=" * 60)
    print(f"📥 Equity Data Download")
    print("=" * 60)
    print(f"Provider: {args.provider}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Interval: {args.interval}")
    print("=" * 60)
    print()
    
    # Initialize client based on provider
    if args.provider == 'tiingo':
        api_key = settings.TIINGO_API_KEY
        if not api_key:
            print("❌ Tiingo API key not found in environment.")
            print("💡 Set TIINGO_API_KEY in .env file")
            return 1
        client = TiingoClient(api_key=api_key)
    else:
        print(f"❌ Unsupported provider: {args.provider}")
        return 1
    
    # Download data for each symbol
    coverage_table = []
    
    for symbol in symbols:
        print(f"📊 Processing {symbol}...")
        
        try:
            # Check existing coverage
            has_full, first_date, last_date = has_coverage(
                'equity', symbol, args.interval, start_date, end_date
            )
            
            if has_full:
                print(f"  ✅ Full coverage already exists ({first_date} to {last_date})")
                coverage_table.append({
                    'symbol': symbol,
                    'status': '✅ Cached',
                    'first_date': first_date.strftime('%Y-%m-%d'),
                    'last_date': last_date.strftime('%Y-%m-%d'),
                    'rows': 'N/A'
                })
                continue
            
            # Fetch data
            print(f"  📡 Fetching from {args.provider}...")
            
            if args.interval == '1d':
                df = client.fetch_daily(symbol, start=start_date, end=end_date)
            elif args.interval == '1h':
                df = client.fetch_intraday(symbol, start=start_date, end=end_date, interval='1hour')
            else:
                raise ValueError(f"Unsupported interval: {args.interval}")
            
            if df.empty:
                print(f"  ⚠️  No data returned")
                coverage_table.append({
                    'symbol': symbol,
                    'status': '⚠️  Empty',
                    'first_date': 'N/A',
                    'last_date': 'N/A',
                    'rows': 0
                })
                continue
            
            # Validate and clean
            print(f"  🔍 Validating data...")
            df = clean_and_validate(df, args.interval, validate_ohlc=True)
            
            # Save to cache
            print(f"  💾 Saving to cache...")
            save_parquet(df, 'equity', symbol, args.interval)
            
            # Record coverage
            print(f"  ✅ Downloaded {len(df)} rows ({df.index.min()} to {df.index.max()})")
            coverage_table.append({
                'symbol': symbol,
                'status': '✅ Downloaded',
                'first_date': df.index.min().strftime('%Y-%m-%d'),
                'last_date': df.index.max().strftime('%Y-%m-%d'),
                'rows': len(df)
            })
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            coverage_table.append({
                'symbol': symbol,
                'status': f'❌ Error',
                'first_date': 'N/A',
                'last_date': 'N/A',
                'rows': 'N/A'
            })
    
    # Print coverage table
    print()
    print("=" * 60)
    print("📋 Coverage Summary")
    print("=" * 60)
    
    if coverage_table:
        df_coverage = pd.DataFrame(coverage_table)
        print(df_coverage.to_string(index=False))
    
    print()
    print("✅ Download complete!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
