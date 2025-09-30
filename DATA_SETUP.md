# Data Setup Guide

This guide covers data provider setup, caching, and download procedures.

## Overview

The quant-bot uses a **Parquet-based caching system** to efficiently store and retrieve market data. Data is organized by:

```
data/cache/{asset}/{symbol}/{interval}/{YYYY}/{YYYY-MM}.parquet
```

Example:
```
data/cache/equity/SPY/1d/2024/2024-01.parquet
data/cache/crypto/btc/1h/2024/2024-06.parquet
```

## Data Providers

### Equity Providers

#### Tiingo (Recommended)

**Pros:**
- Excellent free tier
- Covers stocks, ETFs, mutual funds
- Adjusted prices included
- Both daily and intraday data

**Setup:**
1. Sign up at https://www.tiingo.com/
2. Get API key from Account → API
3. Add to `.env`:
   ```
   TIINGO_API_KEY=your_key_here
   ```

**Rate Limits:**
- Free: 500 requests/hour, 20,000/month
- Paid: Higher limits

**Usage:**
```bash
python -m data_tools.download_equities \
    --provider tiingo \
    --symbols SPY,AAPL,MSFT \
    --start 2023-01-01 \
    --end 2024-09-01 \
    --interval 1d
```

#### Polygon.io

**Pros:**
- High-quality data
- Real-time and historical
- Options data available

**Cons:**
- Limited free tier (5 API calls/minute)

**Setup:**
1. Sign up at https://polygon.io/
2. Get API key
3. Add to `.env`: `POLYGON_API_KEY=your_key`

**Note:** Not yet implemented in download CLI (coming soon)

#### Alpha Vantage

**Pros:**
- Simple API
- Free tier available

**Cons:**
- Strict rate limits (5 calls/minute)

**Note:** Not yet implemented in download CLI (coming soon)

### Crypto Providers

#### CoinGecko (Recommended)

**Pros:**
- Excellent free tier
- Comprehensive coin coverage
- No authentication required for free tier

**Setup:**
1. Optional: Sign up for Pro API at https://www.coingecko.com/
2. Add to `.env` (or leave empty for public API):
   ```
   COINGECKO_API_KEY=your_key_here
   ```

**Rate Limits:**
- Public: ~10-50 calls/minute
- Pro: Higher limits with API key

**Usage:**
```bash
python -m data_tools.download_crypto \
    --provider coingecko \
    --symbols btc,eth,sol \
    --start 2024-01-01 \
    --end 2024-09-01 \
    --interval 1h
```

**Supported Intervals:**
- `15m`: Last 1 day only (CoinGecko limitation)
- `1h`: Last 90 days
- `1d`: Last 365 days

**Coin Symbol Mapping:**
```
btc  → bitcoin
eth  → ethereum
bnb  → binancecoin
sol  → solana
ada  → cardano
xrp  → ripple
dot  → polkadot
(see data_providers/coingecko_client.py for full list)
```

#### CryptoCompare

**Pros:**
- Good free tier
- Historical data

**Note:** Not yet implemented in download CLI (coming soon)

## Data Download Procedures

### Quick Start (Makefile)

```bash
# Download sample crypto data (BTC, ETH, 1h bars, Jan-Sep 2024)
make data.crypto

# Download sample equity data (SPY, AAPL, daily, 2023-2024)
make data.equities
```

### Manual Downloads

#### Equities

```bash
# Daily bars (recommended for backtesting)
python -m data_tools.download_equities \
    --provider tiingo \
    --symbols SPY,QQQ,IWM,AAPL,MSFT,GOOGL \
    --start 2020-01-01 \
    --end 2024-09-01 \
    --interval 1d

# Hourly bars (for intraday strategies)
python -m data_tools.download_equities \
    --provider tiingo \
    --symbols SPY,QQQ \
    --start 2024-08-01 \
    --end 2024-09-01 \
    --interval 1h
```

**Supported Intervals:** `1d`, `1h`, `30m`, `15m`, `5m`, `1m`

#### Crypto

```bash
# Hourly bars (good balance of granularity and history)
python -m data_tools.download_crypto \
    --provider coingecko \
    --symbols btc,eth,bnb,sol,ada \
    --start 2024-01-01 \
    --end 2024-09-01 \
    --interval 1h

# Daily bars (for longer backtests)
python -m data_tools.download_crypto \
    --provider coingecko \
    --symbols btc,eth \
    --start 2020-01-01 \
    --end 2024-09-01 \
    --interval 1d
```

**Supported Intervals:** `15m`, `1h`, `1d`

### Data Validation

All downloaded data is automatically validated:

✅ **Checks performed:**
- Timezone normalization (all timestamps converted to UTC)
- Duplicate removal (keeps most recent)
- OHLCV sanity checks (high >= low, etc.)
- Monotonic timestamp verification
- Gap detection (warns about forward-fill leakage)

❌ **Will fail if:**
- High < Low
- Negative prices
- NaN values in OHLCV
- Excessive forward-fill detected

## Cache Management

### Check Coverage

```python
from data.cache_io import get_cache

cache = get_cache()
has_cov, ratio, actual, expected = cache.has_coverage(
    asset="equity",
    symbol="SPY",
    interval="1d",
    start=datetime(2024, 1, 1),
    end=datetime(2024, 9, 1),
    min_coverage=0.95
)

print(f"Coverage: {ratio*100:.1f}% ({actual}/{expected} bars)")
```

### Load Cached Data

```python
from data.cache_io import get_cache
from datetime import datetime

cache = get_cache()
df = cache.load_range(
    asset="equity",
    symbol="SPY",
    interval="1d",
    start=datetime(2024, 1, 1),
    end=datetime(2024, 9, 1)
)

print(df.head())
```

### Cache Location

```
data/cache/           # All cached data
├── equity/
│   ├── SPY/
│   │   ├── 1d/
│   │   │   ├── 2023/
│   │   │   │   ├── 2023-01.parquet
│   │   │   │   ├── 2023-02.parquet
│   │   │   │   └── ...
│   │   │   └── 2024/
│   │   │       └── ...
│   │   └── 1h/
│   └── AAPL/
└── crypto/
    ├── btc/
    └── eth/
```

### Cache Size

Approximate sizes:
- Daily equity bars: ~10 KB per symbol per year
- Hourly crypto bars: ~500 KB per coin per year
- 15-minute crypto bars: ~2 MB per coin per year

## Data Quality Best Practices

### 1. Prevent Data Leakage

✅ **Do:**
- Use train/test split chronologically
- Never backfill NaN values with future data
- Use proper forward-looking labels

❌ **Don't:**
- Train on future data
- Forward-fill missing bars excessively
- Use non-causal indicators

### 2. Handle Gaps Properly

```python
from data_tools.validate import forbid_forward_fill_leakage

# This will raise error if gaps suggest improper forward-fill
df = forbid_forward_fill_leakage(df, max_gap_bars=3, interval="1h")
```

### 3. Align to Bar Intervals

```python
from data_tools.validate import align_to_bar

# Align timestamps to expected intervals
df = align_to_bar(df, interval="1h", column="ts", tolerance_seconds=60)
```

## Recommended Data Sets

### For Backtesting

**Equities (2020-2024):**
```bash
python -m data_tools.download_equities \
    --provider tiingo \
    --symbols SPY,QQQ,IWM,DIA,AAPL,MSFT,GOOGL,AMZN,NVDA,TSLA \
    --start 2020-01-01 \
    --end 2024-09-01 \
    --interval 1d
```

**Crypto (2020-2024):**
```bash
python -m data_tools.download_crypto \
    --provider coingecko \
    --symbols btc,eth,bnb,ada,sol \
    --start 2020-01-01 \
    --end 2024-09-01 \
    --interval 1d
```

### For Paper Trading

**Recent Data (Last 90 days):**
```bash
# Hourly equity bars
python -m data_tools.download_equities \
    --provider tiingo \
    --symbols SPY,QQQ \
    --start 2024-06-01 \
    --end 2024-09-01 \
    --interval 1h

# Hourly crypto bars
python -m data_tools.download_crypto \
    --provider coingecko \
    --symbols btc,eth \
    --start 2024-06-01 \
    --end 2024-09-01 \
    --interval 1h
```

## Troubleshooting

### Download Failures

**Symptom:** API errors, timeouts, rate limit exceeded

**Solutions:**
1. Check API key is valid
2. Wait for rate limit reset
3. Reduce batch size (download fewer symbols)
4. Use longer interval between requests

### Missing Data

**Symptom:** Coverage report shows < 95%

**Common Causes:**
- Weekends/holidays (equities)
- Exchange downtime
- Delisted symbols

**Fix:** Generally acceptable if > 90% coverage

### Validation Errors

**Symptom:** "High < Low" or other validation errors

**Cause:** Bad data from provider

**Fix:**
1. Try different date range
2. Report to provider
3. Use alternative provider

### Large Cache Size

**Symptom:** `data/cache/` directory too large

**Solutions:**
```bash
# Remove old data
rm -rf data/cache/equity/*/1d/2020/
rm -rf data/cache/equity/*/1d/2021/

# Keep only recent data
# (Manually delete old year directories)
```

## Next Steps

After data setup:

1. Run a backtest: `make backtest.quick`
2. Start Discord bot for monitoring: `python -m ui.discord_bot.main`
3. See [OBSERVABILITY.md](OBSERVABILITY.md) for monitoring setup

## Provider API Key Registration

### Tiingo
→ https://www.tiingo.com/account/api

### CoinGecko Pro
→ https://www.coingecko.com/en/api/pricing

### Polygon.io
→ https://polygon.io/dashboard/signup

### Alpha Vantage
→ https://www.alphavantage.co/support/#api-key

### CryptoCompare
→ https://www.cryptocompare.com/cryptopian/api-keys

### FRED (Federal Reserve)
→ https://fred.stlouisfed.org/docs/api/api_key.html
