# Data Sources Configuration

This document describes the data providers supported by quant-bot and how to configure them.

## Overview

Quant-bot supports multiple data providers for different asset classes:

- **Equities**: Polygon, Alpha Vantage, Tiingo
- **Crypto**: CoinGecko, Binance (via adapters)
- **Economic Data**: FRED (Federal Reserve Economic Data)
- **Forex**: Alpha Vantage

All providers include:
- Automatic retry with exponential backoff
- Rate limiting to respect free tier limits
- Parquet caching to reduce API calls
- Unified OHLCV interface

## Configuration

Add API keys to your `.env` file:

```bash
# Equities data
POLYGON_API_KEY=your_key_here
ALPHA_VANTAGE_API_KEY=your_key_here
TIINGO_API_KEY=your_key_here

# Crypto data
COINGECKO_API_KEY=your_key_here
CRYPTOCOMPARE_API_KEY=your_key_here

# Economic data
FRED_API_KEY=your_key_here
```

## Provider Details

### Polygon.io

**Best for**: Real-time and historical stock data, high quality

**Free Tier**:
- 5 API calls per minute
- Delayed data (15 minutes)
- Stocks, options, forex, crypto

**Rate Limiting**: 12 seconds between requests (conservative)

**Get API Key**: https://polygon.io/dashboard/signup

**Usage**:
```python
from data.providers import PolygonAdapter

polygon = PolygonAdapter()
df = polygon.get_bars("SPY", start_date, end_date, timeframe="1D")
```

**Supported Timeframes**: 1T, 5T, 15T, 30T, 1H, 4H, 1D, 1W, 1M

### Alpha Vantage

**Best for**: Broad coverage (stocks, forex, crypto), technical indicators

**Free Tier**:
- 5 API calls per minute
- 500 API calls per day
- Full history included

**Rate Limiting**: 12 seconds between requests

**Get API Key**: https://www.alphavantage.co/support/#api-key

**Usage**:
```python
from data.providers import AlphaVantageAdapter

alpha_vantage = AlphaVantageAdapter()
df = alpha_vantage.get_bars("AAPL", start_date, end_date, timeframe="1H")
```

**Supported Timeframes**: 1T, 5T, 15T, 30T, 60T, 1D, 1W, 1M

### Tiingo

**Best for**: Reliable historical equity data, IEX real-time

**Free Tier**:
- 50 unique symbols per month
- 500 API calls per hour
- End-of-day data

**Rate Limiting**: Built into adapter (1 request per second)

**Get API Key**: https://api.tiingo.com/account/api/token

**Usage**:
```python
from data_providers.tiingo_client import TiingoClient

tiingo = TiingoClient()
df = tiingo.get_historical("SPY", start_date, end_date)
```

### CoinGecko

**Best for**: Crypto prices and market data

**Free Tier**:
- 10-50 calls per minute (depending on endpoint)
- No API key required for basic usage
- Pro features require paid plan

**Get API Key**: https://www.coingecko.com/en/api/pricing

**Usage**:
```python
from data_providers.coingecko_client import CoinGeckoClient

coingecko = CoinGeckoClient()
df = coingecko.get_historical("bitcoin", start_date, end_date)
```

### FRED (Federal Reserve Economic Data)

**Best for**: Macroeconomic indicators, interest rates, inflation

**Free Tier**:
- No rate limits
- 500,000+ time series
- Daily updates

**Rate Limiting**: 0.5 seconds between requests (conservative)

**Get API Key**: https://fred.stlouisfed.org/docs/api/api_key.html

**Usage**:
```python
from data.providers import FREDAdapter

fred = FREDAdapter()
df = fred.get_bars("DFF", start_date, end_date)  # Federal Funds Rate
```

**Common Series IDs**:
```python
# Interest Rates
DFF          # Federal Funds Effective Rate
DGS10        # 10-Year Treasury Constant Maturity Rate
DGS2         # 2-Year Treasury Constant Maturity Rate
T10Y2Y       # 10-Year minus 2-Year Treasury Spread

# Volatility
VIXCLS       # CBOE Volatility Index (VIX)

# Economic Indicators
UNRATE       # Unemployment Rate
CPIAUCSL     # Consumer Price Index
GDP          # Gross Domestic Product

# Credit Spreads
BAMLH0A0HYM2 # High Yield Option-Adjusted Spread
BAMLC0A0CM   # Corporate Option-Adjusted Spread
```

## Caching

All data is cached using Parquet files in `data/cache/` to reduce API calls.

**Cache Settings**:
- Default TTL: 7 days
- Force refresh: Use `force_refresh=True` parameter
- Cache location: `data/cache/*.parquet` (gitignored)

**Usage**:
```python
from data.cache import CacheManager

cache = CacheManager(cache_dir="data/cache", max_age_days=7)

# Automatic caching
df = cache.get(
    key="SPY_1D_2024-01-01_2024-12-31",
    fetch_fn=lambda: polygon.get_bars("SPY", start, end, "1D"),
    force_refresh=False
)

# Check cache stats
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")
```

## Best Practices

### 1. Provider Selection

**For Backtesting (Historical Data)**:
- Primary: Polygon or Alpha Vantage (stocks)
- Secondary: Tiingo (equities), CoinGecko (crypto)
- Macro: FRED (economic indicators)

**For Paper Trading**:
- Tiingo (IEX real-time for free)
- Alpha Vantage (15min delayed)

**For Live Trading**:
- Broker's data feed (Alpaca, Interactive Brokers)
- Polygon (real-time with paid plan)

### 2. Rate Limiting

Always respect provider rate limits to avoid bans:

```python
# Configure rate limits in provider initialization
polygon = PolygonAdapter()  # Auto rate-limited to 12s/request

# Use caching to minimize API calls
cache = CacheManager()
df = cache.get("SPY_1D", lambda: polygon.get_bars(...))
```

### 3. Error Handling

Providers include automatic retry logic:

```python
from data.providers import PolygonAdapter

try:
    polygon = PolygonAdapter()
    df = polygon.get_bars("SPY", start, end)
    
    if df.empty:
        print("No data returned (may be invalid symbol or date range)")
except Exception as e:
    print(f"Failed to fetch data: {e}")
```

### 4. Data Quality Checks

Always validate data before use:

```python
from data_tools.validate import validate_bars

# Load data
df = polygon.get_bars("SPY", start, end)

# Validate
is_valid, issues = validate_bars(df)
if not is_valid:
    print(f"Data quality issues: {issues}")
```

See `data_tools/validate.py` for validation functions.

## Cost Optimization

### Free Tier Strategy

1. **Use caching aggressively**: 7-day TTL for daily data
2. **Download once, use many times**: Cache full history
3. **Combine providers**: Use different providers for different assets
4. **Monitor API usage**: Track calls per provider

### Paid Tier Considerations

When to upgrade:

- **Polygon Pro** ($99/mo): Real-time data, unlimited calls, options/forex
- **Alpha Vantage Premium** ($49/mo): 75-1200 calls/min, extended history
- **Tiingo Pro** ($30/mo): Real-time streaming, unlimited symbols
- **CoinGecko Analyst** ($129/mo): Enterprise crypto data

## Troubleshooting

### "API key not configured" Warning

```bash
# Check your .env file
cat .env | grep POLYGON_API_KEY

# Make sure .env is loaded
python -c "from config.settings import settings; print(settings.POLYGON_API_KEY)"
```

### "Rate limit exceeded" Error

Increase `rate_limit_delay` in provider initialization:

```python
polygon = PolygonAdapter()
polygon.rate_limit_delay = 15.0  # Increase to 15 seconds
```

### Empty DataFrame Returned

Common causes:
1. Invalid symbol
2. Invalid date range (weekends, holidays)
3. No data available for timeframe
4. API key invalid or expired

Check logs with:
```bash
export LOG_LEVEL=DEBUG
python your_script.py
```

### Cache Not Working

Clear cache and retry:

```python
from data.cache import CacheManager

cache = CacheManager()
cache.clear_all()  # Clear all cached data
```

## Provider Comparison

| Provider      | Free Calls | Delay   | Best For           | Quality |
|---------------|------------|---------|--------------------| ------- |
| Polygon       | 5/min      | 15 min  | Stocks, Options    | ★★★★★   |
| Alpha Vantage | 5/min      | None    | All Asset Classes  | ★★★★☆   |
| Tiingo        | 500/hour   | EOD     | Historical Equities| ★★★★☆   |
| CoinGecko     | ~10/min    | ~5 min  | Crypto             | ★★★★☆   |
| FRED          | Unlimited  | N/A     | Economic Data      | ★★★★★   |

## See Also

- [SETUP_ENV.md](SETUP_ENV.md) - Environment configuration
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues
- [data_tools/validate.py](data_tools/validate.py) - Data validation utilities
