"""
CoinGecko data provider client.
Fetches crypto OHLCV data with optional Pro API support.
"""

import time
from datetime import datetime, timedelta
from typing import Literal, Optional
import pandas as pd
import requests
from loguru import logger

from config.settings import settings


class CoinGeckoClient:
    """CoinGecko API client for crypto market data."""
    
    # CoinGecko coin ID mapping
    COIN_IDS = {
        'btc': 'bitcoin',
        'eth': 'ethereum',
        'bnb': 'binancecoin',
        'ada': 'cardano',
        'sol': 'solana',
        'xrp': 'ripple',
        'dot': 'polkadot',
        'doge': 'dogecoin',
        'avax': 'avalanche-2',
        'matic': 'matic-network',
        'link': 'chainlink',
        'uni': 'uniswap',
        'ltc': 'litecoin',
        'atom': 'cosmos',
        'etc': 'ethereum-classic',
    }
    
    def __init__(self, api_key: Optional[str] = None, rate_limit_delay: float = 1.2):
        """
        Initialize CoinGecko client.
        
        Args:
            api_key: CoinGecko Pro API key (optional)
            rate_limit_delay: Delay between requests in seconds
        """
        self.api_key = api_key or settings.COINGECKO_API_KEY
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0.0
        
        # Use Pro API if key provided
        if self.api_key:
            self.base_url = "https://pro-api.coingecko.com/api/v3"
        else:
            self.base_url = "https://api.coingecko.com/api/v3"
    
    def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    def _get_coin_id(self, symbol: str) -> str:
        """
        Convert symbol to CoinGecko coin ID.
        
        Args:
            symbol: Symbol like 'btc', 'eth'
            
        Returns:
            CoinGecko coin ID
        """
        symbol_lower = symbol.lower().replace('usd', '').replace('usdt', '')
        
        if symbol_lower in self.COIN_IDS:
            return self.COIN_IDS[symbol_lower]
        
        # Fallback: assume symbol is coin ID
        logger.warning(f"Unknown symbol {symbol}, using as coin ID directly")
        return symbol_lower
    
    def fetch_ohlcv(
        self,
        coin_id: str,
        vs_currency: str = "usd",
        interval: Literal["15m", "1h", "1d"] = "1h",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from CoinGecko.
        
        Args:
            coin_id: CoinGecko coin ID or symbol (btc, eth, etc.)
            vs_currency: Quote currency (usd, eur, etc.)
            interval: Time interval (15m, 1h, 1d)
            start: Start datetime (UTC)
            end: End datetime (UTC)
            
        Returns:
            DataFrame with columns [ts, open, high, low, close, volume, provider]
        """
        # Convert symbol to coin ID if needed
        if coin_id.lower() in self.COIN_IDS:
            coin_id = self._get_coin_id(coin_id)
        
        # Default date range: last 90 days
        if end is None:
            end = datetime.utcnow()
        if start is None:
            start = end - timedelta(days=90)
        
        # Map interval to days parameter
        # CoinGecko has different endpoints for different granularities
        interval_days_map = {
            '15m': 1,    # Last 1 day for minutely data
            '1h': 90,    # Last 90 days for hourly data
            '1d': 365    # Last 365 days for daily data
        }
        
        if interval not in interval_days_map:
            raise ValueError(f"Unsupported interval: {interval}. Use 15m, 1h, or 1d")
        
        days = interval_days_map[interval]
        
        # Determine endpoint based on interval
        if interval in ['15m', '1h']:
            # Use OHLC endpoint for intraday data
            endpoint = f"{self.base_url}/coins/{coin_id}/ohlc"
            params = {
                'vs_currency': vs_currency,
                'days': days
            }
        else:
            # Use market_chart endpoint for daily data
            endpoint = f"{self.base_url}/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': vs_currency,
                'days': days,
                'interval': 'daily'
            }
        
        # Add API key header if available
        headers = {}
        if self.api_key:
            headers['x-cg-pro-api-key'] = self.api_key
        
        # Apply rate limiting
        self._rate_limit()
        
        # Make request
        try:
            response = requests.get(endpoint, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"CoinGecko API request failed: {e}")
            raise
        
        # Parse response based on endpoint
        if interval in ['15m', '1h']:
            # OHLC endpoint returns [[timestamp, open, high, low, close], ...]
            if not data:
                logger.warning(f"No OHLC data returned for {coin_id}")
                return pd.DataFrame()
            
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
            df['volume'] = 0.0  # OHLC endpoint doesn't provide volume
        else:
            # market_chart endpoint returns {prices: [[ts, price], ...], ...}
            if 'prices' not in data or not data['prices']:
                logger.warning(f"No price data returned for {coin_id}")
                return pd.DataFrame()
            
            # For daily data, approximate OHLC from prices
            prices = pd.DataFrame(data['prices'], columns=['timestamp', 'close'])
            prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms', utc=True)
            
            # Group by day and create OHLC
            prices['date'] = prices['timestamp'].dt.date
            
            ohlc = prices.groupby('date')['close'].agg([
                ('open', 'first'),
                ('high', 'max'),
                ('low', 'min'),
                ('close', 'last')
            ]).reset_index()
            
            ohlc['timestamp'] = pd.to_datetime(ohlc['date']).astype(int) // 10**6
            ohlc['volume'] = 0.0
            
            df = ohlc[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        # Convert timestamp to datetime
        df['ts'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df.drop(columns=['timestamp'])
        
        # Add provider column
        df['provider'] = 'coingecko'
        
        # Filter to date range
        df = df[(df['ts'] >= start) & (df['ts'] <= end)]
        
        # Sort by timestamp
        df = df.sort_values('ts').reset_index(drop=True)
        
        # Reorder columns
        df = df[['ts', 'open', 'high', 'low', 'close', 'volume', 'provider']]
        
        logger.info(f"Fetched {len(df)} {interval} bars for {coin_id}/{vs_currency}")
        
        return df


# Module-level convenience functions
def fetch_ohlcv(
    coin_id: str,
    vs_currency: str = "usd",
    interval: Literal["15m", "1h", "1d"] = "1h",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None
) -> pd.DataFrame:
    """Convenience function to fetch OHLCV data."""
    client = CoinGeckoClient()
    return client.fetch_ohlcv(coin_id, vs_currency, interval, start, end)
