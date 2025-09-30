"""
CoinGecko client for cryptocurrency OHLCV data.
Supports free and pro API keys.
"""

import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests


class CoinGeckoClient:
    """Client for fetching data from CoinGecko API."""
    
    BASE_URL_FREE = "https://api.coingecko.com/api/v3"
    BASE_URL_PRO = "https://pro-api.coingecko.com/api/v3"
    
    # Coin ID mappings for common symbols
    SYMBOL_TO_ID = {
        'btc': 'bitcoin',
        'eth': 'ethereum',
        'bnb': 'binancecoin',
        'sol': 'solana',
        'ada': 'cardano',
        'xrp': 'ripple',
        'dot': 'polkadot',
        'doge': 'dogecoin',
        'avax': 'avalanche-2',
        'matic': 'matic-network'
    }
    
    def __init__(self, api_key: Optional[str] = None, rate_limit_delay: float = 1.5):
        """
        Initialize CoinGecko client.
        
        Args:
            api_key: Optional API key for pro tier
            rate_limit_delay: Delay between requests in seconds
        """
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0.0
        
        # Use pro URL if API key provided
        self.base_url = self.BASE_URL_PRO if api_key else self.BASE_URL_FREE
    
    def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    def _get_headers(self) -> dict:
        """Get request headers."""
        headers = {'Accept': 'application/json'}
        if self.api_key:
            headers['x-cg-pro-api-key'] = self.api_key
        return headers
    
    def _symbol_to_coin_id(self, symbol: str) -> str:
        """Convert symbol to CoinGecko coin ID."""
        symbol_lower = symbol.lower().replace('usd', '').replace('usdt', '')
        return self.SYMBOL_TO_ID.get(symbol_lower, symbol_lower)
    
    def fetch_ohlcv(
        self,
        coin_id: str,
        vs: str = "usd",
        interval: str = "1h",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        days: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from CoinGecko.
        
        Args:
            coin_id: CoinGecko coin ID (e.g., 'bitcoin') or symbol (e.g., 'btc')
            vs: Quote currency (default: 'usd')
            interval: Interval ('15m', '1h', '1d')
            start: Start datetime (optional if days provided)
            end: End datetime (optional, defaults to now)
            days: Number of days to fetch (alternative to start/end)
            
        Returns:
            DataFrame with columns: timestamp (index), open, high, low, close, volume, provider
        """
        # Convert symbol to coin_id if needed
        coin_id = self._symbol_to_coin_id(coin_id)
        
        # Determine days parameter for API
        if days is None and start is not None:
            if end is None:
                end = datetime.utcnow()
            days = (end - start).days + 1
        
        if days is None:
            days = 30  # Default
        
        # Rate limit
        self._rate_limit()
        
        # CoinGecko only provides OHLC for certain intervals
        # For 15m and 1h, we need to use the /market_chart endpoint
        # For daily, we can use /ohlc
        
        if interval in ['15m', '1h', '4h']:
            return self._fetch_market_chart(coin_id, vs, days, interval)
        elif interval in ['1d', 'daily']:
            return self._fetch_ohlc(coin_id, vs, days)
        else:
            raise ValueError(f"Unsupported interval: {interval}")
    
    def _fetch_ohlc(self, coin_id: str, vs: str, days: int) -> pd.DataFrame:
        """Fetch daily OHLC data."""
        url = f"{self.base_url}/coins/{coin_id}/ohlc"
        params = {
            'vs_currency': vs,
            'days': min(days, 90)  # CoinGecko free tier limit
        }
        
        response = requests.get(url, headers=self._get_headers(), params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            return pd.DataFrame()
        
        # Parse response: [[timestamp_ms, open, high, low, close], ...]
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        df['volume'] = 0  # OHLC endpoint doesn't provide volume
        df['provider'] = 'coingecko'
        
        return df
    
    def _fetch_market_chart(self, coin_id: str, vs: str, days: int, interval: str) -> pd.DataFrame:
        """Fetch intraday data from market_chart endpoint."""
        url = f"{self.base_url}/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': vs,
            'days': min(days, 90),
            'interval': 'hourly' if interval != '15m' else 'minutely'
        }
        
        response = requests.get(url, headers=self._get_headers(), params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Parse prices and volumes
        prices_df = pd.DataFrame(data.get('prices', []), columns=['timestamp', 'close'])
        volumes_df = pd.DataFrame(data.get('total_volumes', []), columns=['timestamp', 'volume'])
        
        if prices_df.empty:
            return pd.DataFrame()
        
        # Merge prices and volumes
        df = prices_df.merge(volumes_df, on='timestamp', how='left')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        
        # For market_chart, we don't have OHLC, so we'll use close for all
        # This is a limitation of the free API
        df['open'] = df['close']
        df['high'] = df['close']
        df['low'] = df['close']
        df['provider'] = 'coingecko'
        
        return df[['open', 'high', 'low', 'close', 'volume', 'provider']]
