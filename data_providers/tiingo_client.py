"""
Tiingo client for equity daily OHLCV data.
"""

import time
from datetime import datetime
from typing import Optional

import pandas as pd
import requests


class TiingoClient:
    """Client for fetching equity data from Tiingo API."""
    
    BASE_URL = "https://api.tiingo.com"
    
    def __init__(self, api_key: str, rate_limit_delay: float = 0.5):
        """
        Initialize Tiingo client.
        
        Args:
            api_key: Tiingo API key (required)
            rate_limit_delay: Delay between requests in seconds
        """
        if not api_key:
            raise ValueError("Tiingo API key is required")
        
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0.0
    
    def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    def fetch_daily(
        self,
        symbol: str,
        start: datetime,
        end: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV data from Tiingo.
        
        Args:
            symbol: Stock symbol (e.g., 'SPY', 'AAPL')
            start: Start date
            end: End date (optional, defaults to today)
            
        Returns:
            DataFrame with columns: timestamp (index), open, high, low, close, volume, provider
        """
        if end is None:
            end = datetime.utcnow()
        
        # Rate limit
        self._rate_limit()
        
        # Format dates
        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')
        
        # Build URL
        url = f"{self.BASE_URL}/tiingo/daily/{symbol.upper()}/prices"
        
        # Parameters
        params = {
            'startDate': start_str,
            'endDate': end_str,
            'token': self.api_key,
            'format': 'json'
        }
        
        # Make request
        response = requests.get(url, params=params, timeout=30)
        
        # Handle errors
        if response.status_code == 404:
            raise ValueError(f"Symbol not found: {symbol}")
        
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            return pd.DataFrame()
        
        # Parse response
        df = pd.DataFrame(data)
        
        # Rename columns to match our schema
        column_map = {
            'date': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }
        
        df = df.rename(columns=column_map)
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df.set_index('timestamp', inplace=True)
        
        # Select and order columns
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
        
        # Add provider
        df['provider'] = 'tiingo'
        
        return df
    
    def fetch_intraday(
        self,
        symbol: str,
        start: datetime,
        end: Optional[datetime] = None,
        interval: str = '1hour'
    ) -> pd.DataFrame:
        """
        Fetch intraday OHLCV data from Tiingo (requires premium subscription).
        
        Args:
            symbol: Stock symbol
            start: Start datetime
            end: End datetime (optional)
            interval: Interval ('1min', '5min', '15min', '30min', '1hour', '4hour')
            
        Returns:
            DataFrame with OHLCV data
        """
        if end is None:
            end = datetime.utcnow()
        
        # Rate limit
        self._rate_limit()
        
        # Format dates
        start_str = start.strftime('%Y-%m-%d %H:%M:%S')
        end_str = end.strftime('%Y-%m-%d %H:%M:%S')
        
        # Map interval format
        interval_map = {
            '1min': '1min',
            '5min': '5min',
            '15min': '15min',
            '30min': '30min',
            '1h': '1hour',
            '1hour': '1hour',
            '4h': '4hour',
            '4hour': '4hour'
        }
        
        tiingo_interval = interval_map.get(interval, interval)
        
        # Build URL
        url = f"{self.BASE_URL}/iex/{symbol.upper()}/prices"
        
        # Parameters
        params = {
            'startDate': start_str,
            'endDate': end_str,
            'resampleFreq': tiingo_interval,
            'token': self.api_key,
            'format': 'json'
        }
        
        # Make request
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            return pd.DataFrame()
        
        # Parse response
        df = pd.DataFrame(data)
        
        # Rename columns
        column_map = {
            'date': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }
        
        df = df.rename(columns=column_map)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df.set_index('timestamp', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
        df['provider'] = 'tiingo'
        
        return df
