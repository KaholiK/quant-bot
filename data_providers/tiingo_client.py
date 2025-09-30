"""
Tiingo data provider client.
Fetches equity OHLCV data.
"""

import time
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import requests
from loguru import logger

from config.settings import settings


class TiingoClient:
    """Tiingo API client for equity market data."""
    
    def __init__(self, api_key: Optional[str] = None, rate_limit_delay: float = 0.5):
        """
        Initialize Tiingo client.
        
        Args:
            api_key: Tiingo API key (required)
            rate_limit_delay: Delay between requests in seconds
        """
        self.api_key = api_key or settings.TIINGO_API_KEY
        
        if not self.api_key:
            raise ValueError("Tiingo API key is required. Set TIINGO_API_KEY environment variable.")
        
        self.base_url = "https://api.tiingo.com"
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
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV data from Tiingo.
        
        Args:
            symbol: Stock symbol (e.g., 'SPY', 'AAPL')
            start: Start date (UTC)
            end: End date (UTC)
            
        Returns:
            DataFrame with columns [ts, open, high, low, close, volume, provider]
        """
        # Default date range: last 2 years
        if end is None:
            end = datetime.utcnow()
        if start is None:
            start = end - timedelta(days=730)
        
        # Format dates for API
        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')
        
        # Build endpoint
        endpoint = f"{self.base_url}/tiingo/daily/{symbol.upper()}/prices"
        
        params = {
            'startDate': start_str,
            'endDate': end_str,
            'token': self.api_key,
            'format': 'json'
        }
        
        # Apply rate limiting
        self._rate_limit()
        
        # Make request
        try:
            response = requests.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Tiingo API request failed for {symbol}: {e}")
            raise
        
        if not data:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Normalize column names
        # Tiingo returns: date, close, high, low, open, volume, adjClose, adjHigh, adjLow, adjOpen, adjVolume, divCash, splitFactor
        column_map = {
            'date': 'ts',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            # Use adjusted values if available
            'adjOpen': 'open',
            'adjHigh': 'high',
            'adjLow': 'low',
            'adjClose': 'close',
            'adjVolume': 'volume',
        }
        
        # Use adjusted values if available, otherwise use raw
        if 'adjClose' in df.columns:
            df['open'] = df['adjOpen']
            df['high'] = df['adjHigh']
            df['low'] = df['adjLow']
            df['close'] = df['adjClose']
            df['volume'] = df['adjVolume']
        
        # Select and rename columns
        df = df.rename(columns={'date': 'ts'})
        
        # Parse timestamp
        df['ts'] = pd.to_datetime(df['ts'], utc=True)
        
        # Add provider column
        df['provider'] = 'tiingo'
        
        # Select final columns
        df = df[['ts', 'open', 'high', 'low', 'close', 'volume', 'provider']]
        
        # Sort by timestamp
        df = df.sort_values('ts').reset_index(drop=True)
        
        logger.info(f"Fetched {len(df)} daily bars for {symbol}")
        
        return df
    
    def fetch_intraday(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        interval: str = "1hour"
    ) -> pd.DataFrame:
        """
        Fetch intraday OHLCV data from Tiingo IEX.
        
        Args:
            symbol: Stock symbol
            start: Start datetime (UTC)
            end: End datetime (UTC)
            interval: Resample frequency (1min, 5min, 15min, 30min, 1hour, 4hour)
            
        Returns:
            DataFrame with columns [ts, open, high, low, close, volume, provider]
        """
        # Default date range: last 7 days
        if end is None:
            end = datetime.utcnow()
        if start is None:
            start = end - timedelta(days=7)
        
        # Format datetimes for API
        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')
        
        # Build endpoint
        endpoint = f"{self.base_url}/iex/{symbol.upper()}/prices"
        
        params = {
            'startDate': start_str,
            'endDate': end_str,
            'resampleFreq': interval,
            'token': self.api_key,
            'format': 'json'
        }
        
        # Apply rate limiting
        self._rate_limit()
        
        # Make request
        try:
            response = requests.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Tiingo IEX API request failed for {symbol}: {e}")
            raise
        
        if not data:
            logger.warning(f"No intraday data returned for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Normalize column names
        df = df.rename(columns={'date': 'ts'})
        
        # Parse timestamp
        df['ts'] = pd.to_datetime(df['ts'], utc=True)
        
        # Add provider column
        df['provider'] = 'tiingo'
        
        # Select final columns
        df = df[['ts', 'open', 'high', 'low', 'close', 'volume', 'provider']]
        
        # Sort by timestamp
        df = df.sort_values('ts').reset_index(drop=True)
        
        logger.info(f"Fetched {len(df)} {interval} bars for {symbol}")
        
        return df


# Module-level convenience functions
def fetch_daily(
    symbol: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None
) -> pd.DataFrame:
    """Convenience function to fetch daily data."""
    client = TiingoClient()
    return client.fetch_daily(symbol, start, end)


def fetch_intraday(
    symbol: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    interval: str = "1hour"
) -> pd.DataFrame:
    """Convenience function to fetch intraday data."""
    client = TiingoClient()
    return client.fetch_intraday(symbol, start, end, interval)
