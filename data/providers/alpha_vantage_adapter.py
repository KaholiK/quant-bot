"""
Alpha Vantage data provider adapter.

Free tier: 5 API calls per minute, 500 per day
Good for stocks, forex, and crypto with technical indicators built-in.
"""

import os
from datetime import datetime
from typing import Any

import pandas as pd
from loguru import logger

from .base import DataProviderBase


class AlphaVantageAdapter(DataProviderBase):
    """
    Alpha Vantage data provider with retry and rate limiting.
    
    Features:
    - Stock, forex, crypto data
    - Built-in technical indicators
    - Fundamental data
    """

    def __init__(self, api_key: str | None = None):
        """
        Initialize Alpha Vantage adapter.
        
        Args:
            api_key: Alpha Vantage API key (from env if not provided)
        """
        api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        if not api_key:
            logger.warning("ALPHA_VANTAGE_API_KEY not configured, adapter may fail")
        
        # Free tier: 5 req/min = 12 sec between requests
        super().__init__(api_key=api_key, rate_limit_delay=12.0)
        self.base_url = "https://www.alphavantage.co/query"

    def get_bars(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1D",
    ) -> pd.DataFrame:
        """
        Fetch time series data from Alpha Vantage.
        
        Args:
            symbol: Ticker symbol
            start_date: Start date (Alpha Vantage returns full history, we filter)
            end_date: End date
            timeframe: Timeframe (1D, 1H, 5T, 15T, 30T, 60T)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Determine function based on timeframe
        function, interval = self._get_function_and_interval(timeframe)
        
        params = {
            "function": function,
            "symbol": symbol,
            "apikey": self.api_key,
            "outputsize": "full",  # Get all available data
            "datatype": "json",
        }
        
        if interval:
            params["interval"] = interval
        
        try:
            data = self._make_request(self.base_url, params=params)
            
            # Check for error messages
            if "Error Message" in data:
                logger.error(f"Alpha Vantage error: {data['Error Message']}")
                return pd.DataFrame()
            
            if "Note" in data:
                logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                return pd.DataFrame()
            
            # Extract time series data
            time_series_key = self._get_time_series_key(data)
            if not time_series_key:
                logger.warning(f"No time series data for {symbol}")
                return pd.DataFrame()
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient="index")
            
            # Rename columns (Alpha Vantage uses "1. open", "2. high", etc.)
            df.columns = [col.split(". ")[1] for col in df.columns]
            df.index.name = "timestamp"
            df.reset_index(inplace=True)
            
            # Convert types
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            
            # Filter by date range
            df = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]
            
            # Normalize and return
            return self.normalize_dataframe(df[["timestamp", "open", "high", "low", "close", "volume"]])
            
        except Exception as e:
            logger.error(f"Failed to fetch bars for {symbol}: {e}")
            return pd.DataFrame()

    def get_quote(self, symbol: str) -> dict[str, Any]:
        """
        Get global quote for a symbol.
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            Dictionary with quote data
        """
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": self.api_key,
        }
        
        try:
            data = self._make_request(self.base_url, params=params)
            
            if "Global Quote" in data:
                quote = data["Global Quote"]
                return {
                    "symbol": quote.get("01. symbol", symbol),
                    "last": float(quote.get("05. price", 0.0)),
                    "volume": int(quote.get("06. volume", 0)),
                    "timestamp": pd.to_datetime(quote.get("07. latest trading day")),
                    "change": float(quote.get("09. change", 0.0)),
                    "change_pct": float(quote.get("10. change percent", "0").rstrip("%")),
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to fetch quote for {symbol}: {e}")
            return {}

    def _get_function_and_interval(self, timeframe: str) -> tuple[str, str | None]:
        """
        Map timeframe to Alpha Vantage function and interval.
        
        Args:
            timeframe: Timeframe string (e.g., "1D", "5T", "1H")
            
        Returns:
            Tuple of (function, interval)
        """
        # Extract multiplier and unit
        multiplier = int("".join(filter(str.isdigit, timeframe)) or "1")
        unit = "".join(filter(str.isalpha, timeframe)).upper()
        
        if unit in ["D", "DAY"]:
            return "TIME_SERIES_DAILY", None
        elif unit in ["W", "WEEK"]:
            return "TIME_SERIES_WEEKLY", None
        elif unit in ["M", "MONTH", "MO"]:
            return "TIME_SERIES_MONTHLY", None
        else:
            # Intraday
            interval_map = {
                1: "1min",
                5: "5min",
                15: "15min",
                30: "30min",
                60: "60min",
            }
            interval = interval_map.get(multiplier, "5min")
            return "TIME_SERIES_INTRADAY", interval

    def _get_time_series_key(self, data: dict[str, Any]) -> str | None:
        """
        Find the time series key in response data.
        
        Args:
            data: API response data
            
        Returns:
            Time series key or None
        """
        for key in data.keys():
            if "Time Series" in key:
                return key
        return None
