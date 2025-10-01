"""
Polygon.io data provider adapter.

High-quality market data for stocks, options, forex, and crypto.
Free tier: 5 requests/minute, delayed data
"""

import os
from datetime import datetime
from typing import Any

import pandas as pd
from loguru import logger

from .base import DataProviderBase


class PolygonAdapter(DataProviderBase):
    """
    Polygon.io data provider with retry and rate limiting.
    
    Features:
    - Stock, options, forex, crypto data
    - Adjustments for splits/dividends
    - Real-time and delayed data
    """

    def __init__(self, api_key: str | None = None):
        """
        Initialize Polygon adapter.
        
        Args:
            api_key: Polygon API key (from env if not provided)
        """
        api_key = api_key or os.getenv("POLYGON_API_KEY")
        if not api_key:
            logger.warning("POLYGON_API_KEY not configured, adapter may fail")
        
        # Free tier: 5 req/min = 12 sec between requests (conservative)
        super().__init__(api_key=api_key, rate_limit_delay=12.0)
        self.base_url = "https://api.polygon.io"

    def get_bars(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1D",
    ) -> pd.DataFrame:
        """
        Fetch aggregated bars (candles) from Polygon.
        
        Args:
            symbol: Ticker symbol
            start_date: Start date
            end_date: End date
            timeframe: Timeframe (1D, 1H, 5T, etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Parse timeframe (e.g., "1D" -> multiplier=1, timespan="day")
        multiplier, timespan = self._parse_timeframe(timeframe)
        
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        
        params = {
            "apiKey": self.api_key,
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
        }
        
        try:
            data = self._make_request(url, params=params)
            
            if "results" not in data or not data["results"]:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data["results"])
            
            # Rename columns to standard format
            df = df.rename(
                columns={
                    "t": "timestamp",
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": "close",
                    "v": "volume",
                    "vw": "vwap",
                    "n": "transactions",
                }
            )
            
            # Convert timestamp from milliseconds
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            
            # Normalize and return
            return self.normalize_dataframe(df[["timestamp", "open", "high", "low", "close", "volume"]])
            
        except Exception as e:
            logger.error(f"Failed to fetch bars for {symbol}: {e}")
            return pd.DataFrame()

    def get_quote(self, symbol: str) -> dict[str, Any]:
        """
        Get last quote for a symbol.
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            Dictionary with quote data
        """
        url = f"{self.base_url}/v2/last/trade/{symbol}"
        
        params = {"apiKey": self.api_key}
        
        try:
            data = self._make_request(url, params=params)
            
            if "results" in data:
                result = data["results"]
                return {
                    "symbol": symbol,
                    "last": result.get("p", 0.0),
                    "size": result.get("s", 0),
                    "timestamp": pd.to_datetime(result.get("t", 0), unit="ms"),
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to fetch quote for {symbol}: {e}")
            return {}

    def _parse_timeframe(self, timeframe: str) -> tuple[int, str]:
        """
        Parse timeframe string to Polygon format.
        
        Args:
            timeframe: Timeframe string (e.g., "1D", "5T", "1H")
            
        Returns:
            Tuple of (multiplier, timespan)
        """
        # Extract multiplier and unit
        multiplier = int("".join(filter(str.isdigit, timeframe)) or "1")
        unit = "".join(filter(str.isalpha, timeframe)).upper()
        
        # Map to Polygon timespans
        timespan_map = {
            "T": "minute",
            "M": "minute",
            "H": "hour",
            "D": "day",
            "W": "week",
            "MO": "month",
        }
        
        timespan = timespan_map.get(unit, "day")
        
        return multiplier, timespan
