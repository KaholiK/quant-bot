"""
FRED (Federal Reserve Economic Data) adapter.

Access to economic data: interest rates, inflation, unemployment, GDP, etc.
Free API with generous rate limits.
"""

import os
from datetime import datetime
from typing import Any

import pandas as pd
from loguru import logger

from .base import DataProviderBase


class FREDAdapter(DataProviderBase):
    """
    FRED data provider for macroeconomic data.
    
    Features:
    - 500,000+ economic time series
    - No rate limits on free tier
    - Daily updates for most series
    
    Useful for:
    - Regime detection (yield curve, VIX, etc.)
    - Risk-off indicators
    - Macro feature engineering
    """

    def __init__(self, api_key: str | None = None):
        """
        Initialize FRED adapter.
        
        Args:
            api_key: FRED API key (from env if not provided)
        """
        api_key = api_key or os.getenv("FRED_API_KEY")
        if not api_key:
            logger.warning("FRED_API_KEY not configured, adapter may fail")
        
        # FRED has generous rate limits, use small delay
        super().__init__(api_key=api_key, rate_limit_delay=0.5)
        self.base_url = "https://api.stlouisfed.org/fred"

    def get_bars(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1D",
    ) -> pd.DataFrame:
        """
        Fetch economic time series from FRED.
        
        Note: FRED uses series IDs (e.g., "DFF" for Federal Funds Rate)
        rather than ticker symbols. Timeframe is ignored as FRED data
        is typically daily or less frequent.
        
        Args:
            symbol: FRED series ID (e.g., "DFF", "T10Y2Y", "VIXCLS")
            start_date: Start date
            end_date: End date
            timeframe: Ignored (FRED determines frequency)
            
        Returns:
            DataFrame with date and value columns
        """
        url = f"{self.base_url}/series/observations"
        
        params = {
            "series_id": symbol,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start_date.strftime("%Y-%m-%d"),
            "observation_end": end_date.strftime("%Y-%m-%d"),
            "sort_order": "asc",
        }
        
        try:
            data = self._make_request(url, params=params)
            
            if "observations" not in data or not data["observations"]:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data["observations"])
            
            # Select and rename columns
            df = df[["date", "value"]]
            df.columns = ["timestamp", "close"]
            
            # Convert types
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            
            # Drop missing values (FRED uses "." for missing)
            df = df.dropna()
            
            # Add placeholder OHLV columns (FRED only has single value)
            df["open"] = df["close"]
            df["high"] = df["close"]
            df["low"] = df["close"]
            df["volume"] = 0
            
            # Normalize and return
            return self.normalize_dataframe(df[["timestamp", "open", "high", "low", "close", "volume"]])
            
        except Exception as e:
            logger.error(f"Failed to fetch series {symbol}: {e}")
            return pd.DataFrame()

    def get_quote(self, symbol: str) -> dict[str, Any]:
        """
        Get latest observation for a series.
        
        Args:
            symbol: FRED series ID
            
        Returns:
            Dictionary with latest value
        """
        url = f"{self.base_url}/series/observations"
        
        params = {
            "series_id": symbol,
            "api_key": self.api_key,
            "file_type": "json",
            "limit": 1,
            "sort_order": "desc",  # Get most recent
        }
        
        try:
            data = self._make_request(url, params=params)
            
            if "observations" in data and data["observations"]:
                obs = data["observations"][0]
                return {
                    "symbol": symbol,
                    "last": float(obs.get("value", 0.0)) if obs.get("value") != "." else None,
                    "timestamp": pd.to_datetime(obs.get("date")),
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to fetch latest for {symbol}: {e}")
            return {}

    def get_series_info(self, series_id: str) -> dict[str, Any]:
        """
        Get metadata about a FRED series.
        
        Args:
            series_id: FRED series ID
            
        Returns:
            Dictionary with series metadata
        """
        url = f"{self.base_url}/series"
        
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
        }
        
        try:
            data = self._make_request(url, params=params)
            
            if "seriess" in data and data["seriess"]:
                return data["seriess"][0]
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to fetch info for {series_id}: {e}")
            return {}


# Common FRED series for quant trading
FRED_SERIES = {
    # Interest Rates
    "DFF": "Federal Funds Effective Rate",
    "DGS10": "10-Year Treasury Constant Maturity Rate",
    "DGS2": "2-Year Treasury Constant Maturity Rate",
    "T10Y2Y": "10-Year Treasury Constant Maturity Minus 2-Year",
    
    # Volatility
    "VIXCLS": "CBOE Volatility Index: VIX",
    
    # Economic Indicators
    "UNRATE": "Unemployment Rate",
    "CPIAUCSL": "Consumer Price Index for All Urban Consumers",
    "GDP": "Gross Domestic Product",
    
    # Credit Spreads
    "BAMLH0A0HYM2": "ICE BofA US High Yield Index Option-Adjusted Spread",
    "BAMLC0A0CM": "ICE BofA US Corporate Index Option-Adjusted Spread",
}
