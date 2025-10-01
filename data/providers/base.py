"""
Base data provider interface with retry, rate limiting, and caching.

Follows López de Prado principles for data quality and robustness.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import pandas as pd
import requests
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class DataProviderBase(ABC):
    """
    Abstract base class for data providers.
    
    Provides:
    - Automatic retry with exponential backoff
    - Rate limiting
    - Request session with connection pooling
    - Unified error handling
    """

    def __init__(
        self,
        api_key: str | None = None,
        rate_limit_delay: float = 0.2,
        max_retries: int = 3,
        timeout: int = 30,
    ):
        """
        Initialize data provider.
        
        Args:
            api_key: API key for the provider
            rate_limit_delay: Minimum delay between requests (seconds)
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self.timeout = timeout
        
        # Setup retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        
        # Create session with retry
        self.session = requests.Session()
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self.last_request_time = 0.0

    @abstractmethod
    def get_bars(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1D",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bar data for a symbol.
        
        Args:
            symbol: Trading symbol
            start_date: Start date for data
            end_date: End date for data
            timeframe: Timeframe (e.g., "1D", "1H", "5T")
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        pass

    @abstractmethod
    def get_quote(self, symbol: str) -> dict[str, Any]:
        """
        Get current quote for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with bid, ask, last, volume, etc.
        """
        pass

    def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        import time
        
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def _make_request(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Make HTTP request with rate limiting and error handling.
        
        Args:
            url: Request URL
            params: Query parameters
            headers: Request headers
            
        Returns:
            Response JSON data
            
        Raises:
            Exception: If request fails after retries
        """
        self._rate_limit()
        
        try:
            response = self.session.get(
                url,
                params=params,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise

    def normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize DataFrame to standard OHLCV format.
        
        Args:
            df: Raw DataFrame from provider
            
        Returns:
            Normalized DataFrame with standard columns
        """
        # Ensure timestamp is datetime index
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
        
        # Sort by timestamp
        df.sort_index(inplace=True)
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep="last")]
        
        return df
