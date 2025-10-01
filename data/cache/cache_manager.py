"""
Cache manager for market data using Parquet files.

Implements intelligent caching with automatic updates and cleanup.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd
from loguru import logger


class CacheManager:
    """
    Manage Parquet cache for market data.
    
    Features:
    - Automatic cache invalidation
    - Incremental updates
    - Efficient storage with Parquet compression
    - Cache hit/miss tracking
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        base_path: str | Path | None = None,
        max_age_days: int = 7,
    ):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for cache files (preferred)
            base_path: Backward compatibility alias for cache_dir
            max_age_days: Maximum age of cached data before refresh
        """
        # Support both cache_dir and base_path for backward compatibility
        path = cache_dir or base_path or "data/cache"
        self.cache_dir = Path(path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age = timedelta(days=max_age_days)
        
        # Track cache statistics
        self.hits = 0
        self.misses = 0

    def get(
        self,
        key: str,
        fetch_fn: Callable[[], pd.DataFrame],
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Get data from cache or fetch if not available/stale.
        
        Args:
            key: Cache key (e.g., "SPY_1D_2024-01-01_2024-12-31")
            fetch_fn: Function to fetch data if cache miss
            force_refresh: Force refresh even if cached
            
        Returns:
            DataFrame with cached or fetched data
        """
        cache_file = self._get_cache_path(key)
        
        # Check if cache exists and is fresh
        if not force_refresh and cache_file.exists():
            age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            
            if age < self.max_age:
                try:
                    df = pd.read_parquet(cache_file)
                    self.hits += 1
                    logger.debug(f"Cache hit: {key}")
                    return df
                except Exception as e:
                    logger.warning(f"Failed to read cache {key}: {e}")
        
        # Cache miss - fetch data
        self.misses += 1
        logger.debug(f"Cache miss: {key}")
        
        try:
            df = fetch_fn()
            
            if not df.empty:
                self._write_cache(key, df)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {key}: {e}")
            return pd.DataFrame()

    def put(self, key: str, df: pd.DataFrame) -> None:
        """
        Store data in cache.
        
        Args:
            key: Cache key
            df: DataFrame to cache
        """
        self._write_cache(key, df)

    def invalidate(self, key: str) -> None:
        """
        Invalidate cached data.
        
        Args:
            key: Cache key to invalidate
        """
        cache_file = self._get_cache_path(key)
        if cache_file.exists():
            cache_file.unlink()
            logger.debug(f"Invalidated cache: {key}")

    def clear_all(self) -> None:
        """Clear all cached data."""
        for file in self.cache_dir.glob("*.parquet"):
            file.unlink()
        logger.info(f"Cleared all cache in {self.cache_dir}")

    def get_stats(self) -> dict[str, int | float]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with hit rate and counts
        """
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate": hit_rate,
        }

    def _get_cache_path(self, key: str) -> Path:
        """
        Get cache file path for key.
        
        Args:
            key: Cache key
            
        Returns:
            Path to cache file
        """
        # Sanitize key for filesystem
        safe_key = key.replace("/", "_").replace(":", "_")
        return self.cache_dir / f"{safe_key}.parquet"

    def _write_cache(self, key: str, df: pd.DataFrame) -> None:
        """
        Write DataFrame to cache.
        
        Args:
            key: Cache key
            df: DataFrame to cache
        """
        try:
            cache_file = self._get_cache_path(key)
            df.to_parquet(cache_file, compression="snappy", index=True)
            logger.debug(f"Cached {len(df)} rows for {key}")
        except Exception as e:
            logger.error(f"Failed to write cache {key}: {e}")

    # Backward compatibility methods for legacy CacheIO interface
    def save_parquet(
        self,
        df: pd.DataFrame,
        asset: str,
        symbol: str,
        interval: str,
    ) -> None:
        """
        Save DataFrame in legacy format.
        
        Args:
            df: DataFrame to save
            asset: Asset type (equity, crypto, etc.)
            symbol: Symbol
            interval: Timeframe interval
        """
        key = f"{asset}_{symbol}_{interval}"
        self.put(key, df)

    def load_range(
        self,
        asset: str,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame | None:
        """
        Load data for a date range.
        
        Args:
            asset: Asset type
            symbol: Symbol
            interval: Timeframe interval
            start: Start date
            end: End date
            
        Returns:
            Filtered DataFrame or None
        """
        key = f"{asset}_{symbol}_{interval}"
        cache_file = self._get_cache_path(key)
        
        if not cache_file.exists():
            return None
        
        try:
            df = pd.read_parquet(cache_file)
            
            # Filter by date range if index is datetime
            if isinstance(df.index, pd.DatetimeIndex):
                df = df[(df.index >= start) & (df.index <= end)]
            
            return df if not df.empty else None
            
        except Exception as e:
            logger.error(f"Failed to load range for {key}: {e}")
            return None

    def has_coverage(
        self,
        asset: str,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        min_coverage: float = 0.95,
    ) -> tuple[bool, float, int, int]:
        """
        Check if cache has sufficient coverage for date range.
        
        Args:
            asset: Asset type
            symbol: Symbol
            interval: Timeframe interval
            start: Start date
            end: End date
            min_coverage: Minimum coverage ratio required
            
        Returns:
            Tuple of (has_coverage, ratio, actual_count, expected_count)
        """
        df = self.load_range(asset, symbol, interval, start, end)
        
        if df is None or df.empty:
            return False, 0.0, 0, 0
        
        # Calculate expected count based on interval
        actual_count = len(df)
        
        # For daily data, count business days
        if interval == "1d":
            expected_count = pd.bdate_range(start, end).size
        else:
            # For intraday, estimate based on time delta
            # This is a simplified calculation
            days = (end - start).days + 1
            expected_count = days
        
        ratio = actual_count / expected_count if expected_count > 0 else 0.0
        has_cov = ratio >= min_coverage
        
        return has_cov, ratio, actual_count, expected_count
