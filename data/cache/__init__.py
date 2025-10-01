"""
Parquet-based caching layer for market data.

Reduces API calls and speeds up backtesting.
"""

from .cache_manager import CacheManager

__all__ = ["CacheManager"]
