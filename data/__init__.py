"""
Data module for market data with providers and caching.

Integrates multiple data providers with Parquet caching layer.
"""

from .cache import CacheManager
from .providers import (
    AlphaVantageAdapter,
    DataProviderBase,
    FREDAdapter,
    PolygonAdapter,
)

__all__ = [
    "CacheManager",
    "DataProviderBase",
    "PolygonAdapter",
    "AlphaVantageAdapter",
    "FREDAdapter",
]
