"""
Backward compatibility shim for cache_io.

Legacy module - use data.cache.CacheManager instead.
"""

from .cache import CacheManager

# Provide backward compatible alias
CacheIO = CacheManager

__all__ = ["CacheIO"]
