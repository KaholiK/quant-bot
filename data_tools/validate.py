"""
Data validation utilities to prevent look-ahead bias and ensure data quality.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional


def assert_monotonic_ts(df: pd.DataFrame) -> None:
    """
    Assert that DataFrame has monotonically increasing timestamp index.
    
    Raises:
        ValueError: If index is not monotonically increasing
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")
    
    if not df.index.is_monotonic_increasing:
        raise ValueError("Timestamp index must be monotonically increasing")


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop duplicate timestamps, keeping the last occurrence.
    
    Args:
        df: DataFrame with DatetimeIndex
        
    Returns:
        DataFrame with duplicates removed
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")
    
    return df[~df.index.duplicated(keep='last')]


def tz_to_utc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert DataFrame index to UTC timezone.
    
    Args:
        df: DataFrame with DatetimeIndex
        
    Returns:
        DataFrame with UTC timezone
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")
    
    # If already UTC, return as-is
    if df.index.tz is not None and df.index.tz.zone == 'UTC':
        return df
    
    # If naive, assume UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        # Convert to UTC
        df.index = df.index.tz_convert('UTC')
    
    return df


def align_to_bar(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    Align timestamps to bar boundaries (e.g., round to nearest 15m).
    
    Args:
        df: DataFrame with DatetimeIndex
        interval: Interval string ('15m', '1h', '1d', etc.)
        
    Returns:
        DataFrame with aligned timestamps
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")
    
    # Parse interval to frequency
    freq_map = {
        '1m': '1min',
        '5m': '5min',
        '15m': '15min',
        '30m': '30min',
        '1h': '1H',
        '4h': '4H',
        '1d': '1D',
        '1w': '1W'
    }
    
    freq = freq_map.get(interval, interval)
    
    # Round timestamps to interval
    df.index = df.index.floor(freq)
    
    # Drop duplicates after rounding
    df = drop_duplicates(df)
    
    return df


def forbid_forward_fill_leakage(df: pd.DataFrame, max_gap_multiplier: float = 3.0) -> pd.DataFrame:
    """
    Validate that there are no suspicious forward fills that could indicate look-ahead bias.
    
    This function checks for large gaps in the data that might have been forward-filled,
    which could introduce look-ahead bias.
    
    Args:
        df: DataFrame with DatetimeIndex
        max_gap_multiplier: Maximum allowed gap as multiple of median interval
        
    Returns:
        DataFrame unchanged (raises if suspicious gaps found)
        
    Raises:
        ValueError: If suspicious forward-fill patterns detected
    """
    if df.empty or len(df) < 3:
        return df
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")
    
    # Calculate time deltas between consecutive points
    deltas = pd.Series(df.index[1:]) - pd.Series(df.index[:-1])
    median_delta = deltas.median()
    
    # Find gaps larger than threshold
    large_gaps = deltas > (median_delta * max_gap_multiplier)
    
    if large_gaps.any():
        gap_count = large_gaps.sum()
        max_gap = deltas.max()
        raise ValueError(
            f"Found {gap_count} suspicious gaps in data. "
            f"Max gap: {max_gap}, median interval: {median_delta}. "
            f"This might indicate forward-fill leakage."
        )
    
    return df


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate OHLCV data for consistency.
    
    Args:
        df: DataFrame with OHLCV columns
        
    Returns:
        DataFrame unchanged (raises if validation fails)
        
    Raises:
        ValueError: If OHLCV data is inconsistent
    """
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Check high >= low
    if (df['high'] < df['low']).any():
        raise ValueError("Found bars where high < low")
    
    # Check high >= open, close
    if (df['high'] < df['open']).any() or (df['high'] < df['close']).any():
        raise ValueError("Found bars where high < open or high < close")
    
    # Check low <= open, close
    if (df['low'] > df['open']).any() or (df['low'] > df['close']).any():
        raise ValueError("Found bars where low > open or low > close")
    
    # Check for negative values
    if (df[required_cols] < 0).any().any():
        raise ValueError("Found negative values in OHLCV data")
    
    # Check for NaN values
    if df[required_cols].isna().any().any():
        raise ValueError("Found NaN values in OHLCV data")
    
    return df


def clean_and_validate(
    df: pd.DataFrame,
    interval: str,
    validate_ohlc: bool = True,
    max_gap_multiplier: float = 3.0
) -> pd.DataFrame:
    """
    Full cleaning and validation pipeline.
    
    Args:
        df: Raw DataFrame
        interval: Interval string
        validate_ohlc: Whether to validate OHLCV consistency
        max_gap_multiplier: Maximum allowed gap multiplier
        
    Returns:
        Cleaned and validated DataFrame
    """
    # Convert to UTC
    df = tz_to_utc(df)
    
    # Drop duplicates
    df = drop_duplicates(df)
    
    # Align to bar
    df = align_to_bar(df, interval)
    
    # Check monotonic
    assert_monotonic_ts(df)
    
    # Validate no forward-fill leakage
    df = forbid_forward_fill_leakage(df, max_gap_multiplier)
    
    # Validate OHLCV if requested
    if validate_ohlc and all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        df = validate_ohlcv(df)
    
    return df
