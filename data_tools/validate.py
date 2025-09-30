"""
Data validation utilities to prevent data leakage and ensure quality.
"""

import pandas as pd
from datetime import datetime
from typing import Optional
from loguru import logger


def assert_monotonic_ts(df: pd.DataFrame, column: str = 'ts') -> None:
    """
    Assert that timestamps are monotonically increasing.
    
    Args:
        df: DataFrame to check
        column: Timestamp column name
        
    Raises:
        ValueError: If timestamps are not monotonic
    """
    if df.empty:
        return
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not df[column].is_monotonic_increasing:
        # Find non-monotonic points
        diffs = df[column].diff()
        bad_indices = diffs[diffs < pd.Timedelta(0)].index.tolist()
        
        raise ValueError(
            f"Timestamps in column '{column}' are not monotonic. "
            f"Found {len(bad_indices)} backwards jumps at indices: {bad_indices[:5]}..."
        )


def drop_duplicates(df: pd.DataFrame, column: str = 'ts', keep: str = 'last') -> pd.DataFrame:
    """
    Drop duplicate timestamps, keeping last by default.
    
    Args:
        df: DataFrame to deduplicate
        column: Timestamp column to check
        keep: Which duplicate to keep ('first', 'last')
        
    Returns:
        Deduplicated DataFrame
    """
    if df.empty:
        return df
    
    original_len = len(df)
    df = df.drop_duplicates(subset=[column], keep=keep)
    
    if len(df) < original_len:
        logger.warning(f"Dropped {original_len - len(df)} duplicate timestamps")
    
    return df


def tz_to_utc(df: pd.DataFrame, column: str = 'ts') -> pd.DataFrame:
    """
    Convert timezone-aware or naive timestamps to UTC.
    
    Args:
        df: DataFrame to convert
        column: Timestamp column
        
    Returns:
        DataFrame with UTC timestamps
    """
    if df.empty:
        return df
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[column]):
        df[column] = pd.to_datetime(df[column])
    
    # Convert to UTC
    if df[column].dt.tz is None:
        # Naive timestamps - assume UTC
        df[column] = df[column].dt.tz_localize('UTC')
        logger.info(f"Localized naive timestamps in '{column}' to UTC")
    else:
        # Convert to UTC if not already
        df[column] = df[column].dt.tz_convert('UTC')
    
    return df


def align_to_bar(
    df: pd.DataFrame,
    interval: str,
    column: str = 'ts',
    tolerance_seconds: int = 60
) -> pd.DataFrame:
    """
    Align timestamps to expected bar intervals.
    Flags bars that are misaligned by more than tolerance.
    
    Args:
        df: DataFrame to align
        interval: Expected interval (15m, 1h, 1d)
        column: Timestamp column
        tolerance_seconds: Maximum allowed misalignment in seconds
        
    Returns:
        DataFrame with aligned timestamps (or original if misaligned)
    """
    if df.empty:
        return df
    
    # Map intervals to frequencies
    interval_map = {
        '1m': '1T', '5m': '5T', '15m': '15T', '30m': '30T',
        '1h': '1H', '4h': '4H', '1d': '1D', '1w': '1W'
    }
    
    if interval not in interval_map:
        logger.warning(f"Unknown interval '{interval}', skipping alignment")
        return df
    
    freq = interval_map[interval]
    
    # Round timestamps to nearest interval
    df['ts_aligned'] = df[column].dt.round(freq)
    
    # Check misalignment
    misalignment = (df[column] - df['ts_aligned']).abs()
    max_misalign = misalignment.max().total_seconds()
    
    if max_misalign > tolerance_seconds:
        logger.warning(
            f"Maximum misalignment: {max_misalign:.1f}s exceeds tolerance {tolerance_seconds}s. "
            f"Keeping original timestamps."
        )
        df = df.drop(columns=['ts_aligned'])
    else:
        # Use aligned timestamps
        df[column] = df['ts_aligned']
        df = df.drop(columns=['ts_aligned'])
        
        if max_misalign > 0:
            logger.info(f"Aligned timestamps (max offset: {max_misalign:.1f}s)")
    
    return df


def forbid_forward_fill_leakage(
    df: pd.DataFrame,
    max_gap_bars: int = 3,
    interval: Optional[str] = None
) -> pd.DataFrame:
    """
    Detect and prevent forward-fill data leakage.
    Raises error if gaps exceed threshold (suggesting forward-fill was used).
    
    Args:
        df: DataFrame to check
        max_gap_bars: Maximum allowed gap in number of bars
        interval: Time interval for gap calculation (optional)
        
    Returns:
        Original DataFrame if validation passes
        
    Raises:
        ValueError: If potential forward-fill leakage detected
    """
    if df.empty or len(df) < 2:
        return df
    
    # Calculate time gaps
    df = df.sort_values('ts')
    gaps = df['ts'].diff()
    
    if interval:
        # Calculate expected interval
        interval_map = {
            '1m': pd.Timedelta(minutes=1),
            '5m': pd.Timedelta(minutes=5),
            '15m': pd.Timedelta(minutes=15),
            '30m': pd.Timedelta(minutes=30),
            '1h': pd.Timedelta(hours=1),
            '4h': pd.Timedelta(hours=4),
            '1d': pd.Timedelta(days=1),
        }
        
        if interval in interval_map:
            expected_gap = interval_map[interval]
            max_allowed_gap = expected_gap * max_gap_bars
            
            # Find large gaps
            large_gaps = gaps[gaps > max_allowed_gap]
            
            if len(large_gaps) > 0:
                logger.info(f"Found {len(large_gaps)} gaps larger than {max_gap_bars} bars")
                
                # Check for suspiciously uniform values (forward-fill indicator)
                # This is a heuristic - check if consecutive close prices are identical
                consecutive_same = (df['close'].diff() == 0).sum()
                same_ratio = consecutive_same / len(df)
                
                if same_ratio > 0.1:  # More than 10% identical consecutive values
                    raise ValueError(
                        f"Potential forward-fill leakage detected: "
                        f"{same_ratio*100:.1f}% consecutive identical close prices. "
                        f"This suggests data was forward-filled improperly."
                    )
    
    return df


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate OHLCV data for basic sanity.
    
    Args:
        df: DataFrame with OHLCV columns
        
    Returns:
        Validated DataFrame
        
    Raises:
        ValueError: If validation fails
    """
    required_cols = ['ts', 'open', 'high', 'low', 'close', 'volume']
    
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Check for negative prices
    for col in ['open', 'high', 'low', 'close']:
        if (df[col] < 0).any():
            raise ValueError(f"Negative values found in '{col}' column")
    
    # Check high >= low
    if (df['high'] < df['low']).any():
        bad_rows = df[df['high'] < df['low']]
        raise ValueError(f"Found {len(bad_rows)} rows where high < low")
    
    # Check high >= open, close
    if ((df['high'] < df['open']) | (df['high'] < df['close'])).any():
        raise ValueError("High must be >= open and close")
    
    # Check low <= open, close
    if ((df['low'] > df['open']) | (df['low'] > df['close'])).any():
        raise ValueError("Low must be <= open and close")
    
    # Check for NaN values
    if df[required_cols].isna().any().any():
        raise ValueError("NaN values found in OHLCV data")
    
    return df
