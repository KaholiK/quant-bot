"""
Tests for data validation functions.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from data_tools.validate import (
    assert_monotonic_ts,
    drop_duplicates,
    tz_to_utc,
    align_to_bar,
    forbid_forward_fill_leakage,
    validate_ohlcv
)


def test_assert_monotonic_ts():
    """Test monotonic timestamp assertion."""
    # Valid monotonic timestamps
    df = pd.DataFrame({
        'ts': pd.date_range('2024-01-01', periods=10, freq='1H')
    })
    assert_monotonic_ts(df)  # Should not raise
    
    # Non-monotonic timestamps
    df_bad = pd.DataFrame({
        'ts': [
            datetime(2024, 1, 1, 10),
            datetime(2024, 1, 1, 11),
            datetime(2024, 1, 1, 9),  # Goes backward
            datetime(2024, 1, 1, 12)
        ]
    })
    
    with pytest.raises(ValueError, match="not monotonic"):
        assert_monotonic_ts(df_bad)


def test_drop_duplicates():
    """Test duplicate removal."""
    df = pd.DataFrame({
        'ts': [
            datetime(2024, 1, 1, 10),
            datetime(2024, 1, 1, 11),
            datetime(2024, 1, 1, 11),  # Duplicate
            datetime(2024, 1, 1, 12)
        ],
        'value': [1, 2, 3, 4]
    })
    
    result = drop_duplicates(df)
    
    assert len(result) == 3
    # Should keep last by default
    assert result[result['ts'] == datetime(2024, 1, 1, 11)]['value'].values[0] == 3


def test_tz_to_utc():
    """Test timezone conversion to UTC."""
    # Naive timestamps
    df = pd.DataFrame({
        'ts': pd.date_range('2024-01-01', periods=5, freq='1H')
    })
    
    result = tz_to_utc(df)
    
    assert result['ts'].dt.tz is not None
    assert str(result['ts'].dt.tz) == 'UTC'
    
    # Already UTC
    df_utc = pd.DataFrame({
        'ts': pd.date_range('2024-01-01', periods=5, freq='1H', tz='UTC')
    })
    
    result = tz_to_utc(df_utc)
    assert str(result['ts'].dt.tz) == 'UTC'


def test_align_to_bar():
    """Test bar alignment."""
    # Slightly misaligned timestamps
    df = pd.DataFrame({
        'ts': [
            pd.Timestamp('2024-01-01 10:00:15', tz='UTC'),  # 15 seconds off
            pd.Timestamp('2024-01-01 11:00:30', tz='UTC'),  # 30 seconds off
            pd.Timestamp('2024-01-01 12:00:00', tz='UTC'),  # Perfect
        ]
    })
    
    result = align_to_bar(df, interval='1h', tolerance_seconds=60)
    
    # Should align to hour boundaries
    assert result['ts'].iloc[0] == pd.Timestamp('2024-01-01 10:00:00', tz='UTC')
    assert result['ts'].iloc[1] == pd.Timestamp('2024-01-01 11:00:00', tz='UTC')
    assert result['ts'].iloc[2] == pd.Timestamp('2024-01-01 12:00:00', tz='UTC')


def test_validate_ohlcv():
    """Test OHLCV validation."""
    # Valid OHLCV
    df = pd.DataFrame({
        'ts': pd.date_range('2024-01-01', periods=5, freq='1D', tz='UTC'),
        'open': [100, 101, 102, 103, 104],
        'high': [105, 106, 107, 108, 109],
        'low': [95, 96, 97, 98, 99],
        'close': [102, 103, 104, 105, 106],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })
    
    validate_ohlcv(df)  # Should not raise
    
    # Invalid: high < low
    df_bad = df.copy()
    df_bad.loc[0, 'high'] = 90  # Less than low
    
    with pytest.raises(ValueError, match="high < low"):
        validate_ohlcv(df_bad)
    
    # Invalid: negative prices
    df_bad = df.copy()
    df_bad.loc[0, 'close'] = -100
    
    with pytest.raises(ValueError, match="Negative values"):
        validate_ohlcv(df_bad)


def test_forbid_forward_fill_leakage():
    """Test forward-fill leakage detection."""
    # Normal data (no suspicious forward-fill)
    df = pd.DataFrame({
        'ts': pd.date_range('2024-01-01', periods=10, freq='1H', tz='UTC'),
        'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    })
    
    # Should pass
    result = forbid_forward_fill_leakage(df, max_gap_bars=3, interval='1h')
    assert len(result) == 10
    
    # Suspicious: many consecutive identical values
    df_bad = pd.DataFrame({
        'ts': pd.date_range('2024-01-01', periods=10, freq='1H', tz='UTC'),
        'close': [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]  # All same
    })
    
    # Should raise due to high ratio of identical consecutive values
    with pytest.raises(ValueError, match="forward-fill leakage"):
        forbid_forward_fill_leakage(df_bad, max_gap_bars=3, interval='1h')


def test_empty_dataframe_handling():
    """Test that validators handle empty DataFrames gracefully."""
    df_empty = pd.DataFrame()
    
    # Should not raise
    assert_monotonic_ts(df_empty)
    result = drop_duplicates(df_empty)
    assert len(result) == 0
