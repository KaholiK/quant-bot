"""
Tests for trend_breakout strategy, specifically testing the momentum calculation bug fix.
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict
from unittest.mock import Mock

# Mock the logger to avoid import issues in tests
import sys
from unittest.mock import MagicMock
sys.modules['loguru'] = MagicMock()


def test_trend_breakout_momentum_unbound_error():
    """Test that momentum calculation doesn't raise UnboundLocalError."""
    from algos.strategies.trend_breakout import TrendBreakoutStrategy
    
    # Mock config
    config = {
        'strategies': {
            'trend_breakout': {
                'enabled': True,
                'lookback_bars': 55,
                'momentum_rank_pct': 0.30,
                'rsi_threshold': 80,
                'momentum_timeframe': '3m'
            }
        }
    }
    
    strategy = TrendBreakoutStrategy(config)
    
    # Create test data that will cause _calculate_momentum to fail
    symbol = "TEST"
    price_data = pd.DataFrame({
        'close': [100, 101, 102, 103, 104],  # Very short history 
        'high': [101, 102, 103, 104, 105],
        'low': [99, 100, 101, 102, 103],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })
    
    # Universe data with one symbol that will fail
    universe_data = {
        'GOOD_SYMBOL': pd.DataFrame({
            'close': np.random.randn(100) + 100,  # Sufficient data
            'high': np.random.randn(100) + 101,
            'low': np.random.randn(100) + 99,
            'volume': np.random.randint(1000, 2000, 100)
        }),
        'BAD_SYMBOL': pd.DataFrame({
            'close': [np.nan, np.nan, np.nan],  # Invalid data
            'high': [np.nan, np.nan, np.nan],
            'low': [np.nan, np.nan, np.nan],
            'volume': [np.nan, np.nan, np.nan]
        })
    }
    
    # Mock the _calculate_momentum method to raise an exception for BAD_SYMBOL
    original_calculate_momentum = strategy._calculate_momentum
    
    def mock_calculate_momentum(data, lookback):
        if data['close'].isna().all():
            raise ValueError("Invalid data")
        return original_calculate_momentum(data, lookback)
    
    strategy._calculate_momentum = mock_calculate_momentum
    
    # This should not raise UnboundLocalError
    result = strategy._check_momentum_condition(symbol, price_data, universe_data)
    
    # Should return 0 (neutral) and not crash
    assert isinstance(result, int)
    assert result in [-1, 0, 1]


def test_trend_breakout_momentum_with_valid_data():
    """Test momentum calculation with valid data."""
    from algos.strategies.trend_breakout import TrendBreakoutStrategy
    
    config = {
        'strategies': {
            'trend_breakout': {
                'enabled': True,
                'lookback_bars': 55,
                'momentum_rank_pct': 0.30,
                'rsi_threshold': 80,
                'momentum_timeframe': '3m'
            }
        }
    }
    
    strategy = TrendBreakoutStrategy(config)
    
    # Create sufficient test data
    symbol = "TEST"
    np.random.seed(42)  # For reproducibility
    
    price_data = pd.DataFrame({
        'close': np.cumsum(np.random.randn(70)) + 100,  # Trending data
        'high': np.cumsum(np.random.randn(70)) + 101,
        'low': np.cumsum(np.random.randn(70)) + 99,
        'volume': np.random.randint(1000, 2000, 70)
    })
    
    universe_data = {
        'SYMBOL1': pd.DataFrame({
            'close': np.cumsum(np.random.randn(70)) + 100,
            'high': np.cumsum(np.random.randn(70)) + 101,  
            'low': np.cumsum(np.random.randn(70)) + 99,
            'volume': np.random.randint(1000, 2000, 70)
        }),
        'SYMBOL2': pd.DataFrame({
            'close': np.cumsum(np.random.randn(70)) + 100,
            'high': np.cumsum(np.random.randn(70)) + 101,
            'low': np.cumsum(np.random.randn(70)) + 99,
            'volume': np.random.randint(1000, 2000, 70)
        })
    }
    
    # Should work without errors
    result = strategy._check_momentum_condition(symbol, price_data, universe_data)
    
    assert isinstance(result, int)
    assert result in [-1, 0, 1]
    
    # Should have stored momentum rank
    assert symbol in strategy.momentum_ranks


def test_trend_breakout_momentum_insufficient_data():
    """Test momentum calculation with insufficient data."""
    from algos.strategies.trend_breakout import TrendBreakoutStrategy
    
    config = {
        'strategies': {
            'trend_breakout': {
                'enabled': True,
                'lookback_bars': 55,
                'momentum_rank_pct': 0.30,
                'rsi_threshold': 80,
                'momentum_timeframe': '3m'
            }
        }
    }
    
    strategy = TrendBreakoutStrategy(config)
    
    # Create insufficient test data (less than 20 bars)
    symbol = "TEST"
    price_data = pd.DataFrame({
        'close': [100, 101, 102],  # Only 3 bars
        'high': [101, 102, 103],
        'low': [99, 100, 101],
        'volume': [1000, 1100, 1200]
    })
    
    universe_data = {
        'SYMBOL1': price_data.copy()
    }
    
    result = strategy._check_momentum_condition(symbol, price_data, universe_data)
    
    # Should return 0 for insufficient data
    assert result == 0


def test_trend_breakout_momentum_no_universe():
    """Test momentum calculation with no universe data."""
    from algos.strategies.trend_breakout import TrendBreakoutStrategy
    
    config = {
        'strategies': {
            'trend_breakout': {
                'enabled': True,
                'lookback_bars': 55,
                'momentum_rank_pct': 0.30,
                'rsi_threshold': 80,
                'momentum_timeframe': '3m'
            }
        }
    }
    
    strategy = TrendBreakoutStrategy(config)
    
    symbol = "TEST"
    price_data = pd.DataFrame({
        'close': [100, 101, 102, 103, 104],
        'high': [101, 102, 103, 104, 105],
        'low': [99, 100, 101, 102, 103],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })
    
    # No universe data
    result = strategy._check_momentum_condition(symbol, price_data, None)
    
    # Should return 0 when no universe data
    assert result == 0


def test_trend_breakout_other_symbols_still_score():
    """Test that valid symbols still get scored even when some fail."""
    from algos.strategies.trend_breakout import TrendBreakoutStrategy
    
    config = {
        'strategies': {
            'trend_breakout': {
                'enabled': True,
                'lookback_bars': 55,
                'momentum_rank_pct': 0.30,
                'rsi_threshold': 80,
                'momentum_timeframe': '3m'
            }
        }
    }
    
    strategy = TrendBreakoutStrategy(config)
    
    symbol = "TEST"
    np.random.seed(42)
    
    # Good price data
    price_data = pd.DataFrame({
        'close': np.cumsum(np.random.randn(70)) + 100,
        'high': np.cumsum(np.random.randn(70)) + 101,
        'low': np.cumsum(np.random.randn(70)) + 99,
        'volume': np.random.randint(1000, 2000, 70)
    })
    
    # Universe with mix of good and bad data
    universe_data = {
        'GOOD_SYMBOL': pd.DataFrame({
            'close': np.cumsum(np.random.randn(70)) + 100,
            'high': np.cumsum(np.random.randn(70)) + 101,
            'low': np.cumsum(np.random.randn(70)) + 99,
            'volume': np.random.randint(1000, 2000, 70)
        }),
        'BAD_SYMBOL': pd.DataFrame({
            'close': [100, 101],  # Insufficient data
            'high': [101, 102],
            'low': [99, 100],
            'volume': [1000, 1100]
        })
    }
    
    result = strategy._check_momentum_condition(symbol, price_data, universe_data)
    
    # Should still work and give a valid result
    assert isinstance(result, int)
    assert result in [-1, 0, 1]
    
    # Should have momentum rank stored
    assert symbol in strategy.momentum_ranks