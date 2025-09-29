"""
Tests for config_loader module.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

# Mock dependencies that might not be available
import sys
sys.modules['loguru'] = MagicMock()


def test_load_config_with_new_format():
    """Test loading config with new trading: key format."""
    from algos.core.config_loader import load_config
    
    config_data = {
        'trading': {
            'universe': {
                'equities': 'SP100',
                'crypto': ['BTCUSD', 'ETHUSD']
            },
            'bars': {
                'equities': '30m',
                'crypto': '15m'
            },
            'risk': {
                'per_trade_risk_pct': 0.01,
                'max_leverage': 2.0,
                'single_name_max_pct': 0.10,
                'sector_max_pct': 0.30,
                'kill_switch_dd': 0.20
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.safe_dump(config_data, f)
        config_path = f.name
    
    try:
        config = load_config(config_path)
        
        assert config.trading.universe.equities == 'SP100'
        assert config.trading.universe.crypto == ['BTCUSD', 'ETHUSD']
        assert config.trading.bars.equities == '30m'
        assert config.trading.bars.crypto == '15m'
        assert config.trading.risk.per_trade_risk_pct == 0.01
        assert config.trading.risk.max_leverage == 2.0
        
    finally:
        Path(config_path).unlink()


def test_load_config_with_legacy_format():
    """Test loading config with legacy flat format and normalization."""
    from algos.core.config_loader import load_config
    
    legacy_config = {
        'universe': {
            'equities': {
                'resolution': 'ThirtyMinute'
            },
            'crypto': {
                'symbols': ['BTCUSD', 'ETHUSD'],
                'resolution': 'FifteenMinute'
            }
        },
        'risk': {
            'max_leverage': 3.0,
            'max_position_pct': 0.15,
            'risk_pct_per_trade': 0.02,
            'kill_switch_dd': 0.25
        },
        'strategies': {
            'trend_breakout': {'enabled': True}
        },
        'models': {
            'classifier_path': 'models/test.joblib'
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.safe_dump(legacy_config, f)
        config_path = f.name
    
    try:
        config = load_config(config_path)
        
        # Should be normalized
        assert config.trading.bars.equities == '30m'  # Converted from ThirtyMinute
        assert config.trading.bars.crypto == '15m'    # Converted from FifteenMinute
        assert config.trading.risk.max_leverage == 3.0
        assert config.trading.risk.single_name_max_pct == 0.15  # Mapped from max_position_pct
        assert config.trading.risk.per_trade_risk_pct == 0.02   # Mapped from risk_pct_per_trade
        assert config.trading.strategies.trend_breakout.enabled == True
        
    finally:
        Path(config_path).unlink()


def test_load_config_validation_errors():
    """Test config validation with invalid values."""
    from algos.core.config_loader import load_config
    
    # Invalid risk percentage (too high)
    invalid_config = {
        'trading': {
            'risk': {
                'per_trade_risk_pct': 0.10  # 10% is too high (max 5%)
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.safe_dump(invalid_config, f)
        config_path = f.name
    
    try:
        with pytest.raises(ValueError, match="Configuration validation failed"):
            load_config(config_path)
    finally:
        Path(config_path).unlink()


def test_load_config_missing_file():
    """Test loading config when file doesn't exist."""
    from algos.core.config_loader import load_config
    
    # Should return default config without error
    config = load_config("nonexistent.yaml")
    
    # Should have default values
    assert config.trading.universe.equities == 'SP100'
    assert config.trading.universe.crypto == ['BTCUSD', 'ETHUSD']
    assert config.trading.risk.per_trade_risk_pct == 0.01
    assert config.trading.risk.max_leverage == 2.0


def test_load_config_empty_file():
    """Test loading empty config file."""
    from algos.core.config_loader import load_config
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("")  # Empty file
        config_path = f.name
    
    try:
        config = load_config(config_path)
        
        # Should use defaults
        assert config.trading.universe.equities == 'SP100'
        assert config.trading.risk.per_trade_risk_pct == 0.01
        
    finally:
        Path(config_path).unlink()


def test_get_legacy_dict():
    """Test conversion back to legacy dictionary format."""
    from algos.core.config_loader import load_config, get_legacy_dict
    
    config_data = {
        'trading': {
            'universe': {
                'equities': 'SP100',
                'crypto': ['BTCUSD', 'ETHUSD']
            },
            'risk': {
                'per_trade_risk_pct': 0.015,
                'max_leverage': 2.5
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.safe_dump(config_data, f)
        config_path = f.name
    
    try:
        config = load_config(config_path)
        legacy_dict = get_legacy_dict(config)
        
        # Should flatten trading section
        assert 'universe' in legacy_dict
        assert 'risk' in legacy_dict
        assert legacy_dict['universe']['equities'] == 'SP100'
        assert legacy_dict['risk']['per_trade_risk_pct'] == 0.015
        
    finally:
        Path(config_path).unlink()


def test_risk_validation_ranges():
    """Test risk parameter validation ranges."""
    from algos.core.config_loader import load_config
    
    # Test kill_switch_dd validation (must be > 0 and <= 0.5)
    invalid_configs = [
        {'trading': {'risk': {'kill_switch_dd': 0.0}}},     # Too low
        {'trading': {'risk': {'kill_switch_dd': 0.6}}},     # Too high
        {'trading': {'risk': {'per_trade_risk_pct': -0.01}}}, # Negative
        {'trading': {'risk': {'max_leverage': 0.5}}},       # Too low
    ]
    
    for i, invalid_config in enumerate(invalid_configs):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.safe_dump(invalid_config, f)
            config_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Configuration validation failed"):
                load_config(config_path)
        finally:
            Path(config_path).unlink()