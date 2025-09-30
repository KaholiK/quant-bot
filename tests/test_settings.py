"""
Tests for config/settings module.
"""

import pytest
from config.settings import Settings


def test_settings_defaults():
    """Test default settings values."""
    settings = Settings()
    
    assert settings.APP_ENV == "dev"
    assert settings.RUN_MODE == "paper"
    assert settings.LOG_LEVEL == "INFO"
    assert settings.TZ == "Pacific/Honolulu"


def test_masked_dict():
    """Test secret masking in masked_dict()."""
    settings = Settings(
        DISCORD_BOT_TOKEN="abc123def456ghi789",
        TIINGO_API_KEY="my_secret_key_123",
        DATABASE_URL="postgresql://user:password@host/db"
    )
    
    masked = settings.masked_dict()
    
    # Should mask secrets
    assert "abc123def456ghi789" not in str(masked)
    assert "my_secret_key_123" not in str(masked)
    assert "password" not in str(masked.get("DATABASE_URL", ""))
    
    # Should show last 4 characters
    assert masked["DISCORD_BOT_TOKEN"] == "***i789"
    assert masked["TIINGO_API_KEY"] == "***_123"


def test_helper_methods():
    """Test helper methods for checking integrations."""
    settings = Settings()
    
    # Without keys
    assert not settings.has_openai()
    assert not settings.has_wandb()
    assert not settings.has_db()
    
    # With keys
    settings = Settings(
        OPENAI_API_KEY="sk-test",
        WANDB_API_KEY="wandb-test",
        DATABASE_URL="postgresql://test"
    )
    
    assert settings.has_openai()
    assert settings.has_wandb()
    assert settings.has_db()


def test_preferred_providers():
    """Test preferred provider selection."""
    # No providers
    settings = Settings()
    assert settings.preferred_equity_provider() is None
    assert settings.preferred_crypto_provider() is None
    
    # With providers (test priority)
    settings = Settings(
        POLYGON_API_KEY="poly",
        TIINGO_API_KEY="tiingo",
        ALPHA_VANTAGE_API_KEY="alpha"
    )
    assert settings.preferred_equity_provider() == "polygon"
    
    settings = Settings(
        TIINGO_API_KEY="tiingo",
        ALPHA_VANTAGE_API_KEY="alpha"
    )
    assert settings.preferred_equity_provider() == "tiingo"
    
    settings = Settings(
        COINGECKO_API_KEY="cg",
        CRYPTOCOMPARE_API_KEY="cc"
    )
    assert settings.preferred_crypto_provider() == "coingecko"


def test_db_url_or_sqlite():
    """Test database URL fallback."""
    # Without DATABASE_URL
    settings = Settings()
    assert settings.db_url_or_sqlite() == "sqlite:///data/runtime/quantbot.db"
    
    # With DATABASE_URL
    settings = Settings(DATABASE_URL="postgresql://user:pass@host/db")
    assert settings.db_url_or_sqlite() == "postgresql://user:pass@host/db"


def test_run_mode_safety():
    """Test that RUN_MODE defaults to paper."""
    settings = Settings()
    assert settings.RUN_MODE == "paper"
    
    # Can be set to paper explicitly
    settings = Settings(RUN_MODE="paper")
    assert settings.RUN_MODE == "paper"


def test_warnings_for_missing_config():
    """Test that warnings are issued for missing Discord config."""
    settings = Settings()
    
    # Should warn but not raise
    with pytest.warns(UserWarning, match="Discord not fully configured"):
        settings.require_for_paper()
    
    # Should warn about no data providers
    with pytest.warns(UserWarning, match="No data providers configured"):
        settings.require_data_any()
