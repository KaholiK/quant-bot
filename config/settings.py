"""
Settings module using pydantic-settings for environment-based configuration.
All secrets are read from .env and never printed raw.
"""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # ===== Runtime =====
    APP_ENV: str = Field(default="dev", description="Application environment")
    RUN_MODE: str = Field(default="paper", description="Run mode: paper or training")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    TZ: str = Field(default="Pacific/Honolulu", description="Timezone")
    
    # ===== Discord (control UI) =====
    DISCORD_BOT_TOKEN: Optional[str] = Field(default=None, description="Discord bot token")
    DISCORD_GUILD_ID: Optional[int] = Field(default=None, description="Discord guild ID")
    DISCORD_REPORTS_CHANNEL_ID: Optional[int] = Field(default=None, description="Discord reports channel ID")
    DISCORD_WEBHOOK_URL: Optional[str] = Field(default=None, description="Discord webhook URL")
    DISCORD_APP_ID: Optional[int] = Field(default=None, description="Discord app ID")
    
    # ===== Broker (paper equities - not used yet, kept optional) =====
    ALPACA_API_KEY: Optional[str] = Field(default=None, description="Alpaca API key")
    ALPACA_API_SECRET: Optional[str] = Field(default=None, description="Alpaca API secret")
    ALPACA_BASE_URL: str = Field(
        default="https://paper-api.alpaca.markets",
        description="Alpaca base URL"
    )
    
    # ===== Equities data =====
    ALPHA_VANTAGE_API_KEY: Optional[str] = Field(default=None, description="Alpha Vantage API key")
    POLYGON_API_KEY: Optional[str] = Field(default=None, description="Polygon API key")
    TIINGO_API_KEY: Optional[str] = Field(default=None, description="Tiingo API key")
    
    # ===== Macro =====
    FRED_API_KEY: Optional[str] = Field(default=None, description="FRED API key")
    QUANDL_API_KEY: Optional[str] = Field(default=None, description="Quandl API key")
    
    # ===== Crypto data =====
    COINGECKO_API_KEY: Optional[str] = Field(default=None, description="CoinGecko API key")
    CRYPTOCOMPARE_API_KEY: Optional[str] = Field(default=None, description="CryptoCompare API key")
    
    # ===== Optional power-ups =====
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API key")
    WANDB_API_KEY: Optional[str] = Field(default=None, description="Weights & Biases API key")
    DATABASE_URL: Optional[str] = Field(default=None, description="Database URL")
    
    def masked_dict(self) -> dict:
        """Return settings dict with secrets masked (show last 4 chars only)."""
        result = {}
        for key, value in self.model_dump().items():
            if value is None:
                result[key] = None
            elif any(secret in key.upper() for secret in ["KEY", "SECRET", "TOKEN", "PASSWORD", "URL"]):
                if isinstance(value, str) and len(value) > 4:
                    result[key] = f"***{value[-4:]}"
                elif isinstance(value, str):
                    result[key] = "***"
                else:
                    result[key] = value
            else:
                result[key] = value
        return result
    
    def db_url_or_sqlite(self) -> str:
        """Get DATABASE_URL or fallback to SQLite."""
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return "sqlite:///data/runtime/quantbot.db"
    
    def has_openai(self) -> bool:
        """Check if OpenAI API key is configured."""
        return bool(self.OPENAI_API_KEY)
    
    def has_wandb(self) -> bool:
        """Check if W&B API key is configured."""
        return bool(self.WANDB_API_KEY)
    
    def has_db(self) -> bool:
        """Check if database URL is configured (not using SQLite fallback)."""
        return bool(self.DATABASE_URL)
    
    def preferred_equity_provider(self) -> str:
        """Return preferred equity data provider based on available keys."""
        if self.TIINGO_API_KEY:
            return "tiingo"
        elif self.POLYGON_API_KEY:
            return "polygon"
        elif self.ALPHA_VANTAGE_API_KEY:
            return "alpha_vantage"
        return "none"
    
    def preferred_crypto_provider(self) -> str:
        """Return preferred crypto data provider based on available keys."""
        if self.COINGECKO_API_KEY:
            return "coingecko"
        elif self.CRYPTOCOMPARE_API_KEY:
            return "cryptocompare"
        return "none"


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
