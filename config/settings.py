"""
Production-grade configuration with pydantic-settings.
All optional integrations auto-enable based on environment variables.
"""

from typing import Optional, Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Main application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="allow"
    )
    
    # Core runtime settings
    APP_ENV: Literal["dev", "prod"] = Field(default="dev", description="Application environment")
    RUN_MODE: Literal["paper", "live"] = Field(default="paper", description="Trading mode (paper only for safety)")
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO", description="Logging level")
    TZ: str = Field(default="Pacific/Honolulu", description="Timezone for operations")
    
    # Discord control UI
    DISCORD_BOT_TOKEN: Optional[str] = Field(default=None, description="Discord bot token")
    DISCORD_GUILD_ID: Optional[int] = Field(default=None, description="Discord guild (server) ID")
    DISCORD_REPORTS_CHANNEL_ID: Optional[int] = Field(default=None, description="Channel for automated reports")
    DISCORD_WEBHOOK_URL: Optional[str] = Field(default=None, description="Webhook for notifications")
    DISCORD_APP_ID: Optional[int] = Field(default=None, description="Discord application ID")
    
    # Paper equities broker (kept optional for now)
    ALPACA_API_KEY: Optional[str] = Field(default=None, description="Alpaca API key")
    ALPACA_API_SECRET: Optional[str] = Field(default=None, description="Alpaca API secret")
    ALPACA_BASE_URL: str = Field(default="https://paper-api.alpaca.markets", description="Alpaca API base URL")
    
    # Equities data providers
    ALPHA_VANTAGE_API_KEY: Optional[str] = Field(default=None, description="Alpha Vantage API key")
    POLYGON_API_KEY: Optional[str] = Field(default=None, description="Polygon.io API key")
    TIINGO_API_KEY: Optional[str] = Field(default=None, description="Tiingo API key")
    
    # Macro / misc data
    FRED_API_KEY: Optional[str] = Field(default=None, description="FRED API key")
    QUANDL_API_KEY: Optional[str] = Field(default=None, description="Quandl API key")
    
    # Crypto data providers
    COINGECKO_API_KEY: Optional[str] = Field(default=None, description="CoinGecko API key")
    CRYPTOCOMPARE_API_KEY: Optional[str] = Field(default=None, description="CryptoCompare API key")
    
    # Optional power-ups
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API key for narration")
    WANDB_API_KEY: Optional[str] = Field(default=None, description="Weights & Biases API key")
    DATABASE_URL: Optional[str] = Field(default=None, description="PostgreSQL connection URL (optional)")
    
    def masked_dict(self) -> dict:
        """
        Return configuration dictionary with secrets masked.
        Shows only last 4 characters of API keys.
        """
        result = {}
        for field_name, field_value in self.model_dump().items():
            if field_value is None:
                result[field_name] = None
            elif "KEY" in field_name.upper() or "TOKEN" in field_name.upper() or "SECRET" in field_name.upper() or "URL" in field_name.upper():
                # Mask sensitive fields
                str_value = str(field_value)
                if len(str_value) > 4:
                    result[field_name] = f"***{str_value[-4:]}"
                else:
                    result[field_name] = "***"
            else:
                result[field_name] = field_value
        return result
    
    def has_openai(self) -> bool:
        """Check if OpenAI integration is available."""
        return self.OPENAI_API_KEY is not None and len(self.OPENAI_API_KEY) > 0
    
    def has_wandb(self) -> bool:
        """Check if W&B integration is available."""
        return self.WANDB_API_KEY is not None and len(self.WANDB_API_KEY) > 0
    
    def has_db(self) -> bool:
        """Check if PostgreSQL database is configured."""
        return self.DATABASE_URL is not None and len(self.DATABASE_URL) > 0
    
    def preferred_equity_provider(self) -> Optional[str]:
        """
        Return preferred equity data provider based on availability.
        Priority: polygon > tiingo > alphavantage
        """
        if self.POLYGON_API_KEY:
            return "polygon"
        elif self.TIINGO_API_KEY:
            return "tiingo"
        elif self.ALPHA_VANTAGE_API_KEY:
            return "alphavantage"
        return None
    
    def preferred_crypto_provider(self) -> Optional[str]:
        """
        Return preferred crypto data provider based on availability.
        Priority: coingecko > cryptocompare
        """
        if self.COINGECKO_API_KEY:
            return "coingecko"
        elif self.CRYPTOCOMPARE_API_KEY:
            return "cryptocompare"
        return None
    
    def db_url_or_sqlite(self) -> str:
        """
        Return DATABASE_URL if set, otherwise fallback to SQLite.
        SQLite database stored in data/runtime/quantbot.db
        """
        if self.has_db():
            return self.DATABASE_URL
        return "sqlite:///data/runtime/quantbot.db"
    
    def require_for_paper(self) -> None:
        """
        Validate requirements for paper trading.
        Warns if Discord not fully configured but doesn't crash.
        """
        issues = []
        
        if not self.DISCORD_BOT_TOKEN:
            issues.append("DISCORD_BOT_TOKEN not set")
        if not self.DISCORD_GUILD_ID:
            issues.append("DISCORD_GUILD_ID not set")
        if not self.DISCORD_REPORTS_CHANNEL_ID:
            issues.append("DISCORD_REPORTS_CHANNEL_ID not set")
        
        if issues:
            import warnings
            warnings.warn(
                f"Discord not fully configured for paper trading: {', '.join(issues)}. "
                "Bot functionality will be limited.",
                UserWarning
            )
    
    def require_data_any(self) -> None:
        """
        Validate that at least some data providers are configured.
        Warns if no providers available.
        """
        equity_provider = self.preferred_equity_provider()
        crypto_provider = self.preferred_crypto_provider()
        
        if not equity_provider and not crypto_provider:
            import warnings
            warnings.warn(
                "No data providers configured. Set at least one of: "
                "POLYGON_API_KEY, TIINGO_API_KEY, ALPHA_VANTAGE_API_KEY, "
                "COINGECKO_API_KEY, or CRYPTOCOMPARE_API_KEY",
                UserWarning
            )


# Global settings instance
settings = Settings()
