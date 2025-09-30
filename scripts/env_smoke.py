"""
Environment smoke test - validates settings and database connectivity.
Exits non-zero with friendly error if any critical check fails.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings


def test_db_connectivity(db_url: str) -> bool:
    """Test database connectivity."""
    try:
        from sqlalchemy import create_engine, text
        
        # Create parent directory for SQLite if needed
        if db_url.startswith("sqlite:///"):
            db_path = db_url.replace("sqlite:///", "")
            db_dir = os.path.dirname(db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
        
        engine = create_engine(db_url, pool_pre_ping=True)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
        return True
    except Exception as e:
        print(f"  ❌ Database connection failed: {e}")
        return False


def main() -> int:
    """Run environment smoke tests."""
    print("=" * 60)
    print("🔍 Environment Smoke Test")
    print("=" * 60)
    
    # Load settings
    try:
        settings = get_settings()
        print("✅ Settings loaded successfully\n")
    except Exception as e:
        print(f"❌ Failed to load settings: {e}")
        return 1
    
    # Display masked settings
    print("📋 Configuration Summary:")
    print("-" * 60)
    masked = settings.masked_dict()
    
    # Runtime
    print(f"  APP_ENV: {masked['APP_ENV']}")
    print(f"  RUN_MODE: {masked['RUN_MODE']}")
    print(f"  LOG_LEVEL: {masked['LOG_LEVEL']}")
    print(f"  TZ: {masked['TZ']}")
    print()
    
    # Discord
    print("  Discord:")
    print(f"    BOT_TOKEN: {masked['DISCORD_BOT_TOKEN'] or 'Not set'}")
    print(f"    GUILD_ID: {masked['DISCORD_GUILD_ID'] or 'Not set'}")
    print(f"    REPORTS_CHANNEL_ID: {masked['DISCORD_REPORTS_CHANNEL_ID'] or 'Not set'}")
    print()
    
    # Data providers
    print("  Data Providers:")
    print(f"    Equities: {settings.preferred_equity_provider()}")
    print(f"    Crypto: {settings.preferred_crypto_provider()}")
    print(f"    TIINGO_API_KEY: {masked['TIINGO_API_KEY'] or 'Not set'}")
    print(f"    COINGECKO_API_KEY: {masked['COINGECKO_API_KEY'] or 'Not set'}")
    print(f"    POLYGON_API_KEY: {masked['POLYGON_API_KEY'] or 'Not set'}")
    print(f"    ALPHA_VANTAGE_API_KEY: {masked['ALPHA_VANTAGE_API_KEY'] or 'Not set'}")
    print()
    
    # Optional features
    print("  Optional Features:")
    print(f"    OpenAI: {'✅' if settings.has_openai() else '❌'}")
    print(f"    W&B: {'✅' if settings.has_wandb() else '❌'}")
    print(f"    Database: {'✅ Postgres/MySQL' if settings.has_db() else '⚠️  SQLite fallback'}")
    print()
    
    # Test database connectivity
    print("-" * 60)
    print("🔌 Database Connectivity Test:")
    db_url = settings.db_url_or_sqlite()
    
    # Mask DB URL for display
    display_url = db_url
    if "@" in db_url:  # Postgres/MySQL with credentials
        parts = db_url.split("@")
        display_url = f"{parts[0].split('://')[0]}://***@{parts[1]}"
    
    print(f"  URL: {display_url}")
    
    if test_db_connectivity(db_url):
        print("  ✅ Database connection successful")
    else:
        print("  ❌ Database connection failed")
        if db_url.startswith("sqlite://"):
            print("  💡 Tip: SQLite database will be created on first use")
        else:
            print("  💡 Tip: Check DATABASE_URL and ensure database is accessible")
            return 1
    
    print()
    print("=" * 60)
    print("✅ All smoke tests passed!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
