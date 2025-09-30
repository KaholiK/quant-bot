#!/usr/bin/env python3
"""
Environment smoke test.
Validates configuration and checks connectivity to configured services.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from sqlalchemy import create_engine, text


console = Console()


def check_db_connectivity() -> tuple[bool, str]:
    """Check database connectivity."""
    db_url = settings.db_url_or_sqlite()
    
    try:
        # Create engine
        engine = create_engine(db_url, pool_pre_ping=True)
        
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        # Check if using Neon pooler (warn about migrations)
        if settings.has_db() and "-pooler" in db_url:
            return True, f"✅ Connected (DB: PostgreSQL)\n⚠️  Note: Using Neon pooler URL. For migrations, use direct connection URL."
        elif settings.has_db():
            return True, "✅ Connected (DB: PostgreSQL)"
        else:
            # Ensure SQLite directory exists
            sqlite_path = Path("data/runtime")
            sqlite_path.mkdir(parents=True, exist_ok=True)
            return True, "✅ Connected (DB: SQLite fallback)"
            
    except Exception as e:
        return False, f"❌ Connection failed: {str(e)}"


def main() -> int:
    """Run environment smoke test."""
    console.print("\n[bold blue]🔍 Environment Smoke Test[/bold blue]\n")
    
    # Display masked configuration
    console.print("[bold]Configuration Summary:[/bold]")
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Setting", style="dim")
    table.add_column("Value")
    table.add_column("Status")
    
    masked = settings.masked_dict()
    
    # Core settings
    table.add_row("APP_ENV", str(masked["APP_ENV"]), "✅")
    table.add_row("RUN_MODE", str(masked["RUN_MODE"]), 
                  "✅" if masked["RUN_MODE"] == "paper" else "⚠️ LIVE MODE")
    table.add_row("LOG_LEVEL", str(masked["LOG_LEVEL"]), "✅")
    table.add_row("TZ", str(masked["TZ"]), "✅")
    
    console.print(table)
    console.print()
    
    # Provider availability
    console.print("[bold]Data Providers:[/bold]")
    provider_table = Table(show_header=True, header_style="bold cyan")
    provider_table.add_column("Provider", style="dim")
    provider_table.add_column("Status")
    provider_table.add_column("Value")
    
    # Equity providers
    equity_prov = settings.preferred_equity_provider()
    provider_table.add_row(
        "Preferred Equity", 
        "✅" if equity_prov else "⚠️  None",
        equity_prov or "N/A"
    )
    provider_table.add_row("Polygon", "✅" if settings.POLYGON_API_KEY else "❌", 
                          masked.get("POLYGON_API_KEY") or "Not set")
    provider_table.add_row("Tiingo", "✅" if settings.TIINGO_API_KEY else "❌",
                          masked.get("TIINGO_API_KEY") or "Not set")
    provider_table.add_row("Alpha Vantage", "✅" if settings.ALPHA_VANTAGE_API_KEY else "❌",
                          masked.get("ALPHA_VANTAGE_API_KEY") or "Not set")
    
    # Crypto providers
    crypto_prov = settings.preferred_crypto_provider()
    provider_table.add_row(
        "Preferred Crypto",
        "✅" if crypto_prov else "⚠️  None", 
        crypto_prov or "N/A"
    )
    provider_table.add_row("CoinGecko", "✅" if settings.COINGECKO_API_KEY else "❌",
                          masked.get("COINGECKO_API_KEY") or "Not set")
    provider_table.add_row("CryptoCompare", "✅" if settings.CRYPTOCOMPARE_API_KEY else "❌",
                          masked.get("CRYPTOCOMPARE_API_KEY") or "Not set")
    
    console.print(provider_table)
    console.print()
    
    # Discord configuration
    console.print("[bold]Discord Control UI:[/bold]")
    discord_table = Table(show_header=True, header_style="bold cyan")
    discord_table.add_column("Setting", style="dim")
    discord_table.add_column("Status")
    discord_table.add_column("Value")
    
    discord_table.add_row("Bot Token", "✅" if settings.DISCORD_BOT_TOKEN else "❌",
                         masked.get("DISCORD_BOT_TOKEN") or "Not set")
    discord_table.add_row("Guild ID", "✅" if settings.DISCORD_GUILD_ID else "❌",
                         str(masked.get("DISCORD_GUILD_ID") or "Not set"))
    discord_table.add_row("Reports Channel", "✅" if settings.DISCORD_REPORTS_CHANNEL_ID else "❌",
                         str(masked.get("DISCORD_REPORTS_CHANNEL_ID") or "Not set"))
    discord_table.add_row("Webhook URL", "✅" if settings.DISCORD_WEBHOOK_URL else "⚪",
                         masked.get("DISCORD_WEBHOOK_URL") or "Not set (optional)")
    
    console.print(discord_table)
    console.print()
    
    # Optional integrations
    console.print("[bold]Optional Integrations:[/bold]")
    opt_table = Table(show_header=True, header_style="bold cyan")
    opt_table.add_column("Integration", style="dim")
    opt_table.add_column("Status")
    opt_table.add_column("Value")
    
    opt_table.add_row("OpenAI (Narration)", "✅" if settings.has_openai() else "⚪",
                     masked.get("OPENAI_API_KEY") or "Not set (will use fallback)")
    opt_table.add_row("W&B (Telemetry)", "✅" if settings.has_wandb() else "⚪",
                     masked.get("WANDB_API_KEY") or "Not set (no-op mode)")
    opt_table.add_row("Alpaca (Broker)", "✅" if settings.ALPACA_API_KEY else "⚪",
                     masked.get("ALPACA_API_KEY") or "Not set (optional)")
    
    console.print(opt_table)
    console.print()
    
    # Database connectivity
    console.print("[bold]Database Connectivity:[/bold]")
    db_ok, db_msg = check_db_connectivity()
    
    if db_ok:
        console.print(Panel(db_msg, style="green"))
    else:
        console.print(Panel(db_msg, style="red"))
        console.print("\n[bold red]❌ Database connectivity check failed![/bold red]")
        return 1
    
    # Final validation
    console.print("\n[bold]Validation Checks:[/bold]")
    
    errors = []
    warnings = []
    
    # Check run mode safety
    if settings.RUN_MODE != "paper":
        errors.append("RUN_MODE must be 'paper' (live trading disabled)")
    
    # Check data providers
    if not equity_prov and not crypto_prov:
        warnings.append("No data providers configured")
    
    # Check Discord for paper trading
    if not settings.DISCORD_BOT_TOKEN or not settings.DISCORD_GUILD_ID or not settings.DISCORD_REPORTS_CHANNEL_ID:
        warnings.append("Discord not fully configured (limited bot functionality)")
    
    if errors:
        console.print("\n[bold red]Errors:[/bold red]")
        for error in errors:
            console.print(f"  ❌ {error}")
        console.print("\n[bold red]❌ Smoke test FAILED - fix errors above[/bold red]\n")
        return 1
    
    if warnings:
        console.print("\n[bold yellow]Warnings:[/bold yellow]")
        for warning in warnings:
            console.print(f"  ⚠️  {warning}")
    
    console.print("\n[bold green]✅ Smoke test PASSED - environment looks good![/bold green]\n")
    
    # Print setup hints
    if warnings:
        console.print("[dim]Hints:[/dim]")
        console.print("[dim]- Set data provider API keys for market data access[/dim]")
        console.print("[dim]- Configure Discord for mobile control UI[/dim]")
        console.print("[dim]- See SETUP_ENV.md for detailed setup instructions[/dim]\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
