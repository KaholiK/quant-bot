#!/usr/bin/env python3
"""
Discord bot for paper trading control with slash commands.
Paper trading mode only - no live trading.
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Literal, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import discord
from discord.ext import commands
from discord import app_commands
from loguru import logger

from config.settings import settings
from storage.db import get_session, EquityPoint, Fill, Run
from ui.discord_bot.cogs.reporting import (
    create_performance_embed, create_trade_table_embed,
    create_equity_chart, create_error_embed, create_success_embed
)
from reporting.narration import summarize

from sqlalchemy import select, func


class QuantBot(commands.Bot):
    """Discord bot for paper trading control."""
    
    def __init__(self):
        """Initialize bot with minimal intents."""
        intents = discord.Intents.default()
        # No message content needed for slash commands
        intents.message_content = False
        
        super().__init__(
            command_prefix="!",  # Not used for slash commands
            intents=intents,
            description="Paper Trading Control Bot"
        )
        
        self.guild_id = settings.DISCORD_GUILD_ID
        self.reports_channel_id = settings.DISCORD_REPORTS_CHANNEL_ID
        self.paper_loop_flag = False  # Flag for /halt command
    
    async def setup_hook(self):
        """Called when bot is starting up."""
        logger.info("Bot setup hook called")
        
        # Sync commands to guild (faster than global)
        if self.guild_id:
            guild = discord.Object(id=self.guild_id)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            logger.info(f"Synced commands to guild {self.guild_id}")
        else:
            # Global sync (slower, can take up to 1 hour)
            await self.tree.sync()
            logger.info("Synced commands globally")
    
    async def on_ready(self):
        """Called when bot is ready."""
        logger.info(f"{self.user} has connected to Discord!")
        logger.info(f"Bot is in {len(self.guilds)} guilds")
        
        # Set activity
        activity = discord.Activity(
            type=discord.ActivityType.watching,
            name="paper markets 📊"
        )
        await self.change_presence(activity=activity)
        
        # Post startup message
        await self.post_boot_message()
    
    async def post_boot_message(self):
        """Post boot confirmation to reports channel."""
        if not self.reports_channel_id:
            logger.warning("Reports channel not configured")
            return
        
        try:
            channel = self.get_channel(self.reports_channel_id)
            if not channel:
                logger.error(f"Reports channel {self.reports_channel_id} not found")
                return
            
            # Create masked env summary
            masked = settings.masked_dict()
            
            providers = []
            if settings.preferred_equity_provider():
                providers.append(f"Equity: {settings.preferred_equity_provider()}")
            if settings.preferred_crypto_provider():
                providers.append(f"Crypto: {settings.preferred_crypto_provider()}")
            
            embed = discord.Embed(
                title="✅ Bot Started (Paper Mode)",
                description="Discord control bot is online",
                color=discord.Color.green(),
                timestamp=datetime.utcnow()
            )
            
            embed.add_field(name="Mode", value=settings.RUN_MODE.upper(), inline=True)
            embed.add_field(name="Environment", value=settings.APP_ENV.upper(), inline=True)
            embed.add_field(name="Database", value="PostgreSQL" if settings.has_db() else "SQLite", inline=True)
            
            if providers:
                embed.add_field(name="Data Providers", value="\n".join(providers), inline=False)
            
            # Optional integrations
            integrations = []
            if settings.has_openai():
                integrations.append("✅ OpenAI")
            if settings.has_wandb():
                integrations.append("✅ W&B")
            
            if integrations:
                embed.add_field(name="Integrations", value=" | ".join(integrations), inline=False)
            
            await channel.send(embed=embed)
            logger.info("Posted boot message to reports channel")
            
        except Exception as e:
            logger.error(f"Failed to post boot message: {e}")


bot = QuantBot()


def check_guild(interaction: discord.Interaction) -> bool:
    """Check if command is from correct guild."""
    if not settings.DISCORD_GUILD_ID:
        return True  # No restriction if not configured
    
    if interaction.guild_id != settings.DISCORD_GUILD_ID:
        return False
    
    return True


@bot.tree.command(name="envcheck", description="Check active data providers and configuration")
async def envcheck(interaction: discord.Interaction):
    """Show active providers and configuration."""
    if not check_guild(interaction):
        await interaction.response.send_message(
            "❌ This command can only be used in the configured guild.",
            ephemeral=True
        )
        return
    
    await interaction.response.defer(ephemeral=True)
    
    masked = settings.masked_dict()
    
    embed = discord.Embed(
        title="🔍 Environment Check",
        color=discord.Color.blue(),
        timestamp=datetime.utcnow()
    )
    
    # Core settings
    embed.add_field(name="Mode", value=masked["RUN_MODE"], inline=True)
    embed.add_field(name="Environment", value=masked["APP_ENV"], inline=True)
    embed.add_field(name="Log Level", value=masked["LOG_LEVEL"], inline=True)
    
    # Data providers
    providers_status = []
    equity_prov = settings.preferred_equity_provider()
    crypto_prov = settings.preferred_crypto_provider()
    
    if equity_prov:
        providers_status.append(f"✅ Equity: {equity_prov}")
    else:
        providers_status.append("❌ Equity: None")
    
    if crypto_prov:
        providers_status.append(f"✅ Crypto: {crypto_prov}")
    else:
        providers_status.append("❌ Crypto: None")
    
    embed.add_field(
        name="Data Providers",
        value="\n".join(providers_status),
        inline=False
    )
    
    # Optional integrations
    integrations = []
    integrations.append(f"{'✅' if settings.has_openai() else '❌'} OpenAI (Narration)")
    integrations.append(f"{'✅' if settings.has_wandb() else '❌'} W&B (Telemetry)")
    integrations.append(f"{'✅' if settings.has_db() else '❌'} PostgreSQL")
    
    embed.add_field(
        name="Integrations",
        value="\n".join(integrations),
        inline=False
    )
    
    await interaction.followup.send(embed=embed, ephemeral=True)


@bot.tree.command(name="pnl", description="Show P&L for time window")
@app_commands.describe(window="Time window for P&L calculation")
@app_commands.choices(window=[
    app_commands.Choice(name="1 Day", value="1d"),
    app_commands.Choice(name="1 Week", value="1w"),
    app_commands.Choice(name="1 Month", value="1m"),
    app_commands.Choice(name="Year to Date", value="ytd"),
])
async def pnl(interaction: discord.Interaction, window: str):
    """Show P&L and equity chart for time window."""
    if not check_guild(interaction):
        await interaction.response.send_message(
            "❌ This command can only be used in the configured guild.",
            ephemeral=True
        )
        return
    
    await interaction.response.defer(ephemeral=True)
    
    # Calculate date range
    end_time = datetime.utcnow()
    
    if window == "1d":
        start_time = end_time - timedelta(days=1)
    elif window == "1w":
        start_time = end_time - timedelta(weeks=1)
    elif window == "1m":
        start_time = end_time - timedelta(days=30)
    elif window == "ytd":
        start_time = datetime(end_time.year, 1, 1)
    else:
        start_time = end_time - timedelta(days=7)
    
    try:
        # Query equity points
        with get_session() as session:
            equity_points = session.execute(
                select(EquityPoint)
                .where(EquityPoint.ts >= start_time)
                .where(EquityPoint.ts <= end_time)
                .order_by(EquityPoint.ts)
            ).scalars().all()
            
            if not equity_points:
                await interaction.followup.send(
                    f"No equity data found for {window}",
                    ephemeral=True
                )
                return
            
            # Calculate P&L and drawdown
            timestamps = [p.ts for p in equity_points]
            equity_values = [p.equity for p in equity_points]
            
            initial_equity = equity_values[0]
            final_equity = equity_values[-1]
            pnl = final_equity - initial_equity
            pnl_pct = (pnl / initial_equity) * 100 if initial_equity > 0 else 0
            
            # Calculate max drawdown
            peak = equity_values[0]
            max_dd = 0.0
            
            for equity in equity_values:
                if equity > peak:
                    peak = equity
                dd = ((equity - peak) / peak) * 100 if peak > 0 else 0
                if dd < max_dd:
                    max_dd = dd
            
            # Create embed
            color = discord.Color.green() if pnl >= 0 else discord.Color.red()
            
            embed = discord.Embed(
                title=f"📊 P&L Report ({window.upper()})",
                color=color,
                timestamp=datetime.utcnow()
            )
            
            embed.add_field(name="Initial Equity", value=f"${initial_equity:,.2f}", inline=True)
            embed.add_field(name="Final Equity", value=f"${final_equity:,.2f}", inline=True)
            embed.add_field(name="P&L", value=f"${pnl:+,.2f} ({pnl_pct:+.2f}%)", inline=True)
            embed.add_field(name="Max Drawdown", value=f"{max_dd:.2f}%", inline=True)
            embed.add_field(name="Data Points", value=str(len(equity_points)), inline=True)
            
            # Create chart
            chart_buf = create_equity_chart(timestamps, equity_values, f"Equity Curve ({window})")
            
            if chart_buf:
                file = discord.File(chart_buf, filename="equity.png")
                embed.set_image(url="attachment://equity.png")
                await interaction.followup.send(embed=embed, file=file, ephemeral=True)
            else:
                await interaction.followup.send(embed=embed, ephemeral=True)
            
    except Exception as e:
        logger.exception("PnL command error")
        await interaction.followup.send(
            embed=create_error_embed("Failed to calculate P&L", str(e)),
            ephemeral=True
        )


@bot.tree.command(name="trades", description="Show recent trades")
@app_commands.describe(
    limit="Number of trades to show (max 50)",
    symbol="Filter by symbol (optional)"
)
async def trades(interaction: discord.Interaction, limit: int = 20, symbol: Optional[str] = None):
    """Show recent trades."""
    if not check_guild(interaction):
        await interaction.response.send_message(
            "❌ This command can only be used in the configured guild.",
            ephemeral=True
        )
        return
    
    await interaction.response.defer(ephemeral=True)
    
    limit = min(limit, 50)  # Cap at 50
    
    try:
        with get_session() as session:
            query = select(Fill).order_by(Fill.ts.desc()).limit(limit)
            
            if symbol:
                query = query.where(Fill.symbol == symbol.upper())
            
            fills = session.execute(query).scalars().all()
            
            if not fills:
                msg = f"No trades found"
                if symbol:
                    msg += f" for {symbol}"
                await interaction.followup.send(msg, ephemeral=True)
                return
            
            # Convert to dict format for embed
            trades_dict = [
                {
                    "ts": f.ts,
                    "symbol": f.symbol,
                    "side": "BUY" if f.qty > 0 else "SELL",
                    "qty": abs(f.qty),
                    "price": f.price,
                    "pnl": 0  # TODO: Calculate PnL if needed
                }
                for f in fills
            ]
            
            embed = create_trade_table_embed(trades_dict, limit)
            await interaction.followup.send(embed=embed, ephemeral=True)
            
    except Exception as e:
        logger.exception("Trades command error")
        await interaction.followup.send(
            embed=create_error_embed("Failed to fetch trades", str(e)),
            ephemeral=True
        )


@bot.tree.command(name="halt", description="Stop paper trading loop")
async def halt(interaction: discord.Interaction):
    """Set halt flag for paper trading loop."""
    if not check_guild(interaction):
        await interaction.response.send_message(
            "❌ This command can only be used in the configured guild.",
            ephemeral=True
        )
        return
    
    bot.paper_loop_flag = True
    
    embed = create_success_embed(
        "Paper Loop Halted",
        "The halt flag has been set. Paper trading loop will stop on next check."
    )
    
    await interaction.response.send_message(embed=embed, ephemeral=True)
    logger.info("Halt flag set via Discord command")


async def main():
    """Main bot entry point."""
    # Validate configuration
    if not settings.DISCORD_BOT_TOKEN:
        logger.error("DISCORD_BOT_TOKEN not configured")
        return 1
    
    if settings.RUN_MODE != "paper":
        logger.error("Bot only runs in paper mode (RUN_MODE=paper)")
        return 1
    
    # Validate Discord settings
    settings.require_for_paper()  # Warn but don't fail
    
    logger.info("Starting Discord bot for paper trading...")
    logger.info(f"Guild ID: {settings.DISCORD_GUILD_ID or 'Not set (commands global)'}")
    logger.info(f"Reports Channel: {settings.DISCORD_REPORTS_CHANNEL_ID or 'Not set'}")
    
    try:
        await bot.start(settings.DISCORD_BOT_TOKEN)
    except discord.LoginFailure:
        logger.error("Invalid Discord bot token")
        return 1
    except Exception as e:
        logger.exception("Bot error")
        return 1
    finally:
        await bot.close()
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
