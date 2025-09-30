"""
Discord bot for quant trading control.
Provides slash commands for mobile-first trading control.
"""

import asyncio
import os
from datetime import datetime, timedelta

import discord
import requests
from discord import app_commands
from discord.ext import commands
from loguru import logger

# Configuration
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
ALLOWED_USER_IDS = [int(uid.strip()) for uid in os.getenv("ALLOWED_USER_IDS", "").split(",") if uid.strip()]
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
ADMIN_API_BASE = os.getenv("ADMIN_API_BASE", "http://localhost:8080")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")


class QuantBot(commands.Bot):
    """Discord bot for quant trading control."""

    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True

        super().__init__(
            command_prefix="!",
            intents=intents,
            description="Quant Trading Bot Control"
        )

    async def setup_hook(self):
        """Called when bot is starting up."""
        logger.info("Bot is starting up...")

        # Sync commands globally (can take up to 1 hour to propagate)
        try:
            synced = await self.tree.sync()
            logger.info(f"Synced {len(synced)} command(s)")
        except Exception as e:
            logger.error(f"Failed to sync commands: {e}")

    async def on_ready(self):
        """Called when bot is ready."""
        logger.info(f"{self.user} has connected to Discord!")
        logger.info(f"Bot is in {len(self.guilds)} guilds")

        # Set activity
        activity = discord.Activity(
            type=discord.ActivityType.watching,
            name="markets 📈"
        )
        await self.change_presence(activity=activity)


bot = QuantBot()


def check_permissions(interaction: discord.Interaction) -> bool:
    """Check if user is authorized to use bot commands."""
    if not ALLOWED_USER_IDS:
        # If no user IDs configured, allow all (not recommended for production)
        logger.warning("No ALLOWED_USER_IDS configured - allowing all users")
        return True

    user_id = interaction.user.id
    if user_id not in ALLOWED_USER_IDS:
        logger.warning(f"Unauthorized user {interaction.user} ({user_id}) attempted command")
        return False

    return True


async def make_api_request(endpoint: str, method: str = "GET", data: dict | None = None) -> dict | None:
    """Make request to Admin API with proper error handling."""
    url = f"{ADMIN_API_BASE.rstrip('/')}/{endpoint.lstrip('/')}"

    headers = {"Content-Type": "application/json"}
    if ADMIN_TOKEN and method != "GET":
        headers["Authorization"] = f"Bearer {ADMIN_TOKEN}"

    try:
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers, timeout=10)
        else:
            raise ValueError(f"Unsupported method: {method}")

        if response.status_code == 200:
            return response.json()
        if response.status_code == 401:
            logger.error("Admin API authentication failed")
            return {"error": "Authentication failed"}
        logger.error(f"Admin API returned {response.status_code}: {response.text}")
        return {"error": f"API error: {response.status_code}"}

    except requests.exceptions.ConnectTimeout:
        logger.error("Admin API connection timeout")
        return {"error": "Connection timeout"}
    except requests.exceptions.ConnectionError:
        logger.error("Failed to connect to Admin API")
        return {"error": "Connection failed"}
    except Exception as e:
        logger.error(f"API request error: {e}")
        return {"error": str(e)}


@bot.tree.command(name="status", description="Get trading bot status")
async def status(interaction: discord.Interaction):
    """Get current trading status."""
    if not check_permissions(interaction):
        await interaction.response.send_message("❌ You don't have permission to use this bot.", ephemeral=True)
        return

    await interaction.response.defer()

    try:
        # Get status from Admin API
        status_data = await make_api_request("status")

        if not status_data or "error" in status_data:
            error_msg = status_data.get("error", "Unknown error") if status_data else "No response"
            await interaction.followup.send(f"❌ Failed to get status: {error_msg}")
            return

        # Format status message
        trading_status = "⏸️ PAUSED" if status_data.get("trading_paused") else "▶️ ACTIVE"
        kill_switch = "🚨 ACTIVE" if status_data.get("kill_switch_active") else "✅ INACTIVE"

        embed = discord.Embed(
            title="🤖 Quant Bot Status",
            color=discord.Color.red() if status_data.get("kill_switch_active") else discord.Color.green(),
            timestamp=datetime.utcnow()
        )

        embed.add_field(name="Trading", value=trading_status, inline=True)
        embed.add_field(name="Kill Switch", value=kill_switch, inline=True)
        embed.add_field(name="Regime", value=status_data.get("current_regime", "unknown").title(), inline=True)

        embed.add_field(
            name="Strategies",
            value=f"{status_data.get('active_strategies', 0)}/{status_data.get('total_strategies', 0)} active",
            inline=True
        )
        embed.add_field(
            name="Equity",
            value=f"${status_data.get('equity', 0):,.2f}",
            inline=True
        )
        embed.add_field(
            name="Drawdown",
            value=f"{status_data.get('drawdown_pct', 0):.2f}%",
            inline=True
        )

        embed.add_field(
            name="Total Trades",
            value=f"{status_data.get('total_trades', 0):,}",
            inline=True
        )
        embed.add_field(
            name="Win Rate",
            value=f"{status_data.get('win_rate', 0)*100:.1f}%",
            inline=True
        )
        embed.add_field(
            name="Total P&L",
            value=f"${status_data.get('total_pnl', 0):,.2f}",
            inline=True
        )

        if status_data.get("last_kill_reason"):
            embed.add_field(
                name="Last Kill Reason",
                value=status_data["last_kill_reason"],
                inline=False
            )

        await interaction.followup.send(embed=embed)

    except Exception as e:
        logger.error(f"Status command error: {e}")
        await interaction.followup.send(f"❌ Error getting status: {e}")


@bot.tree.command(name="toggle_strategy", description="Enable or disable a trading strategy")
@app_commands.describe(
    strategy="Strategy name",
    enabled="Enable (True) or disable (False) the strategy"
)
@app_commands.choices(strategy=[
    app_commands.Choice(name="Scalper Sigma", value="scalper_sigma"),
    app_commands.Choice(name="Trend Breakout", value="trend_breakout"),
    app_commands.Choice(name="Bull Mode", value="bull_mode"),
    app_commands.Choice(name="Market Neutral", value="market_neutral"),
    app_commands.Choice(name="Gamma Reversal", value="gamma_reversal"),
])
async def toggle_strategy(interaction: discord.Interaction, strategy: str, enabled: bool):
    """Toggle strategy on/off."""
    if not check_permissions(interaction):
        await interaction.response.send_message("❌ You don't have permission to use this bot.", ephemeral=True)
        return

    await interaction.response.defer()

    try:
        # Make API request
        result = await make_api_request(
            "toggle_strategy",
            method="POST",
            data={"name": strategy, "enabled": enabled}
        )

        if not result or "error" in result:
            error_msg = result.get("error", "Unknown error") if result else "No response"
            await interaction.followup.send(f"❌ Failed to toggle strategy: {error_msg}")
            return

        status_emoji = "✅" if enabled else "❌"
        status_text = "enabled" if enabled else "disabled"

        embed = discord.Embed(
            title=f"{status_emoji} Strategy {status_text.title()}",
            description=f"**{strategy.replace('_', ' ').title()}** has been **{status_text}**",
            color=discord.Color.green() if enabled else discord.Color.orange(),
            timestamp=datetime.utcnow()
        )

        await interaction.followup.send(embed=embed)

    except Exception as e:
        logger.error(f"Toggle strategy command error: {e}")
        await interaction.followup.send(f"❌ Error toggling strategy: {e}")


@bot.tree.command(name="risk", description="Update a risk parameter")
@app_commands.describe(
    parameter="Risk parameter to update",
    value="New value for the parameter"
)
@app_commands.choices(parameter=[
    app_commands.Choice(name="Per Trade Risk %", value="per_trade_risk_pct"),
    app_commands.Choice(name="Max Leverage", value="max_leverage"),
    app_commands.Choice(name="Single Name Max %", value="single_name_max_pct"),
    app_commands.Choice(name="Sector Max %", value="sector_max_pct"),
    app_commands.Choice(name="Crypto Max Gross %", value="crypto_max_gross_pct"),
    app_commands.Choice(name="Vol Target Annual", value="vol_target_ann"),
    app_commands.Choice(name="Kill Switch DD", value="kill_switch_dd"),
    app_commands.Choice(name="Day Stop DD", value="day_stop_dd"),
    app_commands.Choice(name="Week Stop DD", value="week_stop_dd"),
])
async def risk_set(interaction: discord.Interaction, parameter: str, value: float):
    """Update risk parameter."""
    if not check_permissions(interaction):
        await interaction.response.send_message("❌ You don't have permission to use this bot.", ephemeral=True)
        return

    await interaction.response.defer()

    try:
        # Make API request
        result = await make_api_request(
            "risk",
            method="POST",
            data={"key": parameter, "value": value}
        )

        if not result or "error" in result:
            error_msg = result.get("error", "Unknown error") if result else "No response"
            await interaction.followup.send(f"❌ Failed to update risk parameter: {error_msg}")
            return

        embed = discord.Embed(
            title="⚙️ Risk Parameter Updated",
            description=f"**{parameter.replace('_', ' ').title()}** set to **{value}**",
            color=discord.Color.blue(),
            timestamp=datetime.utcnow()
        )

        await interaction.followup.send(embed=embed)

    except Exception as e:
        logger.error(f"Risk command error: {e}")
        await interaction.followup.send(f"❌ Error updating risk parameter: {e}")


@bot.tree.command(name="kill_switch", description="Activate emergency kill switch")
@app_commands.describe(reason="Reason for activating kill switch")
async def kill_switch(interaction: discord.Interaction, reason: str | None = "Manual Discord kill switch"):
    """Activate kill switch."""
    if not check_permissions(interaction):
        await interaction.response.send_message("❌ You don't have permission to use this bot.", ephemeral=True)
        return

    await interaction.response.defer()

    try:
        # Make API request
        result = await make_api_request(
            "kill_switch",
            method="POST",
            data={"reason": reason}
        )

        if not result or "error" in result:
            error_msg = result.get("error", "Unknown error") if result else "No response"
            await interaction.followup.send(f"❌ Failed to activate kill switch: {error_msg}")
            return

        embed = discord.Embed(
            title="🚨 KILL SWITCH ACTIVATED",
            description=f"**All trading stopped**\n\nReason: {reason}",
            color=discord.Color.red(),
            timestamp=datetime.utcnow()
        )
        embed.add_field(
            name="⚠️ Next Steps",
            value="Use `/resume_trading` when ready to resume",
            inline=False
        )

        await interaction.followup.send(embed=embed)

    except Exception as e:
        logger.error(f"Kill switch command error: {e}")
        await interaction.followup.send(f"❌ Error activating kill switch: {e}")


@bot.tree.command(name="resume_trading", description="Resume trading (deactivate kill switch)")
async def resume_trading(interaction: discord.Interaction):
    """Resume trading."""
    if not check_permissions(interaction):
        await interaction.response.send_message("❌ You don't have permission to use this bot.", ephemeral=True)
        return

    await interaction.response.defer()

    try:
        # Make API request
        result = await make_api_request("resume", method="POST")

        if not result or "error" in result:
            error_msg = result.get("error", "Unknown error") if result else "No response"
            await interaction.followup.send(f"❌ Failed to resume trading: {error_msg}")
            return

        embed = discord.Embed(
            title="✅ Trading Resumed",
            description="Kill switch deactivated - trading resumed",
            color=discord.Color.green(),
            timestamp=datetime.utcnow()
        )

        await interaction.followup.send(embed=embed)

    except Exception as e:
        logger.error(f"Resume trading command error: {e}")
        await interaction.followup.send(f"❌ Error resuming trading: {e}")


@bot.tree.command(name="retrain_now", description="Trigger model retraining")
async def retrain_now(interaction: discord.Interaction):
    """Trigger model retraining."""
    if not check_permissions(interaction):
        await interaction.response.send_message("❌ You don't have permission to use this bot.", ephemeral=True)
        return

    await interaction.response.defer()

    try:
        # Make API request
        result = await make_api_request("retrain_now", method="POST")

        if not result:
            await interaction.followup.send("❌ Failed to trigger retraining: No response")
            return

        if result.get("success"):
            embed = discord.Embed(
                title="🧠 Model Retraining",
                description=result.get("message", "Retraining started"),
                color=discord.Color.blue(),
                timestamp=datetime.utcnow()
            )
        else:
            embed = discord.Embed(
                title="⚠️ Retraining Not Available",
                description=result.get("message", "Retraining not available in this environment"),
                color=discord.Color.orange(),
                timestamp=datetime.utcnow()
            )

        await interaction.followup.send(embed=embed)

    except Exception as e:
        logger.error(f"Retrain command error: {e}")
        await interaction.followup.send(f"❌ Error triggering retraining: {e}")


@bot.tree.command(name="export_trades", description="Export trades to CSV")
@app_commands.describe(
    symbol="Filter by symbol (optional)",
    days="Number of days to export (default: 30)"
)
async def export_trades(interaction: discord.Interaction, symbol: str | None = None, days: int = 30):
    """Export trades to CSV."""
    if not check_permissions(interaction):
        await interaction.response.send_message("❌ You don't have permission to use this bot.", ephemeral=True)
        return

    await interaction.response.defer()

    try:
        # Calculate date range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)

        # Build export URL
        export_url = f"{ADMIN_API_BASE}/trades/export.csv"
        params = {
            "start": start_time.isoformat(),
            "end": end_time.isoformat()
        }
        if symbol:
            params["symbol"] = symbol

        # Make request with auth
        headers = {}
        if ADMIN_TOKEN:
            headers["Authorization"] = f"Bearer {ADMIN_TOKEN}"

        response = requests.get(export_url, params=params, headers=headers, timeout=30)

        if response.status_code == 200:
            # Save to temporary file
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"trades_{symbol or 'all'}_{timestamp}.csv"

            with open(filename, "wb") as f:
                f.write(response.content)

            # Send file
            with open(filename, "rb") as f:
                discord_file = discord.File(f, filename=filename)

                embed = discord.Embed(
                    title="📊 Trades Export",
                    description=f"Exported trades for last {days} days",
                    color=discord.Color.blue(),
                    timestamp=datetime.utcnow()
                )
                if symbol:
                    embed.add_field(name="Symbol", value=symbol, inline=True)

                await interaction.followup.send(embed=embed, file=discord_file)

            # Clean up
            os.remove(filename)

        else:
            await interaction.followup.send(f"❌ Export failed: HTTP {response.status_code}")

    except Exception as e:
        logger.error(f"Export trades command error: {e}")
        await interaction.followup.send(f"❌ Error exporting trades: {e}")


@bot.event
async def on_command_error(ctx, error):
    """Handle command errors."""
    logger.error(f"Command error: {error}")

    if isinstance(error, commands.CommandNotFound):
        return  # Ignore unknown commands

    await ctx.send(f"❌ Error: {error}")


async def main():
    """Main bot entry point."""
    if not DISCORD_BOT_TOKEN:
        logger.error("DISCORD_BOT_TOKEN not configured")
        return

    if not ALLOWED_USER_IDS:
        logger.warning("ALLOWED_USER_IDS not configured - bot will accept commands from any user")

    if not ADMIN_TOKEN:
        logger.warning("ADMIN_TOKEN not configured - some commands may fail")

    logger.info("Starting Discord bot...")

    try:
        await bot.start(DISCORD_BOT_TOKEN)
    except discord.LoginFailure:
        logger.error("Invalid Discord bot token")
    except Exception as e:
        logger.error(f"Bot error: {e}")
    finally:
        await bot.close()


if __name__ == "__main__":
    asyncio.run(main())
