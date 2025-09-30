"""
Discord bot reporting helpers and chart generation.
"""

import io
from typing import Dict, Any, List, Optional
from datetime import datetime
import discord
from loguru import logger

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available, charts will be disabled")


def create_performance_embed(
    title: str,
    metrics: Dict[str, Any],
    color: discord.Color = discord.Color.blue()
) -> discord.Embed:
    """
    Create performance metrics embed.
    
    Args:
        title: Embed title
        metrics: Dictionary of metrics
        color: Embed color
        
    Returns:
        Discord embed
    """
    embed = discord.Embed(
        title=title,
        color=color,
        timestamp=datetime.utcnow()
    )
    
    # Add fields
    if "total_return" in metrics:
        embed.add_field(
            name="Total Return",
            value=f"{metrics['total_return']:.2%}",
            inline=True
        )
    
    if "sharpe" in metrics:
        embed.add_field(
            name="Sharpe Ratio",
            value=f"{metrics['sharpe']:.2f}",
            inline=True
        )
    
    if "sortino" in metrics:
        embed.add_field(
            name="Sortino Ratio",
            value=f"{metrics['sortino']:.2f}",
            inline=True
        )
    
    if "max_dd" in metrics:
        embed.add_field(
            name="Max Drawdown",
            value=f"{metrics['max_dd']:.2%}",
            inline=True
        )
    
    if "win_rate" in metrics:
        embed.add_field(
            name="Win Rate",
            value=f"{metrics['win_rate']:.1%}",
            inline=True
        )
    
    if "total_trades" in metrics:
        embed.add_field(
            name="Total Trades",
            value=f"{metrics['total_trades']:,}",
            inline=True
        )
    
    return embed


def create_trade_table_embed(
    trades: List[Dict[str, Any]],
    limit: int = 20
) -> discord.Embed:
    """
    Create embed with trade table.
    
    Args:
        trades: List of trade dictionaries
        limit: Maximum number of trades to show
        
    Returns:
        Discord embed
    """
    embed = discord.Embed(
        title=f"Recent Trades (Last {min(len(trades), limit)})",
        color=discord.Color.green(),
        timestamp=datetime.utcnow()
    )
    
    if not trades:
        embed.description = "No trades found"
        return embed
    
    # Format trades into table
    lines = []
    for trade in trades[:limit]:
        ts = trade.get('ts', 'N/A')
        if isinstance(ts, datetime):
            ts = ts.strftime('%m/%d %H:%M')
        
        symbol = trade.get('symbol', 'N/A')
        side = trade.get('side', 'N/A')
        qty = trade.get('qty', 0)
        price = trade.get('price', 0)
        pnl = trade.get('pnl', 0)
        
        pnl_emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
        
        line = f"{pnl_emoji} `{ts}` {symbol} {side.upper()} {qty:.2f} @ ${price:.2f}"
        if pnl != 0:
            line += f" PnL: ${pnl:.2f}"
        
        lines.append(line)
    
    embed.description = "\n".join(lines)
    
    return embed


def create_equity_chart(
    timestamps: List[datetime],
    equity_values: List[float],
    title: str = "Equity Curve"
) -> Optional[io.BytesIO]:
    """
    Create equity curve chart as PNG bytes.
    
    Args:
        timestamps: List of timestamps
        equity_values: List of equity values
        title: Chart title
        
    Returns:
        BytesIO object with PNG data, or None if matplotlib unavailable
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Cannot create chart - matplotlib not available")
        return None
    
    if not timestamps or not equity_values:
        logger.warning("Cannot create chart - empty data")
        return None
    
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        
        # Plot equity curve
        ax.plot(timestamps, equity_values, linewidth=2, color='#2E86AB')
        ax.fill_between(timestamps, equity_values, alpha=0.3, color='#2E86AB')
        
        # Formatting
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Equity ($)', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        
        # Format y-axis currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Tight layout
        plt.tight_layout()
        
        # Save to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        
        # Close figure
        plt.close(fig)
        
        return buf
        
    except Exception as e:
        logger.error(f"Failed to create equity chart: {e}")
        return None


def create_error_embed(
    error_message: str,
    details: Optional[str] = None
) -> discord.Embed:
    """
    Create error message embed.
    
    Args:
        error_message: Main error message
        details: Optional details
        
    Returns:
        Discord embed
    """
    embed = discord.Embed(
        title="❌ Error",
        description=error_message,
        color=discord.Color.red(),
        timestamp=datetime.utcnow()
    )
    
    if details:
        embed.add_field(name="Details", value=details, inline=False)
    
    return embed


def create_success_embed(
    title: str,
    message: str,
    fields: Optional[Dict[str, str]] = None
) -> discord.Embed:
    """
    Create success message embed.
    
    Args:
        title: Embed title
        message: Main message
        fields: Optional additional fields
        
    Returns:
        Discord embed
    """
    embed = discord.Embed(
        title=f"✅ {title}",
        description=message,
        color=discord.Color.green(),
        timestamp=datetime.utcnow()
    )
    
    if fields:
        for name, value in fields.items():
            embed.add_field(name=name, value=value, inline=True)
    
    return embed
