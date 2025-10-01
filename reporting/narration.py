"""
Optional OpenAI narration for trading summaries.
Falls back to concise text summary if API key not configured.
"""

from typing import Any

from loguru import logger

from config.settings import settings


def summarize(
    metrics: dict[str, Any],
    recent_trades: list[dict[str, Any]] | None = None,
    max_tokens: int = 300
) -> str:
    """
    Generate narrative summary of trading performance.
    Uses OpenAI if configured, otherwise returns concise fallback.
    
    Args:
        metrics: Dictionary of performance metrics (sharpe, sortino, max_dd, etc.)
        recent_trades: Optional list of recent trade dictionaries
        max_tokens: Maximum tokens for OpenAI response
        
    Returns:
        Narrative summary string
    """
    # If no OpenAI key, use fallback
    if not settings.has_openai():
        return _fallback_summary(metrics, recent_trades)

    # Try OpenAI
    try:
        return _openai_summary(metrics, recent_trades, max_tokens)
    except Exception as e:
        logger.warning(f"OpenAI summary failed, using fallback: {e}")
        return _fallback_summary(metrics, recent_trades)


def _fallback_summary(
    metrics: dict[str, Any],
    recent_trades: list[dict[str, Any]] | None = None
) -> str:
    """
    Generate concise fallback summary without OpenAI.
    
    Args:
        metrics: Performance metrics
        recent_trades: Recent trades
        
    Returns:
        Summary string
    """
    lines = []

    # Performance summary
    sharpe = metrics.get("sharpe", 0.0)
    sortino = metrics.get("sortino", 0.0)
    max_dd = metrics.get("max_dd", 0.0)
    total_return = metrics.get("total_return", 0.0)
    win_rate = metrics.get("win_rate", 0.0)

    lines.append("**Performance Summary**")
    lines.append(f"Total Return: {total_return:.2%}")
    lines.append(f"Sharpe Ratio: {sharpe:.2f} | Sortino Ratio: {sortino:.2f}")
    lines.append(f"Max Drawdown: {max_dd:.2%}")
    lines.append(f"Win Rate: {win_rate:.1%}")

    # Risk assessment
    if sharpe > 2.0:
        lines.append("\n**Assessment:** Strong risk-adjusted returns.")
    elif sharpe > 1.0:
        lines.append("\n**Assessment:** Solid performance with acceptable risk.")
    elif sharpe > 0:
        lines.append("\n**Assessment:** Positive returns but high volatility.")
    else:
        lines.append("\n**Assessment:** Underperforming - review strategy parameters.")

    # Drawdown warning
    if abs(max_dd) > 0.15:
        lines.append(f"⚠️  Significant drawdown ({abs(max_dd):.1%}) - consider risk controls.")

    # Trade summary
    if recent_trades and len(recent_trades) > 0:
        num_trades = len(recent_trades)
        winners = sum(1 for t in recent_trades if t.get("pnl", 0) > 0)
        lines.append(f"\n**Recent Activity:** {num_trades} trades, {winners} winners ({winners/num_trades*100:.1f}%)")

    return "\n".join(lines)


def _openai_summary(
    metrics: dict[str, Any],
    recent_trades: list[dict[str, Any]] | None = None,
    max_tokens: int = 300
) -> str:
    """
    Generate summary using OpenAI.
    
    Args:
        metrics: Performance metrics
        recent_trades: Recent trades
        max_tokens: Maximum response tokens
        
    Returns:
        AI-generated summary
    """
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai package not installed, using fallback")
        return _fallback_summary(metrics, recent_trades)

    # Prepare context
    context = f"""
Analyze this trading performance and provide a concise 2-paragraph summary.

Performance Metrics:
- Total Return: {metrics.get('total_return', 0):.2%}
- Sharpe Ratio: {metrics.get('sharpe', 0):.2f}
- Sortino Ratio: {metrics.get('sortino', 0):.2f}
- Max Drawdown: {metrics.get('max_dd', 0):.2%}
- Win Rate: {metrics.get('win_rate', 0):.1%}
- Total Trades: {metrics.get('total_trades', 0)}
"""

    if recent_trades and len(recent_trades) > 0:
        num_wins = sum(1 for t in recent_trades if t.get("pnl", 0) > 0)
        context += f"\nRecent trades: {len(recent_trades)} total, {num_wins} profitable\n"

    context += """
Provide:
1. Overall performance assessment with key drivers
2. Risk considerations and recommendations
"""

    try:
        client = OpenAI(api_key=settings.OPENAI_API_KEY)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a quantitative trading analyst. Provide concise, actionable insights."
                },
                {
                    "role": "user",
                    "content": context
                }
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )

        summary = response.choices[0].message.content.strip()
        logger.info("Generated OpenAI summary")

        return summary

    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        raise


def summarize_backtest(
    start_date: str,
    end_date: str,
    universe: str,
    kpis: dict[str, Any]
) -> str:
    """
    Generate backtest summary.
    
    Args:
        start_date: Start date string
        end_date: End date string
        universe: Trading universe
        kpis: Backtest KPIs
        
    Returns:
        Summary string
    """
    header = f"**Backtest Results** ({start_date} to {end_date})\n"
    header += f"Universe: {universe}\n\n"

    summary = summarize(kpis)

    return header + summary


def summarize_paper_run(
    duration_hours: float,
    kpis: dict[str, Any],
    trades: list[dict[str, Any]] | None = None
) -> str:
    """
    Generate paper trading run summary.
    
    Args:
        duration_hours: Run duration in hours
        kpis: Performance KPIs
        trades: List of trades
        
    Returns:
        Summary string
    """
    header = f"**Paper Trading Run** ({duration_hours:.1f} hours)\n\n"

    summary = summarize(kpis, trades)

    return header + summary
