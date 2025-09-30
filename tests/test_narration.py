"""
Tests for reporting/narration module.
"""

import pytest
from reporting.narration import (
    summarize,
    summarize_backtest,
    summarize_paper_run,
    _fallback_summary
)


def test_fallback_summary():
    """Test fallback summary generation without OpenAI."""
    metrics = {
        "total_return": 0.15,
        "sharpe": 1.85,
        "sortino": 2.10,
        "max_dd": -0.08,
        "win_rate": 0.65,
        "total_trades": 42
    }
    
    summary = _fallback_summary(metrics)
    
    # Check key elements are present
    assert "Performance Summary" in summary
    assert "15.00%" in summary  # Total return
    assert "1.85" in summary    # Sharpe
    assert "2.10" in summary    # Sortino
    assert "8.00%" in summary   # Max DD
    assert "65.0%" in summary   # Win rate


def test_fallback_summary_with_trades():
    """Test fallback summary with trade information."""
    metrics = {
        "total_return": 0.10,
        "sharpe": 1.5,
        "sortino": 1.8,
        "max_dd": -0.05,
        "win_rate": 0.6,
    }
    
    trades = [
        {"pnl": 100},
        {"pnl": -50},
        {"pnl": 75},
        {"pnl": 25},
    ]
    
    summary = _fallback_summary(metrics, trades)
    
    assert "Recent Activity" in summary
    assert "4 trades" in summary
    assert "3 winners" in summary  # 3 out of 4 profitable


def test_fallback_summary_assessments():
    """Test different performance assessments."""
    # Strong performance
    metrics = {
        "total_return": 0.30,
        "sharpe": 2.5,
        "sortino": 3.0,
        "max_dd": -0.05,
        "win_rate": 0.70
    }
    
    summary = _fallback_summary(metrics)
    assert "Strong risk-adjusted returns" in summary
    
    # Solid performance
    metrics["sharpe"] = 1.5
    summary = _fallback_summary(metrics)
    assert "Solid performance" in summary
    
    # Underperforming
    metrics["sharpe"] = -0.5
    summary = _fallback_summary(metrics)
    assert "Underperforming" in summary


def test_fallback_summary_drawdown_warning():
    """Test that large drawdown triggers warning."""
    metrics = {
        "total_return": 0.20,
        "sharpe": 1.5,
        "sortino": 1.8,
        "max_dd": -0.25,  # Large drawdown
        "win_rate": 0.6
    }
    
    summary = _fallback_summary(metrics)
    assert "⚠️" in summary
    assert "Significant drawdown" in summary


def test_summarize_uses_fallback_without_openai():
    """Test that summarize uses fallback when OpenAI not configured."""
    metrics = {
        "total_return": 0.12,
        "sharpe": 1.6,
        "sortino": 1.9,
        "max_dd": -0.10,
        "win_rate": 0.58
    }
    
    # Without OPENAI_API_KEY, should use fallback
    summary = summarize(metrics)
    
    assert "Performance Summary" in summary
    assert "12.00%" in summary


def test_summarize_backtest():
    """Test backtest summary generation."""
    kpis = {
        "total_return": 0.15,
        "sharpe": 1.85,
        "sortino": 2.10,
        "max_dd": -0.08,
        "win_rate": 0.65
    }
    
    summary = summarize_backtest(
        start_date="2024-01-01",
        end_date="2024-06-01",
        universe="SPY,AAPL,MSFT",
        kpis=kpis
    )
    
    # Check header information
    assert "Backtest Results" in summary
    assert "2024-01-01" in summary
    assert "2024-06-01" in summary
    assert "SPY,AAPL,MSFT" in summary
    
    # Check KPIs are included
    assert "15.00%" in summary


def test_summarize_paper_run():
    """Test paper run summary generation."""
    kpis = {
        "total_return": 0.05,
        "sharpe": 0.8,
        "sortino": 1.0,
        "max_dd": -0.03,
        "win_rate": 0.55,
        "total_trades": 10
    }
    
    trades = [
        {"pnl": 50},
        {"pnl": -25},
        {"pnl": 30}
    ]
    
    summary = summarize_paper_run(
        duration_hours=2.5,
        kpis=kpis,
        trades=trades
    )
    
    # Check header
    assert "Paper Trading Run" in summary
    assert "2.5 hours" in summary
    
    # Check trade info
    assert "3 trades" in summary


def test_empty_metrics():
    """Test handling of empty metrics."""
    metrics = {}
    
    # Should not crash
    summary = _fallback_summary(metrics)
    
    assert "Performance Summary" in summary
    # Should show 0.00% for missing metrics
    assert "0.00%" in summary or "0.0%" in summary
