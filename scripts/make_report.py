#!/usr/bin/env python3
"""
Daily report generator for quant trading bot.
Generates HTML/PDF reports with daily KPIs and performance metrics.
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from jinja2 import Environment, FileSystemLoader

    from storage.trades import get_trade_storage
    HAS_DEPS = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Install with: pip install pandas matplotlib seaborn jinja2")
    HAS_DEPS = False


def calculate_daily_kpis(trades_data: list[dict[str, Any]],
                        equity_data: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate daily KPIs from trade data."""
    if not trades_data:
        return {
            "total_pnl": 0.0,
            "num_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "turnover": 0.0,
            "avg_slippage_bps": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0
        }

    # Convert to DataFrame for easier analysis
    df_trades = pd.DataFrame(trades_data)
    df_trades["time"] = pd.to_datetime(df_trades["time"])

    # Basic statistics
    total_pnl = df_trades["pnl"].sum()
    num_trades = len(df_trades)

    # Win rate
    winners = df_trades["pnl"] > 0
    win_rate = winners.sum() / len(df_trades) if len(df_trades) > 0 else 0.0

    # Profit factor
    gross_profits = df_trades[winners]["pnl"].sum() if winners.any() else 0.0
    gross_losses = abs(df_trades[~winners]["pnl"].sum()) if (~winners).any() else 0.0
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else float("inf")

    # Turnover (total volume traded)
    turnover = (df_trades["qty"] * df_trades["avg_price"]).sum()

    # Average slippage
    avg_slippage_bps = df_trades["slippage_bps"].mean() if "slippage_bps" in df_trades.columns else 0.0

    # Drawdown from equity curve
    max_drawdown = 0.0
    if equity_data:
        df_equity = pd.DataFrame(equity_data)
        if "drawdown" in df_equity.columns:
            max_drawdown = df_equity["drawdown"].max()

    # Risk-adjusted returns (simplified)
    if equity_data and len(equity_data) > 1:
        df_equity = pd.DataFrame(equity_data)
        df_equity["time"] = pd.to_datetime(df_equity["ts"])
        df_equity = df_equity.sort_values("time")

        returns = df_equity["equity"].pct_change().dropna()
        if len(returns) > 0:
            mean_return = returns.mean()
            std_return = returns.std()

            # Sharpe ratio (annualized, assuming daily data)
            sharpe_ratio = (mean_return / std_return) * (252 ** 0.5) if std_return > 0 else 0.0

            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 0 else std_return
            sortino_ratio = (mean_return / downside_std) * (252 ** 0.5) if downside_std > 0 else 0.0
        else:
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
    else:
        sharpe_ratio = 0.0
        sortino_ratio = 0.0

    return {
        "total_pnl": total_pnl,
        "num_trades": num_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "turnover": turnover,
        "avg_slippage_bps": avg_slippage_bps,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio
    }


def create_plots(trades_data: list[dict[str, Any]],
                equity_data: list[dict[str, Any]],
                output_dir: Path) -> list[str]:
    """Create performance plots and return list of plot filenames."""
    plot_files = []

    if not trades_data and not equity_data:
        return plot_files

    plt.style.use("seaborn-v0_8")

    # 1. Equity curve
    if equity_data:
        df_equity = pd.DataFrame(equity_data)
        df_equity["time"] = pd.to_datetime(df_equity["ts"])
        df_equity = df_equity.sort_values("time")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Equity curve
        ax1.plot(df_equity["time"], df_equity["equity"], linewidth=2, color="blue")
        ax1.set_title("Equity Curve", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.grid(True, alpha=0.3)

        # Drawdown
        ax2.fill_between(df_equity["time"], 0, -df_equity["drawdown"] * 100,
                        color="red", alpha=0.7)
        ax2.set_title("Drawdown", fontsize=14, fontweight="bold")
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Date")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        equity_plot = output_dir / "equity_curve.png"
        plt.savefig(equity_plot, dpi=150, bbox_inches="tight")
        plt.close()
        plot_files.append("equity_curve.png")

    # 2. Daily P&L distribution
    if trades_data:
        df_trades = pd.DataFrame(trades_data)
        df_trades["time"] = pd.to_datetime(df_trades["time"])

        # Group by date for daily P&L
        daily_pnl = df_trades.groupby(df_trades["time"].dt.date)["pnl"].sum()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Daily P&L bar chart
        colors = ["green" if pnl >= 0 else "red" for pnl in daily_pnl.values]
        ax1.bar(range(len(daily_pnl)), daily_pnl.values, color=colors, alpha=0.7)
        ax1.set_title("Daily P&L", fontsize=14, fontweight="bold")
        ax1.set_ylabel("P&L ($)")
        ax1.set_xlabel("Trading Days")
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color="black", linestyle="-", alpha=0.5)

        # P&L distribution histogram
        ax2.hist(daily_pnl.values, bins=20, color="blue", alpha=0.7, edgecolor="black")
        ax2.set_title("Daily P&L Distribution", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Daily P&L ($)")
        ax2.set_ylabel("Frequency")
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=0, color="red", linestyle="--", alpha=0.7)

        plt.tight_layout()
        pnl_plot = output_dir / "daily_pnl.png"
        plt.savefig(pnl_plot, dpi=150, bbox_inches="tight")
        plt.close()
        plot_files.append("daily_pnl.png")

    return plot_files


def generate_html_report(kpis: dict[str, Any],
                        plot_files: list[str],
                        report_date: datetime) -> str:
    """Generate HTML report using Jinja2 template."""

    html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Quant Bot Daily Report - {{ report_date.strftime('%Y-%m-%d') }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; text-align: center; margin-bottom: 30px; }
        h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
        .kpi-card { background: #ecf0f1; padding: 15px; border-radius: 8px; text-align: center; }
        .kpi-value { font-size: 24px; font-weight: bold; color: #2980b9; }
        .kpi-label { color: #7f8c8d; margin-top: 5px; }
        .plot { text-align: center; margin: 20px 0; }
        .plot img { max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .positive { color: #27ae60; }
        .negative { color: #e74c3c; }
        .footer { text-align: center; color: #95a5a6; margin-top: 30px; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Quant Bot Daily Report</h1>
        <p style="text-align: center; color: #7f8c8d;">{{ report_date.strftime('%A, %B %d, %Y') }}</p>
        
        <h2>📊 Key Performance Indicators</h2>
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-value {{ 'positive' if kpis.total_pnl >= 0 else 'negative' }}">
                    ${{ "{:,.2f}".format(kpis.total_pnl) }}
                </div>
                <div class="kpi-label">Total P&L</div>
            </div>
            
            <div class="kpi-card">
                <div class="kpi-value">{{ kpis.num_trades }}</div>
                <div class="kpi-label">Total Trades</div>
            </div>
            
            <div class="kpi-card">
                <div class="kpi-value">{{ "{:.1%}".format(kpis.win_rate) }}</div>
                <div class="kpi-label">Win Rate</div>
            </div>
            
            <div class="kpi-card">
                <div class="kpi-value">{{ "{:.2f}".format(kpis.profit_factor) if kpis.profit_factor != float('inf') else '∞' }}</div>
                <div class="kpi-label">Profit Factor</div>
            </div>
            
            <div class="kpi-card">
                <div class="kpi-value negative">{{ "{:.2%}".format(kpis.max_drawdown) }}</div>
                <div class="kpi-label">Max Drawdown</div>
            </div>
            
            <div class="kpi-card">
                <div class="kpi-value">{{ "{:.2f}".format(kpis.sharpe_ratio) }}</div>
                <div class="kpi-label">Sharpe Ratio</div>
            </div>
            
            <div class="kpi-card">
                <div class="kpi-value">${{ "{:,.0f}".format(kpis.turnover) }}</div>
                <div class="kpi-label">Total Turnover</div>
            </div>
            
            <div class="kpi-card">
                <div class="kpi-value">{{ "{:.1f}".format(kpis.avg_slippage_bps) }}</div>
                <div class="kpi-label">Avg Slippage (bps)</div>
            </div>
        </div>
        
        {% if plot_files %}
        <h2>📈 Performance Charts</h2>
        {% for plot_file in plot_files %}
        <div class="plot">
            <img src="{{ plot_file }}" alt="Performance Chart">
        </div>
        {% endfor %}
        {% endif %}
        
        <div class="footer">
            Generated by Quant Bot on {{ datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC') }}
        </div>
    </div>
</body>
</html>
    """

    template = Environment(loader=FileSystemLoader(".")).from_string(html_template)
    return template.render(
        kpis=kpis,
        plot_files=plot_files,
        report_date=report_date,
        datetime=datetime
    )


def send_discord_notification(webhook_url: str, report_path: str, kpis: dict[str, Any]):
    """Send report notification to Discord webhook."""
    try:
        import requests

        # Create embed for Discord
        embed = {
            "title": "📊 Daily Trading Report",
            "description": f"Report generated for {datetime.now().strftime('%Y-%m-%d')}",
            "color": 0x00ff00 if kpis["total_pnl"] >= 0 else 0xff0000,
            "fields": [
                {"name": "Total P&L", "value": f"${kpis['total_pnl']:,.2f}", "inline": True},
                {"name": "Trades", "value": str(kpis["num_trades"]), "inline": True},
                {"name": "Win Rate", "value": f"{kpis['win_rate']:.1%}", "inline": True},
                {"name": "Profit Factor", "value": f"{kpis['profit_factor']:.2f}" if kpis["profit_factor"] != float("inf") else "∞", "inline": True},
                {"name": "Max DD", "value": f"{kpis['max_drawdown']:.2%}", "inline": True},
                {"name": "Sharpe", "value": f"{kpis['sharpe_ratio']:.2f}", "inline": True}
            ],
            "footer": {"text": "Quant Bot Daily Report"},
            "timestamp": datetime.utcnow().isoformat()
        }

        payload = {"embeds": [embed]}

        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()

        print("✅ Discord notification sent successfully")

    except Exception as e:
        print(f"❌ Failed to send Discord notification: {e}")


def main():
    """Main entry point."""
    if not HAS_DEPS:
        print("Missing required dependencies. Please install them first.")
        return 1

    parser = argparse.ArgumentParser(description="Generate daily trading report")
    parser.add_argument("--date", type=str, help="Report date (YYYY-MM-DD), defaults to today")
    parser.add_argument("--output-dir", type=str, default="reports/daily", help="Output directory")
    parser.add_argument("--db-path", type=str, default="data/trading.db", help="Database path")
    parser.add_argument("--days-back", type=int, default=30, help="Days of data to include")

    args = parser.parse_args()

    # Parse report date
    if args.date:
        report_date = datetime.strptime(args.date, "%Y-%m-%d")
    else:
        report_date = datetime.now()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize storage
    trade_storage = get_trade_storage(args.db_path)

    # Calculate date range
    end_date = report_date
    start_date = end_date - timedelta(days=args.days_back)

    print(f"Generating report for {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # Get data
    trades_data = trade_storage.list_trades(
        start_time=start_date.isoformat(),
        end_time=end_date.isoformat(),
        limit=10000
    )

    equity_data = trade_storage.get_equity_curve(
        start_time=start_date.isoformat(),
        end_time=end_date.isoformat()
    )

    print(f"Found {len(trades_data)} trades and {len(equity_data)} equity points")

    # Calculate KPIs
    kpis = calculate_daily_kpis(trades_data, equity_data)

    # Create plots
    plot_files = create_plots(trades_data, equity_data, output_dir)

    # Generate HTML report
    html_content = generate_html_report(kpis, plot_files, report_date)

    # Save HTML report
    html_path = output_dir / f"{report_date.strftime('%Y-%m-%d')}.html"
    with open(html_path, "w") as f:
        f.write(html_content)

    print(f"✅ HTML report saved to: {html_path}")

    # Try to generate PDF if weasyprint is available
    try:
        from weasyprint import HTML
        pdf_path = output_dir / f"{report_date.strftime('%Y-%m-%d')}.pdf"
        HTML(string=html_content, base_url=str(output_dir)).write_pdf(str(pdf_path))
        print(f"✅ PDF report saved to: {pdf_path}")
    except ImportError:
        print("ℹ️  Install weasyprint for PDF generation")
    except Exception as e:
        print(f"❌ Failed to generate PDF: {e}")

    # Send Discord notification if webhook is configured
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if webhook_url:
        send_discord_notification(webhook_url, str(html_path), kpis)

    print("📊 Report generation complete!")
    return 0


if __name__ == "__main__":
    exit(main())
