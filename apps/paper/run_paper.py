#!/usr/bin/env python3
"""
Simple paper trading runner.
Simulates live trading without real orders.
"""

import sys
import argparse
import uuid
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from loguru import logger
from rich.console import Console
from rich.live import Live
from rich.table import Table

from config.settings import settings
from storage.db import get_session, Run, EquityPoint, Order, Fill
from telemetry.wandb_utils import get_wandb
from reporting.narration import summarize_paper_run

console = Console()


class PaperTradingSimulator:
    """Simple paper trading simulator."""
    
    def __init__(self, initial_equity: float = 100000.0):
        """Initialize simulator."""
        self.initial_equity = initial_equity
        self.equity = initial_equity
        self.positions = {}  # symbol -> qty
        self.run_id = str(uuid.uuid4())
        self.trades = []
        self.equity_history = [(datetime.utcnow(), self.equity)]
        
        # Create run in DB
        self.run = Run(
            id=self.run_id,
            started_at=datetime.utcnow(),
            mode="paper",
            params={"initial_equity": initial_equity}
        )
        
        with get_session() as session:
            session.add(self.run)
            session.commit()
        
        logger.info(f"Paper run started: {self.run_id}")
    
    def simulate_tick(self) -> None:
        """Simulate one market tick."""
        # Simple random walk for equity (for demonstration)
        # In real implementation, this would fetch live prices and calculate portfolio value
        
        # Random return between -0.5% and +0.5%
        rand_return = np.random.uniform(-0.005, 0.005)
        self.equity *= (1 + rand_return)
        
        # Record equity point
        now = datetime.utcnow()
        self.equity_history.append((now, self.equity))
        
        # Save to DB every 10 ticks
        if len(self.equity_history) % 10 == 0:
            with get_session() as session:
                point = EquityPoint(
                    run_id=self.run_id,
                    ts=now,
                    equity=self.equity
                )
                session.add(point)
                session.commit()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current stats."""
        pnl = self.equity - self.initial_equity
        pnl_pct = (pnl / self.initial_equity) * 100
        
        # Calculate max drawdown
        equity_values = [e for _, e in self.equity_history]
        peak = max(equity_values)
        current_dd = ((self.equity - peak) / peak) * 100 if peak > 0 else 0
        
        return {
            "equity": self.equity,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "drawdown": current_dd,
            "num_ticks": len(self.equity_history),
            "num_trades": len(self.trades)
        }
    
    def finalize(self) -> Dict[str, Any]:
        """Finalize run and calculate KPIs."""
        # Save final equity
        with get_session() as session:
            point = EquityPoint(
                run_id=self.run_id,
                ts=datetime.utcnow(),
                equity=self.equity
            )
            session.add(point)
            
            # Update run
            run = session.get(Run, self.run_id)
            run.ended_at = datetime.utcnow()
            
            stats = self.get_stats()
            run.kpis = {
                "total_return": stats["pnl_pct"] / 100,
                "final_equity": stats["equity"],
                "pnl": stats["pnl"],
                "num_ticks": stats["num_ticks"],
                "total_trades": stats["num_trades"]
            }
            
            session.commit()
        
        return stats


def run_paper(hours: float = 1.0, tick_interval_secs: int = 5) -> str:
    """
    Run paper trading simulation.
    
    Args:
        hours: Duration in hours
        tick_interval_secs: Seconds between ticks
        
    Returns:
        Run ID
    """
    console.print(f"\n[bold cyan]Starting Paper Trading[/bold cyan]")
    console.print(f"Duration: {hours} hours")
    console.print(f"Tick Interval: {tick_interval_secs} seconds\n")
    
    sim = PaperTradingSimulator()
    
    # Calculate end time
    end_time = datetime.utcnow() + timedelta(hours=hours)
    
    # Import bot to check halt flag
    from ui.discord_bot.main import bot
    
    # Run simulation loop with live display
    try:
        with Live(console=console, refresh_per_second=1) as live:
            while datetime.utcnow() < end_time:
                # Check halt flag
                if bot.paper_loop_flag:
                    console.print("\n[yellow]⚠️  Halt flag detected, stopping...[/yellow]")
                    break
                
                # Simulate tick
                sim.simulate_tick()
                
                # Update display
                stats = sim.get_stats()
                
                table = Table(title="Paper Trading Status")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")
                
                table.add_row("Equity", f"${stats['equity']:,.2f}")
                table.add_row("P&L", f"${stats['pnl']:+,.2f} ({stats['pnl_pct']:+.2f}%)")
                table.add_row("Drawdown", f"{stats['drawdown']:.2f}%")
                table.add_row("Ticks", str(stats['num_ticks']))
                table.add_row("Trades", str(stats['num_trades']))
                
                remaining = (end_time - datetime.utcnow()).total_seconds()
                table.add_row("Time Remaining", f"{remaining/60:.1f} min")
                
                live.update(table)
                
                # Sleep
                time.sleep(tick_interval_secs)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Interrupted by user[/yellow]")
    
    # Finalize
    final_stats = sim.finalize()
    
    console.print(f"\n[bold green]Paper Run Complete[/bold green]")
    console.print(f"Final Equity: ${final_stats['equity']:,.2f}")
    console.print(f"P&L: ${final_stats['pnl']:+,.2f} ({final_stats['pnl_pct']:+.2f}%)")
    console.print(f"Ticks: {final_stats['num_ticks']}\n")
    
    # W&B logging
    wandb = get_wandb()
    if wandb.enabled:
        wandb.init_run(project="quantbot", tags=["paper"], config={
            "duration_hours": hours,
            "tick_interval_secs": tick_interval_secs
        })
        wandb.log_summary(final_stats)
        wandb.finish()
    
    # Generate narration
    kpis = {
        "total_return": final_stats["pnl_pct"] / 100,
        "sharpe": 0.0,  # Not calculated in simple sim
        "sortino": 0.0,
        "max_dd": final_stats["drawdown"] / 100,
        "win_rate": 0.0,
        "total_trades": final_stats["num_trades"]
    }
    
    narration = summarize_paper_run(
        duration_hours=hours,
        kpis=kpis,
        trades=sim.trades
    )
    
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(narration)
    console.print()
    
    return sim.run_id


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run paper trading simulation")
    parser.add_argument("--hours", type=float, default=1.0, help="Duration in hours (default: 1)")
    parser.add_argument("--tick-interval", type=int, default=5, help="Seconds between ticks (default: 5)")
    
    args = parser.parse_args()
    
    if args.hours <= 0:
        console.print("[red]❌ Hours must be positive[/red]")
        return 1
    
    if args.tick_interval < 1:
        console.print("[red]❌ Tick interval must be >= 1 second[/red]")
        return 1
    
    # Run paper trading
    try:
        run_id = run_paper(args.hours, args.tick_interval)
        console.print(f"[green]✅ Paper run complete. Run ID: {run_id}[/green]")
        return 0
    except Exception as e:
        console.print(f"[red]❌ Paper run failed: {e}[/red]")
        logger.exception("Paper run failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
