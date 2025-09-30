"""
SQLAlchemy 2.0 database models and session management.
Supports both PostgreSQL and SQLite with automatic fallback.
"""

import uuid
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import json

from sqlalchemy import (
    create_engine, String, Float, Integer, DateTime, JSON, Text,
    Index, select, func, text
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
from loguru import logger

from config.settings import settings


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class Run(Base):
    """Backtest or paper run metadata."""
    
    __tablename__ = "runs"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    mode: Mapped[str] = mapped_column(String(20), nullable=False)  # 'backtest' or 'paper'
    universe: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # CSV of symbols
    params: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)  # Run parameters
    kpis: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)  # Final KPIs
    
    __table_args__ = (
        Index('idx_runs_started_at', 'started_at'),
        Index('idx_runs_mode', 'mode'),
    )


class Order(Base):
    """Order records."""
    
    __tablename__ = "orders"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)  # FK to runs
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False)  # 'buy' or 'sell'
    qty: Mapped[float] = mapped_column(Float, nullable=False)
    price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False)  # 'pending', 'filled', 'cancelled'
    
    __table_args__ = (
        Index('idx_orders_run_id', 'run_id'),
        Index('idx_orders_ts', 'ts'),
        Index('idx_orders_symbol', 'symbol'),
    )


class Fill(Base):
    """Fill (execution) records."""
    
    __tablename__ = "fills"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    order_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    qty: Mapped[float] = mapped_column(Float, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    commission: Mapped[float] = mapped_column(Float, default=0.0)
    
    __table_args__ = (
        Index('idx_fills_run_id', 'run_id'),
        Index('idx_fills_ts', 'ts'),
        Index('idx_fills_symbol', 'symbol'),
    )


class EquityPoint(Base):
    """Equity curve points."""
    
    __tablename__ = "equity_points"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    equity: Mapped[float] = mapped_column(Float, nullable=False)
    
    __table_args__ = (
        Index('idx_equity_run_id', 'run_id'),
        Index('idx_equity_ts', 'ts'),
    )


class Metric(Base):
    """Custom metrics/KPIs over time."""
    
    __tablename__ = "metrics"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    
    __table_args__ = (
        Index('idx_metrics_run_id', 'run_id'),
        Index('idx_metrics_name', 'name'),
        Index('idx_metrics_ts', 'ts'),
    )


class ErrorLog(Base):
    """Error logging for debugging."""
    
    __tablename__ = "error_logs"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.utcnow())
    level: Mapped[str] = mapped_column(String(20), nullable=False)  # 'ERROR', 'WARNING'
    message: Mapped[str] = mapped_column(Text, nullable=False)
    meta: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    __table_args__ = (
        Index('idx_error_logs_run_id', 'run_id'),
        Index('idx_error_logs_ts', 'ts'),
        Index('idx_error_logs_level', 'level'),
    )


# Database engine and session management
_engine = None
_session_factory = None


def get_engine():
    """Get or create database engine."""
    global _engine
    
    if _engine is None:
        db_url = settings.db_url_or_sqlite()
        
        # Check for Neon pooler and warn
        if "-pooler" in db_url and "postgresql" in db_url:
            logger.warning(
                "⚠️  Detected Neon pooled connection URL. "
                "For migrations and schema changes, use direct connection URL instead. "
                "Proceeding with pooled connection for regular operations."
            )
        
        # Create directory for SQLite if needed
        if db_url.startswith("sqlite"):
            db_path = Path(db_url.replace("sqlite:///", ""))
            db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create engine with SQLAlchemy 2.0 settings
        _engine = create_engine(
            db_url,
            echo=False,
            future=True,
            pool_pre_ping=True,
            pool_size=5 if "postgresql" in db_url else 1,
            max_overflow=10 if "postgresql" in db_url else 0,
        )
        
        logger.info(f"Database engine created: {db_url.split('@')[-1] if '@' in db_url else 'SQLite'}")
    
    return _engine


def get_session() -> Session:
    """Get database session."""
    from sqlalchemy.orm import sessionmaker
    
    global _session_factory
    
    if _session_factory is None:
        engine = get_engine()
        _session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    
    return _session_factory()


def init_db() -> None:
    """Initialize database schema (create all tables)."""
    engine = get_engine()
    
    logger.info("Creating database tables...")
    Base.metadata.create_all(engine)
    logger.info("✅ Database tables created successfully")


def purge_simulation_data() -> None:
    """Purge all simulation/paper trading data (keeps schema)."""
    engine = get_engine()
    
    logger.warning("⚠️  Purging all simulation data...")
    
    with Session(engine) as session:
        # Truncate tables in reverse dependency order
        for table in [ErrorLog, Metric, EquityPoint, Fill, Order, Run]:
            session.execute(text(f"DELETE FROM {table.__tablename__}"))
        
        session.commit()
    
    logger.info("✅ Simulation data purged")


def dump_run_kpis(run_id: str) -> Optional[Dict[str, Any]]:
    """
    Dump KPIs and summary for a specific run.
    
    Args:
        run_id: Run UUID
        
    Returns:
        Dictionary with run details and KPIs, or None if not found
    """
    with get_session() as session:
        # Get run
        run = session.execute(
            select(Run).where(Run.id == run_id)
        ).scalar_one_or_none()
        
        if not run:
            logger.error(f"Run {run_id} not found")
            return None
        
        # Get order count
        order_count = session.execute(
            select(func.count(Order.id)).where(Order.run_id == run_id)
        ).scalar()
        
        # Get fill count
        fill_count = session.execute(
            select(func.count(Fill.id)).where(Fill.run_id == run_id)
        ).scalar()
        
        # Get equity points
        equity_count = session.execute(
            select(func.count(EquityPoint.id)).where(EquityPoint.run_id == run_id)
        ).scalar()
        
        # Get error count
        error_count = session.execute(
            select(func.count(ErrorLog.id)).where(ErrorLog.run_id == run_id)
        ).scalar()
        
        result = {
            "run_id": run.id,
            "mode": run.mode,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "ended_at": run.ended_at.isoformat() if run.ended_at else None,
            "universe": run.universe,
            "params": run.params,
            "kpis": run.kpis,
            "counts": {
                "orders": order_count,
                "fills": fill_count,
                "equity_points": equity_count,
                "errors": error_count,
            }
        }
        
        return result


# CLI interface
def main() -> int:
    """CLI entry point for database management."""
    parser = argparse.ArgumentParser(description="Database management CLI")
    parser.add_argument(
        "command",
        choices=["init", "purge-sim", "dump-kpis"],
        help="Command to execute"
    )
    parser.add_argument(
        "run_id",
        nargs="?",
        help="Run ID (required for dump-kpis)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.command == "init":
            init_db()
            print("✅ Database initialized")
            return 0
        
        elif args.command == "purge-sim":
            response = input("⚠️  This will delete all simulation data. Continue? (yes/no): ")
            if response.lower() == "yes":
                purge_simulation_data()
                print("✅ Simulation data purged")
                return 0
            else:
                print("Cancelled")
                return 1
        
        elif args.command == "dump-kpis":
            if not args.run_id:
                print("❌ Run ID required for dump-kpis")
                return 1
            
            result = dump_run_kpis(args.run_id)
            if result:
                print(json.dumps(result, indent=2))
                return 0
            else:
                print(f"❌ Run {args.run_id} not found")
                return 1
        
    except Exception as e:
        logger.exception("Database command failed")
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
