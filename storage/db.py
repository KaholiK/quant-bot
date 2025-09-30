"""
Database models and utilities using SQLAlchemy 2.0.
Supports PostgreSQL and SQLite with automatic fallback.
"""

import sys
import argparse
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from sqlalchemy import (
    create_engine, Column, String, Integer, Float, DateTime, 
    Text, ForeignKey, JSON, select
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session, relationship
from sqlalchemy.dialects.postgresql import UUID

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class Run(Base):
    """Backtest/paper trading run."""
    __tablename__ = 'runs'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    mode = Column(String(20), nullable=False)  # 'backtest' or 'paper'
    universe = Column(Text, nullable=True)  # CSV of symbols
    params = Column(JSON, nullable=True)  # Strategy parameters
    kpis = Column(JSON, nullable=True)  # Performance metrics
    
    # Relationships
    orders = relationship("Order", back_populates="run", cascade="all, delete-orphan")
    fills = relationship("Fill", back_populates="run", cascade="all, delete-orphan")
    equity_points = relationship("EquityPoint", back_populates="run", cascade="all, delete-orphan")
    metrics = relationship("Metric", back_populates="run", cascade="all, delete-orphan")
    error_logs = relationship("ErrorLog", back_populates="run", cascade="all, delete-orphan")


class Order(Base):
    """Order record."""
    __tablename__ = 'orders'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id = Column(String(36), ForeignKey('runs.id'), nullable=False)
    ts = Column(DateTime, nullable=False)
    symbol = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)  # 'buy' or 'sell'
    qty = Column(Float, nullable=False)
    price = Column(Float, nullable=True)
    status = Column(String(20), nullable=False)  # 'open', 'filled', 'cancelled'
    
    # Relationship
    run = relationship("Run", back_populates="orders")
    fills = relationship("Fill", back_populates="order", cascade="all, delete-orphan")


class Fill(Base):
    """Fill record."""
    __tablename__ = 'fills'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id = Column(String(36), ForeignKey('runs.id'), nullable=False)
    order_id = Column(String(36), ForeignKey('orders.id'), nullable=True)
    ts = Column(DateTime, nullable=False)
    symbol = Column(String(20), nullable=False)
    qty = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    commission = Column(Float, nullable=False, default=0.0)
    
    # Relationships
    run = relationship("Run", back_populates="fills")
    order = relationship("Order", back_populates="fills")


class EquityPoint(Base):
    """Equity curve point."""
    __tablename__ = 'equity_points'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id = Column(String(36), ForeignKey('runs.id'), nullable=False)
    ts = Column(DateTime, nullable=False)
    equity = Column(Float, nullable=False)
    
    # Relationship
    run = relationship("Run", back_populates="equity_points")


class Metric(Base):
    """Custom metric record."""
    __tablename__ = 'metrics'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id = Column(String(36), ForeignKey('runs.id'), nullable=False)
    ts = Column(DateTime, nullable=False)
    name = Column(String(50), nullable=False)
    value = Column(Float, nullable=False)
    
    # Relationship
    run = relationship("Run", back_populates="metrics")


class ErrorLog(Base):
    """Error log record."""
    __tablename__ = 'error_logs'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id = Column(String(36), ForeignKey('runs.id'), nullable=True)
    ts = Column(DateTime, nullable=False, default=datetime.utcnow)
    level = Column(String(20), nullable=False)  # 'ERROR', 'WARNING', 'CRITICAL'
    message = Column(Text, nullable=False)
    meta = Column(JSON, nullable=True)
    
    # Relationship
    run = relationship("Run", back_populates="error_logs")


# Database engine and session factory
_engine = None
_SessionLocal = None


def get_engine():
    """Get or create database engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        db_url = settings.db_url_or_sqlite()
        
        # Create parent directory for SQLite
        if db_url.startswith("sqlite:///"):
            db_path = db_url.replace("sqlite:///", "")
            db_dir = Path(db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
        
        # Log connection info (masked)
        if "postgresql" in db_url or "postgres" in db_url:
            print(f"🔌 Connecting to PostgreSQL database...")
            if "sslmode" not in db_url:
                print("💡 Tip: Add ?sslmode=require for Neon/managed Postgres")
        else:
            print(f"🔌 Using SQLite database at {db_path}")
        
        # Create engine with SQLAlchemy 2.0 style
        _engine = create_engine(
            db_url,
            pool_pre_ping=True,
            echo=False
        )
    
    return _engine


def get_session_factory():
    """Get or create session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine()
        _SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    return _SessionLocal


def init_db():
    """Initialize database tables."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    print("✅ Database tables created successfully")


def purge_sim_data():
    """Purge all simulation data (runs, orders, fills, etc.)."""
    engine = get_engine()
    SessionLocal = get_session_factory()
    
    with SessionLocal() as session:
        # Delete all runs (cascades to related tables)
        session.query(Run).delete()
        session.commit()
    
    print("✅ Simulation data purged")


def dump_kpis(run_id: str):
    """Dump KPIs for a specific run."""
    SessionLocal = get_session_factory()
    
    with SessionLocal() as session:
        run = session.execute(
            select(Run).where(Run.id == run_id)
        ).scalar_one_or_none()
        
        if not run:
            print(f"❌ Run not found: {run_id}")
            return
        
        print("=" * 60)
        print(f"📊 KPIs for Run: {run_id}")
        print("=" * 60)
        print(f"Mode: {run.mode}")
        print(f"Started: {run.started_at}")
        print(f"Ended: {run.ended_at or 'Running'}")
        print(f"Universe: {run.universe or 'N/A'}")
        print()
        
        if run.kpis:
            print("KPIs:")
            for key, value in run.kpis.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        else:
            print("No KPIs recorded")
        
        print()
        
        # Query equity points
        equity_points = session.execute(
            select(EquityPoint)
            .where(EquityPoint.run_id == run_id)
            .order_by(EquityPoint.ts)
        ).scalars().all()
        
        if equity_points:
            print(f"Equity curve: {len(equity_points)} points")
            print(f"  Start: ${equity_points[0].equity:,.2f}")
            print(f"  End: ${equity_points[-1].equity:,.2f}")
            pnl = equity_points[-1].equity - equity_points[0].equity
            pnl_pct = (pnl / equity_points[0].equity) * 100
            print(f"  P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
        
        print()
        
        # Query orders/fills
        orders = session.execute(
            select(Order).where(Order.run_id == run_id)
        ).scalars().all()
        
        fills = session.execute(
            select(Fill).where(Fill.run_id == run_id)
        ).scalars().all()
        
        print(f"Orders: {len(orders)}")
        print(f"Fills: {len(fills)}")
        
        print("=" * 60)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Database management CLI')
    parser.add_argument('--init', action='store_true', help='Initialize database tables')
    parser.add_argument('--purge-sim', action='store_true', help='Purge simulation data')
    parser.add_argument('--dump-kpis', metavar='RUN_ID', help='Dump KPIs for run')
    
    args = parser.parse_args()
    
    if args.init:
        init_db()
    elif args.purge_sim:
        confirm = input("⚠️  This will delete ALL simulation data. Continue? (yes/no): ")
        if confirm.lower() == 'yes':
            purge_sim_data()
        else:
            print("Cancelled")
    elif args.dump_kpis:
        dump_kpis(args.dump_kpis)
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
