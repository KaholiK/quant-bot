"""
Tests for storage/db module.
"""

import pytest
from datetime import datetime
from storage.db import (
    Base, Run, Order, Fill, EquityPoint, Metric, ErrorLog,
    get_engine, init_db
)
from sqlalchemy import select, create_engine
from sqlalchemy.orm import Session


@pytest.fixture
def test_engine():
    """Create in-memory SQLite engine for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def test_session(test_engine):
    """Create test session."""
    return Session(test_engine)


def test_create_run(test_session):
    """Test creating a Run record."""
    run = Run(
        started_at=datetime.utcnow(),
        mode="backtest",
        universe="SPY,AAPL",
        params={"interval": "1d"}
    )
    
    test_session.add(run)
    test_session.commit()
    
    # Query back
    result = test_session.execute(select(Run)).scalar_one()
    
    assert result.mode == "backtest"
    assert result.universe == "SPY,AAPL"
    assert result.params["interval"] == "1d"


def test_create_equity_point(test_session):
    """Test creating EquityPoint records."""
    run = Run(
        started_at=datetime.utcnow(),
        mode="paper"
    )
    test_session.add(run)
    test_session.commit()
    
    # Add equity points
    for i in range(5):
        point = EquityPoint(
            run_id=run.id,
            ts=datetime.utcnow(),
            equity=100000.0 + i * 1000
        )
        test_session.add(point)
    
    test_session.commit()
    
    # Query back
    points = test_session.execute(
        select(EquityPoint).where(EquityPoint.run_id == run.id)
    ).scalars().all()
    
    assert len(points) == 5
    assert points[0].equity == 100000.0


def test_create_order_and_fill(test_session):
    """Test creating Order and Fill records."""
    run = Run(
        started_at=datetime.utcnow(),
        mode="paper"
    )
    test_session.add(run)
    test_session.commit()
    
    # Create order
    order = Order(
        run_id=run.id,
        ts=datetime.utcnow(),
        symbol="SPY",
        side="buy",
        qty=100.0,
        price=450.0,
        status="filled"
    )
    test_session.add(order)
    test_session.commit()
    
    # Create fill
    fill = Fill(
        run_id=run.id,
        order_id=order.id,
        ts=datetime.utcnow(),
        symbol="SPY",
        qty=100.0,
        price=450.5,
        commission=1.0
    )
    test_session.add(fill)
    test_session.commit()
    
    # Query back
    orders = test_session.execute(
        select(Order).where(Order.run_id == run.id)
    ).scalars().all()
    
    fills = test_session.execute(
        select(Fill).where(Fill.run_id == run.id)
    ).scalars().all()
    
    assert len(orders) == 1
    assert len(fills) == 1
    assert orders[0].symbol == "SPY"
    assert fills[0].qty == 100.0


def test_create_metric(test_session):
    """Test creating Metric records."""
    run = Run(
        started_at=datetime.utcnow(),
        mode="backtest"
    )
    test_session.add(run)
    test_session.commit()
    
    # Add metrics
    metrics_data = [
        ("sharpe", 1.85),
        ("sortino", 2.10),
        ("max_dd", -0.15)
    ]
    
    for name, value in metrics_data:
        metric = Metric(
            run_id=run.id,
            ts=datetime.utcnow(),
            name=name,
            value=value
        )
        test_session.add(metric)
    
    test_session.commit()
    
    # Query back
    metrics = test_session.execute(
        select(Metric).where(Metric.run_id == run.id)
    ).scalars().all()
    
    assert len(metrics) == 3
    
    # Check specific metric
    sharpe = test_session.execute(
        select(Metric)
        .where(Metric.run_id == run.id)
        .where(Metric.name == "sharpe")
    ).scalar_one()
    
    assert sharpe.value == 1.85


def test_create_error_log(test_session):
    """Test creating ErrorLog records."""
    run = Run(
        started_at=datetime.utcnow(),
        mode="paper"
    )
    test_session.add(run)
    test_session.commit()
    
    # Add error
    error = ErrorLog(
        run_id=run.id,
        ts=datetime.utcnow(),
        level="ERROR",
        message="Test error message",
        meta={"traceback": "line 1\nline 2"}
    )
    test_session.add(error)
    test_session.commit()
    
    # Query back
    errors = test_session.execute(
        select(ErrorLog).where(ErrorLog.run_id == run.id)
    ).scalars().all()
    
    assert len(errors) == 1
    assert errors[0].level == "ERROR"
    assert errors[0].message == "Test error message"
    assert "traceback" in errors[0].meta


def test_run_kpis_json():
    """Test that KPIs can be stored as JSON."""
    run = Run(
        started_at=datetime.utcnow(),
        mode="backtest",
        kpis={
            "sharpe": 1.85,
            "sortino": 2.10,
            "max_dd": -0.15,
            "win_rate": 0.65
        }
    )
    
    assert run.kpis["sharpe"] == 1.85
    assert run.kpis["win_rate"] == 0.65


def test_database_init():
    """Test database initialization."""
    # Create in-memory engine
    engine = create_engine("sqlite:///:memory:", echo=False)
    
    # Initialize
    Base.metadata.create_all(engine)
    
    # Check tables exist
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    assert "runs" in tables
    assert "orders" in tables
    assert "fills" in tables
    assert "equity_points" in tables
    assert "metrics" in tables
    assert "error_logs" in tables
