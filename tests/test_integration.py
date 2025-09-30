"""
Integration tests for the full backtest workflow.
"""

import shutil
import tempfile
from datetime import datetime, timedelta

import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from data.cache_io import CacheIO
from storage.db import Base, EquityPoint, Run


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_db_engine():
    """Create test database engine."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    return engine


def create_sample_data(cache: CacheIO, symbol: str, asset: str = "equity"):
    """Create sample OHLCV data in cache."""
    # Generate 30 days of data
    dates = pd.date_range("2024-01-01", periods=30, freq="1D", tz="UTC")

    # Simple uptrend
    close_prices = 100 + pd.Series(range(30)) * 0.5

    df = pd.DataFrame({
        "ts": dates,
        "open": close_prices - 0.5,
        "high": close_prices + 1.0,
        "low": close_prices - 1.0,
        "close": close_prices,
        "volume": [1000000] * 30,
        "provider": ["test"] * 30
    })

    cache.save_parquet(df, asset=asset, symbol=symbol, interval="1d")
    return df


def test_cache_io_save_and_load(temp_cache_dir):
    """Test saving and loading data from cache."""
    cache = CacheIO(base_path=temp_cache_dir)

    # Create sample data
    df_original = create_sample_data(cache, "SPY", "equity")

    # Load it back
    df_loaded = cache.load_range(
        asset="equity",
        symbol="SPY",
        interval="1d",
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 31)
    )

    assert df_loaded is not None
    assert len(df_loaded) == 30
    assert df_loaded["close"].iloc[0] == 100.0
    assert df_loaded["close"].iloc[-1] == 114.5


def test_cache_coverage_check(temp_cache_dir):
    """Test cache coverage checking."""
    cache = CacheIO(base_path=temp_cache_dir)

    # Create sample data
    create_sample_data(cache, "SPY", "equity")

    # Check coverage (should be 100% for daily data)
    has_cov, ratio, actual, expected = cache.has_coverage(
        asset="equity",
        symbol="SPY",
        interval="1d",
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 30),
        min_coverage=0.95
    )

    assert has_cov is True
    assert ratio >= 0.95
    assert actual == 30


def test_database_run_lifecycle(test_db_engine):
    """Test creating and querying a complete run."""
    session = Session(test_db_engine)

    # Create run
    run = Run(
        started_at=datetime.utcnow(),
        mode="backtest",
        universe="SPY,AAPL",
        params={"interval": "1d", "initial_equity": 100000}
    )
    session.add(run)
    session.commit()

    # Add equity points
    base_equity = 100000
    for i in range(10):
        point = EquityPoint(
            run_id=run.id,
            ts=datetime.utcnow() + timedelta(hours=i),
            equity=base_equity + i * 100
        )
        session.add(point)

    session.commit()

    # Update run with KPIs
    run.ended_at = datetime.utcnow()
    run.kpis = {
        "total_return": 0.01,
        "sharpe": 1.5,
        "max_dd": -0.02
    }
    session.commit()

    # Query and verify
    retrieved_run = session.execute(
        select(Run).where(Run.id == run.id)
    ).scalar_one()

    assert retrieved_run.mode == "backtest"
    assert retrieved_run.universe == "SPY,AAPL"
    assert retrieved_run.kpis["sharpe"] == 1.5

    # Query equity points
    points = session.execute(
        select(EquityPoint)
        .where(EquityPoint.run_id == run.id)
        .order_by(EquityPoint.ts)
    ).scalars().all()

    assert len(points) == 10
    assert points[0].equity == 100000
    assert points[-1].equity == 100900


def test_end_to_end_simple_backtest(temp_cache_dir, test_db_engine):
    """Test a simple end-to-end backtest workflow."""
    # Setup
    cache = CacheIO(base_path=temp_cache_dir)
    session = Session(test_db_engine)

    # Create sample data for two symbols
    create_sample_data(cache, "SPY", "equity")
    create_sample_data(cache, "AAPL", "equity")

    # Verify data can be loaded
    spy_data = cache.load_range(
        asset="equity",
        symbol="SPY",
        interval="1d",
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 30)
    )

    aapl_data = cache.load_range(
        asset="equity",
        symbol="AAPL",
        interval="1d",
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 30)
    )

    assert spy_data is not None
    assert aapl_data is not None
    assert len(spy_data) == 30
    assert len(aapl_data) == 30

    # Simulate simple backtest: create run and log results
    run = Run(
        started_at=datetime.utcnow(),
        mode="backtest",
        universe="SPY,AAPL"
    )
    session.add(run)
    session.commit()

    # Calculate simple returns (just use the data trend)
    initial_equity = 100000
    final_equity = 100000 * (1 + (spy_data["close"].iloc[-1] - spy_data["close"].iloc[0]) / spy_data["close"].iloc[0])

    # Log equity curve
    for i in range(len(spy_data)):
        equity = initial_equity + (spy_data["close"].iloc[i] - spy_data["close"].iloc[0]) / spy_data["close"].iloc[0] * initial_equity
        point = EquityPoint(
            run_id=run.id,
            ts=spy_data["ts"].iloc[i],
            equity=equity
        )
        session.add(point)

    session.commit()

    # Update run with final results
    total_return = (final_equity - initial_equity) / initial_equity
    run.ended_at = datetime.utcnow()
    run.kpis = {
        "total_return": total_return,
        "final_equity": final_equity
    }
    session.commit()

    # Verify results can be queried
    result = session.execute(
        select(Run).where(Run.id == run.id)
    ).scalar_one()

    assert result.kpis["total_return"] > 0  # Should be profitable given uptrend

    equity_points = session.execute(
        select(EquityPoint).where(EquityPoint.run_id == run.id)
    ).scalars().all()

    assert len(equity_points) == 30
    assert equity_points[-1].equity > equity_points[0].equity  # Upward trend
