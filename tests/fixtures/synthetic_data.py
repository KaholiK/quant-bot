"""
Synthetic data generators for testing.
"""
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

def generate_ohlcv(
    symbol: str = "TEST",
    periods: int = 100,
    trend: str = "random",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    np.random.seed(seed)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    timestamps = [start + timedelta(hours=i) for i in range(periods)]
    
    base_price = 100.0
    returns = np.random.normal(0.001 if trend == "up" else 0.0, 0.02, periods)
    close_prices = base_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        "ts": timestamps,
        "open": close_prices * (1 + np.random.uniform(-0.01, 0.01, periods)),
        "high": close_prices * (1 + abs(np.random.normal(0, 0.02, periods))),
        "low": close_prices * (1 - abs(np.random.normal(0, 0.02, periods))),
        "close": close_prices,
        "volume": np.random.randint(1_000_000, 2_000_000, periods),
        "provider": ["synthetic"] * periods,
    })
    
    return df
