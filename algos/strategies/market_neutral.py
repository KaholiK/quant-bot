"""
Market Neutral Strategy - Pairs trading with cointegration and beta-neutral sizing.
Uses statistical arbitrage between cointegrated pairs with mean-reverting spreads.
"""

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import coint


class MarketNeutralStrategy:
    """
    Market neutral pairs trading strategy.
    
    Strategy characteristics:
    - Identifies cointegrated pairs using Engle-Granger test
    - Enters positions when spread z-score >= 2.0
    - Exits when spread z-score <= 0.0  
    - Beta-neutral position sizing
    - Market-neutral portfolio construction
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize market neutral strategy."""
        self.config = config
        self.strategy_config = config["strategies"]["market_neutral"]

        # Strategy parameters
        self.enabled = self.strategy_config.get("enabled", True)
        self.z_score_entry = self.strategy_config.get("z_score_entry", 2.0)
        self.z_score_exit = self.strategy_config.get("z_score_exit", 0.0)
        self.cointegration_window = self.strategy_config.get("cointegration_window", 252)

        # Pair selection parameters
        self.min_correlation = 0.7
        self.max_pairs = 10
        self.rebalance_frequency = "weekly"  # How often to update pairs
        self.significance_level = 0.05

        # State tracking
        self.active_pairs: dict[str, dict] = {}
        self.cointegrated_pairs: list[tuple[str, str, float]] = []
        self.pair_spreads: dict[str, pd.Series] = {}
        self.hedge_ratios: dict[str, float] = {}
        self.position_history: list[dict] = []
        self.last_rebalance = None

        logger.info(f"Market Neutral strategy initialized: enabled={self.enabled}")

    def find_pairs(self,
                   universe_data: dict[str, pd.DataFrame],
                   min_observations: int = None) -> list[tuple[str, str, float, float]]:
        """
        Find cointegrated pairs from universe.
        
        Args:
            universe_data: Dictionary of symbol -> price DataFrame
            min_observations: Minimum observations required
            
        Returns:
            List of (symbol1, symbol2, p_value, correlation) tuples
        """
        if min_observations is None:
            min_observations = self.cointegration_window

        symbols = list(universe_data.keys())
        pairs_tested = []
        cointegrated_pairs = []

        logger.info(f"Testing {len(symbols)} symbols for cointegration")

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]

                # Skip crypto-equity pairs (different market dynamics)
                crypto_symbols = {"BTCUSD", "ETHUSD"}
                if (symbol1 in crypto_symbols) != (symbol2 in crypto_symbols):
                    continue

                try:
                    data1 = universe_data[symbol1]
                    data2 = universe_data[symbol2]

                    # Align data
                    common_index = data1.index.intersection(data2.index)
                    if len(common_index) < min_observations:
                        continue

                    prices1 = data1.loc[common_index]["close"]
                    prices2 = data2.loc[common_index]["close"]

                    # Check correlation first (screening)
                    correlation = prices1.corr(prices2)
                    if abs(correlation) < self.min_correlation:
                        continue

                    # Test for cointegration
                    score, p_value, _ = coint(prices1, prices2)

                    pairs_tested.append((symbol1, symbol2, p_value, correlation))

                    if p_value < self.significance_level:
                        cointegrated_pairs.append((symbol1, symbol2, p_value, correlation))
                        logger.info(f"Cointegrated pair found: {symbol1}-{symbol2} "
                                   f"(p_value={p_value:.4f}, corr={correlation:.3f})")

                except Exception as e:
                    logger.debug(f"Cointegration test failed for {symbol1}-{symbol2}: {e}")
                    continue

        logger.info(f"Found {len(cointegrated_pairs)} cointegrated pairs from {len(pairs_tested)} tests")

        # Sort by p-value (most significant first)
        cointegrated_pairs.sort(key=lambda x: x[2])

        # Limit number of pairs
        return cointegrated_pairs[:self.max_pairs]

    def calculate_spread_and_hedge_ratio(self,
                                       prices1: pd.Series,
                                       prices2: pd.Series) -> tuple[pd.Series, float]:
        """
        Calculate spread and hedge ratio between two price series.
        
        Args:
            prices1: Price series for asset 1
            prices2: Price series for asset 2
            
        Returns:
            Tuple of (spread_series, hedge_ratio)
        """
        # Align series
        aligned1, aligned2 = prices1.align(prices2, join="inner")

        if len(aligned1) == 0:
            return pd.Series(dtype=float), 0.0

        # Calculate hedge ratio using OLS regression
        X = aligned2.values.reshape(-1, 1)
        y = aligned1.values

        reg = LinearRegression()
        reg.fit(X, y)

        hedge_ratio = reg.coef_[0]

        # Calculate spread: Asset1 - hedge_ratio * Asset2
        spread = aligned1 - hedge_ratio * aligned2

        return spread, hedge_ratio

    def calculate_signals(self,
                         universe_data: dict[str, pd.DataFrame],
                         current_time: datetime | None = None) -> dict[str, Any]:
        """
        Calculate pair trading signals.
        
        Args:
            universe_data: Dictionary of symbol -> price DataFrame
            current_time: Current timestamp
            
        Returns:
            Dictionary with signals for all pairs
        """
        if not self.enabled:
            return {"pairs": {}, "total_pairs": 0}

        if current_time is None:
            current_time = datetime.now()

        # Update pairs if needed
        should_rebalance = (
            self.last_rebalance is None or
            (current_time - self.last_rebalance).days >= 7  # Weekly rebalance
        )

        if should_rebalance:
            self.cointegrated_pairs = self.find_pairs(universe_data)
            self.last_rebalance = current_time

        pair_signals = {}

        for symbol1, symbol2, p_value, correlation in self.cointegrated_pairs:
            try:
                signal_info = self._calculate_pair_signal(
                    symbol1, symbol2, universe_data, p_value, correlation
                )
                pair_key = f"{symbol1}_{symbol2}"
                pair_signals[pair_key] = signal_info

            except Exception as e:
                logger.warning(f"Signal calculation failed for {symbol1}-{symbol2}: {e}")
                continue

        return {
            "pairs": pair_signals,
            "total_pairs": len(pair_signals),
            "timestamp": current_time
        }

    def _calculate_pair_signal(self,
                              symbol1: str,
                              symbol2: str,
                              universe_data: dict[str, pd.DataFrame],
                              p_value: float,
                              correlation: float) -> dict[str, Any]:
        """Calculate signal for a specific pair."""
        data1 = universe_data[symbol1]
        data2 = universe_data[symbol2]

        # Get latest prices
        current_price1 = data1["close"].iloc[-1]
        current_price2 = data2["close"].iloc[-1]

        # Calculate spread and hedge ratio
        lookback_data = min(len(data1), len(data2), self.cointegration_window)
        recent_prices1 = data1["close"].tail(lookback_data)
        recent_prices2 = data2["close"].tail(lookback_data)

        spread, hedge_ratio = self.calculate_spread_and_hedge_ratio(
            recent_prices1, recent_prices2
        )

        if len(spread) == 0:
            return {"signal": 0, "confidence": 0.0, "reason": "no_spread_data"}

        # Calculate z-score
        spread_mean = spread.mean()
        spread_std = spread.std()

        if spread_std <= 0:
            return {"signal": 0, "confidence": 0.0, "reason": "zero_spread_volatility"}

        current_spread = current_price1 - hedge_ratio * current_price2
        z_score = (current_spread - spread_mean) / spread_std

        # Generate signals
        signal = 0
        confidence = 0.0
        reason = ""

        pair_key = f"{symbol1}_{symbol2}"

        # Entry signals
        if abs(z_score) >= self.z_score_entry:
            if z_score > 0:
                # Spread is high: short symbol1, long symbol2
                signal = -1
                reason = f"spread_high_z_{z_score:.2f}"
            else:
                # Spread is low: long symbol1, short symbol2
                signal = 1
                reason = f"spread_low_z_{z_score:.2f}"

            confidence = min(1.0, abs(z_score) / 3.0)  # Cap at 3-sigma

        # Exit signals for existing positions
        elif pair_key in self.active_pairs and abs(z_score) <= self.z_score_exit:
            signal = 0  # Close position
            confidence = 1.0
            reason = f"spread_mean_reversion_z_{z_score:.2f}"

        # Calculate position sizes (beta-neutral)
        position_sizes = self._calculate_position_sizes(
            symbol1, symbol2, current_price1, current_price2, hedge_ratio, signal
        )

        # Store spread data
        self.pair_spreads[pair_key] = spread
        self.hedge_ratios[pair_key] = hedge_ratio

        return {
            "signal": signal,
            "confidence": confidence,
            "z_score": z_score,
            "spread": current_spread,
            "spread_mean": spread_mean,
            "spread_std": spread_std,
            "hedge_ratio": hedge_ratio,
            "position_sizes": position_sizes,
            "p_value": p_value,
            "correlation": correlation,
            "reason": reason,
            "symbol1": symbol1,
            "symbol2": symbol2,
            "current_price1": current_price1,
            "current_price2": current_price2
        }

    def _calculate_position_sizes(self,
                                 symbol1: str,
                                 symbol2: str,
                                 price1: float,
                                 price2: float,
                                 hedge_ratio: float,
                                 signal: int,
                                 base_notional: float = 10000.0) -> dict[str, float]:
        """
        Calculate beta-neutral position sizes.
        
        Args:
            symbol1, symbol2: Trading symbols
            price1, price2: Current prices
            hedge_ratio: Hedge ratio from regression
            signal: Trading signal (-1, 0, 1)
            base_notional: Base notional amount for sizing
            
        Returns:
            Dictionary with position sizes for each symbol
        """
        if signal == 0:
            return {symbol1: 0.0, symbol2: 0.0}

        # Calculate dollar-neutral positions
        if signal == 1:
            # Long symbol1, short symbol2
            size1 = base_notional / price1
            size2 = -(hedge_ratio * base_notional) / price2
        else:
            # Short symbol1, long symbol2
            size1 = -base_notional / price1
            size2 = (hedge_ratio * base_notional) / price2

        return {symbol1: size1, symbol2: size2}

    def update_pair_position(self,
                            pair_key: str,
                            symbol1: str,
                            symbol2: str,
                            size1: float,
                            size2: float,
                            prices: dict[str, float],
                            timestamp: datetime | None = None):
        """Update position tracking for a pair."""
        if timestamp is None:
            timestamp = datetime.now()

        if size1 == 0 and size2 == 0:
            # Position closed
            if pair_key in self.active_pairs:
                del self.active_pairs[pair_key]
        else:
            # Position opened/updated
            self.active_pairs[pair_key] = {
                "symbol1": symbol1,
                "symbol2": symbol2,
                "size1": size1,
                "size2": size2,
                "entry_prices": prices.copy(),
                "entry_time": timestamp,
                "hedge_ratio": self.hedge_ratios.get(pair_key, 1.0),
                "unrealized_pnl": 0.0
            }

        # Log position update
        self.position_history.append({
            "timestamp": timestamp,
            "pair": pair_key,
            "action": "close" if size1 == 0 else "open",
            "size1": size1,
            "size2": size2,
            "prices": prices.copy()
        })

        logger.debug(f"Pair position updated: {pair_key}, sizes=({size1:.2f}, {size2:.2f})")

    def update_unrealized_pnl(self, current_prices: dict[str, float]):
        """Update unrealized PnL for all active pairs."""
        for pair_key, position in self.active_pairs.items():
            symbol1 = position["symbol1"]
            symbol2 = position["symbol2"]

            if symbol1 not in current_prices or symbol2 not in current_prices:
                continue

            # Calculate PnL for each leg
            entry_price1 = position["entry_prices"][symbol1]
            entry_price2 = position["entry_prices"][symbol2]
            current_price1 = current_prices[symbol1]
            current_price2 = current_prices[symbol2]

            pnl1 = position["size1"] * (current_price1 - entry_price1)
            pnl2 = position["size2"] * (current_price2 - entry_price2)

            total_pnl = pnl1 + pnl2
            position["unrealized_pnl"] = total_pnl

    def get_portfolio_exposure(self, current_prices: dict[str, float]) -> dict[str, Any]:
        """Calculate net exposure across all pairs."""
        net_exposure = {}
        gross_exposure = 0.0
        total_pnl = 0.0

        for pair_key, position in self.active_pairs.items():
            symbol1 = position["symbol1"]
            symbol2 = position["symbol2"]

            # Add to net exposure
            if symbol1 not in net_exposure:
                net_exposure[symbol1] = 0.0
            if symbol2 not in net_exposure:
                net_exposure[symbol2] = 0.0

            net_exposure[symbol1] += position["size1"]
            net_exposure[symbol2] += position["size2"]

            # Calculate gross exposure
            if symbol1 in current_prices and symbol2 in current_prices:
                gross_exposure += abs(position["size1"] * current_prices[symbol1])
                gross_exposure += abs(position["size2"] * current_prices[symbol2])

                total_pnl += position["unrealized_pnl"]

        return {
            "net_exposure": net_exposure,
            "gross_exposure": gross_exposure,
            "total_unrealized_pnl": total_pnl,
            "active_pairs": len(self.active_pairs),
            "market_neutrality": self._calculate_market_neutrality(net_exposure, current_prices)
        }

    def _calculate_market_neutrality(self,
                                    net_exposure: dict[str, float],
                                    current_prices: dict[str, float]) -> float:
        """Calculate how market-neutral the portfolio is (0 = perfect neutral, 1 = fully exposed)."""
        total_long = 0.0
        total_short = 0.0

        for symbol, size in net_exposure.items():
            if symbol in current_prices:
                value = size * current_prices[symbol]
                if value > 0:
                    total_long += value
                else:
                    total_short += abs(value)

        total_exposure = total_long + total_short
        if total_exposure == 0:
            return 0.0

        net_exposure_value = abs(total_long - total_short)
        market_neutrality = net_exposure_value / total_exposure

        return market_neutrality

    def get_strategy_stats(self) -> dict[str, Any]:
        """Get strategy performance statistics."""
        stats = {
            "strategy_name": "market_neutral",
            "enabled": self.enabled,
            "active_pairs": len(self.active_pairs),
            "total_cointegrated_pairs": len(self.cointegrated_pairs),
            "z_score_entry": self.z_score_entry,
            "z_score_exit": self.z_score_exit
        }

        if self.active_pairs:
            total_pnl = sum(pos["unrealized_pnl"] for pos in self.active_pairs.values())
            stats["total_unrealized_pnl"] = total_pnl

            # Pair statistics
            hedge_ratios = [pos["hedge_ratio"] for pos in self.active_pairs.values()]
            if hedge_ratios:
                stats["avg_hedge_ratio"] = np.mean(hedge_ratios)
                stats["hedge_ratio_range"] = [min(hedge_ratios), max(hedge_ratios)]

        if self.position_history:
            recent_trades = [t for t in self.position_history
                           if (datetime.now() - t["timestamp"]).days <= 30]
            stats["trades_last_30_days"] = len(recent_trades)

        return stats

    def should_trade_pair(self, symbol1: str, symbol2: str, market_conditions: dict[str, Any]) -> bool:
        """Determine if a pair should be traded given market conditions."""
        if not self.enabled:
            return False

        # Avoid trading pairs during high volatility periods
        volatility = market_conditions.get("volatility", 0.0)
        if volatility > 0.25:  # 25% annualized volatility
            return False

        # Limit total number of active pairs
        if len(self.active_pairs) >= self.max_pairs:
            return False

        return True
