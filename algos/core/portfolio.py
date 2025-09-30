"""
Portfolio management and optimization for quantitative trading.
Implements HRP allocation, portfolio construction, and rebalancing logic.
"""

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


class Portfolio:
    """Enhanced portfolio management system with capacity constraints and ES throttle."""

    def __init__(self,
                 initial_capital: float = 100000.0,
                 rebalance_frequency: str = "monthly",
                 max_weight: float = 0.1,
                 config: dict[str, Any] | None = None):
        """
        Initialize portfolio manager.
        
        Args:
            initial_capital: Starting capital
            rebalance_frequency: How often to rebalance ('daily', 'weekly', 'monthly')
            max_weight: Maximum weight per asset
            config: Trading configuration with capacity and risk settings
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency
        self.max_weight = max_weight
        self.config = config or {}

        # Capacity management
        self.capacity_config = self.config.get("capacity", {})
        self.adv_cap_pct = self.capacity_config.get("adv_cap_pct", 0.05)
        self.adv_data: dict[str, float] = {}  # symbol -> average daily volume
        self.spread_data: dict[str, float] = {}  # symbol -> average spread

        # Expected Shortfall throttle
        self.es_lookback = 30  # days for ES calculation
        self.es_threshold = -0.05  # 5% ES threshold for throttling
        self.es_throttle_factor = 0.5  # Reduce gross exposure by this factor when ES triggered
        self.pnl_history: list[float] = []

        # Portfolio state
        self.positions: dict[str, float] = {}  # symbol -> shares
        self.weights: dict[str, float] = {}    # symbol -> weight
        self.prices: dict[str, float] = {}     # symbol -> last price

        # Performance tracking
        self.value_history: list[tuple[pd.Timestamp, float]] = []
        self.weight_history: list[dict[str, float]] = []
        self.rebalance_dates: list[pd.Timestamp] = []

        logger.info(f"Portfolio initialized with ${initial_capital:,.2f}, "
                   f"ADV cap: {self.adv_cap_pct:.1%}, ES threshold: {self.es_threshold:.1%}")

    def update_prices(self, price_data: dict[str, float]):
        """Update current prices for all assets."""
        self.prices.update(price_data)

        # Calculate current portfolio value
        portfolio_value = self._calculate_portfolio_value()
        self.value_history.append((pd.Timestamp.now(), portfolio_value))

    def _calculate_portfolio_value(self) -> float:
        """Calculate current portfolio value."""
        total_value = 0.0

        for symbol, shares in self.positions.items():
            if symbol in self.prices and shares != 0:
                total_value += shares * self.prices[symbol]

        # Add any cash position
        cash_position = self.positions.get("CASH", 0.0)
        total_value += cash_position

        return total_value

    def get_current_weights(self) -> dict[str, float]:
        """Calculate current portfolio weights."""
        portfolio_value = self._calculate_portfolio_value()

        if portfolio_value <= 0:
            return {}

        current_weights = {}

        for symbol, shares in self.positions.items():
            if symbol in self.prices and symbol != "CASH":
                weight = (shares * self.prices[symbol]) / portfolio_value
                current_weights[symbol] = weight

        return current_weights

    def update_adv_data(self, symbol: str, volume: float, price: float) -> None:
        """Update average daily volume data for capacity calculations."""
        if symbol not in self.adv_data:
            self.adv_data[symbol] = volume
        else:
            # Simple exponential moving average
            alpha = 0.1
            self.adv_data[symbol] = alpha * volume + (1 - alpha) * self.adv_data[symbol]

    def update_spread_data(self, symbol: str, spread: float) -> None:
        """Update average spread data for capacity calculations."""
        if symbol not in self.spread_data:
            self.spread_data[symbol] = spread
        else:
            # Simple exponential moving average
            alpha = 0.1
            self.spread_data[symbol] = alpha * spread + (1 - alpha) * self.spread_data[symbol]

    def apply_capacity_constraints(self, target_weights: dict[str, float]) -> dict[str, float]:
        """Apply capacity constraints based on ADV and spread costs."""
        constrained_weights = target_weights.copy()
        portfolio_value = self._calculate_portfolio_value()

        for symbol, target_weight in target_weights.items():
            if symbol == "CASH":
                continue

            # Skip if no ADV data available
            if symbol not in self.adv_data or symbol not in self.prices:
                continue

            target_notional = abs(target_weight) * portfolio_value
            adv = self.adv_data[symbol]
            price = self.prices[symbol]

            # Calculate maximum notional based on ADV capacity
            max_notional_adv = adv * price * self.adv_cap_pct

            # Additional constraint based on spread costs
            spread = self.spread_data.get(symbol, 0.001)  # Default 10bps spread
            # Reduce capacity further if spreads are wide
            spread_penalty = max(0.5, 1.0 - spread * 100)  # Reduce if spread > 1%
            max_notional_final = max_notional_adv * spread_penalty

            # Apply constraint
            if target_notional > max_notional_final:
                reduction_factor = max_notional_final / target_notional
                constrained_weights[symbol] = target_weight * reduction_factor

                logger.warning(f"Capacity constraint applied to {symbol}: "
                             f"target_weight={target_weight:.3f} -> "
                             f"constrained_weight={constrained_weights[symbol]:.3f} "
                             f"(ADV=${adv*price:,.0f}, max_notional=${max_notional_final:,.0f})")

        return constrained_weights

    def update_pnl_history(self, daily_pnl: float) -> None:
        """Update daily PnL history for ES calculation."""
        self.pnl_history.append(daily_pnl)

        # Keep only recent history
        if len(self.pnl_history) > self.es_lookback * 2:
            self.pnl_history = self.pnl_history[-self.es_lookback:]

    def calculate_expected_shortfall(self, confidence_level: float = 0.05) -> float:
        """Calculate Expected Shortfall (CVaR) from PnL history."""
        if len(self.pnl_history) < 10:  # Need sufficient history
            return 0.0

        pnl_array = np.array(self.pnl_history[-self.es_lookback:])

        # Calculate percentile threshold
        var_threshold = np.percentile(pnl_array, confidence_level * 100)

        # Calculate ES as mean of losses beyond VaR
        tail_losses = pnl_array[pnl_array <= var_threshold]

        if len(tail_losses) == 0:
            return 0.0

        es = np.mean(tail_losses)
        return es

    def apply_es_throttle(self, target_weights: dict[str, float]) -> tuple[dict[str, float], bool]:
        """Apply Expected Shortfall throttle to reduce gross exposure."""
        es = self.calculate_expected_shortfall()

        if es < self.es_threshold:
            # ES indicates tail risk - throttle gross exposure
            throttled_weights = {}
            for symbol, weight in target_weights.items():
                if symbol == "CASH":
                    throttled_weights[symbol] = weight
                else:
                    throttled_weights[symbol] = weight * self.es_throttle_factor

            logger.warning(f"ES throttle activated: ES={es:.3%} < threshold={self.es_threshold:.3%}, "
                         f"reducing gross exposure by {(1-self.es_throttle_factor):.1%}")

            return throttled_weights, True

        return target_weights, False

    def rebalance_to_target(self, target_weights: dict[str, float]) -> dict[str, float]:
        """
        Rebalance portfolio to target weights with capacity and ES constraints.
        
        Args:
            target_weights: Dictionary of symbol -> target weight
            
        Returns:
            Dictionary of required trades (symbol -> dollar amount)
        """
        # Apply capacity constraints
        capacity_constrained_weights = self.apply_capacity_constraints(target_weights)

        # Apply ES throttle
        final_weights, es_throttled = self.apply_es_throttle(capacity_constrained_weights)

        current_value = self._calculate_portfolio_value()
        current_weights = self.get_current_weights()

        # Calculate required trades
        trades = {}

        for symbol, target_weight in final_weights.items():
            current_weight = current_weights.get(symbol, 0.0)
            weight_diff = target_weight - current_weight
            dollar_diff = weight_diff * current_value

            if abs(dollar_diff) > 10:  # Only trade if difference > $10
                trades[symbol] = dollar_diff

        logger.info(f"Rebalancing portfolio: {len(trades)} trades required")
        return trades

    def execute_trades(self, trades: dict[str, float]) -> bool:
        """
        Execute portfolio trades.
        
        Args:
            trades: Dictionary of symbol -> dollar amount to trade
            
        Returns:
            True if all trades executed successfully
        """
        executed_successfully = True

        for symbol, dollar_amount in trades.items():
            if symbol not in self.prices:
                logger.warning(f"No price available for {symbol}")
                executed_successfully = False
                continue

            # Calculate shares to trade
            price = self.prices[symbol]
            shares_to_trade = dollar_amount / price

            # Update positions
            current_shares = self.positions.get(symbol, 0.0)
            self.positions[symbol] = current_shares + shares_to_trade

            # Update cash position
            cash_impact = -dollar_amount
            current_cash = self.positions.get("CASH", 0.0)
            self.positions["CASH"] = current_cash + cash_impact

            logger.info(f"Executed trade: {shares_to_trade:.2f} shares of {symbol} "
                       f"at ${price:.2f} (${dollar_amount:.2f})")

        # Update weights after trades
        self.weights = self.get_current_weights()
        self.weight_history.append(self.weights.copy())
        self.rebalance_dates.append(pd.Timestamp.now())

        return executed_successfully


class HRPOptimizer:
    """Hierarchical Risk Parity portfolio optimization."""

    def __init__(self, lookback_days: int = 252):
        """Initialize HRP optimizer."""
        self.lookback_days = lookback_days

    def calculate_hrp_weights(self, returns: pd.DataFrame) -> pd.Series:
        """
        Calculate Hierarchical Risk Parity weights.
        
        Args:
            returns: DataFrame of asset returns
            
        Returns:
            Series of optimal weights
        """
        # Step 1: Calculate distance matrix
        corr_matrix = returns.corr()
        distance_matrix = np.sqrt((1 - corr_matrix) / 2)

        # Step 2: Hierarchical clustering
        linkage_matrix = linkage(squareform(distance_matrix.values), method="single")

        # Step 3: Quasi-diagonalization
        sort_idx = self._get_quasi_diagonal_order(linkage_matrix, corr_matrix.index)

        # Step 4: Recursive bisection
        weights = self._recursive_bisection(returns[sort_idx])

        logger.info(f"HRP optimization complete: {len(weights)} assets")
        return weights

    def _get_quasi_diagonal_order(self, linkage_matrix: np.ndarray, labels: pd.Index) -> list[str]:
        """Get quasi-diagonal ordering from hierarchical clustering."""
        # Get cluster ordering
        cluster_order = dendrogram(linkage_matrix, labels=labels, no_plot=True)["leaves"]
        return [labels[i] for i in cluster_order]

    def _recursive_bisection(self, returns: pd.DataFrame) -> pd.Series:
        """Apply recursive bisection to allocate weights."""
        weights = pd.Series(1.0, index=returns.columns)
        clustered_alphas = [returns.columns.tolist()]

        while len(clustered_alphas) > 0:
            clustered_alphas = [cluster[i:j] for cluster in clustered_alphas
                              for i, j in ((0, len(cluster) // 2), (len(cluster) // 2, len(cluster)))
                              if len(cluster) > 1]

            for i in range(0, len(clustered_alphas), 2):
                if i + 1 >= len(clustered_alphas):
                    break

                left_cluster = clustered_alphas[i]
                right_cluster = clustered_alphas[i + 1]

                # Calculate cluster variances
                left_var = self._calculate_cluster_variance(returns[left_cluster])
                right_var = self._calculate_cluster_variance(returns[right_cluster])

                # Allocate weights based on inverse variance
                total_var = left_var + right_var
                if total_var > 0:
                    left_weight = right_var / total_var
                    right_weight = left_var / total_var
                else:
                    left_weight = right_weight = 0.5

                # Update weights
                for asset in left_cluster:
                    weights[asset] *= left_weight
                for asset in right_cluster:
                    weights[asset] *= right_weight

        # Normalize weights
        weights = weights / weights.sum()
        return weights

    def _calculate_cluster_variance(self, cluster_returns: pd.DataFrame) -> float:
        """Calculate variance of a cluster (equal-weighted portfolio)."""
        if len(cluster_returns.columns) == 1:
            return cluster_returns.var().iloc[0]

        # Equal weights
        equal_weights = np.ones(len(cluster_returns.columns)) / len(cluster_returns.columns)
        cov_matrix = cluster_returns.cov()
        cluster_variance = np.dot(equal_weights, np.dot(cov_matrix, equal_weights))

        return cluster_variance


class MeanReversionOptimizer:
    """Mean reversion portfolio optimization for pairs trading."""

    def __init__(self,
                 lookback_days: int = 252,
                 entry_zscore: float = 2.0,
                 exit_zscore: float = 0.0):
        """Initialize mean reversion optimizer."""
        self.lookback_days = lookback_days
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore

    def find_cointegrated_pairs(self,
                               price_data: pd.DataFrame,
                               significance_level: float = 0.05) -> list[tuple[str, str, float]]:
        """
        Find cointegrated pairs using Engle-Granger test.
        
        Returns:
            List of (asset1, asset2, p_value) tuples
        """
        from statsmodels.tsa.stattools import coint

        symbols = price_data.columns.tolist()
        cointegrated_pairs = []

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]

                try:
                    # Test for cointegration
                    score, p_value, _ = coint(price_data[symbol1], price_data[symbol2])

                    if p_value < significance_level:
                        cointegrated_pairs.append((symbol1, symbol2, p_value))

                except Exception as e:
                    logger.debug(f"Cointegration test failed for {symbol1}-{symbol2}: {e}")
                    continue

        logger.info(f"Found {len(cointegrated_pairs)} cointegrated pairs")
        return cointegrated_pairs

    def calculate_spread(self,
                        price1: pd.Series,
                        price2: pd.Series) -> tuple[pd.Series, float]:
        """
        Calculate spread between two assets and hedge ratio.
        
        Returns:
            Tuple of (spread_series, hedge_ratio)
        """
        # Calculate hedge ratio using OLS regression
        from sklearn.linear_model import LinearRegression

        # Prepare data
        X = price2.values.reshape(-1, 1)
        y = price1.values

        # Fit regression
        reg = LinearRegression()
        reg.fit(X, y)

        hedge_ratio = reg.coef_[0]

        # Calculate spread
        spread = price1 - hedge_ratio * price2

        return spread, hedge_ratio

    def generate_signals(self,
                        spread: pd.Series,
                        lookback: int = None) -> pd.DataFrame:
        """
        Generate mean reversion signals based on z-score.
        
        Returns:
            DataFrame with columns: ['z_score', 'entry_long', 'entry_short', 'exit']
        """
        if lookback is None:
            lookback = self.lookback_days

        # Calculate rolling statistics
        rolling_mean = spread.rolling(window=lookback).mean()
        rolling_std = spread.rolling(window=lookback).std()

        # Calculate z-score
        z_score = (spread - rolling_mean) / rolling_std

        # Generate signals
        signals = pd.DataFrame(index=spread.index)
        signals["z_score"] = z_score
        signals["entry_long"] = z_score < -self.entry_zscore  # Enter long when z < -2
        signals["entry_short"] = z_score > self.entry_zscore   # Enter short when z > 2
        signals["exit"] = abs(z_score) < self.exit_zscore      # Exit when z near 0

        return signals


class VolatilityTargeting:
    """Volatility targeting for dynamic position sizing."""

    def __init__(self,
                 target_vol: float = 0.12,
                 lookback_days: int = 60,
                 max_leverage: float = 2.0):
        """
        Initialize volatility targeting.
        
        Args:
            target_vol: Target annualized volatility
            lookback_days: Lookback period for vol estimation
            max_leverage: Maximum allowed leverage
        """
        self.target_vol = target_vol
        self.lookback_days = lookback_days
        self.max_leverage = max_leverage

    def calculate_vol_scalar(self, returns: pd.Series) -> float:
        """
        Calculate volatility scaling factor.
        
        Args:
            returns: Series of portfolio returns
            
        Returns:
            Scaling factor to apply to positions
        """
        if len(returns) < self.lookback_days:
            logger.warning(f"Insufficient data for vol targeting: {len(returns)} < {self.lookback_days}")
            return 1.0

        # Calculate realized volatility
        recent_returns = returns.tail(self.lookback_days)
        realized_vol = recent_returns.std() * np.sqrt(252)  # Annualized

        if realized_vol <= 0:
            return 1.0

        # Calculate scaling factor
        vol_scalar = self.target_vol / realized_vol

        # Apply leverage limit
        vol_scalar = min(vol_scalar, self.max_leverage)
        vol_scalar = max(vol_scalar, 0.1)  # Minimum 10% allocation

        logger.debug(f"Vol targeting: realized={realized_vol:.1%}, "
                    f"target={self.target_vol:.1%}, scalar={vol_scalar:.2f}")

        return vol_scalar

    def apply_vol_targeting(self,
                           base_weights: dict[str, float],
                           returns: pd.Series) -> dict[str, float]:
        """
        Apply volatility targeting to base portfolio weights.
        
        Args:
            base_weights: Base portfolio weights
            returns: Portfolio returns series
            
        Returns:
            Vol-adjusted weights
        """
        vol_scalar = self.calculate_vol_scalar(returns)

        # Scale all weights
        adjusted_weights = {symbol: weight * vol_scalar
                          for symbol, weight in base_weights.items()}

        # Normalize if total exceeds 100%
        total_weight = sum(adjusted_weights.values())
        if total_weight > 1.0:
            adjusted_weights = {symbol: weight / total_weight
                              for symbol, weight in adjusted_weights.items()}

        return adjusted_weights
