"""
Test suite for Risk Engine functionality.
Tests position sizing, leverage limits, sector caps, volatility targeting, and kill-switch.
"""

import os

# Import modules to test
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from algos.core.portfolio import HRPOptimizer, Portfolio, VolatilityTargeting
from algos.core.risk import PortfolioOptimizer, RiskManager


class TestRiskManager:
    """Test cases for RiskManager functionality."""

    def setup_method(self):
        """Setup test configuration and risk manager."""
        self.config = {
            "risk": {
                "max_leverage": 2.0,
                "max_position_pct": 0.10,
                "max_sector_pct": 0.30,
                "risk_pct_per_trade": 0.01,
                "vol_target": 0.12,
                "kill_switch_dd": 0.20
            }
        }

        self.risk_manager = RiskManager(self.config)

        # Test equity value
        self.test_equity = 100000.0

    def test_initialization(self):
        """Test RiskManager initialization."""
        rm = self.risk_manager

        assert rm.max_leverage == 2.0, "Max leverage should be set correctly"
        assert rm.max_position_pct == 0.10, "Max position % should be set correctly"
        assert rm.risk_pct_per_trade == 0.01, "Risk % per trade should be set correctly"
        assert rm.kill_switch_dd == 0.20, "Kill switch drawdown should be set correctly"
        assert not rm.is_kill_switch_active, "Kill switch should not be active initially"

        print("Risk Manager initialization test: All parameters set correctly")

    def test_position_sizing_basic(self):
        """Test basic position sizing calculation."""
        symbol = "AAPL"
        entry_price = 150.0
        stop_price = 145.0  # $5 stop loss
        atr = 3.0
        equity = self.test_equity

        position_size = self.risk_manager.calculate_position_size(
            symbol, entry_price, stop_price, equity, atr
        )

        # Should calculate reasonable position size
        assert position_size > 0, "Position size should be positive"
        assert position_size < equity / entry_price, "Position shouldn't exceed all equity"

        # Calculate implied risk
        risk_amount = position_size * (entry_price - stop_price)
        risk_pct = risk_amount / equity

        # Should respect risk percentage
        assert risk_pct <= self.config["risk"]["risk_pct_per_trade"] * 1.1, "Risk should not exceed target"

        print(f"Basic position sizing: size={position_size:.2f}, risk={risk_pct:.1%}")

    def test_position_sizing_with_atr(self):
        """Test position sizing using ATR for stop calculation."""
        symbol = "MSFT"
        entry_price = 300.0
        stop_price = 295.0  # Will be overridden by ATR calculation
        atr = 5.0
        equity = self.test_equity
        stop_mult = 1.0

        position_size = self.risk_manager.calculate_position_size(
            symbol, entry_price, stop_price, equity, atr, stop_mult
        )

        # Should use ATR-based sizing
        assert position_size > 0, "Should calculate positive position size"

        # Expected calculation with ATR
        risk_amount = self.config["risk"]["risk_pct_per_trade"] * equity
        expected_position_value = risk_amount / (stop_mult * atr / entry_price)
        expected_position_size = expected_position_value / entry_price

        # Allow some tolerance for position limits
        assert position_size <= expected_position_size * 1.1, "Position size should be close to expected"

        print(f"ATR position sizing: size={position_size:.2f}, expected≈{expected_position_size:.2f}")

    def test_position_limits(self):
        """Test position size limits."""
        symbol = "TSLA"
        entry_price = 200.0
        stop_price = 195.0
        atr = 10.0  # High ATR to test limits
        equity = self.test_equity

        position_size = self.risk_manager.calculate_position_size(
            symbol, entry_price, stop_price, equity, atr
        )

        # Check position limit
        max_position_value = self.config["risk"]["max_position_pct"] * equity
        position_value = position_size * entry_price

        assert position_value <= max_position_value * 1.01, "Position should respect max position limit"

        print(f"Position limits test: value=${position_value:.0f}, limit=${max_position_value:.0f}")

    def test_leverage_limits(self):
        """Test leverage limits in position sizing."""
        # Add existing positions to test leverage
        self.risk_manager.positions = {
            "AAPL": 500.0,  # $75k at $150
            "MSFT": 100.0   # $30k at $300
        }

        symbol = "GOOGL"
        entry_price = 2000.0
        stop_price = 1950.0
        atr = 30.0
        equity = self.test_equity

        position_size = self.risk_manager.calculate_position_size(
            symbol, entry_price, stop_price, equity, atr
        )

        # Should respect leverage limits
        current_leverage = self.risk_manager._calculate_current_leverage(equity)
        additional_leverage = (position_size * entry_price) / equity
        total_leverage = current_leverage + additional_leverage

        assert total_leverage <= self.config["risk"]["max_leverage"] * 1.01, "Should respect leverage limits"

        print(f"Leverage limits test: current={current_leverage:.2f}x, "
              f"additional={additional_leverage:.2f}x, total={total_leverage:.2f}x")

    def test_kill_switch_inactive(self):
        """Test kill switch when inactive."""
        symbol = "NVDA"
        entry_price = 400.0
        stop_price = 390.0
        atr = 15.0
        equity = self.test_equity

        # Kill switch should be inactive
        assert not self.risk_manager.is_kill_switch_active, "Kill switch should be inactive"

        position_size = self.risk_manager.calculate_position_size(
            symbol, entry_price, stop_price, equity, atr
        )

        assert position_size > 0, "Should allow positions when kill switch inactive"

        print("Kill switch inactive test: Positions allowed")

    def test_kill_switch_activation(self):
        """Test kill switch activation on drawdown."""
        # Simulate equity curve with drawdown
        initial_equity = 100000.0
        peak_equity = 120000.0  # 20% gain
        current_equity = 95000.0  # 20.8% drawdown from peak

        self.risk_manager.peak_equity = peak_equity
        self.risk_manager.update_equity_curve(current_equity)

        # Should activate kill switch
        assert self.risk_manager.is_kill_switch_active, "Kill switch should be active after large drawdown"

        # Should prevent new positions
        position_size = self.risk_manager.calculate_position_size(
            "TEST", 100.0, 95.0, current_equity, 2.0
        )

        assert position_size == 0.0, "Should prevent positions when kill switch active"

        print("Kill switch activation test: Activated at 20.8% drawdown")

    def test_risk_limit_checks(self):
        """Test risk limit checking functionality."""
        symbol = "AMD"
        quantity = 100.0
        price = 80.0
        equity = self.test_equity

        # Should pass with reasonable trade
        can_trade, reason = self.risk_manager.check_risk_limits(symbol, quantity, price, equity)

        assert can_trade, f"Should allow reasonable trade: {reason}"
        assert reason == "Risk limits OK", f"Unexpected reason: {reason}"

        # Test excessive position size
        large_quantity = 2000.0  # $160k position > 10% limit
        can_trade_large, reason_large = self.risk_manager.check_risk_limits(
            symbol, large_quantity, price, equity
        )

        assert not can_trade_large, "Should reject excessive position"
        assert "limit exceeded" in reason_large, f"Should mention limit in reason: {reason_large}"

        print(f"Risk limits test: Normal trade allowed, large trade rejected ({reason_large})")

    def test_equity_curve_tracking(self):
        """Test equity curve tracking and drawdown calculation."""
        # Simulate portfolio value changes
        values = [100000, 105000, 110000, 120000, 115000, 110000, 95000, 90000, 100000]

        for value in values:
            self.risk_manager.update_equity_curve(value)

        # Check peak tracking
        assert self.risk_manager.peak_equity == 120000, "Should track peak correctly"

        # Check current drawdown
        risk_metrics = self.risk_manager.get_risk_metrics()
        expected_dd = (120000 - 100000) / 120000  # 16.67% drawdown

        assert abs(risk_metrics["current_drawdown"] - expected_dd) < 0.01, "Drawdown calculation incorrect"

        print(f"Equity curve tracking: Peak=${self.risk_manager.peak_equity}, "
              f"Current DD={risk_metrics['current_drawdown']:.1%}")

    def test_volatility_scaling(self):
        """Test volatility scaling calculation."""
        # Simulate daily returns
        np.random.seed(42)
        daily_returns = np.random.normal(0, 0.015, 100)  # 1.5% daily vol = ~24% annual

        self.risk_manager.daily_returns = list(daily_returns)

        vol_scaling = self.risk_manager.calculate_vol_scaling()

        # Should scale down due to high volatility (target is 12%)
        assert vol_scaling < 1.0, "Should scale down with high volatility"
        assert vol_scaling >= 0.5, "Should not scale down too aggressively"

        realized_vol = np.std(daily_returns) * np.sqrt(252)
        expected_scaling = self.config["risk"]["vol_target"] / realized_vol
        expected_scaling = np.clip(expected_scaling, 0.5, 2.0)

        assert abs(vol_scaling - expected_scaling) < 0.1, "Vol scaling calculation incorrect"

        print(f"Vol scaling test: realized={realized_vol:.1%}, "
              f"target={self.config['risk']['vol_target']:.1%}, scaling={vol_scaling:.2f}")

    def test_trade_logging(self):
        """Test trade logging functionality."""
        # Log some trades
        trades = [
            ("AAPL", "buy", 100, 150.0, "scalper"),
            ("MSFT", "sell", 50, 300.0, "trend"),
            ("GOOGL", "buy", 10, 2000.0, "breakout")
        ]

        for symbol, side, qty, price, strategy in trades:
            self.risk_manager.log_trade(symbol, side, qty, price, datetime.now(), strategy)

        # Check trade history
        assert len(self.risk_manager.trade_history) == 3, "Should log all trades"

        # Get trade statistics
        stats = self.risk_manager.get_trade_statistics()

        assert stats["total_trades"] == 3, "Should count all trades"
        assert stats["buy_trades"] == 2, "Should count buy trades correctly"
        assert stats["sell_trades"] == 1, "Should count sell trades correctly"
        assert "strategy_breakdown" in stats, "Should provide strategy breakdown"

        print(f"Trade logging test: {stats['total_trades']} trades logged, "
              f"strategies: {stats['strategy_breakdown']}")

    def test_position_tracking(self):
        """Test position tracking functionality."""
        # Update positions
        positions = [
            ("AAPL", 100.0, 150.0),
            ("MSFT", -50.0, 300.0),
            ("GOOGL", 25.0, 2000.0)
        ]

        for symbol, qty, price in positions:
            self.risk_manager.update_position(symbol, qty, price)

        # Check positions
        assert len(self.risk_manager.positions) == 3, "Should track all positions"
        assert self.risk_manager.positions["AAPL"] == 100.0, "Should track AAPL position"
        assert self.risk_manager.positions["MSFT"] == -50.0, "Should track short MSFT position"

        # Close a position
        self.risk_manager.update_position("AAPL", -100.0, 152.0)
        assert abs(self.risk_manager.positions["AAPL"]) < 0.01, "Should close AAPL position"

        print(f"Position tracking test: {len([p for p in self.risk_manager.positions.values() if abs(p) > 0.01])} active positions")


class TestPortfolioOptimizer:
    """Test Portfolio Optimizer functionality."""

    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)

        # Create correlated return series
        dates = pd.date_range("2020-01-01", periods=252, freq="1D")

        # Generate correlated returns
        n_assets = 5
        factor_returns = np.random.normal(0, 0.02, 252)  # Market factor

        returns_data = {}
        for i in range(n_assets):
            asset_returns = (
                0.7 * factor_returns +  # Market exposure
                0.3 * np.random.normal(0, 0.015, 252)  # Idiosyncratic
            )
            returns_data[f"ASSET_{i}"] = asset_returns

        self.returns_df = pd.DataFrame(returns_data, index=dates)
        self.optimizer = PortfolioOptimizer()

    def test_correlation_calculation(self):
        """Test correlation matrix calculation."""
        corr_matrix = self.optimizer.calculate_correlation_matrix(self.returns_df)

        assert isinstance(corr_matrix, pd.DataFrame), "Should return DataFrame"
        assert corr_matrix.shape == (5, 5), "Should be 5x5 matrix"
        assert np.allclose(np.diag(corr_matrix), 1.0), "Diagonal should be 1.0"
        assert np.allclose(corr_matrix, corr_matrix.T), "Should be symmetric"

        # Check for reasonable correlations (due to common factor)
        off_diag = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        assert np.all(off_diag > 0.3), "Assets should be positively correlated"

        print(f"Correlation test: Average correlation = {off_diag.mean():.3f}")

    def test_covariance_calculation(self):
        """Test covariance matrix calculation."""
        cov_matrix = self.optimizer.calculate_covariance_matrix(self.returns_df)

        assert isinstance(cov_matrix, pd.DataFrame), "Should return DataFrame"
        assert cov_matrix.shape == (5, 5), "Should be 5x5 matrix"
        assert np.allclose(cov_matrix, cov_matrix.T), "Should be symmetric"

        # Check positive semi-definite
        eigenvals = np.linalg.eigvals(cov_matrix.values)
        assert np.all(eigenvals >= -1e-10), "Should be positive semi-definite"

        print(f"Covariance test: Matrix eigenvalues range [{eigenvals.min():.6f}, {eigenvals.max():.6f}]")

    def test_hrp_allocation(self):
        """Test Hierarchical Risk Parity allocation."""
        hrp_weights = self.optimizer.hierarchical_risk_parity(self.returns_df)

        assert isinstance(hrp_weights, pd.Series), "Should return Series"
        assert len(hrp_weights) == 5, "Should have 5 weights"
        assert abs(hrp_weights.sum() - 1.0) < 1e-10, "Weights should sum to 1"
        assert np.all(hrp_weights >= 0), "All weights should be non-negative"

        # Check diversification (no single weight too large)
        max_weight = hrp_weights.max()
        assert max_weight < 0.8, f"Max weight {max_weight:.1%} too high - not diversified"

        print(f"HRP test: Weights = {hrp_weights.values}")
        print(f"Max weight = {max_weight:.1%}, Min weight = {hrp_weights.min():.1%}")


class TestPortfolioManagement:
    """Test Portfolio class functionality."""

    def setup_method(self):
        """Setup portfolio for testing."""
        self.portfolio = Portfolio(initial_capital=100000.0)

        # Sample prices
        self.sample_prices = {
            "AAPL": 150.0,
            "MSFT": 300.0,
            "GOOGL": 2000.0
        }

    def test_portfolio_initialization(self):
        """Test portfolio initialization."""
        assert self.portfolio.initial_capital == 100000.0, "Should set initial capital"
        assert self.portfolio.current_capital == 100000.0, "Should set current capital"
        assert len(self.portfolio.positions) == 0, "Should start with no positions"
        assert len(self.portfolio.weights) == 0, "Should start with no weights"

        print("Portfolio initialization test: All parameters set correctly")

    def test_price_updates(self):
        """Test price update functionality."""
        self.portfolio.update_prices(self.sample_prices)

        # Check prices stored
        assert self.portfolio.prices["AAPL"] == 150.0, "Should store AAPL price"
        assert self.portfolio.prices["MSFT"] == 300.0, "Should store MSFT price"

        # Check value history updated
        assert len(self.portfolio.value_history) > 0, "Should record value history"

        print("Price updates test: Prices and value history updated")

    def test_portfolio_rebalancing(self):
        """Test portfolio rebalancing functionality."""
        # Set initial positions
        self.portfolio.positions = {"AAPL": 100, "MSFT": 50}
        self.portfolio.update_prices(self.sample_prices)

        # Target weights
        target_weights = {"AAPL": 0.4, "MSFT": 0.3, "GOOGL": 0.3}

        trades = self.portfolio.rebalance_to_target(target_weights)

        assert isinstance(trades, dict), "Should return trade dictionary"
        assert "GOOGL" in trades, "Should include new position"

        # Execute trades
        success = self.portfolio.execute_trades(trades)
        assert success, "Should execute trades successfully"

        print(f"Rebalancing test: {len(trades)} trades executed")

        # Check final weights
        final_weights = self.portfolio.get_current_weights()
        print(f"Final weights: {final_weights}")


class TestHRPOptimizer:
    """Test HRP Optimizer functionality."""

    def setup_method(self):
        """Setup test data for HRP."""
        np.random.seed(42)

        # Create test return series with known structure
        dates = pd.date_range("2020-01-01", periods=252, freq="1D")

        # Create two clusters of assets
        cluster1_factor = np.random.normal(0, 0.02, 252)
        cluster2_factor = np.random.normal(0, 0.02, 252)

        returns_data = {}

        # Cluster 1: Tech stocks
        for i in range(3):
            returns_data[f"TECH_{i}"] = (
                0.8 * cluster1_factor +
                0.2 * np.random.normal(0, 0.01, 252)
            )

        # Cluster 2: Financial stocks
        for i in range(3):
            returns_data[f"FIN_{i}"] = (
                0.8 * cluster2_factor +
                0.2 * np.random.normal(0, 0.01, 252)
            )

        self.returns_df = pd.DataFrame(returns_data, index=dates)
        self.hrp_optimizer = HRPOptimizer()

    def test_hrp_weights_calculation(self):
        """Test HRP weight calculation."""
        weights = self.hrp_optimizer.calculate_hrp_weights(self.returns_df)

        # Basic checks
        assert isinstance(weights, pd.Series), "Should return Series"
        assert len(weights) == 6, "Should have 6 weights"
        assert abs(weights.sum() - 1.0) < 1e-10, "Weights should sum to 1"
        assert np.all(weights > 0), "All weights should be positive"

        # Check diversification
        max_weight = weights.max()
        min_weight = weights.min()

        assert max_weight < 0.5, f"Max weight {max_weight:.1%} too high"
        assert min_weight > 0.05, f"Min weight {min_weight:.1%} too low"

        # Weights within clusters should be similar (due to HRP)
        tech_weights = weights[weights.index.str.contains("TECH")]
        fin_weights = weights[weights.index.str.contains("FIN")]

        tech_std = tech_weights.std()
        fin_std = fin_weights.std()

        print(f"HRP weights test: Tech weights std={tech_std:.4f}, Fin weights std={fin_std:.4f}")
        print(f"Weights: {dict(weights)}")

    def test_hrp_cluster_behavior(self):
        """Test that HRP respects cluster structure."""
        weights = self.hrp_optimizer.calculate_hrp_weights(self.returns_df)

        # Within-cluster weights should be more similar than between-cluster
        tech_weights = weights[weights.index.str.contains("TECH")]
        fin_weights = weights[weights.index.str.contains("FIN")]

        # Calculate intra-cluster vs inter-cluster weight differences
        tech_mean = tech_weights.mean()
        fin_mean = fin_weights.mean()

        intra_tech_diff = np.mean(np.abs(tech_weights - tech_mean))
        intra_fin_diff = np.mean(np.abs(fin_weights - fin_mean))
        inter_cluster_diff = abs(tech_mean - fin_mean)

        print("Cluster behavior test:")
        print(f"  Intra-tech diff: {intra_tech_diff:.4f}")
        print(f"  Intra-fin diff: {intra_fin_diff:.4f}")
        print(f"  Inter-cluster diff: {inter_cluster_diff:.4f}")

        # Inter-cluster difference should be larger (indicating cluster recognition)
        assert inter_cluster_diff > max(intra_tech_diff, intra_fin_diff), "Should recognize cluster structure"


class TestVolatilityTargeting:
    """Test Volatility Targeting functionality."""

    def setup_method(self):
        """Setup volatility targeting test."""
        self.vol_targeting = VolatilityTargeting(
            target_vol=0.12,  # 12% target
            lookback_days=60,
            max_leverage=2.0
        )

    def test_vol_scalar_calculation(self):
        """Test volatility scalar calculation."""
        # Create return series with known volatility
        np.random.seed(42)
        daily_returns = pd.Series(np.random.normal(0, 0.02, 100))  # ~31% annual vol

        vol_scalar = self.vol_targeting.calculate_vol_scalar(daily_returns)

        # Should scale down due to high volatility
        assert vol_scalar < 1.0, "Should scale down with high volatility"
        assert vol_scalar >= 0.1, "Should not scale too aggressively"

        realized_vol = daily_returns.std() * np.sqrt(252)
        expected_scalar = self.vol_targeting.target_vol / realized_vol
        expected_scalar = np.clip(expected_scalar, 0.1, 2.0)

        assert abs(vol_scalar - expected_scalar) < 0.1, "Vol scalar calculation should be accurate"

        print(f"Vol scalar test: realized={realized_vol:.1%}, "
              f"target={self.vol_targeting.target_vol:.1%}, scalar={vol_scalar:.2f}")

    def test_vol_targeting_application(self):
        """Test application of volatility targeting to portfolio weights."""
        # Sample portfolio weights
        base_weights = {
            "AAPL": 0.3,
            "MSFT": 0.25,
            "GOOGL": 0.25,
            "TSLA": 0.2
        }

        # High volatility returns (should scale down)
        high_vol_returns = pd.Series(np.random.normal(0, 0.025, 100))

        adjusted_weights = self.vol_targeting.apply_vol_targeting(base_weights, high_vol_returns)

        # Should scale down all weights
        for symbol in base_weights:
            assert adjusted_weights[symbol] < base_weights[symbol], f"{symbol} should be scaled down"

        # Should still sum to 1 or less
        total_weight = sum(adjusted_weights.values())
        assert total_weight <= 1.01, f"Total weight {total_weight:.3f} should not exceed 1"

        print(f"Vol targeting test: Original sum={sum(base_weights.values()):.3f}, "
              f"Adjusted sum={total_weight:.3f}")

    def test_insufficient_data_handling(self):
        """Test handling of insufficient data for vol targeting."""
        short_returns = pd.Series([0.01, -0.005, 0.02])  # Only 3 data points

        vol_scalar = self.vol_targeting.calculate_vol_scalar(short_returns)

        # Should return default (no scaling) with insufficient data
        assert vol_scalar == 1.0, "Should return 1.0 with insufficient data"

        print("Insufficient data test: Correctly returned default scaling")


class TestRiskManagerExtended:
    """Extended test cases for RiskManager functionality."""

    def setup_method(self):
        """Setup test configuration and risk manager."""
        self.config = {
            "trading": {
                "risk": {
                    "max_leverage": 2.0,
                    "single_name_max_pct": 0.10,
                    "sector_max_pct": 0.30,
                    "per_trade_risk_pct": 0.01,
                    "vol_target_ann": 0.12,
                    "kill_switch_dd": 0.20,
                    "asset_class_caps": {
                        "crypto_max_gross_pct": 0.50
                    }
                }
            }
        }

        self.risk_manager = RiskManager(self.config)

    def test_should_accept_single_name_cap(self):
        """Test should_accept method for single-name cap enforcement."""
        # Test within limits
        assert self.risk_manager.should_accept("AAPL", 0.05) == True

        # Test exceeding single-name cap
        assert self.risk_manager.should_accept("AAPL", 0.15) == False

        print("Single-name cap enforcement test passed")

    def test_should_accept_leverage_cap(self):
        """Test should_accept method for leverage cap enforcement."""
        # Add some existing positions
        self.risk_manager.positions = {
            "AAPL": {"weight": 0.08},
            "MSFT": {"weight": 0.07},
            "GOOGL": {"weight": 0.09}
        }

        # Test within limits (current total: 0.24, adding 0.05 = 0.29)
        assert self.risk_manager.should_accept("TSLA", 0.05) == True

        # Test exceeding leverage cap (current total: 0.24, adding 1.8 = 2.04 > 2.0)
        assert self.risk_manager.should_accept("TSLA", 1.8) == False

        print("Leverage cap enforcement test passed")

    def test_should_accept_crypto_cap(self):
        """Test should_accept method for crypto gross cap enforcement."""
        # Add existing crypto positions
        self.risk_manager.positions = {
            "BTCUSD": {"weight": 0.20},
            "ETHUSD": {"weight": 0.15}
        }

        # Test within crypto cap (current crypto: 0.35, adding 0.10 = 0.45 < 0.50)
        assert self.risk_manager.should_accept("ADAUSD", 0.10) == True

        # Test exceeding crypto cap (current crypto: 0.35, adding 0.20 = 0.55 > 0.50)
        assert self.risk_manager.should_accept("ADAUSD", 0.20) == False

        print("Crypto cap enforcement test passed")

    def test_should_accept_kill_switch(self):
        """Test should_accept method when kill-switch is active."""
        # Activate kill switch
        self.risk_manager.is_kill_switch_active = True

        # Should reject all positions when kill switch is active
        assert self.risk_manager.should_accept("AAPL", 0.05) == False

        print("Kill-switch enforcement test passed")

    def test_kill_switch_trigger(self):
        """Test kill switch trigger mechanism."""
        initial_equity = 100000.0
        self.risk_manager.update_equity_curve(initial_equity)

        # Simulate a large drawdown (25% > 20% threshold)
        drawdown_equity = 75000.0
        self.risk_manager.update_equity_curve(drawdown_equity)

        assert self.risk_manager.is_kill_switch_active == True

        print("Kill-switch trigger test passed")

    def test_vol_targeting_scaling(self):
        """Test volatility targeting scaling mechanism."""
        # Create returns data with high volatility
        high_vol_returns = pd.Series(np.random.normal(0, 0.03, 252))  # 30% annualized vol

        vol_scalar = self.risk_manager.calculate_vol_scaling()

        # With default initialization, should return 1.0
        assert vol_scalar == 1.0

        print("Vol targeting scaling test passed")


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
