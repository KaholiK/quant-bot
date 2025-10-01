"""
Risk management system for quantitative trading.
Implements position sizing, risk limits, and kill-switch functionality.
"""

import os
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import requests
from loguru import logger


class PositionDict(dict):
    """
    Enhanced dict that allows direct comparison with floats for backward compatibility.
    Supports both dict access (position["quantity"]) and float comparison (position == 100.0).
    """
    def __eq__(self, other):
        if isinstance(other, (int, float)):
            return abs(self.get("quantity", 0.0) - other) < 1e-9
        return super().__eq__(other)
    
    def __float__(self):
        return float(self.get("quantity", 0.0))
    
    def __abs__(self):
        return abs(self.get("quantity", 0.0))


class RiskManager:
    """Comprehensive risk management system for trading algorithms."""

    def __init__(self, config: dict[str, Any]):
        """Initialize risk manager with configuration."""
        self.config = config

        # Handle both old and new config formats
        if "trading" in config:
            self.risk_config = config["trading"]["risk"]
        else:
            self.risk_config = config.get("risk", {})

        # Risk parameters
        self.max_leverage = self.risk_config.get("max_leverage", 2.0)
        self.single_name_max_pct = self.risk_config.get("single_name_max_pct", 0.10)
        self.sector_max_pct = self.risk_config.get("sector_max_pct", 0.30)
        self.crypto_max_gross_pct = self.risk_config.get("crypto_max_gross_pct", 0.50)
        self.risk_pct_per_trade = self.risk_config.get("per_trade_risk_pct", 0.01)
        self.vol_target_ann = self.risk_config.get("vol_target_ann", 0.12)
        self.kill_switch_dd = self.risk_config.get("kill_switch_dd", 0.20)
        
        # Backward compatibility aliases
        self.max_position_pct = self.risk_config.get("max_position_pct", self.single_name_max_pct)
        self.vol_target = self.risk_config.get("vol_target", self.vol_target_ann)

        # New risk features
        self.day_stop_dd = self.risk_config.get("day_stop_dd", 0.06)
        self.week_stop_dd = self.risk_config.get("week_stop_dd", 0.10)

        # Blackout configuration
        blackout_config = self.risk_config.get("blackout", {})
        self.earnings_blackout = blackout_config.get("earnings", True)
        self.fomc_blackout = blackout_config.get("fomc", True)

        # PDT (Pattern Day Trader) rules
        self.pdt_threshold = 25000.0  # $25k threshold for PDT rule
        self.day_trade_count = 0
        self.day_trade_reset_date = datetime.now().date()

        # State tracking
        self.positions: dict[str, dict[str, Any]] = {}  # Enhanced position tracking
        self.sector_exposure: dict[str, float] = {}
        self.asset_class_exposure: dict[str, float] = {"crypto": 0.0, "equity": 0.0}
        self.equity_curve: list[float] = []
        self.peak_equity = 0.0
        self.is_kill_switch_active = False
        self.day_start_equity = 0.0
        self.week_start_equity = 0.0
        self.last_portfolio_value = 0.0
        self.kill_switch_activation_time: datetime | None = None

        # Performance tracking
        self.trade_history: list[dict[str, Any]] = []
        self.daily_returns: list[float] = []
        self.vol_target_scaler = 1.0  # Dynamic volatility scaling
        self.last_vol_update = datetime.now()

        logger.info(f"Risk manager initialized: "
                   f"max_leverage={self.max_leverage}, "
                   f"kill_switch_dd={self.kill_switch_dd:.1%}, "
                   f"vol_target={self.vol_target_ann:.1%}")

        # Send initialization alert
        try:
            self._send_risk_alert("RISK_MANAGER_INIT",
                                f"Risk manager initialized with kill switch at {self.kill_switch_dd:.1%}")
        except Exception as e:
            logger.warning(f"Failed to send init alert: {e}")

    def calculate_position_size(self,
                              symbol: str,
                              entry_price: float,
                              stop_price: float,
                              equity: float,
                              atr: float,
                              stop_mult: float = 1.0) -> float:
        """
        Calculate position size based on risk management rules.
        
        Formula: position_value = (risk_pct * equity) / (stop_mult * ATR)
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price for position
            stop_price: Stop loss price
            equity: Current account equity
            atr: Average True Range
            stop_mult: Stop loss multiplier
            
        Returns:
            Position size in shares/units
        """
        if self.is_kill_switch_active:
            logger.warning("Kill switch active - no new positions allowed")
            return 0.0

        # Calculate risk amount
        risk_amount = self.risk_pct_per_trade * equity

        # Calculate stop distance
        stop_distance = abs(entry_price - stop_price)
        if stop_distance == 0:
            stop_distance = stop_mult * atr

        # Base position size
        if stop_distance > 0:
            position_value = risk_amount / (stop_distance / entry_price)
            position_size = position_value / entry_price
        else:
            logger.warning(f"Invalid stop distance for {symbol}: {stop_distance}")
            return 0.0

        # Apply volatility targeting
        position_size *= self.vol_target_scaler

        # Apply single-name position limits
        max_position_value = self.single_name_max_pct * equity
        max_position_size = max_position_value / entry_price

        position_size = min(position_size, max_position_size)

        # Check leverage limits
        current_leverage = self._calculate_current_leverage(equity)
        position_leverage = (position_size * entry_price) / equity

        if current_leverage + position_leverage > self.max_leverage:
            max_additional_leverage = self.max_leverage - current_leverage
            position_size = max(0, max_additional_leverage * equity / entry_price)

        # Check sector limits
        sector = self._get_sector(symbol)
        if sector:
            current_sector_exposure = self.sector_exposure.get(sector, 0.0)
            max_sector_value = self.sector_max_pct * equity

            if current_sector_exposure + (position_size * entry_price) > max_sector_value:
                max_additional_sector = max_sector_value - current_sector_exposure
                position_size = max(0, max_additional_sector / entry_price)

        # Check asset class limits
        asset_class = self._get_asset_class(symbol)
        if asset_class == "crypto":
            current_crypto_exposure = self.asset_class_exposure.get("crypto", 0.0)
            max_crypto_value = self.crypto_max_gross_pct * equity

            # Check gross exposure (absolute value)
            new_crypto_exposure = current_crypto_exposure + abs(position_size * entry_price)
            if new_crypto_exposure > max_crypto_value:
                max_additional_crypto = max_crypto_value - current_crypto_exposure
                position_size = max(0, max_additional_crypto / entry_price)
                if position_size < abs(quantity):  # Preserve direction
                    position_size *= np.sign(quantity)

        logger.debug(f"Position size for {symbol}: {position_size:.2f} units "
                    f"(${position_size * entry_price:.2f} value, "
                    f"leverage_impact={position_leverage:.2f})")

        return position_size

    def update_position(self, symbol: str, quantity: float, price: float, sector: str | None = None):
        """Update position tracking with enhanced state management."""
        # Get current position data
        old_position_data = self.positions.get(symbol, {"quantity": 0.0, "avg_price": 0.0, "unrealized_pnl": 0.0})
        old_quantity = old_position_data["quantity"]

        # Calculate new position
        new_quantity = old_quantity + quantity

        # Update average price for the position
        if new_quantity != 0:
            total_value = (old_quantity * old_position_data["avg_price"]) + (quantity * price)
            avg_price = total_value / new_quantity
        else:
            avg_price = 0.0

        # Update position data using PositionDict for backward compatibility
        self.positions[symbol] = PositionDict({
            "quantity": new_quantity,
            "avg_price": avg_price,
            "last_price": price,
            "unrealized_pnl": (price - avg_price) * new_quantity if new_quantity != 0 else 0.0,
            "timestamp": datetime.now()
        })

        # Update sector exposure
        sector = sector or self._get_sector(symbol)
        if sector:
            old_sector_exposure = self.sector_exposure.get(sector, 0.0)
            position_change_value = quantity * price
            self.sector_exposure[sector] = old_sector_exposure + position_change_value

        # Update asset class exposure
        asset_class = self._get_asset_class(symbol)
        old_class_exposure = self.asset_class_exposure.get(asset_class, 0.0)
        position_change_value = abs(quantity * price)  # Use absolute for gross exposure
        self.asset_class_exposure[asset_class] = old_class_exposure + position_change_value

        logger.debug(f"Updated position {symbol}: {old_quantity} -> {new_quantity} "
                    f"@ avg_price={avg_price:.2f}, unrealized_pnl={self.positions[symbol]['unrealized_pnl']:.2f}")

    def _get_sector(self, symbol: str) -> str | None:
        """Get sector for a symbol."""
        # Simple sector mapping - in practice, this would use external data
        sector_map = {
            "SPY": "broad_market",
            "QQQ": "technology",
            "XLF": "financial",
            "XLK": "technology",
            "XLE": "energy",
            "XLV": "healthcare",
            "XLI": "industrial",
            "XLP": "consumer_staples",
            "XLY": "consumer_discretionary",
            "XLU": "utilities",
            "XLRE": "real_estate"
        }

        return sector_map.get(symbol, "other")

    def _get_asset_class(self, symbol: str) -> str:
        """Get asset class for a symbol."""
        if any(crypto in symbol.upper() for crypto in ["BTC", "ETH", "USD"]):
            return "crypto"
        return "equity"

    def _calculate_current_leverage(self, equity: float) -> float:
        """Calculate current portfolio leverage."""
        total_position_value = sum(
            abs(pos_data["quantity"] * pos_data["last_price"])
            for pos_data in self.positions.values()
            if pos_data["quantity"] != 0
        )

        return total_position_value / equity if equity > 0 else 0.0

    def check_day_stop(self, current_equity: float) -> bool:
        """Check if daily stop loss is triggered."""
        if self.day_start_equity == 0.0:
            self.day_start_equity = current_equity
            return False

        daily_drawdown = (self.day_start_equity - current_equity) / self.day_start_equity

        if daily_drawdown >= self.day_stop_dd:
            logger.warning(f"Daily stop loss triggered: {daily_drawdown:.2%} >= {self.day_stop_dd:.2%}")
            return True

        return False

    def check_week_stop(self, current_equity: float) -> bool:
        """Check if weekly stop loss is triggered."""
        if self.week_start_equity == 0.0:
            self.week_start_equity = current_equity
            return False

        weekly_drawdown = (self.week_start_equity - current_equity) / self.week_start_equity

        if weekly_drawdown >= self.week_stop_dd:
            logger.warning(f"Weekly stop loss triggered: {weekly_drawdown:.2%} >= {self.week_stop_dd:.2%}")
            return True

        return False

    def reset_daily_counters(self) -> None:
        """Reset daily counters (call at start of each day)."""
        today = datetime.now().date()

        if today != self.day_trade_reset_date:
            self.day_trade_count = 0
            self.day_trade_reset_date = today

            # Reset daily equity tracking
            if self.equity_curve:
                self.day_start_equity = self.equity_curve[-1]

            logger.debug(f"Daily counters reset for {today}")

    def reset_weekly_counters(self) -> None:
        """Reset weekly counters (call at start of each week)."""
        if self.equity_curve:
            self.week_start_equity = self.equity_curve[-1]

        logger.debug("Weekly counters reset")

    def check_pdt_rules(self, symbol: str, quantity: float, current_equity: float) -> tuple[bool, str]:
        """
        Check Pattern Day Trader (PDT) rules compliance.
        
        Args:
            symbol: Trading symbol
            quantity: Proposed trade quantity
            current_equity: Current account equity
            
        Returns:
            Tuple of (is_allowed, reason)
        """
        # PDT rules only apply to equity accounts under $25k
        if current_equity >= self.pdt_threshold:
            return True, "Account above PDT threshold"

        # Check if this would be a day trade
        current_position = self.positions.get(symbol, {}).get("quantity", 0.0)

        # Day trade occurs when opening and closing a position on the same day
        if current_position == 0 and quantity != 0:
            # Opening new position - check if we might close it today
            if self.day_trade_count >= 3:
                return False, f"PDT limit reached: {self.day_trade_count}/3 day trades used"
        elif current_position != 0 and (current_position > 0) != (current_position + quantity > 0):
            # Position flip (long to short or vice versa) counts as day trade
            if self.day_trade_count >= 3:
                return False, f"PDT limit reached: {self.day_trade_count}/3 day trades used"

        return True, "PDT rules compliant"

    def record_day_trade(self, symbol: str) -> None:
        """Record a day trade for PDT tracking."""
        self.day_trade_count += 1
        logger.info(f"Day trade recorded for {symbol}: {self.day_trade_count}/3 used")

    def check_blackout_periods(self, symbol: str, event_features: dict[str, Any] | None = None) -> tuple[bool, str]:
        """
        Check if trading is blocked due to blackout periods.
        
        Args:
            symbol: Trading symbol
            event_features: Features containing earnings_proximity and fomc_day flags
            
        Returns:
            Tuple of (is_allowed, reason)
        """
        if event_features is None:
            return True, "No event data available"

        # Check earnings blackout
        if self.earnings_blackout and event_features.get("earnings_proximity", 0):
            return False, f"Earnings blackout active for {symbol}"

        # Check FOMC blackout
        if self.fomc_blackout and event_features.get("fomc_day", 0):
            return False, "FOMC blackout active"

        return True, "No blackout restrictions"

    def calculate_atr_position_size(self,
                                  symbol: str,
                                  price: float,
                                  atr: float,
                                  equity: float,
                                  stop_loss_atr_mult: float = 1.0) -> float:
        """
        Calculate position size based on ATR and risk per trade.
        
        Args:
            symbol: Trading symbol
            price: Current price
            atr: Average True Range
            equity: Current account equity
            stop_loss_atr_mult: Stop loss distance in ATR units
            
        Returns:
            Position size in shares
        """
        if atr <= 0 or price <= 0 or equity <= 0:
            return 0.0

        # Risk amount in dollars
        risk_amount = equity * self.risk_pct_per_trade

        # Stop distance in price terms
        stop_distance = atr * stop_loss_atr_mult

        # Position size = risk amount / stop distance
        position_size = risk_amount / stop_distance

        # Convert to number of shares
        shares = position_size / price

        logger.debug(f"ATR position sizing for {symbol}: "
                    f"risk_amount=${risk_amount:.2f}, stop_distance=${stop_distance:.2f}, "
                    f"shares={shares:.2f}")

        return shares

    def check_risk_limits(self,
                         symbol: str,
                         quantity: float,
                         price: float,
                         equity: float,
                         sector: str | None = None) -> tuple[bool, str]:
        """
        Check if a proposed trade violates risk limits.
        
        Returns:
            Tuple of (is_allowed, reason)
        """
        if self.is_kill_switch_active:
            return False, "Kill switch is active"

        if quantity == 0:
            return True, "No position change"

        # Check position limits
        new_position_value = abs((self.positions.get(symbol, 0) + quantity) * price)
        max_position_value = self.max_position_pct * equity

        if new_position_value > max_position_value:
            return False, f"Position limit exceeded: {new_position_value:.2f} > {max_position_value:.2f}"

        # Check leverage limits
        current_leverage = self._calculate_current_leverage(equity)
        position_impact = abs(quantity * price) / equity

        if current_leverage + position_impact > self.max_leverage:
            return False, f"Leverage limit exceeded: {current_leverage + position_impact:.2f} > {self.max_leverage:.2f}"

        # Check sector limits
        if sector:
            current_sector = self.sector_exposure.get(sector, 0.0)
            sector_impact = quantity * price
            new_sector_exposure = abs(current_sector + sector_impact)
            max_sector_value = self.max_sector_pct * equity

            if new_sector_exposure > max_sector_value:
                return False, f"Sector limit exceeded: {new_sector_exposure:.2f} > {max_sector_value:.2f}"

        return True, "Risk limits OK"

    def should_accept(self, symbol: str, proposed_weight: float) -> bool:
        """
        Check if a proposed weight for a symbol should be accepted.
        
        Returns False when any cap is violated - use before order placement.
        
        Args:
            symbol: Trading symbol
            proposed_weight: Proposed portfolio weight for the symbol
            
        Returns:
            bool: True if position should be accepted, False otherwise
        """
        if self.is_kill_switch_active:
            logger.warning(f"Kill switch active - rejecting {symbol}")
            return False

        # Check single-name cap
        if abs(proposed_weight) > self.single_name_max_pct:
            logger.warning(f"Single-name cap violated for {symbol}: "
                          f"{abs(proposed_weight):.3f} > {self.single_name_max_pct:.3f}")
            return False

        # Check sector cap (if we can determine sector)
        sector = self._get_sector(symbol)
        if sector:
            # Calculate current sector exposure plus this position
            current_sector_weight = sum(
                abs(pos_info.get("weight", 0.0))
                for sym, pos_info in self.positions.items()
                if self._get_sector(sym) == sector and sym != symbol
            )
            new_sector_weight = current_sector_weight + abs(proposed_weight)

            if new_sector_weight > self.sector_max_pct:
                logger.warning(f"Sector cap violated for {symbol} ({sector}): "
                              f"{new_sector_weight:.3f} > {self.sector_max_pct:.3f}")
                return False

        # Check asset class caps
        asset_class = self._get_asset_class(symbol)
        if asset_class == "crypto":
            # Calculate current crypto exposure plus this position
            current_crypto_weight = sum(
                abs(pos_info.get("weight", 0.0))
                for sym, pos_info in self.positions.items()
                if self._get_asset_class(sym) == "crypto" and sym != symbol
            )
            new_crypto_weight = current_crypto_weight + abs(proposed_weight)

            if new_crypto_weight > self.crypto_max_gross_pct:
                logger.warning(f"Crypto gross cap violated for {symbol}: "
                              f"{new_crypto_weight:.3f} > {self.crypto_max_gross_pct:.3f}")
                return False

        # Check leverage cap (approximate)
        total_gross_weight = sum(
            abs(pos_info.get("weight", 0.0))
            for pos_info in self.positions.values()
        ) + abs(proposed_weight)

        if total_gross_weight > self.max_leverage:
            logger.warning(f"Leverage cap violated for {symbol}: "
                          f"{total_gross_weight:.3f} > {self.max_leverage:.3f}")
            return False

        return True

    def update_equity_curve(self, portfolio_value: float):
        """Update equity curve and check for kill switch."""
        self.equity_curve.append(portfolio_value)

        # Update peak equity
        self.peak_equity = max(self.peak_equity, portfolio_value)

        # Calculate drawdown
        current_dd = 0.0
        if self.peak_equity > 0:
            current_dd = (self.peak_equity - portfolio_value) / self.peak_equity

            # Check kill switch activation
            if current_dd >= self.kill_switch_dd and not self.is_kill_switch_active:
                self.activate_kill_switch(current_dd)

            # Check kill switch deactivation (optional recovery mechanism)
            elif current_dd < self.kill_switch_dd * 0.5 and self.is_kill_switch_active:
                # Deactivate if drawdown reduces to half the threshold
                self._deactivate_kill_switch(current_dd)

        # Calculate daily return
        if self.last_portfolio_value > 0:
            daily_return = (portfolio_value - self.last_portfolio_value) / self.last_portfolio_value
            self.daily_returns.append(daily_return)

            # Update volatility targeting (weekly)
            if len(self.daily_returns) % 7 == 0:  # Update weekly
                self._update_volatility_targeting()

        self.last_portfolio_value = portfolio_value

        # Keep equity curve manageable (last 1000 points)
        if len(self.equity_curve) > 1000:
            self.equity_curve = self.equity_curve[-1000:]

        # Keep daily returns manageable (last 252 trading days)
        if len(self.daily_returns) > 252:
            self.daily_returns = self.daily_returns[-252:]

        logger.debug(f"Equity update: value={portfolio_value:.2f}, "
                    f"peak={self.peak_equity:.2f}, dd={current_dd:.2%}, "
                    f"kill_switch={self.is_kill_switch_active}")

    def activate_kill_switch(self, drawdown: float):
        """Activate kill switch due to excessive drawdown."""
        self.is_kill_switch_active = True
        self.kill_switch_activation_time = datetime.now()

        message = f"KILL SWITCH ACTIVATED: Drawdown {drawdown:.2%} >= {self.kill_switch_dd:.2%}"
        logger.error(message)

        # Send Discord alert
        self._send_risk_alert("KILL_SWITCH_ACTIVE", message, "CRITICAL")

        # Log position summary
        total_exposure = sum(
            abs(pos_data["quantity"] * pos_data["last_price"])
            for pos_data in self.positions.values()
        )

        logger.error(f"Kill switch activated with ${total_exposure:.2f} total exposure across "
                    f"{len([p for p in self.positions.values() if p['quantity'] != 0])} positions")

    def _deactivate_kill_switch(self, current_dd: float):
        """Deactivate kill switch when drawdown improves."""
        self.is_kill_switch_active = False
        activation_duration = datetime.now() - self.kill_switch_activation_time if self.kill_switch_activation_time else timedelta(0)

        message = f"Kill switch deactivated: drawdown improved to {current_dd:.2%} (was active for {activation_duration})"
        logger.info(message)

        # Send recovery alert
        self._send_risk_alert("KILL_SWITCH_DEACTIVATED", message, "INFO")

    def _update_volatility_targeting(self):
        """Update volatility targeting scaler based on recent performance."""
        if len(self.daily_returns) < 20:
            return  # Need sufficient data

        # Calculate realized volatility (annualized)
        recent_returns = np.array(self.daily_returns[-20:])  # Last 20 days
        realized_vol = np.std(recent_returns) * np.sqrt(252)  # Annualized

        # Calculate target scaler
        if realized_vol > 0:
            target_scaler = self.vol_target_ann / realized_vol

            # Smooth the adjustment (don't change too quickly)
            adjustment_speed = 0.1
            self.vol_target_scaler = (
                (1 - adjustment_speed) * self.vol_target_scaler +
                adjustment_speed * target_scaler
            )

            # Reasonable bounds
            self.vol_target_scaler = np.clip(self.vol_target_scaler, 0.2, 3.0)

            logger.debug(f"Vol targeting: realized={realized_vol:.2%}, "
                        f"target={self.vol_target_ann:.2%}, "
                        f"scaler={self.vol_target_scaler:.2f}")

        self.last_vol_update = datetime.now()

    def _send_risk_alert(self, event_type: str, message: str, severity: str = "WARNING"):
        """Send risk management alert."""
        try:
            # Import here to avoid circular imports
            from .alerts import send_risk_alert
            send_risk_alert(event_type, message, severity)
        except Exception as e:
            logger.warning(f"Failed to send risk alert: {e}")

    def flatten_all_positions(self) -> list[dict[str, Any]]:
        """Generate orders to flatten all positions (used by kill switch)."""
        flatten_orders = []

        for symbol, pos_data in self.positions.items():
            if pos_data["quantity"] != 0:
                flatten_orders.append({
                    "symbol": symbol,
                    "quantity": -pos_data["quantity"],  # Opposite direction
                    "order_type": "market",
                    "reason": "kill_switch_flatten"
                })

        logger.info(f"Generated {len(flatten_orders)} flatten orders for kill switch")
        return flatten_orders

    def get_risk_metrics(self) -> dict[str, Any]:
        """Get comprehensive risk metrics."""
        current_dd = 0.0
        if self.peak_equity > 0 and len(self.equity_curve) > 0:
            current_dd = (self.peak_equity - self.equity_curve[-1]) / self.peak_equity

        # Calculate realized volatility
        realized_vol = 0.0
        if len(self.daily_returns) >= 20:
            realized_vol = np.std(self.daily_returns[-20:]) * np.sqrt(252)

        # Position summary
        active_positions = sum(1 for pos in self.positions.values() if pos["quantity"] != 0)
        total_exposure = sum(
            abs(pos["quantity"] * pos["last_price"])
            for pos in self.positions.values()
        )

        current_leverage = 0.0
        if len(self.equity_curve) > 0:
            current_leverage = total_exposure / self.equity_curve[-1] if self.equity_curve[-1] > 0 else 0.0

        return {
            "current_drawdown": current_dd,
            "peak_equity": self.peak_equity,
            "kill_switch_active": self.is_kill_switch_active,
            "kill_switch_threshold": self.kill_switch_dd,
            "realized_volatility_ann": realized_vol,
            "vol_target_ann": self.vol_target_ann,
            "vol_target_scaler": self.vol_target_scaler,
            "current_leverage": current_leverage,
            "max_leverage": self.max_leverage,
            "active_positions": active_positions,
            "total_exposure": total_exposure,
            "sector_exposure": self.sector_exposure.copy(),
            "asset_class_exposure": self.asset_class_exposure.copy(),
            "total_trades": len(self.trade_history)
        }

    def deactivate_kill_switch(self):
        """Manually deactivate kill switch."""
        self.is_kill_switch_active = False
        logger.info("Kill switch manually deactivated")

    def calculate_vol_scaling(self) -> float:
        """Calculate volatility scaling factor for portfolio."""
        if len(self.daily_returns) < 30:  # Need at least 30 days
            return 1.0

        # Calculate realized volatility (annualized)
        returns_array = np.array(self.daily_returns[-252:])  # Last year
        realized_vol = np.std(returns_array) * np.sqrt(252)

        if realized_vol <= 0:
            return 1.0

        # Calculate scaling factor
        vol_scaling = self.vol_target / realized_vol

        # Limit scaling to prevent extreme adjustments
        vol_scaling = np.clip(vol_scaling, 0.5, 2.0)

        logger.debug(f"Vol scaling: realized={realized_vol:.1%}, target={self.vol_target:.1%}, scaling={vol_scaling:.2f}")

        return vol_scaling

    def get_risk_metrics(self) -> dict[str, Any]:
        """Calculate current risk metrics."""
        if not self.equity_curve:
            return {}

        current_equity = self.equity_curve[-1]

        metrics = {
            "current_equity": current_equity,
            "peak_equity": self.peak_equity,
            "current_drawdown": (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0,
            "kill_switch_active": self.is_kill_switch_active,
            "current_leverage": self._calculate_current_leverage(current_equity),
            "position_count": len([p for p in self.positions.values() if p != 0]),
            "total_position_value": sum(abs(pos * 100) for pos in self.positions.values()),  # Assume $100 price
        }

        if len(self.daily_returns) > 0:
            returns_array = np.array(self.daily_returns)
            metrics.update({
                "daily_vol": np.std(returns_array),
                "annualized_vol": np.std(returns_array) * np.sqrt(252),
                "sharpe_ratio": np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0,
                "max_daily_loss": np.min(returns_array) if len(returns_array) > 0 else 0,
                "max_daily_gain": np.max(returns_array) if len(returns_array) > 0 else 0,
            })

        return metrics

    def _calculate_current_leverage(self, equity: float) -> float:
        """Calculate current gross leverage."""
        if equity <= 0:
            return 0.0

        total_exposure = sum(abs(pos * 100) for pos in self.positions.values())  # Assume $100 price
        return total_exposure / equity

    def _get_sector(self, symbol: str) -> str:
        """Get sector for symbol (simplified implementation)."""
        # This would typically come from a security master database
        sector_mapping = {
            "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
            "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
            "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
            "JNJ": "Healthcare", "PFE": "Healthcare", "UNH": "Healthcare",
            "BTCUSD": "Crypto", "ETHUSD": "Crypto"
        }
        return sector_mapping.get(symbol, "Other")

    def _send_discord_notification(self, message: str):
        """Send Discord notification if webhook URL is configured."""
        webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
        if not webhook_url:
            return

        try:
            payload = {
                "content": message,
                "username": "Quant Bot Risk Manager"
            }

            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()

            logger.info("Discord notification sent successfully")

        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")

    def log_trade(self,
                  symbol: str,
                  side: str,
                  quantity: float,
                  price: float,
                  timestamp: datetime,
                  strategy: str = "unknown"):
        """Log trade for performance tracking."""
        trade = {
            "timestamp": timestamp,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "value": quantity * price,
            "strategy": strategy
        }

        self.trade_history.append(trade)
        logger.info(f"Trade logged: {side} {quantity} {symbol} @ {price} ({strategy})")

    def get_trade_statistics(self) -> dict[str, Any]:
        """Calculate trade statistics."""
        if not self.trade_history:
            return {}

        df = pd.DataFrame(self.trade_history)

        stats = {
            "total_trades": len(df),
            "buy_trades": len(df[df["side"] == "buy"]),
            "sell_trades": len(df[df["side"] == "sell"]),
            "avg_trade_size": df["value"].mean(),
            "largest_trade": df["value"].max(),
            "smallest_trade": df["value"].min(),
            "most_traded_symbol": df["symbol"].value_counts().index[0] if len(df) > 0 else None,
            "strategy_breakdown": df["strategy"].value_counts().to_dict(),
        }

        return stats


class PortfolioOptimizer:
    """Portfolio optimization utilities for risk management."""

    def __init__(self, lookback_days: int = 252):
        """Initialize portfolio optimizer."""
        self.lookback_days = lookback_days

    def calculate_correlation_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix of returns."""
        return returns.corr()

    def calculate_covariance_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate covariance matrix of returns."""
        return returns.cov() * 252  # Annualized

    def hierarchical_risk_parity(self, returns: pd.DataFrame) -> pd.Series:
        """
        Implement Hierarchical Risk Parity (HRP) allocation.
        
        This is a simplified implementation of the HRP algorithm
        that builds a hierarchy of assets based on correlations.
        """
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import squareform

        # Calculate correlation matrix
        corr_matrix = self.calculate_correlation_matrix(returns)

        # Convert correlation to distance matrix
        distance_matrix = np.sqrt((1 - corr_matrix) / 2)

        # Hierarchical clustering
        linkage_matrix = linkage(squareform(distance_matrix.values), method="single")

        # Initialize equal weights
        weights = pd.Series(1.0 / len(returns.columns), index=returns.columns)

        # This is a simplified HRP - in practice, you'd implement the full tree-based allocation
        # For now, we'll use inverse volatility weighting as a proxy
        vol = returns.std()
        inv_vol_weights = (1 / vol) / (1 / vol).sum()

        return inv_vol_weights
