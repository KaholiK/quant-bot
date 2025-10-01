"""
Gamma Reversal Strategy - High-frequency mean reversion strategy for crypto markets.
Uses 1-5 minute bars with tight stops and quick profit targets for scalping reversals.
"""

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


class GammaReversalStrategy:
    """
    High-frequency gamma reversal strategy for crypto markets.
    
    Strategy characteristics:
    - Operates on 1-5 minute timeframes
    - Tight profit targets: 0.5-0.75 x ATR
    - Strict stop loss: 0.5 x ATR
    - Minimum holding period to avoid overtrading
    - Focus on crypto pairs with high gamma (volatility of volatility)
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize gamma reversal strategy."""
        self.config = config
        self.strategy_config = config["strategies"]["gamma_reversal"]

        # Strategy parameters
        self.enabled = self.strategy_config.get("enabled", True)
        self.timeframes = self.strategy_config.get("timeframes", ["1m", "5m"])
        self.take_profit_mult = self.strategy_config.get("take_profit_mult", [0.5, 0.75])
        self.stop_loss_mult = self.strategy_config.get("stop_loss_mult", 0.5)
        self.min_hold_seconds = self.strategy_config.get("min_hold_seconds", 60)

        # Gamma detection parameters
        self.vol_lookback = 20          # Period for volatility calculation
        self.gamma_threshold = 0.8      # Minimum gamma score for entry
        self.mean_reversion_periods = [5, 10, 20]  # Multiple timeframes for mean reversion
        self.volume_spike_threshold = 1.5   # Volume spike multiplier

        # State tracking
        self.positions: dict[str, dict] = {}
        self.gamma_scores: dict[str, float] = {}
        self.last_trade_time: dict[str, datetime] = {}
        self.signal_history: list[dict] = []
        self.microstructure_data: dict[str, dict] = {}

        # Performance tracking
        self.trade_count = 0
        self.win_count = 0
        self.total_pnl = 0.0

        logger.info(f"Gamma Reversal strategy initialized: enabled={self.enabled}")

    def calculate_gamma_score(self,
                             price_data: pd.DataFrame,
                             features: pd.DataFrame) -> float:
        """
        Calculate gamma score (volatility of volatility).
        
        Args:
            price_data: OHLCV data
            features: Technical features including volatility measures
            
        Returns:
            Gamma score (0-1, higher = more gamma)
        """
        if len(price_data) < self.vol_lookback * 2:
            return 0.0

        # Calculate rolling volatility
        returns = price_data["close"].pct_change()
        rolling_vol = returns.rolling(window=self.vol_lookback).std()

        if len(rolling_vol.dropna()) < self.vol_lookback:
            return 0.0

        # Calculate volatility of volatility (gamma)
        vol_of_vol = rolling_vol.pct_change().std()

        # Normalize gamma score
        gamma_score = min(1.0, vol_of_vol * 100)  # Scale appropriately

        # Additional gamma indicators
        if "atr" in features.columns:
            # ATR acceleration
            atr_series = features["atr"].tail(10)
            atr_trend = atr_series.pct_change().mean() if len(atr_series) > 1 else 0.0
            gamma_score += min(0.3, abs(atr_trend) * 10)

        # Volume-price divergence
        if "volume" in price_data.columns and len(price_data) >= 5:
            price_change = price_data["close"].pct_change(5).iloc[-1]
            volume_change = price_data["volume"].pct_change(5).iloc[-1]

            if abs(price_change) > 0.001 and abs(volume_change) > 0.1:
                # High volume with small price change = potential reversal setup
                divergence_score = abs(volume_change) / (abs(price_change) * 100)
                gamma_score += min(0.2, divergence_score)

        return min(1.0, gamma_score)

    def calculate_signals(self,
                         symbol: str,
                         price_data_1m: pd.DataFrame,
                         price_data_5m: pd.DataFrame,
                         features: pd.DataFrame,
                         orderbook_data: dict | None = None) -> dict[str, Any]:
        """
        Calculate high-frequency reversal signals.
        
        Args:
            symbol: Trading symbol (should be crypto)
            price_data_1m: 1-minute OHLCV data
            price_data_5m: 5-minute OHLCV data  
            features: Technical features
            orderbook_data: Optional order book data
            
        Returns:
            Dictionary with signal information
        """
        if not self.enabled:
            return {"signal": 0, "confidence": 0.0}

        # Only trade crypto symbols
        if symbol not in ["BTCUSD", "ETHUSD"]:
            return {"signal": 0, "confidence": 0.0, "reason": "non_crypto_symbol"}

        # Use 1m data as primary for signals
        primary_data = price_data_1m

        if len(primary_data) < max(self.mean_reversion_periods):
            return {"signal": 0, "confidence": 0.0, "reason": "insufficient_data"}

        # Calculate gamma score
        gamma_score = self.calculate_gamma_score(primary_data, features)
        self.gamma_scores[symbol] = gamma_score

        if gamma_score < self.gamma_threshold:
            return {"signal": 0, "confidence": 0.0, "reason": f"low_gamma_{gamma_score:.2f}"}

        # Get current market data
        current_price = primary_data["close"].iloc[-1]
        current_volume = primary_data["volume"].iloc[-1] if "volume" in primary_data.columns else 0
        atr = features["atr"].iloc[-1] if "atr" in features.columns else None

        if pd.isna(current_price) or pd.isna(atr) or atr <= 0:
            return {"signal": 0, "confidence": 0.0, "reason": "invalid_data"}

        # Multi-timeframe mean reversion signals
        mr_signals = []
        for period in self.mean_reversion_periods:
            if len(primary_data) >= period:
                mean_price = primary_data["close"].tail(period).mean()
                std_price = primary_data["close"].tail(period).std()

                if std_price > 0:
                    z_score = (current_price - mean_price) / std_price
                    mr_signals.append(z_score)

        if not mr_signals:
            return {"signal": 0, "confidence": 0.0, "reason": "no_mean_reversion_signals"}

        # Combine mean reversion signals
        avg_z_score = np.mean(mr_signals)
        max_z_score = max(mr_signals, key=abs)

        # Check microstructure conditions
        microstructure_signal = self._check_microstructure(symbol, primary_data, orderbook_data)

        # Apply filters
        signal_filters = self._apply_hf_filters(symbol, primary_data, features)
        if not signal_filters["passed"]:
            return {"signal": 0, "confidence": 0.0, "reason": signal_filters["reason"]}

        # Generate signals
        signal = 0
        confidence = 0.0
        reason = ""

        reversal_threshold = 1.5  # Lower threshold for high-frequency

        if max_z_score < -reversal_threshold and microstructure_signal >= 0:
            # Oversold, signal to buy
            signal = 1
            confidence = min(1.0, abs(max_z_score) / 3.0 * gamma_score)
            reason = f"oversold_reversal_z_{max_z_score:.2f}_gamma_{gamma_score:.2f}"

        elif max_z_score > reversal_threshold and microstructure_signal <= 0:
            # Overbought, signal to sell
            signal = -1
            confidence = min(1.0, abs(max_z_score) / 3.0 * gamma_score)
            reason = f"overbought_reversal_z_{max_z_score:.2f}_gamma_{gamma_score:.2f}"

        # Check exit conditions for existing positions
        if symbol in self.positions and self.positions[symbol]["size"] != 0:
            exit_signal = self._check_exit_conditions(symbol, current_price, atr, avg_z_score)
            if exit_signal:
                signal = -np.sign(self.positions[symbol]["size"])
                confidence = 1.0
                reason = "exit_condition_met"

        # Store signal information
        signal_info = {
            "timestamp": datetime.now(),
            "symbol": symbol,
            "signal": signal,
            "confidence": confidence,
            "gamma_score": gamma_score,
            "z_scores": mr_signals,
            "avg_z_score": avg_z_score,
            "max_z_score": max_z_score,
            "current_price": current_price,
            "atr": atr,
            "microstructure_signal": microstructure_signal,
            "reason": reason
        }

        self.signal_history.append(signal_info)

        # Keep last 1000 signals
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]

        return {
            "signal": signal,
            "confidence": confidence,
            "gamma_score": gamma_score,
            "z_scores": mr_signals,
            "reason": reason,
            "stop_price": self._calculate_stop_price(current_price, signal, atr),
            "take_profit_prices": self._calculate_take_profit_prices(current_price, signal, atr),
            "atr": atr,
            "timeframe": "1m"
        }

    def _check_microstructure(self,
                             symbol: str,
                             price_data: pd.DataFrame,
                             orderbook_data: dict | None) -> int:
        """
        Check microstructure conditions for reversal signals.
        
        Returns:
            1: Bullish microstructure, 0: neutral, -1: bearish microstructure
        """
        microstructure_score = 0

        # Volume spike detection
        if "volume" in price_data.columns and len(price_data) >= 10:
            avg_volume = price_data["volume"].tail(10).mean()
            current_volume = price_data["volume"].iloc[-1]

            if current_volume > self.volume_spike_threshold * avg_volume:
                # Volume spike often precedes reversals
                microstructure_score += 1

        # Order book imbalance (if available)
        if orderbook_data:
            bid_size = orderbook_data.get("bid_size", 0)
            ask_size = orderbook_data.get("ask_size", 0)
            total_size = bid_size + ask_size

            if total_size > 0:
                imbalance = (bid_size - ask_size) / total_size
                if imbalance > 0.2:  # Strong bid support
                    microstructure_score += 1
                elif imbalance < -0.2:  # Strong ask pressure
                    microstructure_score -= 1

        # Price action patterns
        if len(price_data) >= 5:
            recent_highs = price_data["high"].tail(5)
            recent_lows = price_data["low"].tail(5)
            current_close = price_data["close"].iloc[-1]

            # Check for double bottom/top patterns
            if (current_close <= recent_lows.quantile(0.2) and
                recent_lows.iloc[-2:].min() < recent_lows.iloc[-5:-2].min()):
                microstructure_score += 1  # Potential bottom

            elif (current_close >= recent_highs.quantile(0.8) and
                  recent_highs.iloc[-2:].max() > recent_highs.iloc[-5:-2].max()):
                microstructure_score -= 1  # Potential top

        return np.clip(microstructure_score, -1, 1)

    def _apply_hf_filters(self,
                         symbol: str,
                         price_data: pd.DataFrame,
                         features: pd.DataFrame) -> dict[str, Any]:
        """Apply high-frequency specific filters."""

        # Filter 1: Minimum holding period
        if symbol in self.last_trade_time:
            time_since_last = (datetime.now() - self.last_trade_time[symbol]).total_seconds()
            if time_since_last < self.min_hold_seconds:
                return {"passed": False, "reason": "min_hold_period_not_met"}

        # Filter 2: Spread filter (critical for HF trading)
        if "spread" in features.columns:
            spread = features["spread"].iloc[-1]
            if pd.notna(spread) and spread > 0.002:  # 0.2% spread
                return {"passed": False, "reason": "wide_spread"}

        # Filter 3: Avoid low liquidity periods
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:  # Very low liquidity hours for crypto
            return {"passed": False, "reason": "low_liquidity_hours"}

        # Filter 4: Volatility spike filter
        if "atr" in features.columns and len(features) >= 5:
            current_atr = features["atr"].iloc[-1]
            avg_atr = features["atr"].tail(5).mean()

            if current_atr > 2.0 * avg_atr:  # ATR spike
                return {"passed": False, "reason": "volatility_spike"}

        # Filter 5: Consecutive loss prevention
        if symbol in self.positions:
            # Check recent trade history to avoid revenge trading
            recent_trades = [s for s in self.signal_history
                           if s["symbol"] == symbol and
                           (datetime.now() - s["timestamp"]).total_seconds() < 3600]  # Last hour

            if len(recent_trades) >= 5:  # Too many trades in short period
                return {"passed": False, "reason": "overtrading_prevention"}

        return {"passed": True, "reason": "all_filters_passed"}

    def _check_exit_conditions(self,
                              symbol: str,
                              current_price: float,
                              atr: float,
                              z_score: float) -> bool:
        """Check high-frequency exit conditions."""
        position = self.positions[symbol]
        entry_price = position["entry_price"]
        position_size = position["size"]
        entry_time = position["entry_time"]

        # Time-based exit: strict minimum holding
        time_held = (datetime.now() - entry_time).total_seconds()
        if time_held < self.min_hold_seconds:
            return False

        # Quick mean reversion exit
        if abs(z_score) < 0.3:  # Back to mean
            logger.info(f"Mean reversion exit for {symbol}: z-score {z_score:.2f}")
            return True

        # Tight stop loss
        if position_size > 0:  # Long position
            stop_price = entry_price - self.stop_loss_mult * atr
            if current_price <= stop_price:
                logger.info(f"Stop loss for {symbol}: {current_price} <= {stop_price}")
                return True
        else:  # Short position
            stop_price = entry_price + self.stop_loss_mult * atr
            if current_price >= stop_price:
                logger.info(f"Stop loss for {symbol}: {current_price} >= {stop_price}")
                return True

        # Quick take profit
        for i, tp_mult in enumerate(self.take_profit_mult):
            if position_size > 0:  # Long position
                tp_price = entry_price + tp_mult * atr
                if current_price >= tp_price:
                    logger.info(f"Take profit {i+1} for {symbol}: {current_price} >= {tp_price}")
                    return True
            else:  # Short position
                tp_price = entry_price - tp_mult * atr
                if current_price <= tp_price:
                    logger.info(f"Take profit {i+1} for {symbol}: {current_price} <= {tp_price}")
                    return True

        # Time-based exit: maximum holding period (5 minutes for gamma strategy)
        if time_held > 300:  # 5 minutes max hold
            logger.info(f"Time-based exit for {symbol} after {time_held} seconds")
            return True

        return False

    def _calculate_stop_price(self, current_price: float, signal: int, atr: float) -> float | None:
        """Calculate tight stop loss for high-frequency strategy."""
        if signal == 0 or atr <= 0:
            return None

        if signal > 0:  # Long position
            return current_price - self.stop_loss_mult * atr
        # Short position
        return current_price + self.stop_loss_mult * atr

    def _calculate_take_profit_prices(self, current_price: float, signal: int, atr: float) -> list[float]:
        """Calculate tight take profit levels."""
        if signal == 0 or atr <= 0:
            return []

        tp_prices = []

        for tp_mult in self.take_profit_mult:
            if signal > 0:  # Long position
                tp_price = current_price + tp_mult * atr
            else:  # Short position
                tp_price = current_price - tp_mult * atr

            tp_prices.append(tp_price)

        return tp_prices

    def update_position(self,
                       symbol: str,
                       size: float,
                       entry_price: float,
                       timestamp: datetime | None = None):
        """Update position tracking."""
        if timestamp is None:
            timestamp = datetime.now()

        if size == 0:
            # Position closed - update statistics
            if symbol in self.positions:
                old_position = self.positions[symbol]
                pnl = old_position.get("unrealized_pnl", 0.0)

                self.trade_count += 1
                self.total_pnl += pnl

                if pnl > 0:
                    self.win_count += 1

                del self.positions[symbol]

            self.last_trade_time[symbol] = timestamp
        else:
            # Position opened
            self.positions[symbol] = {
                "size": size,
                "entry_price": entry_price,
                "entry_time": timestamp,
                "unrealized_pnl": 0.0
            }
            self.last_trade_time[symbol] = timestamp

        logger.debug(f"Gamma reversal position updated for {symbol}: size={size}")

    def update_unrealized_pnl(self, symbol: str, current_price: float):
        """Update unrealized PnL for high-frequency position."""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        position_size = position["size"]
        entry_price = position["entry_price"]

        if position_size > 0:  # Long position
            unrealized_pnl = (current_price - entry_price) * position_size
        else:  # Short position
            unrealized_pnl = (entry_price - current_price) * abs(position_size)

        position["unrealized_pnl"] = unrealized_pnl

    def get_strategy_stats(self) -> dict[str, Any]:
        """Get high-frequency strategy statistics."""
        win_rate = self.win_count / self.trade_count if self.trade_count > 0 else 0.0
        avg_pnl = self.total_pnl / self.trade_count if self.trade_count > 0 else 0.0

        stats = {
            "strategy_name": "gamma_reversal",
            "enabled": self.enabled,
            "active_positions": len(self.positions),
            "total_trades": self.trade_count,
            "win_count": self.win_count,
            "win_rate": win_rate,
            "total_pnl": self.total_pnl,
            "avg_pnl_per_trade": avg_pnl,
            "gamma_threshold": self.gamma_threshold
        }

        if self.positions:
            total_unrealized = sum(pos["unrealized_pnl"] for pos in self.positions.values())
            stats["total_unrealized_pnl"] = total_unrealized

            # Average holding time
            current_time = datetime.now()
            holding_times = [(current_time - pos["entry_time"]).total_seconds()
                           for pos in self.positions.values()]
            stats["avg_holding_time_seconds"] = np.mean(holding_times) if holding_times else 0

        if self.signal_history:
            df = pd.DataFrame(self.signal_history)
            stats.update({
                "signal_distribution": df["signal"].value_counts().to_dict(),
                "avg_gamma_score": df["gamma_score"].mean(),
                "avg_confidence": df["confidence"].mean()
            })

        return stats

    def should_trade_symbol(self, symbol: str, market_conditions: dict[str, Any]) -> bool:
        """Determine if symbol should be traded with gamma reversal strategy."""
        if not self.enabled:
            return False

        # Only trade crypto
        if symbol not in ["BTCUSD", "ETHUSD"]:
            return False

        # Check if we have sufficient gamma
        gamma_score = self.gamma_scores.get(symbol, 0.0)
        if gamma_score < self.gamma_threshold:
            return False

        # Avoid extremely volatile periods
        volatility = market_conditions.get("volatility", 0.0)
        if volatility > 0.5:  # 50% annualized vol
            return False

        # Limit concurrent positions for HF strategy
        if len(self.positions) >= 2:  # Max 2 concurrent HF positions
            return False

        # Check spread conditions
        spread = market_conditions.get("spread", 0.0)
        if spread > 0.003:  # 0.3% spread too wide for HF
            return False

        return True
