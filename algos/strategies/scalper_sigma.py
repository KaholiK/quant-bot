"""
Scalper Sigma Strategy - Mean-reversion scalping with tight stops and partial profit targets.
Uses statistical mean-reversion signals with sigma bands for entry/exit decisions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger
from datetime import datetime, timedelta


class ScalperSigmaStrategy:
    """
    Mean-reversion scalping strategy with sigma-based entry/exit logic.
    
    Strategy characteristics:
    - Tight stop loss: 1.0 x ATR
    - Profit targets: 0.75-1.0 x ATR with partial exits
    - Minimum holding period to avoid overtrading
    - Order spacing based on ATR multiples
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize scalper sigma strategy."""
        self.config = config
        self.strategy_config = config['strategies']['scalper_sigma']
        
        # Strategy parameters
        self.enabled = self.strategy_config.get('enabled', True)
        self.stop_mult = self.strategy_config.get('stop_mult', 1.0)
        self.take_profit_mult = self.strategy_config.get('take_profit_mult', [0.75, 1.0])
        self.min_hold_seconds = self.strategy_config.get('min_hold_seconds', 30)
        self.order_spacing_atr = self.strategy_config.get('order_spacing_atr', 0.1)
        
        # Signal parameters
        self.sigma_entry_threshold = 2.0  # Enter when price is 2 sigma away
        self.sigma_exit_threshold = 0.5   # Exit when price returns to 0.5 sigma
        self.lookback_period = 20         # Period for calculating mean and std
        
        # State tracking
        self.positions: Dict[str, Dict] = {}
        self.last_trade_time: Dict[str, datetime] = {}
        self.signal_history: List[Dict] = []
        
        logger.info(f"Scalper Sigma strategy initialized: enabled={self.enabled}")
    
    def calculate_signals(self, 
                         symbol: str,
                         price_data: pd.DataFrame,
                         features: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate scalping signals based on mean reversion.
        
        Args:
            symbol: Trading symbol
            price_data: OHLCV data
            features: Calculated features including ATR
            
        Returns:
            Dictionary with signal information
        """
        if not self.enabled:
            return {'signal': 0, 'confidence': 0.0}
            
        if len(price_data) < self.lookback_period:
            return {'signal': 0, 'confidence': 0.0, 'reason': 'insufficient_data'}
        
        # Get latest data
        current_price = price_data['close'].iloc[-1]
        atr = features['atr'].iloc[-1] if 'atr' in features.columns else None
        
        if pd.isna(current_price) or pd.isna(atr) or atr <= 0:
            return {'signal': 0, 'confidence': 0.0, 'reason': 'invalid_data'}
        
        # Calculate rolling mean and standard deviation
        price_series = price_data['close'].tail(self.lookback_period)
        rolling_mean = price_series.mean()
        rolling_std = price_series.std()
        
        if rolling_std <= 0:
            return {'signal': 0, 'confidence': 0.0, 'reason': 'zero_volatility'}
        
        # Calculate z-score (sigma distance from mean)
        z_score = (current_price - rolling_mean) / rolling_std
        
        # Additional filters
        signal_filters = self._apply_signal_filters(symbol, price_data, features)
        if not signal_filters['passed']:
            return {'signal': 0, 'confidence': 0.0, 'reason': signal_filters['reason']}
        
        # Generate signals
        signal = 0
        confidence = 0.0
        reason = ""
        
        if z_score < -self.sigma_entry_threshold:
            # Price is oversold, signal to buy
            signal = 1
            confidence = min(1.0, abs(z_score) / 3.0)  # Cap confidence at 3-sigma
            reason = f"oversold_z_score_{z_score:.2f}"
            
        elif z_score > self.sigma_entry_threshold:
            # Price is overbought, signal to sell
            signal = -1
            confidence = min(1.0, abs(z_score) / 3.0)
            reason = f"overbought_z_score_{z_score:.2f}"
        
        # Check exit conditions for existing positions
        if symbol in self.positions and self.positions[symbol]['size'] != 0:
            exit_signal = self._check_exit_conditions(symbol, z_score, current_price, atr)
            if exit_signal:
                signal = -np.sign(self.positions[symbol]['size'])
                confidence = 1.0
                reason = "exit_condition_met"
        
        # Log signal generation
        signal_info = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'signal': signal,
            'confidence': confidence,
            'z_score': z_score,
            'current_price': current_price,
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std,
            'atr': atr,
            'reason': reason
        }
        
        self.signal_history.append(signal_info)
        
        # Keep last 1000 signals
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]
        
        return {
            'signal': signal,
            'confidence': confidence,
            'z_score': z_score,
            'reason': reason,
            'stop_price': self._calculate_stop_price(current_price, signal, atr),
            'take_profit_prices': self._calculate_take_profit_prices(current_price, signal, atr),
            'atr': atr
        }
    
    def _apply_signal_filters(self, 
                            symbol: str,
                            price_data: pd.DataFrame, 
                            features: pd.DataFrame) -> Dict[str, Any]:
        """Apply additional filters to avoid bad signals."""
        
        # Filter 1: Minimum holding period
        if symbol in self.last_trade_time:
            time_since_last = (datetime.now() - self.last_trade_time[symbol]).total_seconds()
            if time_since_last < self.min_hold_seconds:
                return {'passed': False, 'reason': 'min_hold_period_not_met'}
        
        # Filter 2: Volume filter (if available)
        if 'volume' in price_data.columns:
            avg_volume = price_data['volume'].tail(20).mean()
            current_volume = price_data['volume'].iloc[-1]
            if current_volume < 0.3 * avg_volume:  # Low volume
                return {'passed': False, 'reason': 'low_volume'}
        
        # Filter 3: Volatility filter
        if 'realized_vol' in features.columns:
            current_vol = features['realized_vol'].iloc[-1]
            if pd.notna(current_vol) and current_vol > 0.5:  # Very high volatility
                return {'passed': False, 'reason': 'excessive_volatility'}
        
        # Filter 4: Spread filter (for crypto)
        if 'spread' in features.columns:
            spread = features['spread'].iloc[-1]
            if pd.notna(spread) and spread > 0.005:  # Wide spread > 0.5%
                return {'passed': False, 'reason': 'wide_spread'}
        
        return {'passed': True, 'reason': 'all_filters_passed'}
    
    def _check_exit_conditions(self, 
                              symbol: str, 
                              z_score: float,
                              current_price: float,
                              atr: float) -> bool:
        """Check if existing position should be exited."""
        position = self.positions[symbol]
        entry_price = position['entry_price']
        position_size = position['size']
        entry_time = position['entry_time']
        
        # Time-based exit: minimum holding period
        time_held = (datetime.now() - entry_time).total_seconds()
        if time_held < self.min_hold_seconds:
            return False
        
        # Mean reversion exit: price returned close to mean
        if abs(z_score) < self.sigma_exit_threshold:
            logger.info(f"Exit signal for {symbol}: z-score returned to {z_score:.2f}")
            return True
        
        # Stop loss
        if position_size > 0:  # Long position
            stop_price = entry_price - self.stop_mult * atr
            if current_price <= stop_price:
                logger.info(f"Stop loss triggered for {symbol}: {current_price} <= {stop_price}")
                return True
        else:  # Short position
            stop_price = entry_price + self.stop_mult * atr
            if current_price >= stop_price:
                logger.info(f"Stop loss triggered for {symbol}: {current_price} >= {stop_price}")
                return True
        
        # Take profit (check partial profit targets)
        for i, tp_mult in enumerate(self.take_profit_mult):
            if position_size > 0:  # Long position
                tp_price = entry_price + tp_mult * atr
                if current_price >= tp_price:
                    logger.info(f"Take profit {i+1} hit for {symbol}: {current_price} >= {tp_price}")
                    return True
            else:  # Short position
                tp_price = entry_price - tp_mult * atr
                if current_price <= tp_price:
                    logger.info(f"Take profit {i+1} hit for {symbol}: {current_price} <= {tp_price}")
                    return True
        
        return False
    
    def _calculate_stop_price(self, current_price: float, signal: int, atr: float) -> Optional[float]:
        """Calculate stop loss price."""
        if signal == 0 or atr <= 0:
            return None
            
        if signal > 0:  # Long position
            return current_price - self.stop_mult * atr
        else:  # Short position
            return current_price + self.stop_mult * atr
    
    def _calculate_take_profit_prices(self, current_price: float, signal: int, atr: float) -> List[float]:
        """Calculate take profit prices."""
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
                       timestamp: Optional[datetime] = None):
        """Update position tracking."""
        if timestamp is None:
            timestamp = datetime.now()
        
        if size == 0:
            # Position closed
            if symbol in self.positions:
                del self.positions[symbol]
            self.last_trade_time[symbol] = timestamp
        else:
            # Position opened or modified
            self.positions[symbol] = {
                'size': size,
                'entry_price': entry_price,
                'entry_time': timestamp,
                'unrealized_pnl': 0.0
            }
            self.last_trade_time[symbol] = timestamp
        
        logger.debug(f"Position updated for {symbol}: size={size}, entry_price={entry_price}")
    
    def update_unrealized_pnl(self, symbol: str, current_price: float):
        """Update unrealized PnL for position."""
        if symbol not in self.positions:
            return
            
        position = self.positions[symbol]
        position_size = position['size']
        entry_price = position['entry_price']
        
        if position_size > 0:  # Long position
            unrealized_pnl = (current_price - entry_price) * position_size
        else:  # Short position
            unrealized_pnl = (entry_price - current_price) * abs(position_size)
        
        position['unrealized_pnl'] = unrealized_pnl
    
    def get_position_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position information for symbol."""
        return self.positions.get(symbol)
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """Get strategy performance statistics."""
        stats = {
            'strategy_name': 'scalper_sigma',
            'enabled': self.enabled,
            'active_positions': len(self.positions),
            'total_signals': len(self.signal_history)
        }
        
        if self.signal_history:
            df = pd.DataFrame(self.signal_history)
            stats.update({
                'signal_distribution': df['signal'].value_counts().to_dict(),
                'avg_confidence': df['confidence'].mean(),
                'avg_z_score': df['z_score'].abs().mean()
            })
        
        # Position statistics
        if self.positions:
            total_unrealized = sum(pos['unrealized_pnl'] for pos in self.positions.values())
            stats['total_unrealized_pnl'] = total_unrealized
            
            long_positions = sum(1 for pos in self.positions.values() if pos['size'] > 0)
            short_positions = sum(1 for pos in self.positions.values() if pos['size'] < 0)
            stats['long_positions'] = long_positions
            stats['short_positions'] = short_positions
        
        return stats
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get risk metrics for the strategy."""
        metrics = {
            'max_position_size': max([abs(pos['size']) for pos in self.positions.values()] + [0]),
            'position_concentration': len(self.positions),
            'avg_holding_time': self._calculate_avg_holding_time(),
        }
        
        return metrics
    
    def _calculate_avg_holding_time(self) -> float:
        """Calculate average holding time for positions."""
        if not self.positions:
            return 0.0
            
        holding_times = []
        current_time = datetime.now()
        
        for position in self.positions.values():
            holding_time = (current_time - position['entry_time']).total_seconds()
            holding_times.append(holding_time)
        
        return np.mean(holding_times) if holding_times else 0.0
    
    def should_trade_symbol(self, symbol: str, market_conditions: Dict[str, Any]) -> bool:
        """Determine if strategy should trade this symbol given market conditions."""
        # Skip if strategy disabled
        if not self.enabled:
            return False
        
        # Skip crypto during low liquidity hours (simplified)
        if symbol in ['BTCUSD', 'ETHUSD']:
            current_hour = datetime.now().hour
            if current_hour < 6 or current_hour > 22:  # Avoid very low liquidity hours
                return False
        
        # Check volatility conditions
        volatility = market_conditions.get('volatility', 0.0)
        if volatility > 0.4:  # Skip in extremely volatile conditions
            return False
        
        # Check if we already have position in this symbol
        if symbol in self.positions and abs(self.positions[symbol]['size']) > 0:
            # Allow position management
            return True
            
        # Limit number of concurrent positions
        if len(self.positions) >= 5:  # Max 5 concurrent scalping positions
            return False
        
        return True