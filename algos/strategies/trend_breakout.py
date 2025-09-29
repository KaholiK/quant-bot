"""
Trend Breakout Strategy - Momentum-based breakout strategy with trend filters.
Uses 55-bar breakout with SMA trend confirmation and momentum ranking.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger
from datetime import datetime, timedelta


class TrendBreakoutStrategy:
    """
    Trend breakout strategy with momentum filters.
    
    Strategy characteristics:
    - 55-bar high/low breakout signals
    - SMA(50) uptrend requirement for longs
    - Momentum rank filter (top 30% only)
    - RSI overbought/oversold blocker
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize trend breakout strategy."""
        self.config = config
        self.strategy_config = config['strategies']['trend_breakout']
        
        # Strategy parameters
        self.enabled = self.strategy_config.get('enabled', True)
        self.lookback_bars = self.strategy_config.get('lookback_bars', 55)
        self.momentum_rank_pct = self.strategy_config.get('momentum_rank_pct', 0.30)
        self.rsi_threshold = self.strategy_config.get('rsi_threshold', 80)
        self.momentum_timeframe = self.strategy_config.get('momentum_timeframe', '3m')
        
        # Trend parameters
        self.trend_sma_period = 50
        self.breakout_confirmation_bars = 2
        
        # State tracking
        self.positions: Dict[str, Dict] = {}
        self.breakout_levels: Dict[str, Dict] = {}
        self.momentum_ranks: Dict[str, float] = {}
        self.signal_history: List[Dict] = []
        
        logger.info(f"Trend Breakout strategy initialized: enabled={self.enabled}")
    
    def calculate_signals(self, 
                         symbol: str,
                         price_data: pd.DataFrame,
                         features: pd.DataFrame,
                         universe_data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, Any]:
        """
        Calculate breakout signals with trend and momentum filters.
        
        Args:
            symbol: Trading symbol
            price_data: OHLCV data
            features: Calculated features
            universe_data: Data for other symbols (for momentum ranking)
            
        Returns:
            Dictionary with signal information
        """
        if not self.enabled:
            return {'signal': 0, 'confidence': 0.0}
            
        if len(price_data) < max(self.lookback_bars, self.trend_sma_period):
            return {'signal': 0, 'confidence': 0.0, 'reason': 'insufficient_data'}
        
        # Get current data
        current_price = price_data['close'].iloc[-1]
        current_high = price_data['high'].iloc[-1]
        current_low = price_data['low'].iloc[-1]
        
        if pd.isna(current_price):
            return {'signal': 0, 'confidence': 0.0, 'reason': 'invalid_price'}
        
        # Calculate breakout levels
        lookback_data = price_data.tail(self.lookback_bars + 1).iloc[:-1]  # Exclude current bar
        breakout_high = lookback_data['high'].max()
        breakout_low = lookback_data['low'].min()
        
        # Update breakout levels tracking
        self.breakout_levels[symbol] = {
            'high': breakout_high,
            'low': breakout_low,
            'timestamp': datetime.now()
        }
        
        # Check trend condition
        trend_signal = self._check_trend_condition(symbol, price_data, features)
        
        # Check momentum condition
        momentum_signal = self._check_momentum_condition(symbol, price_data, universe_data)
        
        # Check RSI condition
        rsi_signal = self._check_rsi_condition(features)
        
        # Generate breakout signals
        signal = 0
        confidence = 0.0
        reason = ""
        
        # Long breakout
        if (current_high > breakout_high and 
            trend_signal >= 0 and 
            momentum_signal >= 0 and 
            rsi_signal != -1):  # Not overbought
            
            signal = 1
            confidence = self._calculate_breakout_confidence(current_price, breakout_high, 'long')
            reason = f"long_breakout_above_{breakout_high:.2f}"
            
        # Short breakout (only if trend allows)
        elif (current_low < breakout_low and 
              trend_signal <= 0 and 
              rsi_signal != 1):  # Not oversold
            
            signal = -1
            confidence = self._calculate_breakout_confidence(current_price, breakout_low, 'short')
            reason = f"short_breakout_below_{breakout_low:.2f}"
        
        # Additional filters
        if signal != 0:
            filters = self._apply_breakout_filters(symbol, price_data, features)
            if not filters['passed']:
                signal = 0
                confidence = 0.0
                reason = filters['reason']
        
        # Check exit conditions for existing positions
        if symbol in self.positions and self.positions[symbol]['size'] != 0:
            exit_signal = self._check_exit_conditions(symbol, price_data, features)
            if exit_signal:
                signal = -np.sign(self.positions[symbol]['size'])
                confidence = 1.0
                reason = "exit_condition_met"
        
        # Log signal
        signal_info = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'signal': signal,
            'confidence': confidence,
            'current_price': current_price,
            'breakout_high': breakout_high,
            'breakout_low': breakout_low,
            'trend_signal': trend_signal,
            'momentum_signal': momentum_signal,
            'rsi_signal': rsi_signal,
            'reason': reason
        }
        
        self.signal_history.append(signal_info)
        
        # Keep last 1000 signals
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]
        
        return {
            'signal': signal,
            'confidence': confidence,
            'breakout_high': breakout_high,
            'breakout_low': breakout_low,
            'reason': reason,
            'stop_price': self._calculate_stop_price(signal, current_price, breakout_high, breakout_low),
            'trend_signal': trend_signal,
            'momentum_rank': self.momentum_ranks.get(symbol, 0.0)
        }
    
    def _check_trend_condition(self, 
                              symbol: str,
                              price_data: pd.DataFrame, 
                              features: pd.DataFrame) -> int:
        """
        Check trend condition using SMA.
        
        Returns:
            1: Strong uptrend, 0: neutral, -1: strong downtrend
        """
        if f'sma_{self.trend_sma_period}' not in features.columns:
            return 0
            
        current_price = price_data['close'].iloc[-1]
        sma_50 = features[f'sma_{self.trend_sma_period}'].iloc[-1]
        sma_200 = features.get('sma_200', pd.Series([np.nan])).iloc[-1]
        
        if pd.isna(sma_50):
            return 0
        
        # Price above SMA50 = bullish
        price_vs_sma50 = 1 if current_price > sma_50 else -1
        
        # SMA50 vs SMA200 (if available)
        sma_trend = 0
        if not pd.isna(sma_200):
            sma_trend = 1 if sma_50 > sma_200 else -1
        
        # Combine signals
        if price_vs_sma50 == 1 and sma_trend >= 0:
            return 1  # Strong uptrend
        elif price_vs_sma50 == -1 and sma_trend <= 0:
            return -1  # Strong downtrend
        else:
            return 0  # Neutral
    
    def _check_momentum_condition(self, 
                                 symbol: str,
                                 price_data: pd.DataFrame,
                                 universe_data: Optional[Dict[str, pd.DataFrame]]) -> int:
        """
        Check momentum condition using relative ranking.
        
        Returns:
            1: Top momentum (top 30%), 0: medium, -1: bottom momentum
        """
        if universe_data is None:
            return 0
            
        try:
            # Calculate momentum for current symbol
            lookback = min(len(price_data), 63)  # ~3 months of daily data
            if lookback < 20:
                return 0
                
            symbol_momentum = self._calculate_momentum(price_data, lookback)
            
            # Calculate momentum for all symbols in universe
            momentum_scores = {}
            for other_symbol, other_data in universe_data.items():
                if len(other_data) >= lookback:
                    momentum_scores[other_symbol] = self._calculate_momentum(other_data, lookback)
            
            # Add current symbol
            momentum_scores[symbol] = symbol_momentum
            
            # Rank symbols by momentum
            sorted_symbols = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
            total_symbols = len(sorted_symbols)
            
            # Find rank of current symbol
            symbol_rank = next(i for i, (s, _) in enumerate(sorted_symbols) if s == symbol)
            rank_percentile = symbol_rank / total_symbols
            
            # Store rank for reference
            self.momentum_ranks[symbol] = 1.0 - rank_percentile
            
            # Return signal based on rank
            if rank_percentile <= self.momentum_rank_pct:  # Top 30%
                return 1
            elif rank_percentile >= (1.0 - self.momentum_rank_pct):  # Bottom 30%
                return -1
            else:
                return 0
                
        except Exception as e:
            logger.warning(f"Momentum calculation failed for {symbol}: {e}")
            return 0
    
    def _calculate_momentum(self, price_data: pd.DataFrame, lookback: int) -> float:
        """Calculate momentum score for a symbol."""
        if len(price_data) < lookback:
            return 0.0
            
        prices = price_data['close'].tail(lookback)
        
        # Simple momentum: total return over period
        if len(prices) < 2:
            return 0.0
            
        total_return = (prices.iloc[-1] / prices.iloc[0]) - 1.0
        
        # Adjust for volatility (Sharpe-like ratio)
        returns = prices.pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            vol_adjusted_momentum = total_return / returns.std()
        else:
            vol_adjusted_momentum = total_return
        
        return vol_adjusted_momentum
    
    def _check_rsi_condition(self, features: pd.DataFrame) -> int:
        """
        Check RSI condition to avoid overbought/oversold entries.
        
        Returns:
            1: RSI oversold (bullish), 0: neutral, -1: RSI overbought (bearish)
        """
        if 'rsi' not in features.columns:
            return 0
            
        rsi = features['rsi'].iloc[-1]
        if pd.isna(rsi):
            return 0
        
        if rsi >= self.rsi_threshold:  # Overbought - avoid longs
            return -1
        elif rsi <= (100 - self.rsi_threshold):  # Oversold - avoid shorts  
            return 1
        else:
            return 0
    
    def _calculate_breakout_confidence(self, 
                                     current_price: float,
                                     breakout_level: float, 
                                     direction: str) -> float:
        """Calculate confidence based on breakout strength."""
        if direction == 'long':
            breakout_strength = (current_price - breakout_level) / breakout_level
        else:  # short
            breakout_strength = (breakout_level - current_price) / breakout_level
        
        # Convert to confidence (0-1)
        confidence = min(1.0, breakout_strength / 0.02)  # Max confidence at 2% breakout
        return max(0.1, confidence)  # Min confidence of 10%
    
    def _apply_breakout_filters(self, 
                               symbol: str,
                               price_data: pd.DataFrame,
                               features: pd.DataFrame) -> Dict[str, Any]:
        """Apply additional filters to breakout signals."""
        
        # Filter 1: Volume confirmation
        if 'volume' in price_data.columns:
            avg_volume = price_data['volume'].tail(20).mean()
            current_volume = price_data['volume'].iloc[-1]
            
            if current_volume < 1.2 * avg_volume:  # Volume should be 20% above average
                return {'passed': False, 'reason': 'insufficient_volume_confirmation'}
        
        # Filter 2: Avoid false breakouts (price should sustain above/below level)
        if len(price_data) >= self.breakout_confirmation_bars:
            recent_closes = price_data['close'].tail(self.breakout_confirmation_bars)
            breakout_high = self.breakout_levels.get(symbol, {}).get('high', 0)
            breakout_low = self.breakout_levels.get(symbol, {}).get('low', float('inf'))
            
            # For long breakout, recent closes should stay above breakout level
            if recent_closes.iloc[-1] > breakout_high:
                if not all(close >= breakout_high * 0.999 for close in recent_closes):
                    return {'passed': False, 'reason': 'failed_breakout_confirmation'}
            
            # For short breakout, recent closes should stay below breakout level  
            elif recent_closes.iloc[-1] < breakout_low:
                if not all(close <= breakout_low * 1.001 for close in recent_closes):
                    return {'passed': False, 'reason': 'failed_breakout_confirmation'}
        
        # Filter 3: Avoid trading in consolidation (low ATR)
        if 'atr' in features.columns:
            atr = features['atr'].iloc[-1]
            current_price = price_data['close'].iloc[-1]
            atr_pct = atr / current_price if current_price > 0 else 0
            
            if atr_pct < 0.01:  # ATR less than 1% = consolidation
                return {'passed': False, 'reason': 'low_volatility_consolidation'}
        
        return {'passed': True, 'reason': 'all_filters_passed'}
    
    def _check_exit_conditions(self, 
                              symbol: str,
                              price_data: pd.DataFrame,
                              features: pd.DataFrame) -> bool:
        """Check exit conditions for existing positions."""
        if symbol not in self.positions:
            return False
            
        position = self.positions[symbol]
        current_price = price_data['close'].iloc[-1]
        position_size = position['size']
        
        # Trend reversal exit
        trend_signal = self._check_trend_condition(symbol, price_data, features)
        
        if position_size > 0 and trend_signal == -1:  # Long position, trend turned down
            logger.info(f"Trend reversal exit for long {symbol}")
            return True
        elif position_size < 0 and trend_signal == 1:  # Short position, trend turned up
            logger.info(f"Trend reversal exit for short {symbol}")
            return True
        
        # RSI extreme exit
        rsi_signal = self._check_rsi_condition(features)
        if position_size > 0 and rsi_signal == -1:  # Long position, RSI overbought
            logger.info(f"RSI overbought exit for long {symbol}")
            return True
        elif position_size < 0 and rsi_signal == 1:  # Short position, RSI oversold
            logger.info(f"RSI oversold exit for short {symbol}")
            return True
        
        # Stop loss based on breakout level
        breakout_level = self.breakout_levels.get(symbol, {})
        if position_size > 0 and 'low' in breakout_level:
            stop_price = breakout_level['low']
            if current_price <= stop_price:
                logger.info(f"Stop loss exit for long {symbol}: {current_price} <= {stop_price}")
                return True
        elif position_size < 0 and 'high' in breakout_level:
            stop_price = breakout_level['high']  
            if current_price >= stop_price:
                logger.info(f"Stop loss exit for short {symbol}: {current_price} >= {stop_price}")
                return True
        
        return False
    
    def _calculate_stop_price(self, 
                             signal: int,
                             current_price: float,
                             breakout_high: float,
                             breakout_low: float) -> Optional[float]:
        """Calculate stop loss price based on breakout levels."""
        if signal == 1:  # Long position
            return breakout_low  # Stop below recent low
        elif signal == -1:  # Short position  
            return breakout_high  # Stop above recent high
        else:
            return None
    
    def update_position(self, 
                       symbol: str,
                       size: float,
                       entry_price: float,
                       timestamp: Optional[datetime] = None):
        """Update position tracking."""
        if timestamp is None:
            timestamp = datetime.now()
            
        if size == 0:
            if symbol in self.positions:
                del self.positions[symbol]
        else:
            self.positions[symbol] = {
                'size': size,
                'entry_price': entry_price,
                'entry_time': timestamp,
                'unrealized_pnl': 0.0
            }
        
        logger.debug(f"Trend breakout position updated for {symbol}: size={size}")
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """Get strategy performance statistics."""
        stats = {
            'strategy_name': 'trend_breakout',
            'enabled': self.enabled,
            'active_positions': len(self.positions),
            'total_signals': len(self.signal_history),
            'lookback_bars': self.lookback_bars
        }
        
        if self.signal_history:
            df = pd.DataFrame(self.signal_history)
            stats.update({
                'signal_distribution': df['signal'].value_counts().to_dict(),
                'avg_confidence': df['confidence'].mean(),
                'avg_momentum_rank': df.get('momentum_signal', pd.Series([0])).mean()
            })
        
        return stats
    
    def should_trade_symbol(self, symbol: str, market_conditions: Dict[str, Any]) -> bool:
        """Determine if strategy should trade this symbol."""
        if not self.enabled:
            return False
            
        # Only trade liquid symbols for breakout strategy
        if symbol not in ['SPY', 'QQQ', 'BTCUSD', 'ETHUSD'] and len(self.positions) >= 3:
            return False  # Limit positions in less liquid names
            
        # Avoid trading in very low volatility environments
        volatility = market_conditions.get('volatility', 0.0)
        if volatility < 0.05:  # Less than 5% volatility
            return False
        
        return True