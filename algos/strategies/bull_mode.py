"""
Bull Mode Strategy - Higher utilization strategy that activates in bullish market regimes.
Uses SMA trend filters and realized volatility conditions to identify bull markets.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from loguru import logger
from datetime import datetime, timedelta


class BullModeStrategy:
    """
    Bull mode strategy that increases position sizing and risk in bullish regimes.
    
    Strategy characteristics:
    - Only active when SMA(50) > SMA(200) AND low realized volatility
    - Higher position sizing multiplier during bull markets
    - Combines with other strategies for enhanced returns
    - Automatic risk reduction when bull conditions end
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize bull mode strategy."""
        self.config = config
        self.strategy_config = config['strategies']['bull_mode']
        
        # Strategy parameters
        self.enabled = self.strategy_config.get('enabled', True)
        self.sma_periods = self.strategy_config.get('sma_bull_condition', [50, 200])
        self.vol_threshold = self.strategy_config.get('vol_threshold', 0.15)
        
        # Bull mode parameters
        self.bull_multiplier = 1.5  # Increase position sizes by 50% in bull mode
        self.min_bull_duration = 5   # Minimum days in bull mode before activation
        self.vol_lookback = 30       # Days to calculate realized volatility
        
        # State tracking
        self.is_bull_mode = False
        self.bull_mode_start = None
        self.bull_mode_history: List[Dict] = []
        self.regime_signals: List[Dict] = []
        
        logger.info(f"Bull Mode strategy initialized: enabled={self.enabled}")
    
    def detect_bull_regime(self, 
                          index_data: pd.DataFrame,
                          features: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect if market is in bullish regime.
        
        Args:
            index_data: Market index data (e.g., SPY)
            features: Calculated features including SMAs
            
        Returns:
            Dictionary with regime information
        """
        if not self.enabled or len(index_data) < max(self.sma_periods):
            return {'bull_mode': False, 'confidence': 0.0, 'reason': 'insufficient_data'}
        
        current_time = datetime.now()
        
        # Condition 1: SMA trend (SMA50 > SMA200)
        sma_short_col = f'sma_{self.sma_periods[0]}'
        sma_long_col = f'sma_{self.sma_periods[1]}'
        
        if sma_short_col not in features.columns or sma_long_col not in features.columns:
            return {'bull_mode': False, 'confidence': 0.0, 'reason': 'missing_sma_data'}
        
        sma_short = features[sma_short_col].iloc[-1]
        sma_long = features[sma_long_col].iloc[-1]
        
        if pd.isna(sma_short) or pd.isna(sma_long):
            return {'bull_mode': False, 'confidence': 0.0, 'reason': 'invalid_sma_data'}
        
        sma_bull_condition = sma_short > sma_long
        sma_strength = (sma_short / sma_long - 1.0) if sma_long > 0 else 0.0
        
        # Condition 2: Low realized volatility
        if len(index_data) < self.vol_lookback:
            return {'bull_mode': False, 'confidence': 0.0, 'reason': 'insufficient_vol_data'}
        
        returns = index_data['close'].tail(self.vol_lookback).pct_change().dropna()
        if len(returns) == 0:
            return {'bull_mode': False, 'confidence': 0.0, 'reason': 'no_returns_data'}
        
        realized_vol = returns.std() * np.sqrt(252)  # Annualized
        vol_condition = realized_vol < self.vol_threshold
        
        # Condition 3: Price above both SMAs
        current_price = index_data['close'].iloc[-1]
        price_condition = current_price > sma_short and current_price > sma_long
        
        # Overall bull condition
        bull_condition = sma_bull_condition and vol_condition and price_condition
        
        # Calculate confidence
        confidence = 0.0
        if bull_condition:
            # Confidence based on SMA separation and vol level
            sma_confidence = min(1.0, sma_strength / 0.05)  # Max at 5% separation
            vol_confidence = min(1.0, (self.vol_threshold - realized_vol) / 0.05)
            price_confidence = min(1.0, (current_price / sma_short - 1.0) / 0.02)
            
            confidence = np.mean([sma_confidence, vol_confidence, price_confidence])
        
        # Update bull mode state
        previous_bull_mode = self.is_bull_mode
        
        if bull_condition and not self.is_bull_mode:
            # Entering potential bull mode
            if self.bull_mode_start is None:
                self.bull_mode_start = current_time
                logger.info("Potential bull regime detected - monitoring for confirmation")
            else:
                # Check if minimum duration met
                days_in_condition = (current_time - self.bull_mode_start).days
                if days_in_condition >= self.min_bull_duration:
                    self.is_bull_mode = True
                    logger.info(f"Bull mode ACTIVATED after {days_in_condition} days")
                    
        elif not bull_condition and self.is_bull_mode:
            # Exiting bull mode
            self.is_bull_mode = False
            self.bull_mode_start = None
            logger.info("Bull mode DEACTIVATED")
            
        elif not bull_condition:
            # Reset start time if conditions not met
            self.bull_mode_start = None
        
        # Log regime change
        if self.is_bull_mode != previous_bull_mode:
            self.bull_mode_history.append({
                'timestamp': current_time,
                'bull_mode': self.is_bull_mode,
                'sma_strength': sma_strength,
                'realized_vol': realized_vol,
                'confidence': confidence
            })
        
        # Store signal
        regime_info = {
            'timestamp': current_time,
            'bull_mode': self.is_bull_mode,
            'bull_condition': bull_condition,
            'confidence': confidence,
            'sma_bull_condition': sma_bull_condition,
            'vol_condition': vol_condition,
            'price_condition': price_condition,
            'sma_strength': sma_strength,
            'realized_vol': realized_vol,
            'current_price': current_price,
            'sma_short': sma_short,
            'sma_long': sma_long,
            'days_in_condition': (current_time - self.bull_mode_start).days if self.bull_mode_start else 0
        }
        
        self.regime_signals.append(regime_info)
        
        # Keep last 1000 signals
        if len(self.regime_signals) > 1000:
            self.regime_signals = self.regime_signals[-1000:]
        
        return regime_info
    
    def get_position_multiplier(self) -> float:
        """
        Get position size multiplier based on bull mode status.
        
        Returns:
            Multiplier for position sizes (1.0 = normal, 1.5 = bull mode)
        """
        if not self.enabled:
            return 1.0
            
        if self.is_bull_mode:
            return self.bull_multiplier
        else:
            return 1.0
    
    def get_risk_multiplier(self) -> float:
        """
        Get risk multiplier for bull mode.
        
        Returns:
            Multiplier for risk limits
        """
        if not self.enabled:
            return 1.0
            
        if self.is_bull_mode:
            return self.bull_multiplier
        else:
            return 1.0
    
    def should_increase_exposure(self, 
                                current_leverage: float,
                                max_leverage: float) -> bool:
        """
        Check if exposure should be increased in bull mode.
        
        Args:
            current_leverage: Current portfolio leverage
            max_leverage: Maximum allowed leverage
            
        Returns:
            True if exposure can be increased
        """
        if not self.enabled or not self.is_bull_mode:
            return False
        
        # Allow higher leverage in bull mode
        bull_max_leverage = max_leverage * self.bull_multiplier
        
        return current_leverage < bull_max_leverage
    
    def adjust_strategy_allocation(self, 
                                  base_allocation: Dict[str, float],
                                  strategy_performance: Dict[str, float]) -> Dict[str, float]:
        """
        Adjust allocation between strategies based on bull mode.
        
        Args:
            base_allocation: Base strategy allocation weights
            strategy_performance: Recent performance by strategy
            
        Returns:
            Adjusted allocation weights
        """
        if not self.enabled or not self.is_bull_mode:
            return base_allocation
        
        adjusted_allocation = base_allocation.copy()
        
        # In bull mode, favor momentum strategies over mean reversion
        momentum_strategies = ['trend_breakout', 'gamma_reversal']
        mean_reversion_strategies = ['scalper_sigma', 'market_neutral']
        
        # Increase allocation to momentum strategies
        for strategy in momentum_strategies:
            if strategy in adjusted_allocation:
                adjusted_allocation[strategy] *= 1.3
        
        # Decrease allocation to mean reversion strategies
        for strategy in mean_reversion_strategies:
            if strategy in adjusted_allocation:
                adjusted_allocation[strategy] *= 0.8
        
        # Normalize weights
        total_weight = sum(adjusted_allocation.values())
        if total_weight > 0:
            adjusted_allocation = {k: v / total_weight for k, v in adjusted_allocation.items()}
        
        logger.debug(f"Bull mode allocation adjustment: {adjusted_allocation}")
        
        return adjusted_allocation
    
    def get_bull_mode_metrics(self) -> Dict[str, Any]:
        """Get bull mode performance metrics."""
        metrics = {
            'strategy_name': 'bull_mode',
            'enabled': self.enabled,
            'is_bull_mode': self.is_bull_mode,
            'bull_multiplier': self.bull_multiplier,
            'total_regime_changes': len(self.bull_mode_history)
        }
        
        if self.bull_mode_start:
            metrics['days_in_current_regime'] = (datetime.now() - self.bull_mode_start).days
        
        if self.bull_mode_history:
            bull_periods = [h for h in self.bull_mode_history if h['bull_mode']]
            bear_periods = [h for h in self.bull_mode_history if not h['bull_mode']]
            
            metrics.update({
                'bull_periods': len(bull_periods),
                'bear_periods': len(bear_periods),
                'avg_bull_confidence': np.mean([p['confidence'] for p in bull_periods]) if bull_periods else 0.0,
                'avg_realized_vol_in_bull': np.mean([p['realized_vol'] for p in bull_periods]) if bull_periods else 0.0
            })
        
        if self.regime_signals:
            recent_signals = self.regime_signals[-100:]  # Last 100 observations
            metrics.update({
                'bull_mode_percentage': np.mean([s['bull_mode'] for s in recent_signals]),
                'avg_sma_strength': np.mean([s['sma_strength'] for s in recent_signals]),
                'avg_realized_vol': np.mean([s['realized_vol'] for s in recent_signals])
            })
        
        return metrics
    
    def get_regime_summary(self) -> Dict[str, Any]:
        """Get current market regime summary."""
        if not self.regime_signals:
            return {'status': 'no_data'}
        
        latest = self.regime_signals[-1]
        
        summary = {
            'current_regime': 'bull' if latest['bull_mode'] else 'bear',
            'confidence': latest['confidence'],
            'sma_trend': 'bullish' if latest['sma_bull_condition'] else 'bearish',
            'volatility': 'low' if latest['vol_condition'] else 'high',
            'realized_vol': latest['realized_vol'],
            'sma_strength': latest['sma_strength'],
            'position_multiplier': self.get_position_multiplier(),
            'days_in_regime': latest['days_in_condition']
        }
        
        return summary
    
    def backtest_regime_detection(self, 
                                 historical_data: pd.DataFrame,
                                 historical_features: pd.DataFrame) -> pd.DataFrame:
        """
        Backtest bull mode detection on historical data.
        
        Args:
            historical_data: Historical price data
            historical_features: Historical features
            
        Returns:
            DataFrame with regime detection results
        """
        results = []
        
        # Reset state for backtesting
        original_state = (self.is_bull_mode, self.bull_mode_start)
        self.is_bull_mode = False
        self.bull_mode_start = None
        
        try:
            min_length = max(self.sma_periods) + self.vol_lookback
            
            for i in range(min_length, len(historical_data)):
                # Get data up to current point
                data_slice = historical_data.iloc[:i+1]
                features_slice = historical_features.iloc[:i+1]
                
                # Detect regime
                regime_info = self.detect_bull_regime(data_slice, features_slice)
                regime_info['index'] = i
                regime_info['date'] = historical_data.index[i] if hasattr(historical_data.index, 'to_pydatetime') else i
                
                results.append(regime_info)
        
        finally:
            # Restore original state
            self.is_bull_mode, self.bull_mode_start = original_state
        
        return pd.DataFrame(results)
    
    def calculate_bull_mode_returns(self, 
                                   portfolio_returns: pd.Series,
                                   regime_history: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate returns during bull vs bear periods.
        
        Args:
            portfolio_returns: Portfolio return series
            regime_history: DataFrame with regime detection history
            
        Returns:
            Dictionary with performance statistics
        """
        if len(regime_history) == 0:
            return {}
        
        # Align returns and regimes
        aligned_data = []
        
        for i, regime_row in regime_history.iterrows():
            date = regime_row.get('date')
            if date in portfolio_returns.index:
                aligned_data.append({
                    'date': date,
                    'return': portfolio_returns[date],
                    'bull_mode': regime_row['bull_mode']
                })
        
        if not aligned_data:
            return {}
        
        df = pd.DataFrame(aligned_data)
        
        # Calculate statistics
        bull_returns = df[df['bull_mode']]['return']
        bear_returns = df[~df['bull_mode']]['return']
        
        stats = {}
        
        if len(bull_returns) > 0:
            stats['bull_avg_return'] = bull_returns.mean()
            stats['bull_vol'] = bull_returns.std()
            stats['bull_sharpe'] = bull_returns.mean() / bull_returns.std() * np.sqrt(252) if bull_returns.std() > 0 else 0
            stats['bull_periods'] = len(bull_returns)
        
        if len(bear_returns) > 0:
            stats['bear_avg_return'] = bear_returns.mean()
            stats['bear_vol'] = bear_returns.std()
            stats['bear_sharpe'] = bear_returns.mean() / bear_returns.std() * np.sqrt(252) if bear_returns.std() > 0 else 0
            stats['bear_periods'] = len(bear_returns)
        
        if len(bull_returns) > 0 and len(bear_returns) > 0:
            stats['return_ratio'] = bull_returns.mean() / bear_returns.mean() if bear_returns.mean() != 0 else np.inf
        
        return stats