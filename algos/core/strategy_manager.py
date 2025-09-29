"""
Strategy Manager for aggregating signals and meta-labeling.
Coordinates multiple strategies, resolves conflicts, and applies meta-filtering.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger
import joblib
import os
from datetime import datetime, timedelta


class StrategyManager:
    """
    Lightweight strategy manager for signal aggregation and meta-filtering.
    
    Responsibilities:
    - Aggregate signals from multiple strategies
    - Resolve conflicts using priority/confidence
    - Apply meta-labeling filter to reject low-quality trades
    - Allocate capital by strategy performance
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize strategy manager."""
        self.config = config
        self.trading_config = config if 'trading' not in config else config['trading']
        
        # Strategy priorities (higher = more priority)
        self.strategy_priorities = {
            'scalper_sigma': 3,
            'trend_breakout': 4,
            'bull_mode': 2,
            'market_neutral': 1,
            'gamma_reversal': 5
        }
        
        # Strategy performance tracking
        self.strategy_performance = {
            name: {'trades': 0, 'wins': 0, 'total_pnl': 0.0, 'weight': 1.0}
            for name in self.strategy_priorities.keys()
        }
        
        # Meta-filter model
        self.meta_model = None
        self.meta_threshold = 0.5  # Threshold for meta-filter acceptance
        
        # Load meta-filter if available
        self._load_meta_model()
        
        # Signal history for analysis
        self.signal_history: List[Dict] = []
        
        logger.info(f"StrategyManager initialized with {len(self.strategy_priorities)} strategies")
    
    def _load_meta_model(self):
        """Load meta-filter model if available."""
        try:
            model_path = self.trading_config.get('models', {}).get('meta_model_path', 'models/meta_filter.joblib')
            if os.path.exists(model_path):
                self.meta_model = joblib.load(model_path)
                logger.info(f"Meta-filter model loaded from {model_path}")
            else:
                logger.warning(f"Meta-filter model not found at {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load meta-filter model: {e}")
    
    def aggregate_signals(self, 
                         symbol: str,
                         strategy_signals: Dict[str, Dict[str, Any]],
                         market_features: Dict[str, float],
                         price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Aggregate signals from multiple strategies.
        
        Args:
            symbol: Trading symbol
            strategy_signals: Dict of {strategy_name: signal_dict}
            market_features: Current market features
            price_data: Recent price data
            
        Returns:
            Aggregated signal with meta-filter decision
        """
        if not strategy_signals:
            return {'signal': 0, 'confidence': 0.0, 'reason': 'no_signals'}
        
        # Filter enabled strategies
        enabled_signals = {
            name: signal for name, signal in strategy_signals.items()
            if self._is_strategy_enabled(name) and signal.get('signal', 0) != 0
        }
        
        if not enabled_signals:
            return {'signal': 0, 'confidence': 0.0, 'reason': 'no_enabled_signals'}
        
        # Resolve conflicts and select best signal
        primary_signal = self._resolve_signal_conflicts(enabled_signals)
        
        # Apply meta-filter
        meta_decision = self._apply_meta_filter(
            symbol, primary_signal, market_features, price_data
        )
        
        # Calculate final signal
        if meta_decision['accept']:
            final_signal = primary_signal['signal']
            final_confidence = primary_signal['confidence'] * meta_decision['confidence']
            reason = f"{primary_signal.get('strategy', 'unknown')}_meta_approved"
        else:
            final_signal = 0
            final_confidence = 0.0
            reason = f"meta_filter_rejected: {meta_decision['reason']}"
        
        # Track signal for analysis
        signal_info = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'primary_strategy': primary_signal.get('strategy', 'unknown'),
            'primary_signal': primary_signal['signal'],
            'primary_confidence': primary_signal['confidence'],
            'meta_accept': meta_decision['accept'],
            'meta_confidence': meta_decision['confidence'],
            'final_signal': final_signal,
            'final_confidence': final_confidence,
            'reason': reason,
            'num_strategies': len(enabled_signals)
        }
        
        self.signal_history.append(signal_info)
        
        # Keep last 1000 signals
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]
        
        return {
            'signal': final_signal,
            'confidence': final_confidence,
            'reason': reason,
            'primary_strategy': primary_signal.get('strategy', 'unknown'),
            'meta_decision': meta_decision,
            'contributing_strategies': list(enabled_signals.keys())
        }
    
    def _is_strategy_enabled(self, strategy_name: str) -> bool:
        """Check if strategy is enabled in config."""
        strategies_config = self.trading_config.get('strategies', {})
        strategy_config = strategies_config.get(strategy_name, {})
        return strategy_config.get('enabled', False)
    
    def _resolve_signal_conflicts(self, strategy_signals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Resolve conflicts between strategy signals.
        
        Priority resolution:
        1. Strategy priority (configured)
        2. Signal confidence
        3. Strategy performance (weighted)
        """
        if not strategy_signals:
            return {'signal': 0, 'confidence': 0.0, 'strategy': 'none'}
        
        if len(strategy_signals) == 1:
            # Single strategy - return with strategy name
            strategy_name, signal = list(strategy_signals.items())[0]
            signal['strategy'] = strategy_name
            return signal
        
        # Multiple strategies - resolve conflicts
        scored_signals = []
        
        for strategy_name, signal in strategy_signals.items():
            # Calculate composite score
            priority_score = self.strategy_priorities.get(strategy_name, 1)
            confidence_score = signal.get('confidence', 0.0)
            performance_weight = self.strategy_performance[strategy_name]['weight']
            
            # Composite score
            composite_score = (
                priority_score * 0.4 +
                confidence_score * 0.4 +
                performance_weight * 0.2
            )
            
            scored_signals.append({
                'strategy': strategy_name,
                'signal': signal,
                'score': composite_score
            })
        
        # Sort by score (descending)
        scored_signals.sort(key=lambda x: x['score'], reverse=True)
        
        # Check for conflicting directions
        primary = scored_signals[0]
        primary_signal = primary['signal']['signal']
        
        # Look for opposing signals
        opposing_signals = [
            s for s in scored_signals[1:]
            if np.sign(s['signal']['signal']) != np.sign(primary_signal)
        ]
        
        if opposing_signals:
            # Reduce confidence due to conflict
            conflict_penalty = 0.1 * len(opposing_signals)
            primary['signal']['confidence'] *= (1.0 - conflict_penalty)
            
            logger.debug(f"Signal conflict detected for {primary['strategy']}: "
                        f"confidence reduced by {conflict_penalty:.1%}")
        
        # Return primary signal with strategy name
        result = primary['signal'].copy()
        result['strategy'] = primary['strategy']
        
        return result
    
    def _apply_meta_filter(self, 
                          symbol: str,
                          primary_signal: Dict[str, Any],
                          market_features: Dict[str, float],
                          price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Apply meta-filter to accept/reject trades.
        
        Meta-filter uses features like:
        - Primary signal probability/confidence
        - ATR ratio (current vs historical)
        - Volatility regime
        - Market conditions
        """
        if self.meta_model is None:
            # No meta-filter - accept all signals
            return {'accept': True, 'confidence': 1.0, 'reason': 'no_meta_model'}
        
        try:
            # Build meta-features
            meta_features = self._build_meta_features(
                symbol, primary_signal, market_features, price_data
            )
            
            # Get prediction from meta-model
            if hasattr(self.meta_model, 'predict_proba'):
                # Classification model
                proba = self.meta_model.predict_proba([meta_features])[0]
                meta_prob = proba[1] if len(proba) > 1 else proba[0]
            else:
                # Regression model
                meta_prob = self.meta_model.predict([meta_features])[0]
            
            # Ensure probability is in valid range
            meta_prob = np.clip(meta_prob, 0.0, 1.0)
            
            # Accept if above threshold
            accept = meta_prob >= self.meta_threshold
            
            logger.debug(f"Meta-filter for {symbol}: prob={meta_prob:.3f}, "
                        f"threshold={self.meta_threshold:.3f}, accept={accept}")
            
            return {
                'accept': accept,
                'confidence': meta_prob,
                'reason': f"meta_prob_{meta_prob:.3f}" if accept else f"below_threshold_{meta_prob:.3f}"
            }
            
        except Exception as e:
            logger.warning(f"Meta-filter failed for {symbol}: {e}")
            # Default to accept on error
            return {'accept': True, 'confidence': 0.5, 'reason': 'meta_filter_error'}
    
    def _build_meta_features(self, 
                           symbol: str,
                           primary_signal: Dict[str, Any],
                           market_features: Dict[str, float],
                           price_data: pd.DataFrame) -> List[float]:
        """Build feature vector for meta-filter."""
        features = []
        
        # Primary signal features
        features.append(primary_signal.get('confidence', 0.0))
        features.append(abs(primary_signal.get('signal', 0)))  # Signal strength
        
        # Market features
        features.append(market_features.get('volatility', 0.2))
        features.append(market_features.get('volume_ratio', 1.0))
        features.append(market_features.get('spread', 0.01))
        
        # Price action features (if available)
        if len(price_data) >= 20:
            returns = price_data['close'].pct_change().dropna()
            features.append(returns.std())  # Recent volatility
            features.append(returns.mean())  # Recent drift
            features.append(len([r for r in returns.tail(5) if abs(r) > 0.02]))  # Large moves
        else:
            features.extend([0.2, 0.0, 0])  # Default values
        
        # ATR ratio (current vs historical)
        current_atr = market_features.get('atr', 0.0)
        if len(price_data) >= 20:
            historical_atr = price_data['close'].diff().abs().rolling(20).mean().iloc[-1]
            atr_ratio = current_atr / historical_atr if historical_atr > 0 else 1.0
        else:
            atr_ratio = 1.0
        features.append(atr_ratio)
        
        # Time features
        now = datetime.now()
        features.append(now.hour / 24.0)  # Time of day
        features.append(now.weekday() / 6.0)  # Day of week
        
        # Strategy performance
        strategy_name = primary_signal.get('strategy', 'unknown')
        if strategy_name in self.strategy_performance:
            perf = self.strategy_performance[strategy_name]
            win_rate = perf['wins'] / max(perf['trades'], 1)
            features.append(win_rate)
            features.append(perf['weight'])
        else:
            features.extend([0.5, 1.0])  # Default performance
        
        return features
    
    def update_strategy_performance(self, 
                                  strategy_name: str,
                                  trade_pnl: float,
                                  was_winner: bool):
        """Update strategy performance tracking."""
        if strategy_name not in self.strategy_performance:
            return
        
        perf = self.strategy_performance[strategy_name]
        perf['trades'] += 1
        perf['total_pnl'] += trade_pnl
        
        if was_winner:
            perf['wins'] += 1
        
        # Update weight based on recent performance
        win_rate = perf['wins'] / perf['trades']
        avg_pnl = perf['total_pnl'] / perf['trades']
        
        # Weight formula: combine win rate and profitability
        perf['weight'] = (win_rate * 0.6 + np.tanh(avg_pnl * 0.01) * 0.4)
        perf['weight'] = np.clip(perf['weight'], 0.1, 2.0)  # Reasonable bounds
        
        logger.debug(f"Updated {strategy_name} performance: "
                    f"trades={perf['trades']}, win_rate={win_rate:.1%}, "
                    f"weight={perf['weight']:.2f}")
    
    def get_capital_allocation(self, total_capital: float) -> Dict[str, float]:
        """
        Allocate capital by strategy based on performance.
        
        Returns:
            Dict of {strategy_name: allocated_capital}
        """
        # Get total weight
        total_weight = sum(
            perf['weight'] for name, perf in self.strategy_performance.items()
            if self._is_strategy_enabled(name)
        )
        
        if total_weight <= 0:
            # Equal allocation if no performance data
            enabled_strategies = [
                name for name in self.strategy_priorities.keys()
                if self._is_strategy_enabled(name)
            ]
            equal_weight = 1.0 / len(enabled_strategies) if enabled_strategies else 0.0
            return {name: total_capital * equal_weight for name in enabled_strategies}
        
        # Allocate proportionally by weight
        allocation = {}
        for name, perf in self.strategy_performance.items():
            if self._is_strategy_enabled(name):
                allocation[name] = total_capital * (perf['weight'] / total_weight)
        
        return allocation
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get strategy manager statistics."""
        stats = {
            'total_signals': len(self.signal_history),
            'meta_model_loaded': self.meta_model is not None,
            'meta_threshold': self.meta_threshold,
            'strategy_performance': self.strategy_performance.copy()
        }
        
        if self.signal_history:
            df = pd.DataFrame(self.signal_history)
            stats.update({
                'meta_accept_rate': df['meta_accept'].mean(),
                'avg_final_confidence': df['final_confidence'].mean(),
                'signal_distribution': df['final_signal'].value_counts().to_dict(),
                'strategy_usage': df['primary_strategy'].value_counts().to_dict()
            })
        
        return stats