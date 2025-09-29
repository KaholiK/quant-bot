"""
Risk management system for quantitative trading.
Implements position sizing, risk limits, and kill-switch functionality.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from loguru import logger
import requests
import os
from datetime import datetime, timedelta


class RiskManager:
    """Comprehensive risk management system for trading algorithms."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize risk manager with configuration."""
        self.config = config
        self.risk_config = config['risk']
        
        # Risk parameters
        self.max_leverage = self.risk_config['max_leverage']
        self.max_position_pct = self.risk_config['max_position_pct']
        self.max_sector_pct = self.risk_config['max_sector_pct']
        self.risk_pct_per_trade = self.risk_config['risk_pct_per_trade']
        self.vol_target = self.risk_config['vol_target']
        self.kill_switch_dd = self.risk_config['kill_switch_dd']
        
        # State tracking
        self.positions: Dict[str, float] = {}
        self.sector_exposure: Dict[str, float] = {}
        self.equity_curve: List[float] = []
        self.peak_equity = 0.0
        self.is_kill_switch_active = False
        self.last_portfolio_value = 0.0
        
        # Performance tracking
        self.trade_history: List[Dict[str, Any]] = []
        self.daily_returns: List[float] = []
        
        logger.info(f"Risk manager initialized with kill switch at {self.kill_switch_dd:.1%} drawdown")
        
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
            
        # Apply position limits
        max_position_value = self.max_position_pct * equity
        max_position_size = max_position_value / entry_price
        
        position_size = min(position_size, max_position_size)
        
        # Check leverage limits
        current_leverage = self._calculate_current_leverage(equity)
        position_leverage = (position_size * entry_price) / equity
        
        if current_leverage + position_leverage > self.max_leverage:
            max_additional_leverage = self.max_leverage - current_leverage
            position_size = max(0, max_additional_leverage * equity / entry_price)
            
        # Check sector limits
        if hasattr(self, '_get_sector'):
            sector = self._get_sector(symbol)
            current_sector_exposure = self.sector_exposure.get(sector, 0.0)
            max_sector_value = self.max_sector_pct * equity
            
            if current_sector_exposure + (position_size * entry_price) > max_sector_value:
                max_additional_sector = max_sector_value - current_sector_exposure
                position_size = max(0, max_additional_sector / entry_price)
        
        logger.debug(f"Position size for {symbol}: {position_size:.2f} units "
                    f"(${position_size * entry_price:.2f} value)")
        
        return position_size
    
    def update_position(self, symbol: str, quantity: float, price: float, sector: Optional[str] = None):
        """Update position tracking."""
        old_position = self.positions.get(symbol, 0.0)
        self.positions[symbol] = old_position + quantity
        
        # Update sector exposure
        if sector:
            old_sector = self.sector_exposure.get(sector, 0.0)
            self.sector_exposure[sector] = old_sector + (quantity * price)
            
        logger.debug(f"Updated position {symbol}: {old_position} -> {self.positions[symbol]}")
    
    def check_risk_limits(self, 
                         symbol: str, 
                         quantity: float, 
                         price: float, 
                         equity: float,
                         sector: Optional[str] = None) -> Tuple[bool, str]:
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
    
    def update_equity_curve(self, portfolio_value: float):
        """Update equity curve and check for kill switch."""
        self.equity_curve.append(portfolio_value)
        
        # Update peak equity
        if portfolio_value > self.peak_equity:
            self.peak_equity = portfolio_value
            
        # Calculate drawdown
        if self.peak_equity > 0:
            current_dd = (self.peak_equity - portfolio_value) / self.peak_equity
            
            # Check kill switch
            if current_dd >= self.kill_switch_dd and not self.is_kill_switch_active:
                self.activate_kill_switch(current_dd)
                
        # Calculate daily return
        if self.last_portfolio_value > 0:
            daily_return = (portfolio_value - self.last_portfolio_value) / self.last_portfolio_value
            self.daily_returns.append(daily_return)
            
        self.last_portfolio_value = portfolio_value
    
    def activate_kill_switch(self, drawdown: float):
        """Activate kill switch due to excessive drawdown."""
        self.is_kill_switch_active = True
        
        message = f"🚨 KILL SWITCH ACTIVATED - Drawdown: {drawdown:.1%} (Limit: {self.kill_switch_dd:.1%})"
        logger.critical(message)
        
        # Send Discord notification if configured
        self._send_discord_notification(message)
        
        # Log all current positions
        logger.critical(f"Current positions at kill switch: {self.positions}")
        
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
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Calculate current risk metrics."""
        if not self.equity_curve:
            return {}
            
        current_equity = self.equity_curve[-1]
        
        metrics = {
            'current_equity': current_equity,
            'peak_equity': self.peak_equity,
            'current_drawdown': (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0,
            'kill_switch_active': self.is_kill_switch_active,
            'current_leverage': self._calculate_current_leverage(current_equity),
            'position_count': len([p for p in self.positions.values() if p != 0]),
            'total_position_value': sum(abs(pos * 100) for pos in self.positions.values()),  # Assume $100 price
        }
        
        if len(self.daily_returns) > 0:
            returns_array = np.array(self.daily_returns)
            metrics.update({
                'daily_vol': np.std(returns_array),
                'annualized_vol': np.std(returns_array) * np.sqrt(252),
                'sharpe_ratio': np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0,
                'max_daily_loss': np.min(returns_array) if len(returns_array) > 0 else 0,
                'max_daily_gain': np.max(returns_array) if len(returns_array) > 0 else 0,
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
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
            'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary',
            'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials',
            'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare',
            'BTCUSD': 'Crypto', 'ETHUSD': 'Crypto'
        }
        return sector_mapping.get(symbol, 'Other')
    
    def _send_discord_notification(self, message: str):
        """Send Discord notification if webhook URL is configured."""
        webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
        if not webhook_url:
            return
            
        try:
            payload = {
                'content': message,
                'username': 'Quant Bot Risk Manager'
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
            'timestamp': timestamp,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'value': quantity * price,
            'strategy': strategy
        }
        
        self.trade_history.append(trade)
        logger.info(f"Trade logged: {side} {quantity} {symbol} @ {price} ({strategy})")
    
    def get_trade_statistics(self) -> Dict[str, Any]:
        """Calculate trade statistics."""
        if not self.trade_history:
            return {}
            
        df = pd.DataFrame(self.trade_history)
        
        stats = {
            'total_trades': len(df),
            'buy_trades': len(df[df['side'] == 'buy']),
            'sell_trades': len(df[df['side'] == 'sell']),
            'avg_trade_size': df['value'].mean(),
            'largest_trade': df['value'].max(),
            'smallest_trade': df['value'].min(),
            'most_traded_symbol': df['symbol'].value_counts().index[0] if len(df) > 0 else None,
            'strategy_breakdown': df['strategy'].value_counts().to_dict(),
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
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import squareform
        
        # Calculate correlation matrix
        corr_matrix = self.calculate_correlation_matrix(returns)
        
        # Convert correlation to distance matrix
        distance_matrix = np.sqrt((1 - corr_matrix) / 2)
        
        # Hierarchical clustering
        linkage_matrix = linkage(squareform(distance_matrix.values), method='single')
        
        # Initialize equal weights
        weights = pd.Series(1.0 / len(returns.columns), index=returns.columns)
        
        # This is a simplified HRP - in practice, you'd implement the full tree-based allocation
        # For now, we'll use inverse volatility weighting as a proxy
        vol = returns.std()
        inv_vol_weights = (1 / vol) / (1 / vol).sum()
        
        return inv_vol_weights