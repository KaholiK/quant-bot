"""
Feature engineering pipeline for quantitative trading.
Generates lag-safe features with no look-ahead bias.
"""

import numpy as np
import pandas as pd
import ta
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger


class FeaturePipeline:
    """Feature engineering pipeline with lag-safe feature generation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize feature pipeline with configuration."""
        self.config = config
        
        # Safe extraction with fallbacks to defaults
        features_config = config.get('features', {})
        
        self.returns_periods = features_config.get('returns_periods', [1, 5, 20, 60])
        self.sma_periods = features_config.get('sma_periods', [20, 50, 200])
        self.ema_periods = features_config.get('ema_periods', [12, 26])
        self.macd_params = features_config.get('macd_params', {"fast": 12, "slow": 26, "signal": 9})
        self.atr_period = features_config.get('atr_period', 14)
        self.bollinger_period = features_config.get('bollinger_period', 20)
        self.vol_window = features_config.get('vol_window', 20)
        
        # Cache for feature calculations
        self._feature_cache: Dict[str, pd.DataFrame] = {}
        
    def calculate_returns(self, prices: pd.Series) -> pd.DataFrame:
        """Calculate multiple period returns."""
        returns_df = pd.DataFrame(index=prices.index)
        
        for period in self.returns_periods:
            returns_df[f'ret_{period}'] = prices.pct_change(period)
            
        return returns_df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators using TA library."""
        features = pd.DataFrame(index=df.index)
        
        # Ensure required columns exist
        required_cols = ['close', 'high', 'low']
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"Missing required column '{col}' for technical indicators")
                return features
        
        try:
            # Simple Moving Averages
            for period in self.sma_periods:
                if len(df) >= period:
                    features[f'sma_{period}'] = ta.trend.sma_indicator(df['close'], window=period)
                    features[f'price_to_sma_{period}'] = df['close'] / features[f'sma_{period}'] - 1.0
                else:
                    features[f'sma_{period}'] = np.nan
                    features[f'price_to_sma_{period}'] = np.nan
                    
            # Exponential Moving Averages
            for period in self.ema_periods:
                if len(df) >= period:
                    features[f'ema_{period}'] = ta.trend.ema_indicator(df['close'], window=period)
                else:
                    features[f'ema_{period}'] = np.nan
            
            # MACD
            if len(df) >= max(self.macd_params['slow'], self.macd_params['fast'], self.macd_params['signal']):
                macd_line = ta.trend.macd_diff(df['close'], 
                                               window_slow=self.macd_params['slow'],
                                               window_fast=self.macd_params['fast'],
                                               window_sign=self.macd_params['signal'])
                macd_signal = ta.trend.macd_signal(df['close'],
                                                  window_slow=self.macd_params['slow'], 
                                                  window_fast=self.macd_params['fast'],
                                                  window_sign=self.macd_params['signal'])
                features['macd_diff'] = macd_line
                features['macd_signal'] = macd_signal
                features['macd_histogram'] = macd_line - macd_signal
            else:
                features['macd_diff'] = np.nan
                features['macd_signal'] = np.nan
                features['macd_histogram'] = np.nan
            
            # ATR
            if len(df) >= self.atr_period:
                features['atr'] = ta.volatility.average_true_range(
                    df['high'], df['low'], df['close'], window=self.atr_period
                )
            else:
                features['atr'] = np.nan
            
            # Bollinger Bands
            if len(df) >= self.bollinger_period:
                bb_high = ta.volatility.bollinger_hband(df['close'], window=self.bollinger_period)
                bb_low = ta.volatility.bollinger_lband(df['close'], window=self.bollinger_period)
                bb_mid = ta.volatility.bollinger_mavg(df['close'], window=self.bollinger_period)
                features['bb_upper'] = bb_high
                features['bb_lower'] = bb_low
                features['bb_mid'] = bb_mid
                features['bb_width'] = (bb_high - bb_low) / bb_mid
                features['bb_position'] = (df['close'] - bb_low) / (bb_high - bb_low)
            else:
                for col in ['bb_upper', 'bb_lower', 'bb_mid', 'bb_width', 'bb_position']:
                    features[col] = np.nan
                    
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            # Return empty dataframe on error but maintain index
            return pd.DataFrame(index=df.index)
        
        # ATR
        features['atr'] = ta.volatility.average_true_range(
            df['high'], df['low'], df['close'], window=self.atr_period
        )
        
        # Bollinger Bands
        bb_high = ta.volatility.bollinger_hband(df['close'], window=self.bollinger_period)
        bb_low = ta.volatility.bollinger_lband(df['close'], window=self.bollinger_period)
        bb_mid = ta.volatility.bollinger_mavg(df['close'], window=self.bollinger_period)
        
        features['bb_width'] = (bb_high - bb_low) / bb_mid
        features['bb_position'] = (df['close'] - bb_low) / (bb_high - bb_low)
        
        # RSI
        features['rsi'] = ta.momentum.rsi(df['close'], window=14)
        
        # Realized Volatility (rolling standard deviation of returns)
        returns = df['close'].pct_change()
        features['realized_vol'] = returns.rolling(window=self.vol_window).std() * np.sqrt(252)
        
        return features
    
    def calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based features."""
        features = pd.DataFrame(index=df.index)
        
        if 'volume' in df.columns:
            # Volume moving averages
            features['volume_sma_20'] = ta.volume.volume_sma(df['close'], df['volume'], window=20)
            features['volume_ratio'] = df['volume'] / features['volume_sma_20']
            
            # Volume price trend
            features['vpt'] = ta.volume.volume_price_trend(df['close'], df['volume'])
            
            # Accumulation/Distribution Line
            features['ad_line'] = ta.volume.acc_dist_index(
                df['high'], df['low'], df['close'], df['volume']
            )
        
        return features
    
    def calculate_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate crypto microstructure features (no-op if data unavailable)."""
        features = pd.DataFrame(index=df.index)
        
        try:
            # These would be populated from order book data if available
            if 'bid_price' in df.columns and 'ask_price' in df.columns:
                features['spread'] = (df['ask_price'] - df['bid_price']) / df['close']
                features['mid_price'] = (df['bid_price'] + df['ask_price']) / 2
                features['price_to_mid'] = df['close'] / features['mid_price'] - 1.0
                
            if 'trade_count' in df.columns:
                features['trade_intensity'] = df['trade_count'].rolling(window=20).mean()
                
            if 'bid_size' in df.columns and 'ask_size' in df.columns:
                total_size = df['bid_size'] + df['ask_size']
                features['order_imbalance'] = (df['bid_size'] - df['ask_size']) / total_size
            
        except KeyError:
            # Fill with zeros if microstructure data not available
            for col in ['spread', 'mid_price', 'price_to_mid', 'trade_intensity', 'order_imbalance']:
                features[col] = 0.0
                
        return features
    
    def calculate_cross_sectional_features(self, 
                                         symbol_data: pd.DataFrame,
                                         universe_data: Dict[str, pd.DataFrame],
                                         sector_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """Calculate cross-sectional features vs universe."""
        features = pd.DataFrame(index=symbol_data.index)
        
        if not universe_data:
            # Return empty features if no universe data
            for col in ['momentum_rank_1m', 'momentum_rank_3m', 'sector_zscore', 'beta_to_index', 'residual_return']:
                features[col] = 0.0
            return features
        
        try:
            # Get symbol returns
            symbol_rets_1m = symbol_data['close'].pct_change(20)  # ~1 month 
            symbol_rets_3m = symbol_data['close'].pct_change(60)  # ~3 months
            
            # Calculate cross-sectional momentum ranks
            for i, date in enumerate(symbol_data.index):
                if i < 60:  # Need enough history
                    continue
                    
                # Get contemporary returns for all symbols
                universe_rets_1m = []
                universe_rets_3m = []
                
                for other_symbol, other_data in universe_data.items():
                    if date in other_data.index and i < len(other_data):
                        other_1m = other_data['close'].pct_change(20).loc[date] if not other_data['close'].pct_change(20).loc[date:date].empty else np.nan
                        other_3m = other_data['close'].pct_change(60).loc[date] if not other_data['close'].pct_change(60).loc[date:date].empty else np.nan
                        
                        if not np.isnan(other_1m):
                            universe_rets_1m.append(other_1m)
                        if not np.isnan(other_3m):
                            universe_rets_3m.append(other_3m)
                
                # Calculate ranks
                if len(universe_rets_1m) > 10:  # Need sufficient universe
                    current_1m = symbol_rets_1m.loc[date] if not np.isnan(symbol_rets_1m.loc[date]) else 0
                    features.loc[date, 'momentum_rank_1m'] = (np.array(universe_rets_1m) < current_1m).mean()
                else:
                    features.loc[date, 'momentum_rank_1m'] = 0.5
                    
                if len(universe_rets_3m) > 10:
                    current_3m = symbol_rets_3m.loc[date] if not np.isnan(symbol_rets_3m.loc[date]) else 0
                    features.loc[date, 'momentum_rank_3m'] = (np.array(universe_rets_3m) < current_3m).mean()
                else:
                    features.loc[date, 'momentum_rank_3m'] = 0.5
            
            # Sector-neutral features if sector mapping available
            if sector_map:
                # This would need sector-specific universe data
                # For now, use simple rolling z-score vs own history as proxy
                returns_20d = symbol_data['close'].pct_change(20)
                features['sector_zscore'] = (returns_20d - returns_20d.rolling(252).mean()) / returns_20d.rolling(252).std()
            else:
                features['sector_zscore'] = 0.0
            
            # Beta calculation vs index (using SPY as proxy)
            if 'SPY' in universe_data:
                index_data = universe_data['SPY']
                symbol_rets = symbol_data['close'].pct_change()
                index_rets = index_data['close'].pct_change()
                
                # Rolling beta calculation
                window = 60
                betas = []
                residuals = []
                
                for i in range(len(symbol_data)):
                    if i < window:
                        betas.append(1.0)  # Default beta
                        residuals.append(0.0)
                        continue
                    
                    # Get aligned returns for regression
                    start_idx = max(0, i - window)
                    y = symbol_rets.iloc[start_idx:i+1].dropna()
                    x = index_rets.iloc[start_idx:i+1].dropna()
                    
                    # Align series
                    aligned_data = pd.concat([y, x], axis=1, keys=['symbol', 'index']).dropna()
                    
                    if len(aligned_data) > 10:
                        # Simple linear regression
                        x_vals = aligned_data['index'].values
                        y_vals = aligned_data['symbol'].values
                        
                        if np.std(x_vals) > 1e-8:  # Avoid division by zero
                            beta = np.cov(x_vals, y_vals)[0,1] / np.var(x_vals)
                            alpha = np.mean(y_vals) - beta * np.mean(x_vals)
                            
                            # Residual return (alpha)
                            current_x = index_rets.iloc[i] if not np.isnan(index_rets.iloc[i]) else 0
                            current_y = symbol_rets.iloc[i] if not np.isnan(symbol_rets.iloc[i]) else 0
                            residual = current_y - (alpha + beta * current_x)
                        else:
                            beta = 1.0
                            residual = 0.0
                    else:
                        beta = 1.0
                        residual = 0.0
                    
                    betas.append(beta)
                    residuals.append(residual)
                
                features['beta_to_index'] = betas
                features['residual_return'] = residuals
            else:
                features['beta_to_index'] = 1.0
                features['residual_return'] = 0.0
                
        except Exception as e:
            logger.error(f"Error calculating cross-sectional features: {e}")
            # Fill with safe defaults
            for col in ['momentum_rank_1m', 'momentum_rank_3m', 'sector_zscore', 'beta_to_index', 'residual_return']:
                features[col] = 0.0
                
        return features
    
    def calculate_event_features(self, symbol_data: pd.DataFrame, symbol: str = '') -> pd.DataFrame:
        """Calculate event-aware features for earnings and FOMC."""
        features = pd.DataFrame(index=symbol_data.index)
        
        try:
            # Earnings proximity features
            # This is a simplified implementation - in practice would need earnings calendar
            features['earnings_proximity'] = 0  # Default: not near earnings
            
            # FOMC meeting features  
            # This is a simplified implementation - in practice would need FOMC calendar
            features['fomc_day'] = 0  # Default: not FOMC day
            
            # Pattern-based proxies for events (simplified)
            if len(symbol_data) > 20:
                # High volume might indicate earnings
                if 'volume' in symbol_data.columns:
                    vol_ma = symbol_data['volume'].rolling(20).mean()
                    vol_spike = symbol_data['volume'] / vol_ma
                    features['earnings_proximity'] = (vol_spike > 2.0).astype(int)
                
                # Wednesday pattern for FOMC (simplified)
                features['fomc_day'] = (symbol_data.index.dayofweek == 2).astype(int)  # Wednesday
                
        except Exception as e:
            logger.error(f"Error calculating event features: {e}")
            features['earnings_proximity'] = 0
            features['fomc_day'] = 0
            
        return features
    
    def to_live_row(self, df: pd.DataFrame, 
                   universe_data: Optional[Dict[str, pd.DataFrame]] = None,
                   sector_map: Optional[Dict[str, str]] = None,
                   symbol: str = '') -> pd.Series:
        """
        Generate features for live trading using only t-1 information.
        Ensures no look-ahead bias.
        """
        if len(df) < 100:  # Need sufficient history
            logger.warning(f"Insufficient data for live feature generation: {len(df)}")
            return pd.Series()
        
        try:
            # Use all data except the very last bar to avoid any potential look-ahead
            historical_data = df.iloc[:-1].copy()  # Everything up to t-1
            
            # Calculate all feature components using historical data
            all_features = []
            
            # Basic features
            returns_features = self.calculate_returns(historical_data['close'])
            tech_features = self.calculate_technical_indicators(historical_data)
            volume_features = self.calculate_volume_features(historical_data)
            micro_features = self.calculate_microstructure_features(historical_data)
            
            all_features.extend([returns_features, tech_features, volume_features, micro_features])
            
            # Cross-sectional features
            if universe_data:
                cross_features = self.calculate_cross_sectional_features(
                    historical_data, universe_data, sector_map
                )
                all_features.append(cross_features)
            
            # Event features
            event_features = self.calculate_event_features(historical_data, symbol)
            all_features.append(event_features)
            
            # Combine and get latest row
            if all_features:
                feature_df = pd.concat([f for f in all_features if not f.empty], axis=1)
                if not feature_df.empty:
                    # Return the last row (which represents t-1 information)
                    latest_row = feature_df.iloc[-1].fillna(0)
                    
                    # Verify time alignment
                    assert latest_row.name < df.index[-1], "Feature timestamp must be before current time"
                    
                    return latest_row
                    
        except Exception as e:
            logger.error(f"Error generating live features: {e}")
            
        return pd.Series()
    
    def calculate_relative_strength(self, symbol_data: pd.DataFrame, 
                                   sector_data: Optional[pd.DataFrame] = None,
                                   index_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Calculate relative strength vs sector/index."""
        features = pd.DataFrame(index=symbol_data.index)
        
        symbol_returns = symbol_data['close'].pct_change(20)  # 20-period returns
        
        if sector_data is not None:
            sector_returns = sector_data['close'].pct_change(20)
            features['relative_strength_sector'] = symbol_returns - sector_returns
            
        if index_data is not None:
            index_returns = index_data['close'].pct_change(20)
            features['relative_strength_index'] = symbol_returns - index_returns
            
        return features
    
    def build_features(self, df: pd.DataFrame, 
                      sector_data: Optional[pd.DataFrame] = None,
                      index_data: Optional[pd.DataFrame] = None,
                      universe_data: Optional[Dict[str, pd.DataFrame]] = None,
                      sector_map: Optional[Dict[str, str]] = None,
                      symbol: str = '') -> pd.DataFrame:
        """Build complete feature set for a symbol."""
        all_features = []
        
        # Returns
        returns_features = self.calculate_returns(df['close'])
        all_features.append(returns_features)
        
        # Technical indicators
        tech_features = self.calculate_technical_indicators(df)
        all_features.append(tech_features)
        
        # Volume features
        volume_features = self.calculate_volume_features(df)
        all_features.append(volume_features)
        
        # Microstructure features
        micro_features = self.calculate_microstructure_features(df)
        all_features.append(micro_features)
        
        # Relative strength
        rel_strength = self.calculate_relative_strength(df, sector_data, index_data)
        all_features.append(rel_strength)
        
        # Cross-sectional features
        if universe_data:
            cross_features = self.calculate_cross_sectional_features(df, universe_data, sector_map)
            all_features.append(cross_features)
        
        # Event-aware features
        event_features = self.calculate_event_features(df, symbol)
        all_features.append(event_features)
        
        # Combine all features
        non_empty_features = [f for f in all_features if not f.empty]
        if non_empty_features:
            feature_df = pd.concat(non_empty_features, axis=1)
            
            # Forward fill missing values and then drop remaining NaNs
            feature_df = feature_df.fillna(method='ffill').dropna()
            
            logger.info(f"Generated {len(feature_df.columns)} features for {len(feature_df)} bars")
            return feature_df
        else:
            return pd.DataFrame(index=df.index)
    
    def build_latest_row(self, df: pd.DataFrame, 
                        sector_data: Optional[pd.DataFrame] = None,
                        index_data: Optional[pd.DataFrame] = None) -> pd.Series:
        """Build features for the latest bar only (for live inference)."""
        # Use last N bars to calculate features, return only the last row
        lookback = max(max(self.sma_periods), self.bollinger_period, 100)
        
        if len(df) < lookback:
            logger.warning(f"Insufficient data for feature calculation: {len(df)} < {lookback}")
            return pd.Series()
            
        # Take last N bars
        recent_df = df.tail(lookback).copy()
        recent_sector = sector_data.tail(lookback).copy() if sector_data is not None else None
        recent_index = index_data.tail(lookback).copy() if index_data is not None else None
        
        # Build features
        features = self.build_features(recent_df, recent_sector, recent_index)
        
        if len(features) == 0:
            logger.warning("No features calculated for latest row")
            return pd.Series()
            
        # Return only the latest row
        return features.iloc[-1]


def create_lagged_features(df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
    """Create lagged versions of features to prevent look-ahead bias."""
    lagged_df = pd.DataFrame(index=df.index)
    
    for lag in lags:
        for col in df.columns:
            lagged_df[f"{col}_lag_{lag}"] = df[col].shift(lag)
            
    return lagged_df


def validate_features(features: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate feature DataFrame for common issues."""
    issues = []
    
    # Check for infinite values
    if np.isinf(features.select_dtypes(include=[np.number])).any().any():
        issues.append("Infinite values detected in features")
        
    # Check for excessive missing values
    missing_pct = features.isnull().mean()
    high_missing = missing_pct[missing_pct > 0.5]
    if len(high_missing) > 0:
        issues.append(f"High missing values in features: {list(high_missing.index)}")
        
    # Check for constant features
    constant_features = []
    for col in features.select_dtypes(include=[np.number]).columns:
        if features[col].nunique() <= 1:
            constant_features.append(col)
    if constant_features:
        issues.append(f"Constant features detected: {constant_features}")
        
    return len(issues) == 0, issues