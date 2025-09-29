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
        self.returns_periods = config['features']['returns_periods']
        self.sma_periods = config['features']['sma_periods']
        self.ema_periods = config['features']['ema_periods']
        self.macd_params = config['features']['macd_params']
        self.atr_period = config['features']['atr_period']
        self.bollinger_period = config['features']['bollinger_period']
        self.vol_window = config['features']['vol_window']
        
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
        
        # Simple Moving Averages
        for period in self.sma_periods:
            features[f'sma_{period}'] = ta.trend.sma_indicator(df['close'], window=period)
            features[f'price_to_sma_{period}'] = df['close'] / features[f'sma_{period}'] - 1.0
            
        # Exponential Moving Averages
        for period in self.ema_periods:
            features[f'ema_{period}'] = ta.trend.ema_indicator(df['close'], window=period)
            
        # MACD
        macd_line = ta.trend.macd_diff(df['close'], 
                                       window_slow=self.macd_params[1],
                                       window_fast=self.macd_params[0],
                                       window_sign=self.macd_params[2])
        macd_signal = ta.trend.macd_signal(df['close'],
                                          window_slow=self.macd_params[1], 
                                          window_fast=self.macd_params[0],
                                          window_sign=self.macd_params[2])
        features['macd_diff'] = macd_line
        features['macd_signal'] = macd_signal
        features['macd_histogram'] = macd_line - macd_signal
        
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
                      index_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
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
        
        # Combine all features
        feature_df = pd.concat(all_features, axis=1)
        
        # Forward fill missing values and then drop remaining NaNs
        feature_df = feature_df.fillna(method='ffill').dropna()
        
        logger.info(f"Generated {len(feature_df.columns)} features for {len(feature_df)} bars")
        return feature_df
    
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