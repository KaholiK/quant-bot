"""
Triple barrier labeling system for quantitative trading.
Implements vectorized triple barrier method and meta-labeling utilities.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from loguru import logger


def apply_triple_barrier(prices: pd.Series, 
                        atr: pd.Series,
                        profit_take_mult: float = 1.75,
                        stop_loss_mult: float = 1.0, 
                        horizon_bars: int = 5) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Apply triple barrier method to generate labels.
    
    Args:
        prices: Price series (typically close prices)
        atr: Average True Range series for dynamic barriers
        profit_take_mult: Multiplier for profit take barrier (e.g., 1.75 * ATR)
        stop_loss_mult: Multiplier for stop loss barrier (e.g., 1.0 * ATR) 
        horizon_bars: Maximum holding period in bars
        
    Returns:
        Tuple of (labels, touch_times, barriers_df)
        - labels: -1 (stop loss), 0 (timeout), 1 (profit take)
        - touch_times: Time index when barrier was touched
        - barriers_df: DataFrame with barrier levels for each position
    """
    if len(prices) != len(atr):
        raise ValueError("Prices and ATR series must have same length")
        
    if len(prices) < horizon_bars:
        logger.warning(f"Insufficient data: {len(prices)} < {horizon_bars}")
        return pd.Series(dtype=int), pd.Series(dtype='datetime64[ns]'), pd.DataFrame()
    
    # Initialize outputs
    labels = pd.Series(index=prices.index, dtype=int)
    touch_times = pd.Series(index=prices.index, dtype='datetime64[ns]')
    
    # Store barrier levels for analysis
    barriers_data = []
    
    # Vectorized barrier calculation
    for i in range(len(prices) - horizon_bars):
        entry_time = prices.index[i]
        entry_price = prices.iloc[i]
        entry_atr = atr.iloc[i]
        
        if pd.isna(entry_price) or pd.isna(entry_atr) or entry_atr <= 0:
            continue
            
        # Define barriers
        profit_barrier = entry_price * (1 + profit_take_mult * entry_atr / entry_price)
        loss_barrier = entry_price * (1 - stop_loss_mult * entry_atr / entry_price)
        
        # Look ahead for barrier touches
        end_idx = min(i + horizon_bars + 1, len(prices))
        future_prices = prices.iloc[i+1:end_idx]
        
        if len(future_prices) == 0:
            continue
            
        # Check for barrier touches
        profit_touch = future_prices >= profit_barrier
        loss_touch = future_prices <= loss_barrier
        
        # Find first touch
        profit_touch_idx = profit_touch.idxmax() if profit_touch.any() else None
        loss_touch_idx = loss_touch.idxmax() if loss_touch.any() else None
        
        # Determine which barrier was hit first
        label = 0  # Default: timeout
        touch_time = None
        
        if profit_touch.any() and loss_touch.any():
            # Both barriers touched, use the first one
            profit_time = future_prices.index.get_loc(profit_touch_idx)
            loss_time = future_prices.index.get_loc(loss_touch_idx)
            
            if profit_time < loss_time:
                label = 1
                touch_time = profit_touch_idx
            else:
                label = -1
                touch_time = loss_touch_idx
                
        elif profit_touch.any():
            label = 1
            touch_time = profit_touch_idx
            
        elif loss_touch.any():
            label = -1
            touch_time = loss_touch_idx
        else:
            # No barrier touched within horizon
            touch_time = future_prices.index[-1]
            
        labels.loc[entry_time] = label
        touch_times.loc[entry_time] = touch_time
        
        # Store barrier info
        barriers_data.append({
            'entry_time': entry_time,
            'entry_price': entry_price,
            'profit_barrier': profit_barrier,
            'loss_barrier': loss_barrier,
            'touch_time': touch_time,
            'label': label
        })
    
    barriers_df = pd.DataFrame(barriers_data)
    if len(barriers_df) > 0:
        barriers_df = barriers_df.set_index('entry_time')
    
    # Remove NaN labels
    labels = labels.dropna()
    touch_times = touch_times.dropna()
    
    logger.info(f"Generated {len(labels)} triple barrier labels")
    logger.info(f"Label distribution: {labels.value_counts().to_dict()}")
    
    return labels, touch_times, barriers_df


def get_barrier_statistics(barriers_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate statistics from barrier analysis."""
    if len(barriers_df) == 0:
        return {}
        
    stats = {}
    
    # Hit rates
    total_trades = len(barriers_df)
    profit_hits = (barriers_df['label'] == 1).sum()
    loss_hits = (barriers_df['label'] == -1).sum()
    timeouts = (barriers_df['label'] == 0).sum()
    
    stats['total_trades'] = total_trades
    stats['profit_hit_rate'] = profit_hits / total_trades if total_trades > 0 else 0
    stats['loss_hit_rate'] = loss_hits / total_trades if total_trades > 0 else 0
    stats['timeout_rate'] = timeouts / total_trades if total_trades > 0 else 0
    
    # Barrier efficiency (how often barriers are actually hit)
    stats['barrier_efficiency'] = (profit_hits + loss_hits) / total_trades if total_trades > 0 else 0
    
    return stats


def create_meta_labels(primary_predictions: pd.Series, 
                      true_labels: pd.Series,
                      threshold: float = 0.5) -> pd.Series:
    """
    Create meta-labels for meta-labeling approach.
    
    Meta-labeling asks: given that the primary model predicts a trade,
    should we take that trade?
    
    Args:
        primary_predictions: Predicted probabilities from primary model
        true_labels: True triple barrier labels
        threshold: Threshold for primary model predictions
        
    Returns:
        Meta-labels: 1 if should take trade, 0 if should skip
    """
    # Align series
    aligned_preds, aligned_labels = primary_predictions.align(true_labels, join='inner')
    
    if len(aligned_preds) == 0:
        logger.warning("No aligned predictions and labels for meta-labeling")
        return pd.Series(dtype=int)
    
    # Create binary predictions from primary model
    binary_preds = (aligned_preds > threshold).astype(int)
    
    # Meta-label: 1 if primary prediction matches true label, 0 otherwise
    meta_labels = (binary_preds == aligned_labels).astype(int)
    
    logger.info(f"Created {len(meta_labels)} meta-labels")
    logger.info(f"Meta-label acceptance rate: {meta_labels.mean():.3f}")
    
    return meta_labels


def validate_labels(labels: pd.Series, prices: pd.Series) -> Tuple[bool, list]:
    """Validate label series for common issues."""
    issues = []
    
    # Check for valid label values
    valid_labels = {-1, 0, 1}
    invalid_labels = set(labels.unique()) - valid_labels
    if invalid_labels:
        issues.append(f"Invalid label values found: {invalid_labels}")
    
    # Check alignment with prices
    if len(labels) == 0:
        issues.append("Empty label series")
    elif not labels.index.equals(prices.index[:len(labels)]):
        issues.append("Label index does not align with price index")
    
    # Check for reasonable label distribution
    label_counts = labels.value_counts()
    if len(label_counts) == 1:
        issues.append(f"All labels have same value: {label_counts.index[0]}")
    
    # Check for temporal ordering
    if not labels.index.is_monotonic_increasing:
        issues.append("Label index is not monotonically increasing")
    
    return len(issues) == 0, issues


def fractional_differentiation(series: pd.Series, d: float = 0.5, threshold: float = 1e-4) -> pd.Series:
    """
    Apply fractional differentiation to make a series stationary while preserving memory.
    
    Useful for price series that need to be made stationary for ML models
    while retaining some memory of past values.
    
    Args:
        series: Input time series
        d: Fractional differentiation parameter (0 < d < 1)
        threshold: Threshold for coefficient truncation
        
    Returns:
        Fractionally differentiated series
    """
    if not 0 < d < 1:
        raise ValueError("Fractional differentiation parameter d must be between 0 and 1")
    
    # Calculate fractional differentiation weights
    weights = [1.0]
    k = 1
    
    while True:
        weight = -weights[-1] * (d - k + 1) / k
        if abs(weight) < threshold:
            break
        weights.append(weight)
        k += 1
    
    weights = np.array(weights)
    
    # Apply fractional differentiation
    frac_diff = pd.Series(index=series.index, dtype=float)
    
    for i in range(len(weights), len(series)):
        frac_diff.iloc[i] = np.dot(weights, series.iloc[i-len(weights)+1:i+1].values[::-1])
    
    return frac_diff.dropna()


class TripleBarrierLabeler:
    """Class-based interface for triple barrier labeling with configuration."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        # Safe extraction with fallbacks to defaults
        labels_config = config.get('labels', {})
        
        self.profit_take_mult = labels_config.get('tp_atr_mult', 1.75)
        self.stop_loss_mult = labels_config.get('sl_atr_mult', 1.00)
        self.horizon_bars = labels_config.get('horizon_bars', 5)
        
    def fit_transform(self, prices: pd.Series, atr: pd.Series) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        """Apply triple barrier labeling."""
        return apply_triple_barrier(
            prices=prices,
            atr=atr,
            profit_take_mult=self.profit_take_mult,
            stop_loss_mult=self.stop_loss_mult,
            horizon_bars=self.horizon_bars
        )
    
    def get_label_info(self, labels: pd.Series) -> Dict[str, Any]:
        """Get information about generated labels."""
        if len(labels) == 0:
            return {"error": "No labels generated"}
            
        info = {
            "total_labels": len(labels),
            "label_distribution": labels.value_counts().to_dict(),
            "label_proportions": labels.value_counts(normalize=True).to_dict(),
            "first_label_time": str(labels.index[0]),
            "last_label_time": str(labels.index[-1])
        }
        
        return info