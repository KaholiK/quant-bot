"""
Cross-validation utilities for time series machine learning.
Implements purged k-fold and walk-forward validation with embargo periods.
"""

import numpy as np
import pandas as pd
from typing import Iterator, Tuple, List, Dict, Any, Optional
from sklearn.model_selection import BaseCrossValidator
from loguru import logger


class PurgedKFold(BaseCrossValidator):
    """
    Purged K-Fold cross-validation for time series with overlapping samples.
    
    This addresses the data leakage problem in time series where:
    1. Training samples might overlap with test samples temporally
    2. Future information might leak into past predictions
    
    The purging process removes training samples that are too close 
    temporally to test samples to prevent leakage.
    """
    
    def __init__(self, 
                 n_splits: int = 5,
                 embargo_frac: float = 0.02,
                 random_state: Optional[int] = None):
        """
        Initialize PurgedKFold.
        
        Args:
            n_splits: Number of folds
            embargo_frac: Fraction of total samples to use as embargo period
            random_state: Random state for reproducibility
        """
        self.n_splits = n_splits
        self.embargo_frac = embargo_frac
        self.random_state = random_state
        
    def _get_test_ranges(self, X: pd.DataFrame) -> List[Tuple[int, int]]:
        """Get test ranges for each fold."""
        n_samples = len(X)
        test_size = n_samples // self.n_splits
        
        test_ranges = []
        for i in range(self.n_splits):
            start_idx = i * test_size
            end_idx = start_idx + test_size if i < self.n_splits - 1 else n_samples
            test_ranges.append((start_idx, end_idx))
            
        return test_ranges
    
    def _purge_train_indices(self, 
                           train_indices: np.ndarray,
                           test_start: int, 
                           test_end: int,
                           X: pd.DataFrame) -> np.ndarray:
        """Remove training samples that are too close to test period."""
        embargo_size = int(self.embargo_frac * len(X))
        
        # Remove training samples within embargo period of test set
        purged_mask = (train_indices < test_start - embargo_size) | (train_indices >= test_end + embargo_size)
        purged_indices = train_indices[purged_mask]
        
        logger.debug(f"Purged {len(train_indices) - len(purged_indices)} samples from training set")
        return purged_indices
    
    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test splits with purging."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame with datetime index")
            
        if not isinstance(X.index, pd.DatetimeIndex):
            logger.warning("X index is not DatetimeIndex, assuming it's ordered chronologically")
            
        test_ranges = self._get_test_ranges(X)
        
        for fold_idx, (test_start, test_end) in enumerate(test_ranges):
            # Test indices
            test_indices = np.arange(test_start, test_end)
            
            # All other indices as potential training
            train_indices = np.concatenate([
                np.arange(0, test_start),
                np.arange(test_end, len(X))
            ])
            
            # Purge training indices
            train_indices = self._purge_train_indices(train_indices, test_start, test_end, X)
            
            if len(train_indices) == 0:
                logger.warning(f"No training samples left after purging for fold {fold_idx}")
                continue
                
            logger.info(f"Fold {fold_idx}: {len(train_indices)} train, {len(test_indices)} test samples")
            yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return number of splits."""
        return self.n_splits


def walk_forward_splits(X: pd.DataFrame,
                       train_months: int = 24,
                       test_months: int = 3,
                       step_months: int = 3) -> Iterator[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Generate walk-forward splits for time series validation.
    
    Args:
        X: DataFrame with datetime index
        train_months: Number of months for training window
        test_months: Number of months for test window  
        step_months: Number of months to step forward for next split
        
    Yields:
        Tuple of (train_index, test_index) for each split
    """
    if not isinstance(X.index, pd.DatetimeIndex):
        raise ValueError("X must have DatetimeIndex")
        
    start_date = X.index.min()
    end_date = X.index.max()
    
    current_date = start_date + pd.DateOffset(months=train_months)
    split_count = 0
    
    while current_date + pd.DateOffset(months=test_months) <= end_date:
        # Training period
        train_start = current_date - pd.DateOffset(months=train_months)
        train_end = current_date
        
        # Test period
        test_start = current_date
        test_end = current_date + pd.DateOffset(months=test_months)
        
        # Get indices for these periods
        train_mask = (X.index >= train_start) & (X.index < train_end)
        test_mask = (X.index >= test_start) & (X.index < test_end)
        
        train_idx = X.index[train_mask]
        test_idx = X.index[test_mask]
        
        if len(train_idx) > 0 and len(test_idx) > 0:
            logger.info(f"Walk-forward split {split_count}: "
                       f"train {len(train_idx)} samples ({train_start.date()} to {train_end.date()}), "
                       f"test {len(test_idx)} samples ({test_start.date()} to {test_end.date()})")
            yield train_idx, test_idx
            split_count += 1
        
        current_date += pd.DateOffset(months=step_months)


def calculate_cv_scores(cv_results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate cross-validation score statistics.
    
    Args:
        cv_results: List of dictionaries containing scores for each fold
        
    Returns:
        Dictionary with mean, std, min, max for each metric
    """
    if not cv_results:
        return {}
        
    # Get all metric names
    metrics = set()
    for result in cv_results:
        metrics.update(result.keys())
    
    score_stats = {}
    
    for metric in metrics:
        scores = [result.get(metric, np.nan) for result in cv_results]
        scores = [s for s in scores if not np.isnan(s)]  # Remove NaN values
        
        if scores:
            score_stats[metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'count': len(scores)
            }
    
    return score_stats


def validate_time_series_split(X: pd.DataFrame, 
                              train_idx: np.ndarray, 
                              test_idx: np.ndarray,
                              min_gap_days: int = 1) -> Tuple[bool, List[str]]:
    """
    Validate that a time series train/test split doesn't have data leakage.
    
    Args:
        X: DataFrame with datetime index
        train_idx: Training indices
        test_idx: Test indices
        min_gap_days: Minimum gap between train and test periods
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check for overlap
    overlap = set(train_idx) & set(test_idx)
    if overlap:
        issues.append(f"Train/test overlap detected: {len(overlap)} samples")
    
    # Check temporal ordering
    if isinstance(X.index, pd.DatetimeIndex):
        train_dates = X.index[train_idx]
        test_dates = X.index[test_idx]
        
        train_max = train_dates.max()
        test_min = test_dates.min()
        
        # Check if there's sufficient gap
        gap = (test_min - train_max).days
        if gap < min_gap_days:
            issues.append(f"Insufficient gap between train and test: {gap} days < {min_gap_days}")
            
        # Check if test period comes after training period
        if train_max >= test_min:
            issues.append("Test period does not come after training period")
    
    # Check for empty sets
    if len(train_idx) == 0:
        issues.append("Empty training set")
    if len(test_idx) == 0:
        issues.append("Empty test set")
        
    return len(issues) == 0, issues


class TimeSeriesSplit:
    """
    Custom time series splitter with various splitting strategies.
    """
    
    def __init__(self, 
                 strategy: str = 'purged_kfold',
                 **kwargs):
        """
        Initialize time series splitter.
        
        Args:
            strategy: Splitting strategy ('purged_kfold', 'walk_forward')
            **kwargs: Strategy-specific parameters
        """
        self.strategy = strategy
        self.kwargs = kwargs
        
        if strategy == 'purged_kfold':
            self.splitter = PurgedKFold(**kwargs)
        elif strategy == 'walk_forward':
            self.splitter = None  # Will use walk_forward_splits function
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test splits."""
        if self.strategy == 'purged_kfold':
            yield from self.splitter.split(X, y)
        elif self.strategy == 'walk_forward':
            for train_idx, test_idx in walk_forward_splits(X, **self.kwargs):
                train_positions = X.index.get_indexer(train_idx)
                test_positions = X.index.get_indexer(test_idx)
                yield train_positions, test_positions
                
    def get_n_splits(self, X=None, y=None) -> int:
        """Return number of splits."""
        if self.strategy == 'purged_kfold':
            return self.splitter.get_n_splits(X, y)
        elif self.strategy == 'walk_forward':
            # Estimate number of splits for walk-forward
            if X is None:
                return 1
            train_months = self.kwargs.get('train_months', 24)
            test_months = self.kwargs.get('test_months', 3)
            step_months = self.kwargs.get('step_months', 3)
            
            total_months = (X.index.max() - X.index.min()).days / 30.44  # Approx months
            available_months = total_months - train_months
            return max(1, int(available_months / step_months))


def combinatorial_purged_cv(X: pd.DataFrame,
                           y: pd.Series,
                           model,
                           n_splits: int = 5,
                           embargo_frac: float = 0.02,
                           sample_weights: Optional[pd.Series] = None) -> Dict[str, Any]:
    """
    Run combinatorial purged cross-validation.
    
    This is an advanced technique that tests multiple train/test combinations
    to get more robust performance estimates.
    
    Args:
        X: Features DataFrame
        y: Target series
        model: Scikit-learn compatible model
        n_splits: Number of splits
        embargo_frac: Embargo fraction
        sample_weights: Optional sample weights
        
    Returns:
        Dictionary with CV results and statistics
    """
    cv = PurgedKFold(n_splits=n_splits, embargo_frac=embargo_frac)
    
    fold_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        # Get train/test data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Get sample weights if provided
        weights_train = sample_weights.iloc[train_idx] if sample_weights is not None else None
        
        # Fit model
        if weights_train is not None and hasattr(model, 'sample_weight'):
            model.fit(X_train, y_train, sample_weight=weights_train)
        else:
            model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        fold_result = {
            'fold': fold_idx,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        if y_pred_proba is not None and len(np.unique(y_test)) > 1:
            fold_result['auc'] = roc_auc_score(y_test, y_pred_proba, average='weighted', multi_class='ovr')
        
        fold_results.append(fold_result)
    
    # Calculate overall statistics
    cv_stats = calculate_cv_scores(fold_results)
    
    return {
        'fold_results': fold_results,
        'cv_stats': cv_stats,
        'n_splits': n_splits,
        'embargo_frac': embargo_frac
    }