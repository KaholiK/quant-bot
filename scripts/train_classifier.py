#!/usr/bin/env python3
"""
Enhanced classifier training script with parallelization and Optuna optimization.
Fetches historical data, builds features, generates labels, and trains XGBoost with purged CV.
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

# Add algos to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import pandas as pd
    import numpy as np
    import yfinance as yf
    import requests
    from sklearn.model_selection import cross_val_score
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import classification_report, accuracy_score, log_loss, brier_score_loss
    import xgboost as xgb
    import joblib
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Optional dependencies
    try:
        import optuna
        from optuna.samplers import TPESampler
        OPTUNA_AVAILABLE = True
    except ImportError:
        OPTUNA_AVAILABLE = False
        
    from algos.core.feature_pipe import FeaturePipeline
    from algos.core.labels import TripleBarrierLabeler
    from algos.core.cv_utils import PurgedKFold
    from algos.core.config_loader import load_config, get_legacy_dict
    
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Missing dependencies for training: {e}")
    DEPENDENCIES_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class DataFetcher:
    """Fetch historical market data from various sources."""
    
    def __init__(self):
        self.sp100_symbols = [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'TSLA', 'NVDA', 'META', 'BRK-B', 'UNH',
            'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'PFE', 'ABBV',
            'BAC', 'COST', 'KO', 'AVGO', 'PEP', 'TMO', 'WMT', 'MRK', 'DHR', 'NEE',
            'LIN', 'VZ', 'ABT', 'ADBE', 'ORCL', 'CRM', 'ACN', 'TXN', 'WFC', 'NKE',
            'RTX', 'PM', 'AMGN', 'HON', 'UNP', 'LOW', 'QCOM', 'ELV', 'SPGI', 'T'
        ]
    
    def fetch_sp100_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch S&P 100 historical data using yfinance.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary of DataFrames keyed by symbol
        """
        logger.info(f"Fetching S&P 100 data from {start_date} to {end_date}")
        
        data = {}
        total_symbols = len(self.sp100_symbols)
        
        for i, symbol in enumerate(self.sp100_symbols):
            try:
                logger.info(f"Fetching {symbol} ({i+1}/{total_symbols})")
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, interval='1h')
                
                if len(df) > 100:  # Ensure sufficient data
                    # Rename columns to match our format
                    df = df.rename(columns={
                        'Open': 'open',
                        'High': 'high', 
                        'Low': 'low',
                        'Close': 'close',
                        'Volume': 'volume'
                    })
                    data[symbol] = df
                    logger.info(f"✓ {symbol}: {len(df)} bars")
                else:
                    logger.warning(f"⚠ {symbol}: insufficient data ({len(df)} bars)")
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"✗ Failed to fetch {symbol}: {e}")
                continue
        
        logger.info(f"Successfully fetched data for {len(data)}/{total_symbols} symbols")
        return data
    
    def fetch_crypto_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch crypto data from Binance public API.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary of DataFrames keyed by symbol
        """
        logger.info(f"Fetching crypto data from {start_date} to {end_date}")
        
        crypto_pairs = ['BTCUSDT', 'ETHUSDT']
        data = {}
        
        for pair in crypto_pairs:
            try:
                logger.info(f"Fetching {pair}")
                
                # Convert dates to timestamps
                start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
                end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
                
                # Binance API endpoint
                url = f"https://api.binance.com/api/v3/klines"
                params = {
                    'symbol': pair,
                    'interval': '1h',
                    'startTime': start_ts,
                    'endTime': end_ts,
                    'limit': 1000
                }
                
                all_data = []
                current_start = start_ts
                
                while current_start < end_ts:
                    params['startTime'] = current_start
                    
                    response = requests.get(url, params=params, timeout=30)
                    response.raise_for_status()
                    
                    klines = response.json()
                    if not klines:
                        break
                        
                    all_data.extend(klines)
                    current_start = klines[-1][6] + 1  # Next candle start time
                    
                    time.sleep(0.2)  # Rate limiting
                
                if all_data:
                    # Convert to DataFrame
                    df = pd.DataFrame(all_data, columns=[
                        'open_time', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                        'taker_buy_quote', 'ignore'
                    ])
                    
                    # Clean up
                    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                    df = df.set_index('open_time')
                    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
                    
                    # Map to our symbol format
                    symbol = pair.replace('USDT', 'USD')
                    data[symbol] = df
                    logger.info(f"✓ {symbol}: {len(df)} bars")
                
            except Exception as e:
                logger.error(f"✗ Failed to fetch {pair}: {e}")
                continue
        
        logger.info(f"Successfully fetched crypto data for {len(data)} pairs")
        return data


class ClassifierTrainer:
    """Train and calibrate XGBoost classifier with purged cross-validation."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize trainer with configuration."""
        self.config = load_config(config_path)
        self.legacy_config = get_legacy_dict(self.config)
        self.feature_pipeline = FeaturePipeline(self.legacy_config)
        self.labeler = TripleBarrierLabeler(self.legacy_config)
        
        # Create output directories
        self.cache_dir = Path("data/cache")
        self.models_dir = Path("models")
        self.reports_dir = Path("reports")
        
        for dir_path in [self.cache_dir, self.models_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Log label settings
        logger.info(f"Label settings: horizon_bars={self.labeler.horizon_bars}, "
                   f"tp_atr_mult={self.labeler.profit_take_mult}, "
                   f"sl_atr_mult={self.labeler.stop_loss_mult}")
        
        # Parallelization settings
        self.n_jobs = min(mp.cpu_count(), 8)  # Limit to 8 cores max
        logger.info(f"Using {self.n_jobs} CPU cores for parallelization")
    
    def prepare_training_data_parallel(self, price_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """
        Prepare features and labels for training using parallel processing.
        
        Args:
            price_data: Dictionary of price DataFrames by symbol
            
        Returns:
            Tuple of (features_df, labels_df, sample_weights)
        """
        logger.info(f"Preparing training data for {len(price_data)} symbols using {self.n_jobs} processes...")
        
        # Check cache first
        cache_file = self.cache_dir / f"training_data_{datetime.now().strftime('%Y%m%d')}.parquet"
        if cache_file.exists():
            logger.info("Loading cached training data...")
            try:
                cached_data = pd.read_parquet(cache_file)
                features_df = cached_data.drop(['label', 'sample_weight'], axis=1)
                labels_df = cached_data[['label']].copy()
                sample_weights = cached_data['sample_weight'].values
                logger.info(f"Loaded cached data: {len(features_df)} samples")
                return features_df, labels_df, sample_weights
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        # Process symbols in parallel
        symbol_chunks = list(price_data.items())
        chunk_size = max(1, len(symbol_chunks) // self.n_jobs)
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # Submit parallel processing tasks
            futures = []
            for i in range(0, len(symbol_chunks), chunk_size):
                chunk = symbol_chunks[i:i+chunk_size]
                future = executor.submit(self._process_symbol_chunk, chunk)
                futures.append(future)
            
            # Collect results
            all_features = []
            all_labels = []
            all_weights = []
            
            for future in as_completed(futures):
                try:
                    chunk_features, chunk_labels, chunk_weights = future.result()
                    all_features.extend(chunk_features)
                    all_labels.extend(chunk_labels)
                    all_weights.extend(chunk_weights)
                except Exception as e:
                    logger.error(f"Chunk processing failed: {e}")
        
        if not all_features:
            raise ValueError("No valid training data generated")
        
        # Combine all data
        features_df = pd.concat(all_features, ignore_index=True)
        labels_df = pd.concat(all_labels, ignore_index=True)
        sample_weights = np.array(all_weights)
        
        # Cache the results
        try:
            cache_data = features_df.copy()
            cache_data['label'] = labels_df['label']
            cache_data['sample_weight'] = sample_weights
            cache_data.to_parquet(cache_file, compression='snappy')
            logger.info(f"Training data cached to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache data: {e}")
        
        logger.info(f"Training data prepared: {len(features_df)} samples, {len(features_df.columns)} features")
        return features_df, labels_df, sample_weights
    
    def _process_symbol_chunk(self, symbol_chunk: List[Tuple[str, pd.DataFrame]]) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[float]]:
        """Process a chunk of symbols for feature generation."""
        chunk_features = []
        chunk_labels = []
        chunk_weights = []
        
        for symbol, df in symbol_chunk:
            if len(df) < 100:  # Skip symbols with insufficient data
                continue
                
            try:
                # Generate features
                features = self.feature_pipeline.build_features(df, symbol=symbol)
                
                # Generate labels
                labels_result = self.labeler.create_labels(df)
                labels = labels_result['labels']
                
                if len(features) != len(labels):
                    # Align lengths
                    min_len = min(len(features), len(labels))
                    features = features.iloc[-min_len:]
                    labels = labels.iloc[-min_len:]
                
                # Add symbol column
                features['symbol'] = symbol
                labels['symbol'] = symbol
                
                # Sample weights (higher weight for cleaner signals)
                weights = np.ones(len(labels))
                if 'confidence' in labels_result:
                    weights = labels_result['confidence'].values
                
                chunk_features.append(features)
                chunk_labels.append(labels)
                chunk_weights.extend(weights)
                
                logger.debug(f"Processed {symbol}: {len(features)} samples")
                
            except Exception as e:
                logger.error(f"Failed to process {symbol}: {e}")
                continue
        
        return chunk_features, chunk_labels, chunk_weights
    
    def optimize_hyperparameters(self, 
                                X: pd.DataFrame, 
                                y: pd.Series, 
                                sample_weights: np.ndarray,
                                n_trials: int = 50) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            X: Features
            y: Labels  
            sample_weights: Sample weights
            n_trials: Number of optimization trials
            
        Returns:
            Best hyperparameters
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, using default hyperparameters")
            return self._get_default_params()
        
        logger.info(f"Optimizing hyperparameters with {n_trials} trials...")
        
        def objective(trial):
            """Optuna objective function."""
            params = {
                'objective': 'multi:softprob',
                'num_class': 3,
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': 42,
                'n_jobs': 1  # Use single job within each trial
            }
            
            # Create and train model
            model = xgb.XGBClassifier(**params)
            
            # Purged cross-validation
            try:
                cv = PurgedKFold(n_splits=5, embargo_frac=0.02)
                cv_scores = cross_val_score(
                    model, X, y,
                    cv=cv,
                    scoring='accuracy',
                    fit_params={'sample_weight': sample_weights},
                    n_jobs=1
                )
                return cv_scores.mean()
            except Exception:
                # Fallback to standard CV
                cv_scores = cross_val_score(
                    model, X, y, 
                    cv=5, 
                    scoring='accuracy',
                    fit_params={'sample_weight': sample_weights},
                    n_jobs=1
                )
                return cv_scores.mean()
        
        # Create study and optimize
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best accuracy: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        
        # Add fixed parameters
        best_params = study.best_params.copy()
        best_params.update({
            'objective': 'multi:softprob',
            'num_class': 3,
            'random_state': 42,
            'n_jobs': -1
        })
        
        return best_params
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default XGBoost parameters."""
        return {
            'objective': 'multi:softprob',
            'num_class': 3,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 150,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1
        }
                
                # Add symbol column
                features['symbol'] = symbol
                labels['symbol'] = symbol
                
                # Sample weights (higher weight for cleaner signals)
                weights = np.ones(len(labels))
                if 'confidence' in labels_result:
                    weights = labels_result['confidence'].values
                
                all_features.append(features)
                all_labels.append(labels)
                all_weights.extend(weights)
                
            except Exception as e:
                logger.error(f"Failed to process {symbol}: {e}")
                continue
        
        if not all_features:
            raise ValueError("No valid training data generated")
        
        # Combine all data
        features_df = pd.concat(all_features, ignore_index=True)
        labels_df = pd.concat(all_labels, ignore_index=True)
        sample_weights = np.array(all_weights)
        
        logger.info(f"Training data prepared: {len(features_df)} samples, {len(features_df.columns)} features")
        return features_df, labels_df, sample_weights
    
    def train_classifier(self, 
                        features_df: pd.DataFrame,
                        labels_df: pd.DataFrame, 
                        sample_weights: np.ndarray,
                        optimize_params: bool = True) -> Dict[str, Any]:
        """
        Train XGBoost classifier with optional hyperparameter optimization.
        
        Args:
            features_df: Features DataFrame
            labels_df: Labels DataFrame
            sample_weights: Sample weights
            optimize_params: Whether to optimize hyperparameters
            
        Returns:
            Dictionary with trained models and metrics
        """
        logger.info("Training XGBoost classifier...")
        
        # Prepare data
        feature_cols = [col for col in features_df.columns if col != 'symbol']
        X = features_df[feature_cols].fillna(0)
        y = labels_df['label'].fillna(0).astype(int)
        
        # Remove invalid labels
        valid_mask = y.isin([-1, 0, 1])
        X = X[valid_mask]
        y = y[valid_mask]
        sample_weights = sample_weights[valid_mask]
        
        # Convert labels to 0, 1, 2 for XGBoost
        y_mapped = y.map({-1: 0, 0: 1, 1: 2})
        
        logger.info(f"Training on {len(X)} samples with {len(feature_cols)} features")
        logger.info(f"Label distribution: {y.value_counts().to_dict()}")
        
        # Optimize hyperparameters if requested
        if optimize_params:
            best_params = self.optimize_hyperparameters(X, y_mapped, sample_weights)
        else:
            best_params = self._get_default_params()
        
        # Initialize model with best parameters
        model = xgb.XGBClassifier(**best_params)
        
        # Purged cross-validation evaluation
        logger.info("Performing purged cross-validation...")
        try:
            cv = PurgedKFold(n_splits=5, embargo_frac=0.02)
            cv_scores = cross_val_score(
                model, X, y_mapped, 
                cv=cv,
                scoring='accuracy',
                fit_params={'sample_weight': sample_weights},
                n_jobs=1  # Avoid nested parallelization
            )
            cv_accuracy = cv_scores.mean()
            cv_std = cv_scores.std()
            logger.info(f"Purged CV Accuracy: {cv_accuracy:.4f} ± {cv_std:.4f}")
        except Exception as e:
            logger.warning(f"Purged CV failed, using standard CV: {e}")
            cv_scores = cross_val_score(
                model, X, y_mapped, 
                cv=5, 
                scoring='accuracy',
                fit_params={'sample_weight': sample_weights}
            )
            cv_accuracy = cv_scores.mean()
            cv_std = cv_scores.std()
            logger.info(f"Standard CV Accuracy: {cv_accuracy:.4f} ± {cv_std:.4f}")
        
        # Train final model
        logger.info("Training final model...")
        model.fit(X, y_mapped, sample_weight=sample_weights)
        
        # Calibrate probabilities
        logger.info("Calibrating probabilities...")
        calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
        calibrated_model.fit(X, y_mapped, sample_weight=sample_weights)
        
        # Evaluate on training set
        y_pred = calibrated_model.predict(X)
        y_proba = calibrated_model.predict_proba(X)
        
        train_accuracy = accuracy_score(y_mapped, y_pred)
        
        # Calculate additional metrics
        try:
            brier_score = brier_score_loss(y_mapped == 2, y_proba[:, 2])  # Probability of class 2 (buy)
            log_loss_score = log_loss(y_mapped, y_proba)
        except Exception as e:
            logger.warning(f"Error calculating additional metrics: {e}")
            brier_score = np.nan
            log_loss_score = np.nan
        
        logger.info(f"Training Accuracy: {train_accuracy:.4f}")
        logger.info(f"Brier Score: {brier_score:.4f}")
        logger.info(f"Log Loss: {log_loss_score:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Save models and generate reports
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save main classifier
        model_path = self.models_dir / 'xgb_classifier.joblib'
        joblib.dump(calibrated_model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save feature importance
        importance_path = self.reports_dir / f'feature_importance_{timestamp}.csv'
        feature_importance.to_csv(importance_path, index=False)
        
        # Generate calibration plots
        self._generate_calibration_plots(y_mapped, y_proba, timestamp)
        
        # Generate feature importance plot
        self._generate_feature_importance_plot(feature_importance.head(20), timestamp)
        
        return {
            'model': calibrated_model,
            'feature_importance': feature_importance,
            'cv_accuracy': cv_accuracy,
            'cv_std': cv_std,
            'train_accuracy': train_accuracy,
            'brier_score': brier_score,
            'log_loss': log_loss_score,
            'best_params': best_params,
            'n_samples': len(X),
            'n_features': len(feature_cols)
        }
    
    def _generate_calibration_plots(self, y_true: np.ndarray, y_proba: np.ndarray, timestamp: str) -> None:
        """Generate calibration plots."""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            for i, class_name in enumerate(['Sell', 'Hold', 'Buy']):
                ax = axes[i]
                
                # Calibration plot for each class
                from sklearn.calibration import calibration_curve
                
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_true == i, y_proba[:, i], n_bins=10, normalize=True
                )
                
                ax.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"{class_name} (Class {i})")
                ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
                ax.set_xlabel("Mean Predicted Probability")
                ax.set_ylabel("Fraction of Positives")
                ax.set_title(f"Calibration Plot - {class_name}")
                ax.legend()
                ax.grid(True)
            
            plt.tight_layout()
            plot_path = self.reports_dir / f'calibration_plot_{timestamp}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Calibration plots saved to {plot_path}")
        except Exception as e:
            logger.error(f"Failed to generate calibration plots: {e}")
    
    def _generate_feature_importance_plot(self, feature_importance: pd.DataFrame, timestamp: str) -> None:
        """Generate feature importance plot."""
        try:
            plt.figure(figsize=(10, 8))
            sns.barplot(data=feature_importance, y='feature', x='importance')
            plt.title('Top 20 Feature Importances')
            plt.xlabel('Importance')
            plt.tight_layout()
            
            plot_path = self.reports_dir / f'feature_importance_{timestamp}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Feature importance plot saved to {plot_path}")
        except Exception as e:
            logger.error(f"Failed to generate feature importance plot: {e}")
        accuracy = accuracy_score(y, y_pred)
        
        logger.info(f"Final accuracy: {accuracy:.4f}")
        logger.info("Classification Report:")
        logger.info(classification_report(y, y_pred))
        
        return {
            'model': calibrated_model,
            'feature_columns': feature_cols,
            'cv_scores': cv_scores,
            'accuracy': accuracy,
            'n_samples': len(X),
            'label_distribution': y.value_counts().to_dict()
        }
    
    def train_meta_model(self, 
                        features_df: pd.DataFrame,
                        labels_df: pd.DataFrame,
                        base_model: Any) -> Dict[str, Any]:
        """
        Train meta-model for trade filtering.
        
        Args:
            features_df: Features DataFrame
            labels_df: Labels DataFrame  
            base_model: Base classifier model
            
        Returns:
            Dictionary with meta-model and metrics
        """
        logger.info("Training meta-model for trade filtering...")
        
        # Use base model predictions as features for meta-model
        feature_cols = [col for col in features_df.columns if col != 'symbol']
        X = features_df[feature_cols].fillna(0)
        y = labels_df['label'].fillna(0).astype(int)
        
        # Get base model probabilities
        base_probs = base_model.predict_proba(X)
        
        # Create meta-features
        meta_features = pd.DataFrame({
            'prob_sell': base_probs[:, 0],
            'prob_hold': base_probs[:, 1], 
            'prob_buy': base_probs[:, 2],
            'max_prob': base_probs.max(axis=1),
            'prob_spread': base_probs.max(axis=1) - base_probs.min(axis=1),
            'volatility': features_df.get('volatility', 0).fillna(0),
            'volume_ratio': features_df.get('volume_ratio', 1).fillna(1)
        })
        
        # Binary classification: trade vs no-trade
        y_binary = (y != 0).astype(int)  # 1 for buy/sell, 0 for hold
        
        # Train meta-classifier
        meta_model = xgb.XGBClassifier(
            objective='binary:logistic',
            max_depth=4,
            learning_rate=0.1,
            n_estimators=50,
            random_state=42
        )
        
        meta_model.fit(meta_features, y_binary)
        
        # Evaluate
        y_meta_pred = meta_model.predict(meta_features)
        meta_accuracy = accuracy_score(y_binary, y_meta_pred)
        
        logger.info(f"Meta-model accuracy: {meta_accuracy:.4f}")
        
        return {
            'model': meta_model,
            'feature_columns': list(meta_features.columns),
            'accuracy': meta_accuracy
        }


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train classifier for quant-bot')
    parser.add_argument('--start', default='2022-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default='2024-01-01', help='End date (YYYY-MM-DD)')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to train on')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--output-dir', default='models', help='Output directory for models')
    
    args = parser.parse_args()
    
    if not DEPENDENCIES_AVAILABLE:
        logger.error("Missing required dependencies. Install: pip install pandas numpy yfinance requests scikit-learn xgboost joblib")
        return 1
    
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        fetcher = DataFetcher()
        trainer = ClassifierTrainer(args.config)
        
        # Fetch data
        logger.info("=" * 50)
        logger.info("FETCHING HISTORICAL DATA")
        logger.info("=" * 50)
        
        if args.symbols:
            # Custom symbols
            data = {}
            for symbol in args.symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(start=args.start, end=args.end, interval='1h')
                    if len(df) > 100:
                        df = df.rename(columns={
                            'Open': 'open', 'High': 'high', 'Low': 'low',
                            'Close': 'close', 'Volume': 'volume'
                        })
                        data[symbol] = df
                except Exception as e:
                    logger.error(f"Failed to fetch {symbol}: {e}")
        else:
            # Fetch S&P 100 and crypto
            equity_data = fetcher.fetch_sp100_data(args.start, args.end)
            crypto_data = fetcher.fetch_crypto_data(args.start, args.end)
            data = {**equity_data, **crypto_data}
        
        if not data:
            logger.error("No data fetched. Exiting.")
            return 1
        
        # Prepare training data
        logger.info("=" * 50)
        logger.info("PREPARING TRAINING DATA")
        logger.info("=" * 50)
        
        features_df, labels_df, sample_weights = trainer.prepare_training_data(data)
        
        # Train classifier
        logger.info("=" * 50)
        logger.info("TRAINING CLASSIFIER")
        logger.info("=" * 50)
        
        classifier_result = trainer.train_classifier(features_df, labels_df, sample_weights)
        
        # Train meta-model
        logger.info("=" * 50)
        logger.info("TRAINING META-MODEL")
        logger.info("=" * 50)
        
        meta_result = trainer.train_meta_model(features_df, labels_df, classifier_result['model'])
        
        # Save models
        logger.info("=" * 50)
        logger.info("SAVING MODELS")
        logger.info("=" * 50)
        
        classifier_path = output_dir / 'xgb_classifier.joblib'
        meta_path = output_dir / 'meta_filter.joblib'
        config_path = output_dir / 'training_config.json'
        
        # Save label settings alongside the model
        import json
        label_settings = {
            'horizon_bars': trainer.labeler.horizon_bars,
            'tp_atr_mult': trainer.labeler.profit_take_mult,
            'sl_atr_mult': trainer.labeler.stop_loss_mult,
            'features_config': get_legacy_dict(trainer.config).get('features', {}),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(config_path, 'w') as f:
            json.dump(label_settings, f, indent=2)
        
        joblib.dump(classifier_result, classifier_path)
        joblib.dump(meta_result, meta_path)
        
        logger.info(f"✓ Classifier saved to: {classifier_path}")
        logger.info(f"✓ Meta-model saved to: {meta_path}")
        logger.info(f"✓ Training config saved to: {config_path}")
        
        # Print summary
        logger.info("=" * 50)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Symbols processed: {len(data)}")
        logger.info(f"Training samples: {classifier_result['n_samples']}")
        logger.info(f"Classifier accuracy: {classifier_result['accuracy']:.4f}")
        logger.info(f"Meta-model accuracy: {meta_result['accuracy']:.4f}")
        logger.info(f"CV score: {classifier_result['cv_scores'].mean():.4f} ± {classifier_result['cv_scores'].std():.4f}")
        
        logger.info("Training completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())