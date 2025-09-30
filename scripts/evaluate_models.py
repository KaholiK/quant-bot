#!/usr/bin/env python3
"""
Model evaluation script for walk-forward analysis.
Compares old and new models using out-of-sample testing.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import sys
from datetime import datetime, timedelta

# Add algos to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.metrics import accuracy_score, classification_report
    import warnings
    warnings.filterwarnings('ignore')
    
    from algos.core.config_loader import load_config
    from algos.core.feature_pipe import FeaturePipeline
    from algos.core.labels import TripleBarrierLabeler
    
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Missing dependencies for evaluation: {e}")
    DEPENDENCIES_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate and compare model performance."""
    
    def __init__(self, config_path: str):
        """Initialize evaluator."""
        self.config = load_config(config_path)
        self.feature_pipeline = FeaturePipeline(self.config.dict())
        self.labeler = TripleBarrierLabeler(self.config.dict())
    
    def evaluate_models(self, 
                       old_models_dir: str,
                       new_models_dir: str,
                       old_policies_dir: str,
                       new_policies_dir: str) -> Dict[str, Any]:
        """
        Evaluate and compare old vs new models.
        
        Returns:
            Dictionary with evaluation results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'classifier': {},
            'ppo': {},
            'comparison': {}
        }
        
        # Evaluate classifiers
        logger.info("Evaluating classifiers...")
        classifier_results = self._evaluate_classifiers(old_models_dir, new_models_dir)
        results['classifier'] = classifier_results
        
        # Evaluate PPO policies
        logger.info("Evaluating PPO policies...")
        ppo_results = self._evaluate_ppo_policies(old_policies_dir, new_policies_dir)
        results['ppo'] = ppo_results
        
        # Create comparison summary
        results['comparison'] = self._create_comparison(classifier_results, ppo_results)
        
        return results
    
    def _evaluate_classifiers(self, old_dir: str, new_dir: str) -> Dict[str, Any]:
        """Evaluate classifier models."""
        old_model_path = Path(old_dir) / "xgb_classifier.joblib"
        new_model_path = Path(new_dir) / "xgb_classifier.joblib"
        
        results = {
            'old_model_available': old_model_path.exists(),
            'new_model_available': new_model_path.exists(),
        }
        
        if not new_model_path.exists():
            logger.warning("New classifier model not found")
            return results
        
        try:
            # Load new model
            new_model = joblib.load(new_model_path)
            
            # Generate test data (last 3 months for out-of-sample testing)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            
            test_data = self._get_test_data(start_date, end_date)
            
            if test_data is None or len(test_data) < 100:
                logger.warning("Insufficient test data for evaluation")
                results['insufficient_data'] = True
                return results
            
            # Evaluate new model
            new_metrics = self._evaluate_classifier_performance(new_model, test_data)
            results['new_model'] = new_metrics
            
            # Evaluate old model if available
            if old_model_path.exists():
                old_model = joblib.load(old_model_path)
                old_metrics = self._evaluate_classifier_performance(old_model, test_data)
                results['old_model'] = old_metrics
                
                # Calculate improvements
                results['improvement'] = {
                    'accuracy': new_metrics['accuracy'] - old_metrics['accuracy'],
                    'oos_sortino': new_metrics.get('oos_sortino', 0) - old_metrics.get('oos_sortino', 0),
                    'oos_profit_factor': new_metrics.get('oos_profit_factor', 0) - old_metrics.get('oos_profit_factor', 0),
                }
            
            # Set OOS metrics for gate checking
            results['oos_sortino'] = new_metrics.get('oos_sortino', 0.0)
            results['oos_profit_factor'] = new_metrics.get('oos_profit_factor', 0.0)
            results['oos_max_dd'] = new_metrics.get('oos_max_dd', 1.0)
            
        except Exception as e:
            logger.error(f"Error evaluating classifiers: {e}")
            results['error'] = str(e)
        
        return results
    
    def _evaluate_ppo_policies(self, old_dir: str, new_dir: str) -> Dict[str, Any]:
        """Evaluate PPO policies."""
        old_policy_path = Path(old_dir) / "ppo_policy.zip"
        new_policy_path = Path(new_dir) / "ppo_policy.zip"
        
        results = {
            'old_policy_available': old_policy_path.exists(),
            'new_policy_available': new_policy_path.exists(),
        }
        
        if not new_policy_path.exists():
            logger.warning("New PPO policy not found")
            return results
        
        try:
            # For PPO evaluation, we'd typically need to run the policy in the environment
            # For now, we'll do a simplified evaluation based on saved training metrics
            
            # Look for training summary files
            new_summary_path = Path(new_dir) / "training_summary.json"
            if new_summary_path.exists():
                with open(new_summary_path, 'r') as f:
                    new_summary = json.load(f)
                results['new_policy'] = new_summary
            
            old_summary_path = Path(old_dir) / "training_summary.json"
            if old_summary_path.exists():
                with open(old_summary_path, 'r') as f:
                    old_summary = json.load(f)
                results['old_policy'] = old_summary
                
                # Calculate improvement
                new_reward = new_summary.get('final_reward', 0.0)
                old_reward = old_summary.get('final_reward', 0.0)
                results['reward_improvement'] = new_reward - old_reward
            
            # Set metrics for gate checking
            results['reward_improvement'] = results.get('reward_improvement', 0.0)
            results['reward_std'] = new_summary.get('reward_std', float('inf'))
            
        except Exception as e:
            logger.error(f"Error evaluating PPO policies: {e}")
            results['error'] = str(e)
        
        return results
    
    def _evaluate_classifier_performance(self, model, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate classifier performance on test data."""
        try:
            # Generate features and labels
            features = self.feature_pipeline.build_features(test_data)
            labels_result = self.labeler.create_labels(test_data)
            labels = labels_result['labels']
            
            # Align data
            min_len = min(len(features), len(labels))
            features = features.iloc[-min_len:]
            labels = labels.iloc[-min_len:]
            
            # Prepare data for model
            feature_cols = [col for col in features.columns if col != 'symbol']
            X = features[feature_cols].fillna(0)
            y = labels['label'].fillna(0).astype(int)
            
            # Remove invalid labels
            valid_mask = y.isin([-1, 0, 1])
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 50:  # Need minimum samples
                return {'error': 'insufficient_samples', 'n_samples': len(X)}
            
            # Convert labels for prediction
            y_mapped = y.map({-1: 0, 0: 1, 1: 2})
            
            # Make predictions
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)
            
            # Basic metrics
            accuracy = accuracy_score(y_mapped, y_pred)
            
            # Simulate trading performance for additional metrics
            trading_metrics = self._simulate_trading_performance(y, y_proba, test_data.iloc[-len(y):])
            
            metrics = {
                'accuracy': accuracy,
                'n_samples': len(X),
                **trading_metrics
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in classifier evaluation: {e}")
            return {'error': str(e)}
    
    def _simulate_trading_performance(self, labels: pd.Series, probabilities: np.ndarray, price_data: pd.DataFrame) -> Dict[str, float]:
        """Simulate trading performance to calculate Sortino ratio, profit factor, etc."""
        try:
            # Simple simulation - use buy/sell signals based on model predictions
            # This is a simplified approach - in practice would be more sophisticated
            
            # Get buy signals (class 2 probability > 0.6)
            buy_signals = probabilities[:, 2] > 0.6
            sell_signals = probabilities[:, 0] > 0.6
            
            if len(price_data) != len(buy_signals):
                return {'oos_sortino': 0.0, 'oos_profit_factor': 1.0, 'oos_max_dd': 0.0}
            
            # Calculate returns
            returns = price_data['close'].pct_change().fillna(0)
            
            # Simple strategy returns
            strategy_returns = []
            position = 0
            
            for i, (buy_sig, sell_sig) in enumerate(zip(buy_signals, sell_signals)):
                if buy_sig and position <= 0:
                    position = 1  # Go long
                elif sell_sig and position >= 0:
                    position = -1  # Go short
                
                # Calculate return
                if i < len(returns):
                    strategy_returns.append(position * returns.iloc[i])
                else:
                    strategy_returns.append(0.0)
            
            strategy_returns = pd.Series(strategy_returns)
            
            # Calculate metrics
            if len(strategy_returns) > 0 and strategy_returns.std() > 0:
                # Sortino ratio (downside deviation)
                downside_returns = strategy_returns[strategy_returns < 0]
                downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.01
                sortino = strategy_returns.mean() * np.sqrt(252) / (downside_std * np.sqrt(252))
                
                # Profit factor
                positive_returns = strategy_returns[strategy_returns > 0].sum()
                negative_returns = abs(strategy_returns[strategy_returns < 0].sum())
                profit_factor = positive_returns / negative_returns if negative_returns > 0 else 1.0
                
                # Maximum drawdown
                cumulative = (1 + strategy_returns).cumprod()
                rolling_max = cumulative.expanding().max()
                drawdown = (cumulative - rolling_max) / rolling_max
                max_dd = abs(drawdown.min())
                
                return {
                    'oos_sortino': float(sortino) if not np.isnan(sortino) else 0.0,
                    'oos_profit_factor': float(profit_factor) if not np.isnan(profit_factor) else 1.0,
                    'oos_max_dd': float(max_dd) if not np.isnan(max_dd) else 0.0,
                }
            else:
                return {'oos_sortino': 0.0, 'oos_profit_factor': 1.0, 'oos_max_dd': 0.0}
                
        except Exception as e:
            logger.error(f"Error in trading simulation: {e}")
            return {'oos_sortino': 0.0, 'oos_profit_factor': 1.0, 'oos_max_dd': 0.0}
    
    def _get_test_data(self, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Get test data for evaluation."""
        try:
            # Use yfinance to get SPY data as proxy
            import yfinance as yf
            
            ticker = yf.Ticker("SPY")
            data = ticker.history(start=start_date, end=end_date, interval="1h")
            
            if data.empty:
                return None
            
            # Normalize columns
            data.columns = [col.lower() for col in data.columns]
            data = data[['open', 'high', 'low', 'close', 'volume']]
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching test data: {e}")
            return None
    
    def _create_comparison(self, classifier_results: Dict[str, Any], ppo_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create comparison summary."""
        comparison = {}
        
        # Classifier comparison
        if 'new_model' in classifier_results and 'old_model' in classifier_results:
            comparison['classifier_accuracy'] = {
                'current': classifier_results['old_model'].get('accuracy', 0.0),
                'new': classifier_results['new_model'].get('accuracy', 0.0)
            }
            comparison['sortino_ratio'] = {
                'current': classifier_results['old_model'].get('oos_sortino', 0.0),
                'new': classifier_results['new_model'].get('oos_sortino', 0.0)
            }
        
        # PPO comparison
        if 'reward_improvement' in ppo_results:
            comparison['ppo_reward_improvement'] = {
                'current': 0.0,
                'new': ppo_results['reward_improvement']
            }
        
        return comparison


def main():
    """Main entry point."""
    if not DEPENDENCIES_AVAILABLE:
        logger.error("Required dependencies not available")
        return 1
    
    parser = argparse.ArgumentParser(description="Evaluate model performance")
    parser.add_argument("--old-models-dir", default="models", help="Directory with old models")
    parser.add_argument("--new-models-dir", default="models_new", help="Directory with new models")
    parser.add_argument("--old-policies-dir", default="policies", help="Directory with old policies")
    parser.add_argument("--new-policies-dir", default="policies_new", help="Directory with new policies")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--output", required=True, help="Output JSON file")
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(args.config)
    results = evaluator.evaluate_models(
        args.old_models_dir,
        args.new_models_dir,
        args.old_policies_dir,
        args.new_policies_dir
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())