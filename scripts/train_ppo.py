#!/usr/bin/env python3
"""
PPO training script for execution optimization.
Creates a simple trading execution environment and trains a PPO agent.
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import time

# Add algos to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import numpy as np
    import pandas as pd
    import gym
    from gym import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
    from stable_baselines3.common.monitor import Monitor
    import torch
    
    from algos.core.config_loader import load_config
    
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Missing dependencies for PPO training: {e}")
    print("Install with: pip install stable-baselines3 torch gymnasium")
    DEPENDENCIES_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingExecutionEnv(gym.Env):
    """
    Simple trading execution environment for PPO training.
    
    State: [price, spread, atr, signal_prob, current_position, time_in_position]
    Action: [position_delta, limit_offset] (continuous)
    Reward: Based on execution quality and slippage minimization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize trading execution environment."""
        super().__init__()
        
        self.config = config or {}
        self.max_episode_steps = 1000
        self.current_step = 0
        
        # State space: [price, spread, atr, signal_prob, current_position, time_in_position]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, -2.0, 0.0]),
            high=np.array([1e6, 1.0, 1.0, 1.0, 2.0, 1000.0]),
            dtype=np.float32
        )
        
        # Action space: [position_delta, limit_offset] 
        # position_delta: -1 to 1 (fraction of max position)
        # limit_offset: 0 to 1 (fraction of ATR for limit orders)
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # Environment state
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_step = 0
        
        # Initialize market state
        self.price = 100.0 + np.random.normal(0, 10)
        self.spread = np.random.uniform(0.001, 0.01)  # 0.1% to 1%
        self.atr = np.random.uniform(0.5, 5.0)
        self.signal_prob = np.random.uniform(0.0, 1.0)
        self.current_position = 0.0
        self.time_in_position = 0.0
        
        # Trading history
        self.executed_trades = []
        self.total_slippage = 0.0
        self.total_transaction_costs = 0.0
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        return np.array([
            self.price / 100.0,  # Normalized price
            self.spread,
            self.atr / self.price,  # ATR as fraction of price
            self.signal_prob,
            self.current_position,
            min(self.time_in_position / 100.0, 1.0)  # Normalized time
        ], dtype=np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one environment step."""
        self.current_step += 1
        
        position_delta = action[0]  # -1 to 1
        limit_offset = action[1]    # 0 to 1
        
        # Calculate execution
        reward = self._execute_trade(position_delta, limit_offset)
        
        # Update market state (simple random walk)
        price_change = np.random.normal(0, self.atr * 0.1)
        self.price = max(1.0, self.price + price_change)
        
        # Update spread and ATR (mean-reverting)
        self.spread = max(0.0001, self.spread + np.random.normal(0, 0.001))
        self.atr = max(0.1, self.atr + np.random.normal(0, 0.1))
        
        # New signal occasionally
        if np.random.random() < 0.1:
            self.signal_prob = np.random.uniform(0.0, 1.0)
        
        # Update time in position
        if abs(self.current_position) > 0.01:
            self.time_in_position += 1
        else:
            self.time_in_position = 0
        
        # Episode termination
        done = (self.current_step >= self.max_episode_steps or 
                abs(self.current_position) > 5.0 or  # Risk limit
                self.total_slippage < -100.0)  # Stop loss
        
        # Info
        info = {
            'total_slippage': self.total_slippage,
            'total_trades': len(self.executed_trades),
            'position': self.current_position,
            'price': self.price
        }
        
        return self._get_observation(), reward, done, info
    
    def _execute_trade(self, position_delta: float, limit_offset: float) -> float:
        """
        Execute trade and return reward.
        
        Args:
            position_delta: Desired position change (-1 to 1)
            limit_offset: Limit order offset as fraction of spread (0 to 1)
            
        Returns:
            Reward for this action
        """
        # Scale position delta
        max_position = 1.0
        target_delta = position_delta * max_position
        
        # No trade if delta too small
        if abs(target_delta) < 0.01:
            return 0.0
        
        # Calculate execution price
        if target_delta > 0:  # Buying
            if limit_offset > 0.5:  # Aggressive limit order
                execution_prob = 0.9
                slippage = self.spread * (1 - limit_offset)
            else:  # Conservative limit order
                execution_prob = 0.6
                slippage = -self.spread * limit_offset
        else:  # Selling
            if limit_offset > 0.5:  # Aggressive limit order
                execution_prob = 0.9
                slippage = self.spread * (1 - limit_offset)
            else:  # Conservative limit order
                execution_prob = 0.6
                slippage = -self.spread * limit_offset
        
        # Execution probability based on market conditions
        if self.spread > 0.005:  # Wide spread reduces execution probability
            execution_prob *= 0.7
        
        executed = False
        if np.random.random() < execution_prob:
            executed = True
            self.current_position += target_delta
            self.total_slippage += slippage * abs(target_delta)
            self.total_transaction_costs += 0.001 * abs(target_delta)  # 0.1% fee
            
            self.executed_trades.append({
                'delta': target_delta,
                'slippage': slippage,
                'price': self.price,
                'limit_offset': limit_offset
            })
        
        # Calculate reward
        reward = 0.0
        
        if executed:
            # Reward for successful execution
            reward += 1.0
            
            # Penalty for slippage
            reward -= abs(slippage) * 100
            
            # Reward for good limit order placement
            if 0.2 < limit_offset < 0.8:  # Sweet spot
                reward += 0.5
            
            # Penalty for excessive position
            if abs(self.current_position) > 2.0:
                reward -= 5.0
        else:
            # Small penalty for failed execution
            reward -= 0.1
        
        # Reward for position alignment with signal
        if self.signal_prob > 0.6 and self.current_position > 0:
            reward += 0.2
        elif self.signal_prob < 0.4 and self.current_position < 0:
            reward += 0.2
        
        return reward


class PPOTrainer:
    """PPO trainer for execution optimization."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize PPO trainer."""
        self.config = load_config(config_path)
        
    def create_env(self) -> gym.Env:
        """Create trading execution environment."""
        return TradingExecutionEnv(self.config)
    
    def train(self, 
              total_timesteps: int = 100000,
              eval_freq: int = 10000,
              save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Train PPO agent.
        
        Args:
            total_timesteps: Total training timesteps
            eval_freq: Evaluation frequency
            save_path: Path to save trained policy
            
        Returns:
            Training results dictionary
        """
        logger.info(f"Training PPO for {total_timesteps} timesteps")
        
        # Create environment
        env = make_vec_env(lambda: Monitor(self.create_env()), n_envs=4)
        
        # Create evaluation environment
        eval_env = Monitor(self.create_env())
        
        # PPO model
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log="./ppo_tensorboard/",
            verbose=1
        )
        
        # Callbacks
        callbacks = []
        
        if eval_freq > 0:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path="./models/best_ppo_model",
                log_path="./models/ppo_logs",
                eval_freq=eval_freq,
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)
        
        # Stop training when reward threshold is reached
        reward_threshold = StopTrainingOnRewardThreshold(
            reward_threshold=50.0,
            verbose=1
        )
        callbacks.append(reward_threshold)
        
        # Train model
        start_time = time.time()
        
        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks
            )
            
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Save final model
            if save_path:
                model.save(save_path)
                logger.info(f"Model saved to: {save_path}")
            
            # Evaluate final performance
            logger.info("Evaluating final performance...")
            final_performance = self.evaluate_model(model, eval_env, n_episodes=10)
            
            return {
                'model': model,
                'training_time': training_time,
                'total_timesteps': total_timesteps,
                'final_performance': final_performance,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                'model': None,
                'error': str(e),
                'success': False
            }
    
    def evaluate_model(self, 
                      model: PPO, 
                      env: gym.Env, 
                      n_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate trained model.
        
        Args:
            model: Trained PPO model
            env: Evaluation environment
            n_episodes: Number of evaluation episodes
            
        Returns:
            Performance metrics
        """
        logger.info(f"Evaluating model over {n_episodes} episodes")
        
        total_rewards = []
        total_slippages = []
        total_trades = []
        
        for episode in range(n_episodes):
            obs = env.reset()
            episode_reward = 0.0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
            
            total_rewards.append(episode_reward)
            total_slippages.append(info.get('total_slippage', 0))
            total_trades.append(info.get('total_trades', 0))
        
        metrics = {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'mean_slippage': np.mean(total_slippages),
            'mean_trades': np.mean(total_trades),
            'success_rate': np.mean([r > 0 for r in total_rewards])
        }
        
        logger.info(f"Evaluation Results:")
        logger.info(f"  Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        logger.info(f"  Mean Slippage: {metrics['mean_slippage']:.4f}")
        logger.info(f"  Mean Trades: {metrics['mean_trades']:.1f}")
        logger.info(f"  Success Rate: {metrics['success_rate']:.2%}")
        
        return metrics


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train PPO for execution optimization')
    parser.add_argument('--timesteps', type=int, default=100000, help='Total training timesteps')
    parser.add_argument('--eval-freq', type=int, default=10000, help='Evaluation frequency')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--output-dir', default='policies', help='Output directory')
    parser.add_argument('--skip-if-no-deps', action='store_true', help='Skip gracefully if dependencies missing')
    
    args = parser.parse_args()
    
    if not DEPENDENCIES_AVAILABLE:
        if args.skip_if_no_deps:
            logger.warning("PPO training dependencies not available, skipping...")
            return 0
        else:
            logger.error("Missing required dependencies for PPO training")
            logger.error("Install with: pip install stable-baselines3 torch gymnasium")
            return 1
    
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Initialize trainer
        trainer = PPOTrainer(args.config)
        
        # Train PPO agent
        logger.info("=" * 50)
        logger.info("TRAINING PPO AGENT")
        logger.info("=" * 50)
        
        save_path = output_dir / "ppo_policy"
        
        results = trainer.train(
            total_timesteps=args.timesteps,
            eval_freq=args.eval_freq,
            save_path=str(save_path)
        )
        
        if results['success']:
            # Print summary
            logger.info("=" * 50)
            logger.info("TRAINING SUMMARY")
            logger.info("=" * 50)
            logger.info(f"Training time: {results['training_time']:.2f} seconds")
            logger.info(f"Total timesteps: {results['total_timesteps']:,}")
            
            perf = results['final_performance']
            logger.info(f"Final performance:")
            logger.info(f"  Mean reward: {perf['mean_reward']:.2f}")
            logger.info(f"  Success rate: {perf['success_rate']:.2%}")
            logger.info(f"  Mean slippage: {perf['mean_slippage']:.4f}")
            
            logger.info(f"✓ PPO policy saved to: {save_path}.zip")
            logger.info("PPO training completed successfully!")
            return 0
        else:
            logger.error("PPO training failed")
            return 1
            
    except Exception as e:
        logger.error(f"PPO training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())