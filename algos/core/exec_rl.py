"""
Reinforcement Learning Execution Engine for Quantitative Trading.
Uses Stable-Baselines3 PPO for intelligent order execution with fallback strategies.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple, List, Union
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
import os
from pathlib import Path


class TradingExecutionEnv(gym.Env):
    """
    Custom Gym environment for training execution policies.
    
    State space includes:
    - Market features (spread, volatility, volume)
    - Position information (current size, target size)
    - Order book features (if available)
    - Time features (time of day, day of week)
    
    Action space:
    - Size delta: how much to change position (-1 to 1, scaled by max_size)
    - Order type: market (0) or limit (1)
    - Limit offset: how far from mid price (0 to 1, scaled by ATR)
    """
    
    def __init__(self, 
                 max_position_size: float = 1000.0,
                 transaction_cost: float = 0.0005,
                 impact_cost: float = 0.001,
                 max_steps: int = 1000):
        """Initialize trading execution environment."""
        super().__init__()
        
        self.max_position_size = max_position_size
        self.transaction_cost = transaction_cost
        self.impact_cost = impact_cost
        self.max_steps = max_steps
        
        # State space: [spread, volatility, volume_ratio, current_pos, target_pos, 
        #                time_of_day, inventory_penalty, pnl, steps_remaining]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(9,), 
            dtype=np.float32
        )
        
        # Action space: [size_delta, order_type, limit_offset]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Environment state
        self.current_step = 0
        self.current_position = 0.0
        self.target_position = 0.0
        self.cash = 0.0
        self.total_pnl = 0.0
        self.total_cost = 0.0
        
        # Market data (would be provided externally)
        self.market_data: Optional[pd.DataFrame] = None
        self.current_price = 100.0
        self.current_spread = 0.01
        self.current_volatility = 0.2
        self.current_volume_ratio = 1.0
        
        logger.info("Trading execution environment initialized")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.current_position = 0.0
        self.target_position = np.random.uniform(-0.5, 0.5) * self.max_position_size
        self.cash = 0.0
        self.total_pnl = 0.0
        self.total_cost = 0.0
        
        # Initialize market conditions
        self.current_price = 100.0 + np.random.normal(0, 5)
        self.current_spread = np.random.uniform(0.005, 0.02)
        self.current_volatility = np.random.uniform(0.1, 0.3)
        self.current_volume_ratio = np.random.uniform(0.5, 2.0)
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        self.current_step += 1
        
        # Parse action
        size_delta_pct, order_type, limit_offset = action
        
        # Calculate actual size delta
        max_delta = min(
            abs(self.target_position - self.current_position),
            self.max_position_size * 0.1  # Max 10% of max size per step
        )
        size_delta = size_delta_pct * max_delta
        
        # Apply size delta
        if abs(self.target_position - self.current_position) > abs(size_delta):
            if self.target_position > self.current_position:
                size_delta = abs(size_delta)
            else:
                size_delta = -abs(size_delta)
        else:
            size_delta = self.target_position - self.current_position
        
        # Calculate execution cost based on order type and market conditions
        if order_type < 0.5:  # Market order
            execution_cost = self.transaction_cost + self.impact_cost * abs(size_delta) / 100
            fill_probability = 1.0
        else:  # Limit order
            execution_cost = self.transaction_cost
            # Fill probability decreases with aggressive limit offset
            fill_probability = max(0.1, 1.0 - limit_offset)
        
        # Simulate order execution
        if np.random.random() < fill_probability:
            executed_size = size_delta
            self.current_position += executed_size
            
            # Update cash and costs
            trade_value = executed_size * self.current_price
            self.cash -= trade_value
            cost = abs(trade_value) * execution_cost
            self.total_cost += cost
            self.cash -= cost
        else:
            executed_size = 0.0
        
        # Update market conditions (simple random walk)
        price_change = np.random.normal(0, self.current_volatility / np.sqrt(252))
        self.current_price += price_change
        self.current_spread = max(0.001, self.current_spread + np.random.normal(0, 0.001))
        self.current_volume_ratio = max(0.1, self.current_volume_ratio + np.random.normal(0, 0.1))
        
        # Calculate PnL
        position_pnl = self.current_position * price_change
        self.total_pnl += position_pnl
        
        # Calculate reward
        reward = self._calculate_reward(executed_size, execution_cost)
        
        # Check if episode is done
        terminated = abs(self.target_position - self.current_position) < 1e-6
        truncated = self.current_step >= self.max_steps
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation."""
        position_pct = self.current_position / self.max_position_size
        target_pct = self.target_position / self.max_position_size
        time_pct = self.current_step / self.max_steps
        inventory_penalty = abs(self.current_position - self.target_position) / self.max_position_size
        pnl_pct = self.total_pnl / (self.max_position_size * self.current_price) if self.current_price > 0 else 0
        
        observation = np.array([
            self.current_spread,
            self.current_volatility, 
            self.current_volume_ratio,
            position_pct,
            target_pct,
            time_pct,
            inventory_penalty,
            pnl_pct,
            (self.max_steps - self.current_step) / self.max_steps
        ], dtype=np.float32)
        
        return observation
    
    def _calculate_reward(self, executed_size: float, execution_cost: float) -> float:
        """Calculate step reward."""
        # Reward for reducing inventory imbalance
        old_imbalance = abs(self.target_position - (self.current_position - executed_size))
        new_imbalance = abs(self.target_position - self.current_position)
        inventory_improvement = old_imbalance - new_imbalance
        
        # Penalty for execution costs
        cost_penalty = abs(executed_size) * self.current_price * execution_cost
        
        # Time penalty to encourage faster execution
        time_penalty = 0.01 * (self.current_step / self.max_steps)
        
        # Reward components
        reward = (
            inventory_improvement * 10.0 -  # Reward for reducing imbalance
            cost_penalty * 100.0 -         # Penalty for costs
            time_penalty                   # Time penalty
        )
        
        return reward
    
    def _get_info(self) -> Dict:
        """Get environment info."""
        return {
            'current_position': self.current_position,
            'target_position': self.target_position,
            'total_pnl': self.total_pnl,
            'total_cost': self.total_cost,
            'current_price': self.current_price,
            'imbalance': abs(self.target_position - self.current_position)
        }


class ExecutionRL:
    """Reinforcement Learning execution engine with PPO policy."""
    
    def __init__(self, 
                 config: Dict[str, Any],
                 model_path: Optional[str] = None):
        """Initialize RL execution engine."""
        self.config = config
        self.model_path = model_path or config.get('models', {}).get('ppo_policy_path', 'policies/ppo_policy.zip')
        
        # Fallback parameters
        self.default_limit_offsets = config.get('execution', {}).get('default_limit_offsets', [0.1, 0.2, 0.3])
        
        # Vector environment flag for training
        self.use_vector_env = config.get('execution', {}).get('use_vector_env', True)
        
        # Load or create policy
        self.policy: Optional[PPO] = None
        self.env: Optional[TradingExecutionEnv] = None
        self._load_policy()
        
        # Execution state
        self.pending_orders: Dict[str, Dict] = {}
        self.execution_history: List[Dict] = []
        
        logger.info(f"Execution RL initialized with policy: {self.model_path}, vector_env: {self.use_vector_env}")
    
    def _load_policy(self):
        """Load trained PPO policy or use fallback."""
        if os.path.exists(self.model_path):
            try:
                self.policy = PPO.load(self.model_path)
                logger.info(f"Loaded PPO policy from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load PPO policy: {e}. Using fallback strategy.")
                self.policy = None
        else:
            logger.info(f"PPO policy not found at {self.model_path}. Using fallback strategy.")
            self.policy = None
    
    def get_execution_decision(self, 
                             symbol: str,
                             target_position: float,
                             current_position: float,
                             market_features: Dict[str, float],
                             atr: float) -> Dict[str, Any]:
        """
        Get execution decision from RL policy or fallback maker ladder.
        
        Args:
            symbol: Trading symbol
            target_position: Target position size
            current_position: Current position size  
            market_features: Market features (spread, vol, etc.)
            atr: Average True Range for sizing
            
        Returns:
            Dictionary with execution parameters:
            - size_delta: Change in position size
            - order_type: 'market' or 'limit'
            - limit_offset: Offset from mid price (for limit orders)
        """
        size_delta = target_position - current_position
        
        if abs(size_delta) < 1e-6:
            return {
                'size_delta': 0.0,
                'order_type': 'market',
                'limit_offset': 0.0
            }
        
        # Guards: if no policy or policy error or out-of-range observation → return fallback
        if self.policy is None:
            logger.debug("No RL policy available, using fallback maker ladder")
            return self._get_fallback_maker_ladder_decision(size_delta, market_features, atr)
        
        try:
            # Attempt RL decision with additional guards
            rl_decision = self._get_rl_decision(size_delta, market_features, atr)
            
            # Validate RL decision bounds
            max_size_delta = self.config.get('execution', {}).get('max_size_delta', 10000.0)
            max_limit_offset = self.config.get('execution', {}).get('max_limit_offset_atr', 0.5) * atr
            
            # Clamp limit_offset and size_delta to config bounds
            rl_decision['limit_offset'] = np.clip(rl_decision['limit_offset'], 
                                                 -max_limit_offset, max_limit_offset)
            rl_decision['size_delta'] = np.clip(rl_decision['size_delta'], 
                                               -max_size_delta, max_size_delta)
            
            return rl_decision
            
        except Exception as e:
            logger.warning(f"RL policy error for {symbol}: {e}. Using fallback maker ladder.")
            return self._get_fallback_maker_ladder_decision(size_delta, market_features, atr)
    
    def _get_rl_decision(self, 
                        size_delta: float,
                        market_features: Dict[str, float], 
                        atr: float) -> Dict[str, Any]:
        """Get execution decision from RL policy."""
        try:
            # Prepare observation
            spread = market_features.get('spread', 0.01)
            volatility = market_features.get('volatility', 0.2)
            volume_ratio = market_features.get('volume_ratio', 1.0)
            
            # Create dummy observation (in practice, this would be from the environment)
            observation = np.array([
                spread,
                volatility,
                volume_ratio,
                0.0,  # current_position (normalized)
                size_delta / 1000.0,  # target_position (normalized)
                0.5,  # time_of_day
                abs(size_delta) / 1000.0,  # inventory_penalty
                0.0,  # pnl
                1.0   # steps_remaining
            ], dtype=np.float32)
            
            # Get action from policy
            action, _ = self.policy.predict(observation, deterministic=True)
            
            # Parse action
            size_delta_pct, order_type_raw, limit_offset = action
            
            # Convert to execution parameters
            actual_size_delta = size_delta * abs(size_delta_pct)
            order_type = 'limit' if order_type_raw > 0.5 else 'market'
            limit_offset_atr = limit_offset * atr
            
            logger.debug(f"RL execution decision: size_delta={actual_size_delta:.2f}, "
                        f"type={order_type}, offset={limit_offset_atr:.4f}")
            
            return {
                'size_delta': actual_size_delta,
                'order_type': order_type,
                'limit_offset': limit_offset_atr
            }
            
        except Exception as e:
            logger.warning(f"RL policy prediction failed: {e}. Using fallback.")
            return self._get_fallback_decision(size_delta, market_features, atr)
    
    def _get_fallback_maker_ladder_decision(self, 
                                           size_delta: float,
                                           market_features: Dict[str, float],
                                           atr: float) -> Dict[str, Any]:
        """Fallback maker ladder execution strategy."""
        spread = market_features.get('spread', 0.01)
        volatility = market_features.get('volatility', 0.2)
        
        # Use market orders for small sizes or high volatility
        if abs(size_delta) < 100 or volatility > 0.3:
            return {
                'size_delta': size_delta,
                'order_type': 'market',
                'limit_offset': 0.0
            }
        
        # Use first level of maker ladder for limit orders
        direction = 1 if size_delta > 0 else -1
        limit_offset = self.default_limit_offsets[0] * atr * direction
        
        return {
            'size_delta': size_delta,
            'order_type': 'limit',
            'limit_offset': limit_offset
        }

    def _get_fallback_decision(self, 
                              size_delta: float,
                              market_features: Dict[str, float],
                              atr: float) -> Dict[str, Any]:
        """Fallback execution strategy using maker ladder with proper spacing."""
        spread = market_features.get('spread', 0.01)
        volatility = market_features.get('volatility', 0.2)
        current_time = market_features.get('timestamp', 0)
        
        # Use market orders for small sizes or high volatility
        if abs(size_delta) < 100 or volatility > 0.3:
            return {
                'size_delta': size_delta,
                'order_type': 'market',
                'limit_offset': 0.0,
                'ladder_orders': []
            }
        
        # Create maker ladder orders
        ladder_orders = []
        total_size = abs(size_delta)
        direction = 1 if size_delta > 0 else -1
        
        # Split size across ladder levels
        ladder_sizes = self._split_size_for_ladder(total_size, len(self.default_limit_offsets))
        
        for i, offset_mult in enumerate(self.default_limit_offsets):
            if i >= len(ladder_sizes):
                break
                
            limit_offset = offset_mult * atr * direction  # direction for buy/sell
            ladder_size = ladder_sizes[i] * direction
            
            # Create idempotent order ID
            order_id = f"ladder_{hash(f'{current_time}_{i}_{ladder_size}_{limit_offset}') % 1000000}"
            
            ladder_orders.append({
                'size': ladder_size,
                'limit_offset': limit_offset,
                'order_id': order_id,
                'level': i
            })
        
        # Primary order is the first ladder level
        primary_order = ladder_orders[0] if ladder_orders else {
            'size': size_delta * 0.7,
            'limit_offset': self.default_limit_offsets[0] * atr * direction,
            'order_id': f"primary_{hash(f'{current_time}_{size_delta}') % 1000000}",
            'level': 0
        }
        
        logger.debug(f"Fallback maker ladder: {len(ladder_orders)} orders, "
                    f"primary_size={primary_order['size']:.2f}, "
                    f"offset={primary_order['limit_offset']:.4f}")
        
        return {
            'size_delta': primary_order['size'],
            'order_type': 'limit',
            'limit_offset': primary_order['limit_offset'],
            'ladder_orders': ladder_orders,
            'order_id': primary_order['order_id']
        }
    
    def _split_size_for_ladder(self, total_size: float, num_levels: int) -> List[float]:
        """Split total size across ladder levels with decreasing amounts."""
        if num_levels <= 0:
            return [total_size]
        
        # Geometric distribution: larger orders at better prices
        weights = [1.0 / (2 ** i) for i in range(num_levels)]
        total_weight = sum(weights)
        
        # Normalize weights and apply to total size
        sizes = [(w / total_weight) * total_size for w in weights]
        
        # Ensure minimum order size
        min_size = max(1.0, total_size * 0.01)  # At least 1% of total
        sizes = [max(size, min_size) for size in sizes]
        
        return sizes
    
    def train_policy(self, 
                    training_data: Optional[pd.DataFrame] = None,
                    total_timesteps: int = 100000,
                    save_path: Optional[str] = None) -> None:
        """
        Train PPO policy for execution.
        
        Args:
            training_data: Historical market data for training
            total_timesteps: Number of training timesteps
            save_path: Path to save trained model
        """
        # Create environment
        env = TradingExecutionEnv()
        check_env(env)
        
        # Create PPO model
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
            verbose=1,
            tensorboard_log="./logs/ppo_execution/"
        )
        
        # Training callback
        callback = ExecutionTrainingCallback()
        
        logger.info(f"Starting PPO training for {total_timesteps} timesteps")
        
        # Train model
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        # Save model
        save_path = save_path or self.model_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        
        logger.info(f"PPO training complete. Model saved to {save_path}")
        
        # Update policy
        self.policy = model
    
    def log_execution(self, 
                     symbol: str,
                     decision: Dict[str, Any],
                     executed_size: float,
                     execution_price: float,
                     timestamp: pd.Timestamp):
        """Log execution for analysis."""
        execution_record = {
            'timestamp': timestamp,
            'symbol': symbol,
            'decision': decision.copy(),
            'executed_size': executed_size,
            'execution_price': execution_price,
            'slippage': 0.0  # Would calculate vs. expected price
        }
        
        self.execution_history.append(execution_record)
        
        # Keep last 10000 records
        if len(self.execution_history) > 10000:
            self.execution_history = self.execution_history[-10000:]
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution performance statistics."""
        if not self.execution_history:
            return {}
            
        df = pd.DataFrame(self.execution_history)
        
        stats = {
            'total_executions': len(df),
            'avg_executed_size': df['executed_size'].mean(),
            'fill_rate': (df['executed_size'] != 0).mean(),
            'market_order_pct': (df['decision'].apply(lambda x: x['order_type']) == 'market').mean(),
            'limit_order_pct': (df['decision'].apply(lambda x: x['order_type']) == 'limit').mean(),
        }
        
        return stats


class ExecutionTrainingCallback(BaseCallback):
    """Custom callback for PPO execution training."""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: List[float] = []
        
    def _on_step(self) -> bool:
        """Called at each training step."""
        # Log training metrics
        if len(self.locals.get('episode_rewards', [])) > 0:
            self.episode_rewards.extend(self.locals['episode_rewards'])
            
        # Log every 1000 steps
        if self.n_calls % 1000 == 0:
            if len(self.episode_rewards) > 0:
                avg_reward = np.mean(self.episode_rewards[-100:])  # Last 100 episodes
                logger.info(f"Step {self.n_calls}: Avg reward = {avg_reward:.4f}")
                
        return True


def create_execution_gym_env(config: Dict[str, Any]) -> TradingExecutionEnv:
    """Factory function to create execution environment."""
    env_config = config.get('execution', {})
    
    env = TradingExecutionEnv(
        max_position_size=env_config.get('max_position_size', 1000.0),
        transaction_cost=env_config.get('transaction_cost', 0.0005),
        impact_cost=env_config.get('impact_cost', 0.001),
        max_steps=env_config.get('max_steps', 1000)
    )
    
    return env