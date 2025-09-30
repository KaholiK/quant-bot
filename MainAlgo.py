"""
Main QuantConnect LEAN Algorithm for Quant Trading Bot.
Orchestrates all strategies, feature generation, model predictions, and execution.
"""

from AlgorithmImports import *
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import yaml
import pickle
import os
from datetime import datetime, timedelta

# Import our modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from algos.core.feature_pipe import FeaturePipeline
from algos.core.labels import TripleBarrierLabeler
from algos.core.risk import RiskManager  
from algos.core.portfolio import Portfolio, HRPOptimizer, VolatilityTargeting
from algos.core.exec_rl import ExecutionRL
from algos.core.strategy_manager import StrategyManager
from algos.core.config_loader import load_config, get_legacy_dict
from algos.core.alerts import send_startup_alert
from algos.core.runtime_state import initialize_runtime_state, get_runtime_state
from algos.core.polling import RuntimeStatePoller
from storage.trades import get_trade_storage
from algos.strategies.scalper_sigma import ScalperSigmaStrategy
from algos.strategies.trend_breakout import TrendBreakoutStrategy
from algos.strategies.bull_mode import BullModeStrategy
from algos.strategies.market_neutral import MarketNeutralStrategy
from algos.strategies.gamma_reversal import GammaReversalStrategy


class MainAlgo(QCAlgorithm):
    """Main QuantConnect algorithm orchestrating all trading strategies."""
    
    def Initialize(self):
        """Initialize the algorithm."""
        
        # Load configuration using new config loader
        self.config_obj = self._load_config()
        self.config = get_legacy_dict(self.config_obj)  # Legacy format for backward compatibility
        
        # Detect live vs backtest mode
        self.is_live_mode = self.LiveMode
        
        # Set dates and cash based on mode
        if not self.is_live_mode:
            # Backtest mode
            self.SetStartDate(2020, 1, 1)
            self.SetEndDate(2024, 1, 1)
            self.SetCash(100000)
            
            # Enable realistic fees and slippage for backtest
            self.SetBrokerageModel(AlphaStreamsBrokerageModel())
            
            # Add realistic transaction costs
            self.Settings.FreePortfolioValuePercentage = 0.05  # 5% buffer
        else:
            # Live mode - no end date
            self.SetCash(100000)
            self.Log("Running in LIVE mode - no end date set")
            
            # Set brokerage based on config
            self._set_live_brokerage()
        
        # Set warmup period (needed for indicators)
        self.SetWarmUp(timedelta(days=60))
        
        # Initialize core components
        self.feature_pipeline = FeaturePipeline(self.config)
        self.risk_manager = RiskManager(self.config)
        self.portfolio_manager = Portfolio(self.GetPortfolio().TotalPortfolioValue)
        self.execution_engine = ExecutionRL(self.config)
        self.strategy_manager = StrategyManager(self.config)
        
        # Initialize runtime state for UI control
        self.runtime_state = initialize_runtime_state(self.config_obj)
        
        # Initialize trade storage
        self.trade_storage = get_trade_storage()
        
        # Initialize polling for QC integration (if enabled)
        self.poller = None
        if self.config_obj.trading.ui.admin_api.enabled:
            try:
                admin_api_config = self.config_obj.trading.ui.admin_api
                self.poller = RuntimeStatePoller(
                    base_url=admin_api_config.host,
                    port=admin_api_config.port,
                    timeout=3,
                    retry_attempts=2
                )
                self.last_poll_time = datetime.now()
                self.polling_interval = timedelta(seconds=admin_api_config.polling_interval_sec)
                self.Log(f"QC polling initialized for {admin_api_config.host}:{admin_api_config.port}")
            except Exception as e:
                self.Log(f"Failed to initialize QC polling: {e}")
                self.poller = None
        
        # Initialize strategies
        self.strategies = {
            'scalper_sigma': ScalperSigmaStrategy(self.config),
            'trend_breakout': TrendBreakoutStrategy(self.config),
            'bull_mode': BullModeStrategy(self.config),
            'market_neutral': MarketNeutralStrategy(self.config),
            'gamma_reversal': GammaReversalStrategy(self.config)
        }
        
        # Load models
        self.models = self._load_models()
        
        # Set up universe
        self.symbols = {}
        self.consolidators = {}
        self.features_data = {}
        self.price_data = {}
        self.last_order_time = {}  # Track order timing for rate limiting
        
        self._setup_universe()
        
        # Performance tracking
        self.trade_count = 0
        self.last_portfolio_value = self.GetPortfolio().TotalPortfolioValue
        self.performance_metrics = {}
        
        # Schedule periodic tasks  
        self.Schedule.On(
            self.DateRules.EveryDay("SPY"),
            self.TimeRules.At(9, 30),
            self.DailyRebalance
        )
        
        # Schedule runtime state polling (if enabled)
        if self.poller:
            self.Schedule.On(
                self.DateRules.EveryDay(),
                self.TimeRules.Every(timedelta(minutes=2)),  # Poll every 2 minutes
                self.PollRuntimeState
            )
        
        # Schedule equity recording
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.Every(timedelta(hours=1)),  # Record equity hourly
            self.RecordEquitySnapshot
        )
        
        self.Log("MainAlgo initialized successfully")
        
        # Send startup alert to Discord
        try:
            send_startup_alert("1.0.0")
        except Exception as e:
            self.Log(f"Failed to send startup alert: {e}")
    
    def _load_config(self):
        """Load configuration using new config loader."""
        try:
            config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
            config_obj = load_config(config_path)
            self.Log("Configuration loaded and validated successfully")
            return config_obj
        except Exception as e:
            self.Log(f"Failed to load config: {e}")
            # Return default configuration
            return load_config("nonexistent.yaml")  # Returns defaults
    
    def _set_live_brokerage(self):
        """Set appropriate brokerage model for live trading."""
        try:
            # Set brokerage based on assets being traded
            universe_config = self.config_obj.trading.universe
            
            # For equities, use Alpaca (if available) or default
            if universe_config.equities:
                try:
                    # Try to use Alpaca for equities
                    self.SetBrokerageModel(AlpacaBrokerageModel())
                    self.Log("Using Alpaca brokerage for equities")
                except:
                    # Fallback to default
                    self.Log("Alpaca not available, using default brokerage")
            
            # For crypto, try Binance/Coinbase
            if universe_config.crypto:
                try:
                    # Note: This depends on QuantConnect's crypto broker support
                    self.Log("Crypto trading configured - using default crypto brokerage")
                except:
                    self.Log("Crypto brokerage setup failed")
                    
        except Exception as e:
            self.Log(f"Brokerage setup failed: {e}")
    
    def OnOrderEvent(self, orderEvent):
        """Handle order events (fills, rejections, etc.)."""
        try:
            order = self.Transactions.GetOrderById(orderEvent.OrderId)
            symbol = order.Symbol.Value
            
            if orderEvent.Status == OrderStatus.Filled:
                # Handle filled orders
                self._handle_order_fill(orderEvent, order)
                
            elif orderEvent.Status == OrderStatus.PartiallyFilled:
                # Handle partial fills
                self._handle_partial_fill(orderEvent, order)
                
            elif orderEvent.Status in [OrderStatus.Invalid, OrderStatus.Canceled]:
                # Handle rejections/cancellations
                self._handle_order_rejection(orderEvent, order)
                
        except Exception as e:
            self.Log(f"Error handling order event: {e}")
    
    def _handle_order_fill(self, orderEvent, order):
        """Handle completed order fills."""
        symbol = order.Symbol.Value
        fill_price = orderEvent.FillPrice
        fill_quantity = orderEvent.FillQuantity
        
        # Update position tracking in strategies
        for strategy in self.strategies.values():
            if hasattr(strategy, 'update_position'):
                strategy.update_position(symbol, fill_quantity, fill_price)
        
        # Update risk manager
        self.risk_manager.update_position(symbol, fill_quantity, fill_price)
        
        # Send Discord alert
        try:
            from algos.core.alerts import send_fill_alert
            side = "BUY" if fill_quantity > 0 else "SELL"
            send_fill_alert(symbol, side, abs(fill_quantity), fill_price, "MainAlgo")
        except Exception as e:
            self.Log(f"Failed to send fill alert: {e}")
        
        self.Log(f"ORDER FILLED: {symbol} {fill_quantity} @ {fill_price}")
        
        # Record trade in storage
        try:
            trade_data = {
                "time": self.Time.isoformat(),
                "symbol": symbol,
                "side": "BUY" if fill_quantity > 0 else "SELL",
                "qty": abs(fill_quantity),
                "avg_price": fill_price,
                "fees": 0.0,  # TODO: Calculate actual fees
                "slippage_bps": 0.0,  # TODO: Calculate slippage
                "pnl": 0.0,  # TODO: Calculate unrealized P&L
                "meta": {
                    "order_id": order.Id,
                    "algorithm": "MainAlgo",
                    "regime": getattr(self.strategy_manager, 'current_regime', 'unknown')
                }
            }
            self.trade_storage.record_fill(trade_data)
        except Exception as e:
            self.Log(f"Failed to record trade: {e}")
    
    def _handle_partial_fill(self, orderEvent, order):
        """Handle partial order fills."""
        symbol = order.Symbol.Value
        fill_price = orderEvent.FillPrice
        fill_quantity = orderEvent.FillQuantity
        
        self.Log(f"PARTIAL FILL: {symbol} {fill_quantity} @ {fill_price}")
        
        # Update positions with partial quantity
        self._handle_order_fill(orderEvent, order)
    
    def _handle_order_rejection(self, orderEvent, order):
        """Handle order rejections and cancellations."""
        symbol = order.Symbol.Value
        
        self.Log(f"ORDER REJECTED/CANCELED: {symbol} - {orderEvent.Message}")
        
        # Send risk alert for rejections
        try:
            from algos.core.alerts import send_risk_alert
            send_risk_alert("ORDER_REJECTION", f"Order rejected for {symbol}: {orderEvent.Message}")
        except Exception as e:
            self.Log(f"Failed to send rejection alert: {e}")
    
    def _load_models(self) -> Dict[str, Any]:
        """Load trained ML models."""
        models = {}
        
        try:
            # Use paths from config
            classifier_path = os.path.join(os.path.dirname(__file__), self.config_obj.trading.models.classifier_path)
            if os.path.exists(classifier_path):
                with open(classifier_path, 'rb') as f:
                    models['classifier'] = pickle.load(f)
                self.Log("XGBoost classifier loaded")
            
            # Load meta-model
            meta_path = os.path.join(os.path.dirname(__file__), self.config_obj.trading.models.meta_model_path)
            if os.path.exists(meta_path):
                with open(meta_path, 'rb') as f:
                    models['meta_model'] = pickle.load(f)
                self.Log("Meta-model loaded")
                
        except Exception as e:
            self.Log(f"Warning: Failed to load models: {e}")
        
        return models
    
    def _setup_universe(self):
        """Set up trading universe and data consolidators."""
        
        # Add SPY as market benchmark
        spy = self.AddEquity("SPY", Resolution.Minute)
        self.symbols['SPY'] = spy.Symbol
        
        # Add major equity ETFs (representing S&P-100 components)
        equity_symbols = ["QQQ", "XLF", "XLK", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLRE"]
        
        for symbol_str in equity_symbols:
            try:
                equity = self.AddEquity(symbol_str, Resolution.Minute)
                self.symbols[symbol_str] = equity.Symbol
                
                # 30-minute consolidator for equities
                consolidator = TradeBarConsolidator(timedelta(minutes=30))
                consolidator.DataConsolidated += self.OnEquityData
                self.SubscriptionManager.AddConsolidator(equity.Symbol, consolidator)
                self.consolidators[symbol_str] = consolidator
                
            except Exception as e:
                self.Log(f"Failed to add equity {symbol_str}: {e}")
        
        # Add crypto pairs
        crypto_symbols = self.config['universe']['crypto']  # Updated for new config format
        for symbol_str in crypto_symbols:
            try:
                crypto = self.AddCrypto(symbol_str, Resolution.Minute)
                self.symbols[symbol_str] = crypto.Symbol
                
                # 15-minute consolidator for crypto
                consolidator_15m = TradeBarConsolidator(timedelta(minutes=15))
                consolidator_15m.DataConsolidated += self.OnCryptoData
                self.SubscriptionManager.AddConsolidator(crypto.Symbol, consolidator_15m)
                
                # 1-minute data for gamma reversal strategy
                consolidator_1m = TradeBarConsolidator(timedelta(minutes=1))
                consolidator_1m.DataConsolidated += self.OnCrypto1mData
                self.SubscriptionManager.AddConsolidator(crypto.Symbol, consolidator_1m)
                
                self.consolidators[f"{symbol_str}_15m"] = consolidator_15m
                self.consolidators[f"{symbol_str}_1m"] = consolidator_1m
                
            except Exception as e:
                self.Log(f"Failed to add crypto {symbol_str}: {e}")
        
        self.Log(f"Universe setup complete: {len(self.symbols)} symbols")
    
    def OnEquityData(self, sender, bar):
        """Handle consolidated equity data (30-minute bars)."""
        symbol_str = str(bar.Symbol).replace(' ', '')
        self._process_bar_data(symbol_str, bar, '30m')
    
    def OnCryptoData(self, sender, bar):
        """Handle consolidated crypto data (15-minute bars)."""
        symbol_str = str(bar.Symbol).replace(' ', '')
        self._process_bar_data(symbol_str, bar, '15m')
    
    def OnCrypto1mData(self, sender, bar):
        """Handle 1-minute crypto data for gamma reversal."""
        symbol_str = str(bar.Symbol).replace(' ', '')
        self._process_bar_data(symbol_str, bar, '1m')
    
    def _process_bar_data(self, symbol_str: str, bar, timeframe: str):
        """Process incoming bar data and generate signals."""
        
        # Store price data
        if symbol_str not in self.price_data:
            self.price_data[symbol_str] = {}
        
        if timeframe not in self.price_data[symbol_str]:
            self.price_data[symbol_str][timeframe] = []
        
        # Add new bar data
        bar_data = {
            'time': bar.Time,
            'open': float(bar.Open),
            'high': float(bar.High),
            'low': float(bar.Low),
            'close': float(bar.Close),
            'volume': float(bar.Volume)
        }
        
        self.price_data[symbol_str][timeframe].append(bar_data)
        
        # Keep last 1000 bars
        if len(self.price_data[symbol_str][timeframe]) > 1000:
            self.price_data[symbol_str][timeframe] = self.price_data[symbol_str][timeframe][-1000:]
        
        # Generate features and signals for primary timeframes
        if (timeframe == '30m' and symbol_str in ['SPY', 'QQQ', 'XLF', 'XLK']) or \
           (timeframe == '15m' and symbol_str in ['BTCUSD', 'ETHUSD']):
            self._generate_signals(symbol_str, timeframe)
    
    def _generate_signals(self, symbol_str: str, timeframe: str):
        """Generate trading signals for a symbol."""
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(self.price_data[symbol_str][timeframe])
            if len(df) < 50:  # Need minimum data
                return
            
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time')
            
            # Generate features
            features = self.feature_pipeline.build_features(df)
            if len(features) == 0:
                return
            
            # Get latest feature row
            latest_features = features.iloc[-1]
            
            # ML Prediction
            prediction_prob = self._get_ml_prediction(latest_features)
            
            # Meta-model filtering
            if 'meta_model' in self.models and prediction_prob is not None:
                meta_features = self._prepare_meta_features(symbol_str, latest_features, prediction_prob)
                should_trade = self.models['meta_model'].predict([meta_features])[0]
                if not should_trade:
                    self.Log(f"Meta-model filtered out trade for {symbol_str}")
                    return
            
            # Strategy signals
            strategy_signals = {}
            market_conditions = self._get_market_conditions(symbol_str, df, features)
            
            # Get signals from each strategy
            for strategy_name, strategy in self.strategies.items():
                if strategy_name == 'scalper_sigma':
                    signal_info = strategy.calculate_signals(symbol_str, df, features)
                elif strategy_name == 'trend_breakout':
                    universe_data = self._get_universe_data_for_strategy(symbol_str)
                    signal_info = strategy.calculate_signals(symbol_str, df, features, universe_data)
                elif strategy_name == 'bull_mode':
                    # Use SPY data for regime detection
                    spy_data = self._get_spy_data()
                    spy_features = self._get_spy_features()
                    signal_info = strategy.detect_bull_regime(spy_data, spy_features)
                elif strategy_name == 'market_neutral':
                    universe_data = self._get_universe_data_for_strategy(symbol_str)
                    signal_info = strategy.calculate_signals(universe_data)
                elif strategy_name == 'gamma_reversal' and symbol_str in ['BTCUSD', 'ETHUSD']:
                    # Get 1m data for gamma reversal
                    df_1m = self._get_1m_data(symbol_str)
                    signal_info = strategy.calculate_signals(symbol_str, df_1m, df, features)
                else:
                    continue
                
                if signal_info.get('signal', 0) != 0:
                    strategy_signals[strategy_name] = signal_info
            
            # Execute trades based on signals
            if strategy_signals:
                self._execute_strategy_signals(symbol_str, strategy_signals, market_conditions)
                
        except Exception as e:
            self.Log(f"Error generating signals for {symbol_str}: {e}")
    
    def _get_ml_prediction(self, features: pd.Series) -> Optional[float]:
        """Get ML model prediction probability."""
        if 'classifier' not in self.models:
            return None
        
        try:
            # Prepare features for prediction
            feature_array = features.values.reshape(1, -1)
            
            # Handle missing values
            feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Get prediction probability
            prob = self.models['classifier'].predict_proba(feature_array)[0, 1]  # Probability of class 1
            
            return float(prob)
            
        except Exception as e:
            self.Log(f"ML prediction failed: {e}")
            return None
    
    def _prepare_meta_features(self, symbol: str, features: pd.Series, prediction_prob: float) -> List[float]:
        """Prepare features for meta-model."""
        meta_features = [
            prediction_prob,
            features.get('atr', 0.0) / features.get('close', 1.0),  # ATR ratio
            features.get('realized_vol', 0.0),
            features.get('rsi', 50.0) / 100.0,  # Normalized RSI
            len(self.GetPortfolio().Keys)  # Position count
        ]
        return meta_features
    
    def _get_market_conditions(self, symbol: str, price_data: pd.DataFrame, features: pd.DataFrame) -> Dict[str, Any]:
        """Get current market conditions."""
        latest_features = features.iloc[-1]
        
        return {
            'volatility': latest_features.get('realized_vol', 0.0),
            'spread': latest_features.get('spread', 0.0),
            'volume_ratio': latest_features.get('volume_ratio', 1.0),
            'atr_pct': latest_features.get('atr', 0.0) / price_data['close'].iloc[-1] if len(price_data) > 0 else 0.0
        }
    
    def _execute_strategy_signals(self, symbol_str: str, strategy_signals: Dict[str, Any], market_conditions: Dict[str, Any]):
        """Execute trades based on strategy signals using StrategyManager."""
        
        # Check runtime state before processing
        if self.runtime_state.trading_paused or self.runtime_state.kill_switch_active:
            self.Debug(f"Trading paused - skipping signals for {symbol_str}")
            return
        
        # Filter signals based on strategy enabled state
        filtered_signals = {}
        for strategy_name, signal_info in strategy_signals.items():
            if self._should_process_strategy(strategy_name):
                filtered_signals[strategy_name] = signal_info
            else:
                self.Debug(f"Strategy {strategy_name} disabled - skipping signal")
        
        if not filtered_signals:
            return
        
        # Get current position
        symbol = self.symbols.get(symbol_str)
        if symbol is None:
            return
        
        current_quantity = self.Portfolio[symbol].Quantity
        current_price = self.Securities[symbol].Price
        
        # Get updated risk parameters from runtime state
        risk_params = self._get_current_risk_params()
        
        # Update risk manager with current parameters if changed
        if risk_params:
            for param, value in risk_params.items():
                if hasattr(self.risk_manager, param):
                    setattr(self.risk_manager, param, value)
        
        # Prepare signals by strategy format for StrategyManager
        signals_by_strategy = {}
        for strategy_name, signal_info in filtered_signals.items():
            if abs(signal_info.get('signal', 0)) > 0:
                signals_by_strategy[strategy_name] = {
                    symbol_str: {
                        'target_weight': signal_info.get('signal', 0) * signal_info.get('confidence', 0.0),
                        'confidence': signal_info.get('confidence', 0.0),
                        'signal': signal_info.get('signal', 0)
                    }
                }
        
        if not signals_by_strategy:
            return
        
        # Prepare meta-features
        meta_features = {
            'volatility': market_conditions.get('volatility', 0.2),
            'volume_ratio': market_conditions.get('volume_ratio', 1.0),
            'spread': market_conditions.get('spread', 0.01),
            'atr_ratio': market_conditions.get('atr_ratio', 1.0),
            f'{symbol_str}_momentum': market_conditions.get('momentum', 0.0),
            f'{symbol_str}_volatility': market_conditions.get('volatility', 0.2)
        }
        
        # Use StrategyManager to aggregate signals and apply meta-gating
        final_signals = self.strategy_manager.aggregate_signals(signals_by_strategy, meta_features)
        
        if symbol_str not in final_signals:
            self.Debug(f"No approved signals for {symbol_str} after meta-gating")
            return
        
        signal_info = final_signals[symbol_str]
        
        # Log decision details
        primary_prob = signal_info.get('confidence', 0.0)
        meta_prob = signal_info.get('meta_prob', 1.0)
        
        self.Log(f"{symbol_str}: Primary prob={primary_prob:.3f}, Meta prob={meta_prob:.3f}, "
                f"Strategy={signal_info.get('strategy', 'unknown')}, Signal ACCEPTED")
        
        # Calculate position size using updated risk parameters
        atr = strategy_signals[list(strategy_signals.keys())[0]].get('atr', 0.01)
        equity = float(self.Portfolio.TotalPortfolioValue)
        
        # Use target weight from strategy manager
        target_weight = signal_info['target_weight']
        
        # Use current risk parameters for position sizing
        per_trade_risk = risk_params.get('per_trade_risk_pct', 0.01)
        
        base_size = self.risk_manager.calculate_position_size(
            symbol_str, current_price, current_price * 0.95, equity, atr
        )
        
        target_size = base_size * target_weight * primary_prob
        size_delta = target_size - current_quantity
        
        if abs(size_delta) < 1:  # Minimum trade size
            return
        
        # Risk management checks with updated parameters
        can_trade, reason = self.risk_manager.check_risk_limits(
            symbol_str, size_delta, current_price, equity
        )
        
        if not can_trade:
            self.Log(f"Risk limits prevent trade for {symbol_str}: {reason}")
            return
        
        # Final runtime state check before execution
        if self.runtime_state.trading_paused or self.runtime_state.kill_switch_active:
            self.Debug("Trading paused during execution - aborting trade")
            return
        
        # Execution decision
        execution_decision = self.execution_engine.get_execution_decision(
            symbol_str, target_size, current_quantity, market_conditions, atr
        )
        
        # Place order
        if execution_decision['order_type'] == 'market':
            order_id = self.MarketOrder(symbol, size_delta)
        else:
            limit_price = current_price + np.sign(size_delta) * execution_decision['limit_offset']
            order_id = self.LimitOrder(symbol, size_delta, limit_price)
        
        if order_id:
            self.last_order_time[symbol_str] = self.Time
            self.trade_count += 1
            
            # Log trade with strategy info
            self.Log(f"TRADE: {signal_info.get('strategy', 'unknown')} -> {symbol_str} "
                    f"weight={target_weight:.2f} conf={primary_prob:.2f} "
                    f"size={size_delta:.0f} price={current_price:.4f}")
            
            # Update tracking
            self.risk_manager.log_trade(symbol_str, 'buy' if size_delta > 0 else 'sell', 
                                       abs(size_delta), current_price, self.Time)
            
            # Update strategy positions
            strategy_name = signal_info.get('strategy', 'unknown')
            if strategy_name in self.strategies and hasattr(self.strategies[strategy_name], 'update_position'):
                self.strategies[strategy_name].update_position(
                    symbol_str, target_size, current_price, self.Time.to_pydatetime()
                )
        else:
            self.Log(f"Failed to place order for {symbol_str}")
    
    def DailyRebalance(self):
        """Daily rebalancing and risk management."""
        
        # Update portfolio value
        portfolio_value = float(self.Portfolio.TotalPortfolioValue)
        self.risk_manager.update_equity_curve(portfolio_value)
        
        # Check for kill switch
        if self.risk_manager.is_kill_switch_active:
            self.Log("Kill switch active - liquidating all positions")
            self.Liquidate()
            return
        
        # Update performance metrics
        daily_return = (portfolio_value - self.last_portfolio_value) / self.last_portfolio_value
        self.last_portfolio_value = portfolio_value
        
        # Log daily stats
        risk_metrics = self.risk_manager.get_risk_metrics()
        self.Log(f"Daily Stats: PnL={portfolio_value:.0f} DD={risk_metrics.get('current_drawdown', 0):.1%} "
                f"Trades={self.trade_count} Positions={len(self.Portfolio.Keys)}")
        
        # Portfolio rebalancing (weekly)
        if self.Time.weekday() == 0:  # Monday
            self._rebalance_portfolio()
    
    def _rebalance_portfolio(self):
        """Weekly portfolio rebalancing using HRP."""
        try:
            # Get recent returns for all positions
            returns_data = {}
            
            for symbol_str, symbol in self.symbols.items():
                if symbol_str in self.price_data and '15m' in self.price_data[symbol_str]:
                    df = pd.DataFrame(self.price_data[symbol_str]['15m'])
                    if len(df) > 100:
                        df['time'] = pd.to_datetime(df['time'])
                        df = df.set_index('time')
                        returns = df['close'].pct_change().dropna()
                        returns_data[symbol_str] = returns.tail(252)  # Last 252 bars
            
            if len(returns_data) < 3:
                return
            
            # Create returns DataFrame
            returns_df = pd.DataFrame(returns_data).fillna(0)
            
            # Calculate HRP weights
            hrp_optimizer = HRPOptimizer()
            target_weights = hrp_optimizer.calculate_hrp_weights(returns_df)
            
            # Apply bull mode adjustments
            target_weights = self.strategies['bull_mode'].adjust_strategy_allocation(
                target_weights.to_dict(), {}
            )
            
            self.Log(f"Weekly rebalance: {len(target_weights)} assets, "
                    f"max weight: {max(target_weights.values()):.1%}")
            
        except Exception as e:
            self.Log(f"Portfolio rebalancing failed: {e}")
    
    def _get_universe_data_for_strategy(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Get universe data for strategy calculations."""
        universe_data = {}
        
        for symbol_str, price_data_dict in self.price_data.items():
            if symbol_str != symbol and '15m' in price_data_dict:
                df = pd.DataFrame(price_data_dict['15m'])
                if len(df) > 50:
                    df['time'] = pd.to_datetime(df['time'])
                    df = df.set_index('time')
                    universe_data[symbol_str] = df
        
        return universe_data
    
    def _get_spy_data(self) -> pd.DataFrame:
        """Get SPY data for regime detection."""
        if 'SPY' in self.price_data and '30m' in self.price_data['SPY']:
            df = pd.DataFrame(self.price_data['SPY']['30m'])
            df['time'] = pd.to_datetime(df['time'])
            return df.set_index('time')
        return pd.DataFrame()
    
    def _get_spy_features(self) -> pd.DataFrame:
        """Get SPY features for regime detection."""
        spy_data = self._get_spy_data()
        if len(spy_data) > 0:
            return self.feature_pipeline.build_features(spy_data)
        return pd.DataFrame()
    
    def _get_1m_data(self, symbol: str) -> pd.DataFrame:
        """Get 1-minute data for gamma reversal strategy."""
        if symbol in self.price_data and '1m' in self.price_data[symbol]:
            df = pd.DataFrame(self.price_data[symbol]['1m'])
            df['time'] = pd.to_datetime(df['time'])
            return df.set_index('time')
        return pd.DataFrame()
    
    def OnEndOfAlgorithm(self):
        """End of algorithm cleanup and final reporting."""
        
        # Final performance metrics
        portfolio_value = float(self.Portfolio.TotalPortfolioValue)
        total_return = (portfolio_value - 100000) / 100000
        
        risk_metrics = self.risk_manager.get_risk_metrics()
        
        self.Log(f"=== FINAL RESULTS ===")
        self.Log(f"Total Return: {total_return:.1%}")
        self.Log(f"Final Portfolio Value: ${portfolio_value:,.2f}")
        self.Log(f"Total Trades: {self.trade_count}")
        self.Log(f"Max Drawdown: {risk_metrics.get('current_drawdown', 0):.1%}")
        
        # Strategy performance
        for strategy_name, strategy in self.strategies.items():
            stats = strategy.get_strategy_stats()
            self.Log(f"{strategy_name}: {stats.get('total_signals', 0)} signals, "
                    f"{stats.get('active_positions', 0)} active positions")
        
        self.Log("Algorithm completed successfully")
    
    def PollRuntimeState(self):
        """Poll for runtime state updates from Admin API."""
        if not self.poller:
            return
        
        try:
            # Check if it's time to poll
            current_time = datetime.now()
            if current_time - self.last_poll_time < self.polling_interval:
                return
            
            # Fetch runtime state
            runtime_state_data = self.poller.fetch(jitter=True)
            
            if runtime_state_data:
                # Apply updates to local runtime state
                success = self.runtime_state.apply_patch(runtime_state_data)
                
                if success:
                    self.Debug("Runtime state updated from Admin API")
                    
                    # Update regime if provided
                    if 'last_regime' in runtime_state_data:
                        self.runtime_state.update_regime(runtime_state_data['last_regime'])
                    
                    # Update heartbeat
                    self.runtime_state.heartbeat()
                else:
                    self.Log("Failed to apply runtime state patch")
            else:
                self.Debug("No runtime state data received")
            
            self.last_poll_time = current_time
            
        except Exception as e:
            self.Log(f"Runtime state polling error: {e}")
    
    def RecordEquitySnapshot(self):
        """Record current equity snapshot to storage."""
        try:
            portfolio_value = float(self.Portfolio.TotalPortfolioValue)
            
            # Calculate drawdown
            current_drawdown = 0.0
            if hasattr(self.risk_manager, 'get_risk_metrics'):
                risk_metrics = self.risk_manager.get_risk_metrics()
                current_drawdown = risk_metrics.get('current_drawdown', 0.0)
            
            # Calculate realized volatility (simplified)
            realized_vol = 0.0
            if self.last_portfolio_value > 0:
                daily_return = (portfolio_value - self.last_portfolio_value) / self.last_portfolio_value
                realized_vol = abs(daily_return) * (252 ** 0.5)  # Annualized
            
            # Record to storage
            self.trade_storage.record_equity(
                ts=self.Time.isoformat(),
                equity=portfolio_value,
                drawdown=current_drawdown,
                realized_vol=realized_vol
            )
            
            self.Debug(f"Recorded equity snapshot: ${portfolio_value:,.2f}")
            
        except Exception as e:
            self.Log(f"Failed to record equity snapshot: {e}")
    
    def OnData(self, data):
        """Main data processing method - enhanced with runtime state checks."""
        try:
            # Check if trading is paused via runtime state
            if self.runtime_state.trading_paused or self.runtime_state.kill_switch_active:
                self.Debug("Trading paused via runtime state")
                return
            
            # Update market regime in runtime state if detected
            if hasattr(self.strategy_manager, 'current_regime'):
                current_regime = getattr(self.strategy_manager, 'current_regime', 'unknown')
                if current_regime != self.runtime_state.last_regime:
                    self.runtime_state.update_regime(current_regime)
            
            # Continue with normal processing...
            super().OnData(data) if hasattr(super(), 'OnData') else None
            
        except Exception as e:
            self.Log(f"Error in OnData: {e}")
    
    def _should_process_strategy(self, strategy_name: str) -> bool:
        """Check if strategy should be processed based on runtime state."""
        return self.runtime_state.get_strategy_enabled(strategy_name)
    
    def _get_current_risk_params(self) -> Dict[str, float]:
        """Get current risk parameters from runtime state."""
        return self.runtime_state.risk_params