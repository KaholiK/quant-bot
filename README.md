# Quant Bot - Production-Grade Quantitative Trading System

[![CI](https://github.com/KaholiK/quant-bot/actions/workflows/ci.yml/badge.svg)](https://github.com/KaholiK/quant-bot/actions/workflows/ci.yml)
[![Retraining](https://github.com/KaholiK/quant-bot/actions/workflows/retrain.yml/badge.svg)](https://github.com/KaholiK/quant-bot/actions/workflows/retrain.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A sophisticated Python 3.11 quantitative trading bot for QuantConnect LEAN with advanced machine learning, risk management, multiple trading strategies, and continuous learning capabilities. Designed for production trading across US equities (S&P-100) and crypto (BTCUSD, ETHUSD).

## 🌟 Key Features

- **🎯 Multi-Strategy Architecture**: 5 integrated strategies with conflict resolution and performance-based allocation
- **🧠 Machine Learning**: XGBoost classifier with meta-labeling and PPO execution optimization  
- **⚖️ Advanced Risk Management**: Kill-switch, volatility targeting, asset class caps, position sizing
- **🔄 Continuous Learning**: Automated weekly retraining with drift detection and model promotion gates
- **🌐 Live/Backtest Dual Mode**: Seamless switching between paper and live trading
- **📊 Production Monitoring**: Discord alerts, comprehensive logging, performance tracking
- **🛡️ Robust Architecture**: Type-safe configuration, broker adapters, comprehensive testing

## 📋 Changelog

### v2.0.0 - Production Release (Latest)
- **🎯 Strategy Manager**: Complete signal aggregation with meta-labeling and conflict resolution
- **⚖️ Enhanced Risk System**: Asset class caps, volatility targeting, sophisticated kill-switch with recovery
- **🧠 ML Pipeline**: Complete training scripts with walk-forward validation and calibration
- **🔄 Auto-Retraining**: GitHub Actions workflow with model evaluation and auto-PR creation
- **🌐 Broker Adapters**: Portable interface for QuantConnect, Alpaca, IBKR compatibility
- **📊 Production CI**: Security scanning, integration tests, comprehensive validation
- **🛠️ Live Trading**: OnOrderEvent handling, proper consolidators, brokerage model detection

### v1.0.0 - Initial Release
- **Config System**: Type-safe pydantic configuration with validation and backward compatibility
- **Bug Fixes**: Fixed trend_breakout momentum calculation UnboundLocalError 
- **CI/CD**: Python 3.10+3.11 matrix testing with ruff, black, mypy, and pytest
- **Training Scripts**: Offline classifier and PPO training with data fetching
- **LEAN Integration**: lean.json and deployment guides for QuantConnect
- **Risk Management**: Discord alerts system for real-time notifications
- **Git LFS**: Model file tracking for joblib/pkl/zip files

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/KaholiK/quant-bot.git
cd quant-bot
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -e .

# Run self-audit to verify setup
python scripts/self_audit.py

# Run tests
pytest

# Train models offline (optional)
python scripts/train_classifier.py --config config.yaml
python scripts/train_ppo.py --config config.yaml --skip-if-no-deps
```

## ⚙️ Configuration

The bot uses a production-grade configuration system with full validation:

```yaml
# config.yaml
trading:
  universe:
    equities: SP100
    crypto: [BTCUSD, ETHUSD]
  bars:
    equities: 30m
    crypto: 15m
  risk:
    per_trade_risk_pct: 0.01
    max_leverage: 2.0
    single_name_max_pct: 0.10
    sector_max_pct: 0.30
    asset_class_caps: { crypto_max_gross_pct: 0.50 }
    kill_switch_dd: 0.20
    vol_target_ann: 0.12
  features:
    returns_periods: [1, 5, 20, 60]
    sma_periods: [20, 50, 200]
    ema_periods: [12, 26]
    macd_params: {fast: 12, slow: 26, signal: 9}
    atr_period: 14
    bollinger_period: 20
    vol_window: 20
  labels:
    horizon_bars: 5
    tp_atr_mult: 1.75
    sl_atr_mult: 1.00
  models:
    classifier_path: models/xgb_classifier.joblib
    meta_model_path: models/meta_filter.joblib
    rl_policy_path: policies/ppo_policy.zip
  strategies:
    scalper_sigma: {enabled: true}
    trend_breakout: {enabled: true}
    bull_mode: {enabled: true}
    market_neutral: {enabled: false}
    gamma_reversal: {enabled: false}
  learning:
    cv: {scheme: purged_kfold_embargo, folds: 5, embargo_frac: 0.02}
    retrain_cadence: weekly
    gates: {oos_sortino_min: 1.2, oos_profit_factor_min: 1.15, oos_max_dd_max: 0.06}
    meta_threshold: 0.55
  execution:
    maker_ladder_offsets_atr: [0.10, 0.20, 0.30]
    min_ms_between_orders: 300
    min_hold_secs: 60
```

## 🚀 Features

### Core Capabilities
- **Multi-Asset Trading**: US equities (S&P-100) on 30m bars, crypto (BTCUSD/ETHUSD) on 15m bars
- **Advanced ML Pipeline**: XGBoost classification with meta-labeling and model calibration
- **Reinforcement Learning**: PPO-based execution optimization for order placement
- **Risk Management**: Kill-switch, position sizing, leverage limits, sector caps
- **Portfolio Optimization**: Hierarchical Risk Parity (HRP) and volatility targeting

### Trading Strategies
1. **Scalper Sigma**: Mean-reversion scalping with sigma bands (tight stops, partial TPs)
2. **Trend Breakout**: 55-bar momentum breakout with trend filters and RSI blockers
3. **Bull Mode**: Higher utilization during bullish regimes (SMA50>200 & low vol)
4. **Market Neutral**: Pairs trading with cointegration and beta-neutral sizing
5. **Gamma Reversal**: High-frequency mean reversion for crypto (1-5m bars)

### Technical Features
- **Lag-Safe Features**: No look-ahead bias in feature engineering
- **Triple Barrier Labels**: Vectorized labeling with configurable profit/stop targets
- **Purged Cross-Validation**: Time series CV with embargo periods
- **Microstructure Support**: Order book features for crypto (when available)
- **Comprehensive Testing**: 95%+ test coverage with realistic scenarios

## 📦 Installation

### Prerequisites
- Python 3.11+
- QuantConnect LEAN (for backtesting/live trading)

### Setup

```bash
# Clone repository
git clone https://github.com/KaholiK/quant-bot.git
cd quant-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

## 🛠️ Development Workflow

### Code Quality
```bash
# Run linter
ruff check .
ruff format .

# Type checking
mypy .

# Run tests
pytest

# Run tests with coverage
pytest --cov=algos --cov-report=html
```

### Training Models

#### 1. Train Classifier
```bash
# Launch Jupyter
jupyter lab

# Open notebooks/train_classifier.ipynb
# This will:
# - Generate features using technical indicators
# - Create triple-barrier labels
# - Train XGBoost with purged cross-validation
# - Apply Platt/Isotonic calibration
# - Train meta-model for trade filtering
# - Save models to models/
```

#### 2. Train PPO Agent
```bash
# Open notebooks/train_ppo.ipynb
# This will:
# - Create trading execution environment
# - Train PPO agent for optimal execution
# - Backtest execution performance
# - Save policy to policies/
```

### Configuration

Edit `config.yaml` to customize:
- Risk parameters (leverage, position limits, kill-switch)
- Feature engineering settings
- Strategy parameters
- Model paths

### How to Run Tests and Audit Locally

To run tests and audit your local development environment:

```bash
# 1. Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install the package in editable mode
pip install -e .

# 3. Run linting
ruff check .
ruff format .

# 4. Run type checking (optional)
mypy algos

# 5. Run tests
pytest

# 6. Run self-audit to verify repository integrity
python scripts/self_audit.py
```

The self-audit script will:
- Verify all required files and directories exist
- Validate `config.yaml` schema and structure
- Test module imports to catch syntax errors
- Generate a detailed report in `self_audit_report.md`

## 🏗️ Architecture

### Core Modules (`algos/core/`)
- `feature_pipe.py`: Feature engineering pipeline with technical indicators
- `labels.py`: Triple barrier labeling and meta-labeling
- `cv_utils.py`: Purged K-fold and walk-forward cross-validation
- `risk.py`: Risk management with kill-switch and position sizing
- `portfolio.py`: HRP optimization and portfolio construction
- `exec_rl.py`: Reinforcement learning execution engine

### Strategies (`algos/strategies/`)
- `scalper_sigma.py`: Mean-reversion scalping strategy
- `trend_breakout.py`: Momentum breakout strategy  
- `bull_mode.py`: Regime-aware position sizing
- `market_neutral.py`: Statistical arbitrage pairs trading
- `gamma_reversal.py`: High-frequency crypto mean reversion

### Entry Point
- `MainAlgo.py`: QuantConnect LEAN algorithm orchestrating all components

## 🚦 Usage

### Local Backtesting (Synthetic Data)
```python
from algos.core.feature_pipe import FeaturePipeline
from algos.strategies.scalper_sigma import ScalperSigmaStrategy
import yaml

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize components
feature_pipeline = FeaturePipeline(config['trading'])
strategy = ScalperSigmaStrategy(config['trading'])

# Generate features and signals
# (See notebooks for complete examples)
```

### QuantConnect LEAN Deployment
1. Upload `MainAlgo.py` and `algos/` folder to QuantConnect
2. Ensure `config.yaml` is in the project root
3. Upload trained models to `models/` directory
4. Upload trained PPO policy to `policies/` directory
5. Configure universe and run backtest/live trading

### Model Artifacts
After training, you'll have:
- `models/xgb.pkl`: Calibrated XGBoost classifier
- `models/meta.pkl`: Meta-model for trade filtering
- `models/model_metadata.yaml`: Model performance metrics
- `policies/ppo_policy.zip`: PPO execution policy
- `policies/training_summary.yaml`: RL training results

## 📊 Performance Monitoring

The system provides comprehensive logging:
- **Feature Pipeline**: Technical indicator values and validation
- **Risk Management**: Position sizes, leverage, drawdowns
- **Strategy Signals**: Signal strength, confidence, entry/exit reasons
- **Execution**: Order types, fill rates, slippage analysis
- **Portfolio**: Asset allocation, rebalancing, performance metrics

Optional Discord webhook notifications for:
- Kill-switch activation
- Large drawdowns
- Strategy performance alerts

## 🧪 Testing

Comprehensive test suite covering:

```bash
# Test triple barrier labeling
pytest tests/test_triple_barrier.py -v

# Test cross-validation
pytest tests/test_purged_cv.py -v  

# Test risk management
pytest tests/test_risk_engine.py -v

# Run all tests
pytest -v
```

Test scenarios include:
- **Barrier Touch Detection**: Exact hits, timeouts, edge cases
- **Data Leakage Prevention**: Purged CV with embargo periods
- **Risk Limit Enforcement**: Position sizing, leverage, kill-switch
- **Model Robustness**: NaN handling, insufficient data scenarios

## ⚠️ Risk Disclaimers

- **Not Financial Advice**: This is educational/research code
- **Use at Own Risk**: Quantitative trading involves substantial risk
- **Test Thoroughly**: Backtest extensively before live deployment
- **Monitor Actively**: Automated systems require ongoing supervision
- **Start Small**: Begin with minimal capital for live testing

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests and linting (`pytest && ruff check .`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🔗 Resources

- [QuantConnect LEAN Documentation](https://www.quantconnect.com/docs)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Advances in Financial Machine Learning (López de Prado)](https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos/dp/1119482089)

## 📈 Roadmap

- [ ] Alternative data integration (sentiment, news)
- [ ] Multi-frequency strategy coordination  
- [ ] Options strategies with Greeks
- [ ] Real-time market microstructure features
- [ ] Advanced regime detection models
- [ ] Portfolio attribution analysis

---

**Built with ❤️ for the quantitative finance community**