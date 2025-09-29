# Quant Bot - Production-Grade Quantitative Trading System

A sophisticated Python 3.11 quantitative trading bot for QuantConnect LEAN that handles US equities (S&P-100) and crypto (BTCUSD, ETHUSD) with advanced machine learning, risk management, and multiple trading strategies.

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