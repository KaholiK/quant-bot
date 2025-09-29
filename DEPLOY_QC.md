# QuantConnect Cloud Deployment Guide

This guide covers deploying the quant-bot to QuantConnect's cloud platform for backtesting, paper trading, and live trading.

## Prerequisites

- QuantConnect account (free or paid)
- Trained models (run training scripts first)
- Configured `config.yaml`

## Step 1: Prepare Files for Upload

### Required Files Structure
```
quant-bot/
├── MainAlgo.py                 # Main algorithm file
├── config.yaml                 # Configuration
├── algos/                      # Core modules
│   ├── core/
│   │   ├── config_loader.py
│   │   ├── feature_pipe.py
│   │   ├── labels.py
│   │   ├── cv_utils.py
│   │   ├── risk.py
│   │   ├── portfolio.py
│   │   ├── exec_rl.py
│   │   └── alerts.py
│   └── strategies/
│       ├── scalper_sigma.py
│       ├── trend_breakout.py
│       ├── bull_mode.py
│       ├── market_neutral.py
│       └── gamma_reversal.py
├── models/                     # Trained models
│   ├── xgb_classifier.joblib
│   └── meta_filter.joblib
└── policies/                   # RL policies
    └── ppo_policy.zip
```

### Pre-Upload Checklist

1. **Train Models Locally**:
```bash
python scripts/train_classifier.py --start 2022-01-01 --end 2024-01-01
python scripts/train_ppo.py --timesteps 50000
```

2. **Verify Configuration**:
```bash
python scripts/self_audit.py
```

3. **Test Locally** (if LEAN CLI is installed):
```bash
lean backtest "quant-bot" --start "2023-01-01" --end "2023-06-01"
```

## Step 2: Upload to QuantConnect

### Method 1: Web IDE (Recommended for beginners)

1. **Login to QuantConnect**:
   - Go to [quantconnect.com](https://www.quantconnect.com)
   - Sign in to your account

2. **Create New Project**:
   - Click "Create New Project"
   - Name: `quant-bot`
   - Language: Python
   - Template: Empty

3. **Upload Files**:
   - Use the file explorer in the web IDE
   - Create folder structure matching local setup
   - Upload all `.py` files and models
   - Copy/paste code from local files

### Method 2: Git Integration (Recommended for advanced users)

1. **Connect Repository**:
   - In QC project settings, connect your GitHub repo
   - Select the branch to sync
   - Configure auto-sync (optional)

2. **File Mapping**:
   - Ensure the main algorithm file is named appropriately
   - Models and config files should be in accessible paths

### Method 3: LEAN CLI Push

```bash
# Configure QC credentials
lean login

# Push project to cloud
lean cloud push --project "quant-bot"
```

## Step 3: Configure Environment Variables

### Required Environment Variables

In QuantConnect project settings, add:

```
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/your_webhook_url
QC_ENVIRONMENT=cloud
```

### Optional Variables

```
LOG_LEVEL=INFO
RISK_OVERRIDE_ENABLED=false
DEBUG_MODE=false
```

## Step 4: Run Backtests

### Basic Backtest

1. **Set Parameters**:
   - Start Date: 2023-01-01
   - End Date: 2024-01-01  
   - Initial Cash: $100,000
   - Node: O1-8 (or higher for faster execution)

2. **Launch Backtest**:
   - Click "Backtest" button
   - Monitor progress in logs
   - Wait for completion (can take 10-60 minutes)

### Parameter Optimization

1. **Create Optimization**:
   - Click "Optimize" instead of "Backtest"
   - Define parameter ranges in code or config
   - Select optimization target (Sharpe, Alpha, etc.)

2. **Example Parameter Setup**:
```python
# In MainAlgo.py Initialize() method
self.risk_per_trade = self.GetParameter("risk_per_trade", 0.01)
self.lookback_bars = int(self.GetParameter("lookback_bars", 55))
```

3. **Optimization Settings**:
   - Parameters: risk_per_trade (0.005-0.02), lookback_bars (20-100)
   - Target: Sharpe Ratio
   - Node: O2-8 or higher

## Step 5: Paper Trading

### Setup Paper Trading

1. **Create Paper Trading**:
   - Go to "Live Trading" section
   - Click "Create Live Algorithm"
   - Select "Paper Trading" brokerage

2. **Configure Settings**:
   - Initial Cash: $100,000
   - Start Date: Current date
   - Node: O1-4 (sufficient for paper trading)

3. **Deploy**:
   - Review configuration
   - Click "Deploy Live"
   - Monitor in real-time dashboard

### Monitor Paper Trading

- **Real-time Dashboard**: Track positions, P&L, orders
- **Logs**: Monitor algorithm decisions and risk events
- **Discord Alerts**: Receive notifications for trades and risks

## Step 6: Live Trading

### Choose Brokerage

Supported brokers (as of 2024):
- Interactive Brokers
- Oanda
- Bitfinex (crypto)
- Coinbase Pro (crypto)
- TD Ameritrade
- Alpaca

### Interactive Brokers Setup (Recommended)

1. **IB Account Requirements**:
   - Active IB account with API access
   - TWS or IB Gateway installed and configured
   - Sufficient margin and permissions

2. **Configure IB in QuantConnect**:
   - Brokerage: Interactive Brokers
   - Account ID: Your IB account number
   - Username/Password: IB credentials
   - Trading Mode: Live

3. **Risk Settings**:
   ```
   Max Leverage: 2.0
   Position Limit: $10,000 per symbol
   Daily Loss Limit: $2,000
   ```

### Live Trading Checklist

- [ ] Adequate account funding ($10,000+ recommended)
- [ ] All required market data subscriptions
- [ ] Risk parameters configured conservatively
- [ ] Discord alerts working
- [ ] Kill switch tested
- [ ] Emergency contact information updated

## Step 7: Risk Management & Monitoring

### Automated Risk Controls

The bot includes several built-in risk controls:

1. **Kill Switch**: Flattens all positions at 20% drawdown
2. **Position Limits**: Max 10% per symbol, 30% per sector
3. **Leverage Limits**: Maximum 2.0x leverage
4. **Stop Losses**: ATR-based stops for all positions

### Manual Risk Controls

1. **Daily Monitoring**:
   - Check P&L and positions each morning
   - Review overnight news and events
   - Verify all systems operational

2. **Weekly Reviews**:
   - Analyze strategy performance
   - Update risk parameters if needed
   - Review and retrain models

3. **Emergency Procedures**:
   - Know how to manually flatten positions
   - Have broker contact information ready
   - Maintain backup communication channels

### Performance Tracking

- **QuantConnect Dashboard**: Real-time metrics
- **Discord Alerts**: Trade and risk notifications  
- **External Tools**: Optional integration with portfolio trackers

## Troubleshooting

### Common Issues

1. **Model Loading Failures**:
   - Verify model files uploaded correctly
   - Check file paths in config.yaml
   - Ensure model dependencies available

2. **Data Issues**:
   - Verify data subscription active
   - Check symbol mappings
   - Review market hours and holidays

3. **Memory/Timeout Issues**:
   - Upgrade to higher node tier
   - Optimize code for cloud execution
   - Reduce universe size if needed

4. **Brokerage Connection Issues**:
   - Verify credentials and permissions
   - Check network connectivity
   - Review brokerage-specific requirements

### Getting Help

1. **QuantConnect Support**:
   - Support ticket system for paid subscribers
   - Community forum for general questions
   - Documentation and tutorials

2. **Bot-Specific Issues**:
   - GitHub issues for code problems
   - Self-audit report for configuration issues
   - Discord alerts for runtime problems

## Required Settings and Configuration

### Brokerage Configuration

When deploying to live trading, ensure your brokerage is properly configured:

1. **Enable Fees and Slippage**: 
   - Go to your algorithm settings in QuantConnect
   - Ensure "Enable Fees" and "Enable Slippage" are checked
   - This provides realistic backtesting and live trading performance

2. **Market Impact Modeling**:
   - Enable market impact simulation for realistic execution costs
   - This helps the algorithm learn proper position sizing

3. **Data Quality Settings**:
   - Use high-quality data feeds for live trading
   - Enable real-time data subscriptions for target symbols

### Discord Notifications Setup

To receive real-time alerts and notifications:

1. **Create Discord Webhook**:
   ```bash
   # In your Discord server, go to:
   # Server Settings > Integrations > Webhooks > New Webhook
   # Copy the webhook URL
   ```

2. **Add to QuantConnect Environment**:
   - In your QuantConnect project settings
   - Add environment variable: `DISCORD_WEBHOOK_URL`
   - Paste your webhook URL as the value

3. **Notification Types**:
   - Startup/shutdown alerts
   - Trade executions
   - Risk limit breaches
   - Model retrain results
   - Kill-switch activations

### Meta-Learning Configuration

The bot includes advanced meta-learning capabilities:

1. **Meta-Threshold Setting**:
   - Default: `learning.meta_threshold: 0.55`
   - Controls how selective the meta-filter is
   - Higher values = more selective (fewer but higher quality trades)
   - Lower values = more permissive (more trades but potentially lower quality)

2. **Features and Labels**:
   - The system automatically generates technical features
   - Labels use triple-barrier method with ATR-based stops
   - All parameters configurable via `config.yaml`

## Best Practices

1. **Start Small**: Begin with paper trading or small live amounts
2. **Monitor Closely**: Especially first few weeks of live trading
3. **Regular Updates**: Keep models and strategies updated
4. **Risk First**: Always prioritize risk management over returns
5. **Backup Plans**: Have manual procedures for emergencies
6. **Documentation**: Keep detailed records of changes and performance

## Performance Expectations

### Realistic Targets
- **Annual Return**: 8-15% (after fees and slippage)
- **Sharpe Ratio**: 1.0-2.0
- **Max Drawdown**: 10-20%
- **Win Rate**: 40-60% (depending on strategy mix)

### Fee Considerations
- QuantConnect: $20-100/month (depending on node tier)
- Brokerage: $0.005-0.01 per share for stocks, varies for crypto
- Data Feeds: $10-50/month for real-time data
- Slippage: 2-5 basis points typical

The bot is designed to be profitable after all costs with proper risk management and realistic expectations.