# LEAN Setup Guide

This guide walks you through setting up QuantConnect LEAN CLI to run the quant-bot locally.

## Prerequisites

- Python 3.10 or 3.11
- .NET SDK 6.0 or later
- Docker (optional, for containerized backtests)

## Install LEAN CLI

### Option 1: Using pip (Recommended)

```bash
pip install lean
```

### Option 2: Using .NET CLI

```bash
dotnet tool install -g QuantConnect.Lean.CLI
```

## Setup LEAN Environment

1. **Initialize LEAN in your project directory:**

```bash
cd quant-bot
lean init
```

2. **Configure LEAN settings:**

Edit the generated `lean.json` file or use the provided one:

```json
{
  "algorithm-type-name": "MainAlgo",
  "algorithm-language": "Python",
  "algorithm-location": "MainAlgo.py",
  "description": "Production-grade quantitative trading bot for LEAN",
  "parameters": {},
  "libraries": [
    "numpy",
    "pandas", 
    "scikit-learn",
    "xgboost",
    "lightgbm",
    "ta",
    "joblib",
    "loguru",
    "statsmodels",
    "matplotlib",
    "quantstats",
    "stable-baselines3",
    "torch",
    "pyyaml"
  ]
}
```

## Download Market Data

LEAN requires historical data for backtesting:

```bash
# Download US equity data (requires QuantConnect account)
lean data download --dataset "US Equity" --data-type "Trade" --resolution "Minute"

# Download crypto data
lean data download --dataset "Crypto" --data-type "Trade" --resolution "Minute"
```

**Note:** You'll need a QuantConnect account and API key for data downloads. Free accounts have limited data access.

## Run Backtests

### Basic Backtest

```bash
lean backtest "quant-bot"
```

### Backtest with Custom Parameters

```bash
lean backtest "quant-bot" \
  --start "2023-01-01" \
  --end "2024-01-01" \
  --cash 100000
```

### Optimization (Parameter Sweep)

```bash
lean optimize "quant-bot" \
  --optimizer-config optimizer.json \
  --optimizer-strategy "Grid Search"
```

Example `optimizer.json`:
```json
{
  "parameters": [
    {
      "name": "risk_per_trade",
      "min": 0.005,
      "max": 0.02,
      "step": 0.005
    },
    {
      "name": "lookback_bars", 
      "min": 20,
      "max": 100,
      "step": 20
    }
  ],
  "target": "Sharpe Ratio"
}
```

## Live Trading Setup

### Paper Trading

```bash
lean live "quant-bot" \
  --brokerage "Paper Trading" \
  --environment "paper"
```

### Live Trading (Example with Interactive Brokers)

```bash
lean live "quant-bot" \
  --brokerage "Interactive Brokers" \
  --environment "live" \
  --ib-user-name "your_username" \
  --ib-account "your_account" \
  --ib-password "your_password"
```

**Security Note:** Use environment variables for credentials:

```bash
export IB_USER_NAME="your_username"
export IB_ACCOUNT="your_account" 
export IB_PASSWORD="your_password"
export DISCORD_WEBHOOK_URL="your_webhook_url"

lean live "quant-bot" \
  --brokerage "Interactive Brokers" \
  --environment "live"
```

## Monitor Performance

LEAN provides several ways to monitor your algorithm:

### Console Output
Real-time logs are displayed in the console during execution.

### Result Files
After backtests, check the generated files:
- `backtests/[timestamp]/report.html` - Performance report
- `backtests/[timestamp]/log.txt` - Detailed logs
- `backtests/[timestamp]/statistics.json` - Performance metrics

### Live Trading Dashboard
For live trading, LEAN creates a web dashboard at `http://localhost:5612`

## Troubleshooting

### Common Issues

1. **Missing Data**: Ensure you've downloaded the required market data
2. **Memory Issues**: Reduce the backtest date range or increase available memory
3. **Import Errors**: Verify all dependencies are installed in the LEAN environment
4. **Permission Errors**: Check file permissions and antivirus software

### Debug Mode

Run backtests in debug mode for more detailed output:

```bash
lean backtest "quant-bot" --debug
```

## Performance Optimization

### Multi-threading
LEAN supports parallel execution:

```bash
lean backtest "quant-bot" --parallel
```

### Cloud Execution
For large backtests, consider using QuantConnect Cloud:

```bash
lean cloud push
lean cloud backtest "quant-bot" --node "O1-8"
```

## Best Practices

1. **Version Control**: Keep your LEAN configuration in git
2. **Environment Separation**: Use different configs for backtest/paper/live
3. **Monitoring**: Set up alerts for live trading issues
4. **Risk Management**: Always use stop-losses and position limits
5. **Data Quality**: Validate data before running strategies

## Resources

- [LEAN Documentation](https://www.quantconnect.com/docs/)
- [LEAN CLI Reference](https://www.quantconnect.com/docs/v2/lean-cli)
- [Community Forum](https://www.quantconnect.com/forum/)
- [Example Algorithms](https://github.com/QuantConnect/Lean/tree/master/Algorithm.Python)

## Support

For issues specific to this quant-bot:
1. Check the [GitHub Issues](https://github.com/KaholiK/quant-bot/issues)
2. Run `python scripts/self_audit.py` to verify setup
3. Review logs in the `logs/` directory