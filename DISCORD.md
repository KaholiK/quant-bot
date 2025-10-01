# Discord Control Interface

Discord bot provides mobile-first control surface for quant-bot operations.

## Setup

### 1. Create Discord Bot

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Click "New Application"
3. Go to "Bot" section
4. Click "Add Bot"
5. Enable "Message Content Intent"
6. Copy bot token

### 2. Invite Bot to Server

Use this URL (replace CLIENT_ID):
```
https://discord.com/api/oauth2/authorize?client_id=CLIENT_ID&permissions=2048&scope=bot%20applications.commands
```

### 3. Configure Environment

Add to `.env`:
```bash
DISCORD_BOT_TOKEN=your_bot_token_here
DISCORD_GUILD_ID=your_server_id
DISCORD_REPORTS_CHANNEL_ID=channel_for_daily_reports
DISCORD_APP_ID=your_application_id

# Optional: Webhook for fallback
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

# Admin user IDs (comma-separated)
ALLOWED_USER_IDS=123456789,987654321
```

### 4. Start Bot

```bash
# Start bot
python -m ui.discord_bot.main

# Or use Docker
docker-compose up discord-bot
```

## Command Reference

### Status & Health

#### `/status`
Show current bot status and mode.

**Example**:
```
/status
```

**Response**:
```
✅ Bot Status: ONLINE

Mode: PAPER
Strategies: 3 enabled (scalper_sigma, trend_breakout, bull_mode)
Open Positions: 5
Equity: $103,450.23 (+3.45%)
Drawdown: -2.1% from peak
Kill Switch: INACTIVE
```

---

#### `/health`
System health check with detailed diagnostics.

**Example**:
```
/health
```

**Response**:
```
🏥 System Health Check

✅ Database: Connected (PostgreSQL)
✅ Data Providers: 3/4 configured
  ✅ Tiingo (equity)
  ✅ CoinGecko (crypto)
  ✅ FRED (macro)
  ❌ Polygon (not configured)
✅ W&B: Connected
✅ OpenAI: Connected
⚠️ Cache: 145 MB (consider cleanup)

Last Update: 2024-01-15 14:23:05 UTC
Uptime: 2d 5h 23m
```

---

### Universe Management

#### `/universe list`
Show current trading universe.

**Example**:
```
/universe list
```

**Response**:
```
📊 Trading Universe

Equities (5):
  SPY - S&P 500 ETF
  QQQ - NASDAQ 100 ETF
  AAPL - Apple Inc.
  MSFT - Microsoft Corp.
  TSLA - Tesla Inc.

Crypto (2):
  BTCUSD - Bitcoin
  ETHUSD - Ethereum
```

---

#### `/universe add <symbol> <asset_type>`
Add symbol to trading universe.

**Parameters**:
- `symbol`: Ticker symbol (e.g., SPY, AAPL)
- `asset_type`: "equity" or "crypto"

**Example**:
```
/universe add NVDA equity
```

**Response**:
```
✅ Added NVDA to equity universe
Downloading historical data...
Ready to trade in next cycle.
```

---

#### `/universe remove <symbol>`
Remove symbol from trading universe.

**Example**:
```
/universe remove TSLA
```

**Response**:
```
✅ Removed TSLA from universe
Current position: FLAT (no action needed)
```

---

### Strategy Management

#### `/strategy list`
Show all strategies and their status.

**Example**:
```
/strategy list
```

**Response**:
```
🎯 Strategies

✅ scalper_sigma - ENABLED
   Mean reversion scalping
   Last 7d: +2.3% | 15 trades | 60% hit rate

✅ trend_breakout - ENABLED
   Momentum breakout with filters
   Last 7d: +1.8% | 8 trades | 62.5% hit rate

✅ bull_mode - ENABLED
   Regime-aware position sizing
   Last 7d: +0.9% | 3 trades | 66% hit rate

❌ market_neutral - DISABLED
   Pairs trading with cointegration

❌ gamma_reversal - DISABLED
   HF crypto mean reversion
```

---

#### `/strategy enable <name>`
Enable a strategy.

**Example**:
```
/strategy enable market_neutral
```

**Response**:
```
✅ Enabled strategy: market_neutral
Will begin trading in next cycle.
```

---

#### `/strategy disable <name>`
Disable a strategy.

**Example**:
```
/strategy disable scalper_sigma
```

**Response**:
```
✅ Disabled strategy: scalper_sigma
Will close existing positions gradually.
```

---

#### `/strategy config <name>`
Show strategy configuration.

**Example**:
```
/strategy config scalper_sigma
```

**Response**:
```
⚙️ scalper_sigma Configuration

lookback_period: 20
entry_z_threshold: 2.0
exit_z_threshold: 0.5
min_hold_bars: 3
max_position_size: 0.05 (5% of portfolio)

Recent Performance:
  7d:  +2.3% | 15 trades | 60% hit rate
  30d: +8.1% | 67 trades | 58% hit rate
  Sharpe: 1.85
```

---

### Risk Management

#### `/risk status`
Show current risk metrics.

**Example**:
```
/risk status
```

**Response**:
```
⚖️ Risk Status

Portfolio Metrics:
  Equity: $103,450.23
  Max Drawdown: -2.1%
  Realized Vol (20d): 11.2% ann.
  Sharpe (60d): 1.92
  Sortino (60d): 2.45

Position Limits:
  Gross Leverage: 1.35x / 2.0x max ✅
  Largest Position: 8.5% / 10% max ✅
  Crypto Exposure: 22% / 50% max ✅

Kill Switch: INACTIVE ✅
Daily Stop: -1.2% / -6% ✅
Weekly Stop: -2.8% / -10% ✅
```

---

#### `/risk set <parameter> <value>`
Update risk parameter.

**Parameters**:
- `per_trade_risk_pct`: Risk per trade (e.g., 0.01 = 1%)
- `max_leverage`: Maximum leverage (e.g., 2.0)
- `vol_target_ann`: Volatility target (e.g., 0.12 = 12%)
- `kill_switch_dd`: Kill switch drawdown (e.g., 0.20 = 20%)

**Example**:
```
/risk set per_trade_risk_pct 0.015
```

**Response**:
```
✅ Updated per_trade_risk_pct: 0.01 → 0.015

New risk per trade: 1.5%
Estimated position sizes will increase ~50%
Change will apply to new trades immediately.
```

---

### Trading Control

#### `/paper start`
Start paper trading.

**Example**:
```
/paper start
```

**Response**:
```
🚀 Starting paper trading...

Mode: PAPER
Initial Equity: $100,000
Strategies: 3 enabled
Universe: 7 symbols

Bot will post updates to #trading-reports
Use /paper stop to halt trading.
```

---

#### `/paper stop`
Stop paper trading.

**Example**:
```
/paper stop
```

**Response**:
```
🛑 Stopping paper trading...

Final Equity: $103,450.23 (+3.45%)
Duration: 5 days 3 hours
Total Trades: 42
Win Rate: 61.9%

Positions will be closed gradually.
Full report posted to #trading-reports
```

---

### Backtesting

#### `/backtest run <start> <end> [params]`
Run backtest with parameters.

**Parameters**:
- `start`: Start date (YYYY-MM-DD)
- `end`: End date (YYYY-MM-DD)
- `universe`: Symbols (optional, defaults to current)
- `strategies`: Strategies to test (optional, defaults to all enabled)

**Example**:
```
/backtest run 2024-01-01 2024-06-30 universe:SPY,QQQ strategies:trend_breakout
```

**Response**:
```
⏳ Running backtest...

Period: 2024-01-01 to 2024-06-30 (181 days)
Universe: SPY, QQQ
Strategies: trend_breakout

Progress will be posted here.
ETA: ~2 minutes
```

**Final Report**:
```
✅ Backtest Complete

Total Return: +12.3%
Sharpe Ratio: 1.85
Sortino Ratio: 2.41
Max Drawdown: -5.2%
Win Rate: 58.3%
Trades: 34

Full report: https://wandb.ai/project/run/abc123
```

---

### Model Management

#### `/model list`
List available models and their status.

**Example**:
```
/model list
```

**Response**:
```
🤖 Models

Classifiers:
  xgb_classifier_v3 [PROD] - XGBoost
    Trained: 2024-01-10
    OOS Sharpe: 1.92
    
  xgb_classifier_v4 [CANDIDATE] - XGBoost
    Trained: 2024-01-15
    OOS Sharpe: 2.15 ⬆️
    
Meta-Filter:
  meta_filter_v2 [PROD] - LightGBM
    Trained: 2024-01-08
    
RL Execution:
  ppo_policy_v3 [PROD] - PPO
    Trained: 2024-01-12
```

---

#### `/model promote <name>`
Promote candidate model to production.

**Example**:
```
/model promote xgb_classifier_v4
```

**Response**:
```
✅ Promoting xgb_classifier_v4 to production...

Validation Checks:
  ✅ OOS Sharpe: 2.15 > 1.2 (min)
  ✅ OOS Profit Factor: 1.28 > 1.15 (min)
  ✅ Max Drawdown: 4.2% < 6% (max)

Backup: xgb_classifier_v3 saved
Active Model: xgb_classifier_v4
Change effective immediately.

Use /model rollback to revert if needed.
```

---

#### `/model rollback <name>`
Rollback to previous model version.

**Example**:
```
/model rollback xgb_classifier_v3
```

**Response**:
```
✅ Rolling back to xgb_classifier_v3

Previous model restored.
Use with caution - investigate why v4 failed first.
```

---

### Retraining

#### `/retrain now [strategies]`
Trigger immediate retraining.

**Example**:
```
/retrain now strategies:all
```

**Response**:
```
🔄 Starting retraining workflow...

Universe: 7 symbols
Lookback: 90 days
Strategies: all

Data download: ⏳ In progress...
Feature engineering: ⏳ Pending
Model training: ⏳ Pending
Validation: ⏳ Pending
Promotion check: ⏳ Pending

ETA: ~15 minutes
Progress will be posted here.
```

---

### Reports

#### `/report daily`
Generate daily P&L report.

**Example**:
```
/report daily
```

**Response**:
```
📊 Daily Report - 2024-01-15

P&L:
  Today: +$1,245.67 (+1.21%)
  Week: +$3,450.23 (+3.45%)
  Month: +$8,123.45 (+8.12%)

Performance:
  Sharpe (30d): 1.92
  Sortino (30d): 2.45
  Max DD: -2.1%
  Hit Rate: 61.9%

Top Winners Today:
  SPY: +$456.78 (trend_breakout)
  QQQ: +$345.12 (scalper_sigma)

Top Losers Today:
  AAPL: -$123.45 (bull_mode)

Trades: 8 executed (5 wins, 3 losses)
Turnover: 24.3%
Avg Slippage: 8.2 bps

What Worked:
  • Trend breakout captured momentum in SPY
  • Scalper sigma performed well in choppy QQQ

What Failed:
  • Bull mode stopped out on AAPL gap down
  • Consider tighter stops for earnings-adjacent trades
```

---

### Kill Switch

#### `/kill-switch halt`
Manually activate kill switch (emergency stop).

**Example**:
```
/kill-switch halt
```

**Response**:
```
🚨 KILL SWITCH ACTIVATED

Trading halted immediately.
All new orders blocked.
Open positions remain (no panic liquidation).

Review system before resuming.
Use /kill-switch resume when ready.
```

---

#### `/kill-switch resume`
Resume trading after kill switch.

**Example**:
```
/kill-switch resume
```

**Response**:
```
⚠️ Confirm Resume Trading

Current Status:
  Drawdown: -21.3%
  Open Positions: 5
  Equity: $78,654.32

Type "confirm" to resume trading.
This action requires admin privileges.
```

---

### Logs

#### `/logs tail [lines]`
Show recent log entries.

**Parameters**:
- `lines`: Number of lines (default: 50)

**Example**:
```
/logs tail 20
```

**Response**:
```
📜 Recent Logs (last 20 lines)

2024-01-15 14:23:05 INFO  | Strategy signal: SPY buy (trend_breakout)
2024-01-15 14:23:06 INFO  | Position sizing: 220 shares
2024-01-15 14:23:07 INFO  | Order submitted: SPY buy 220 @ market
2024-01-15 14:23:08 INFO  | Fill: SPY 220 @ $452.35
...
```

---

### Configuration

#### `/config show [section]`
Show configuration.

**Parameters**:
- `section`: Config section (optional: all, risk, strategies, universe)

**Example**:
```
/config show risk
```

**Response**:
```
⚙️ Risk Configuration

per_trade_risk_pct: 0.01 (1%)
max_leverage: 2.0
single_name_max_pct: 0.10 (10%)
sector_max_pct: 0.30 (30%)
vol_target_ann: 0.12 (12%)
kill_switch_dd: 0.20 (20%)

Use /risk set to modify parameters.
```

---

## Scheduled Reports

Bot automatically posts to `#trading-reports` channel:

### Daily Summary (End of Day)

```
📊 Daily Summary - 2024-01-15

Today: +$1,245.67 (+1.21%) | Week: +$3,450.23 (+3.45%)
Sharpe: 1.92 | Max DD: -2.1% | Hit Rate: 61.9%

🏆 Top Strategy: trend_breakout (+$789.45)
📈 Best Trade: SPY +$456.78 (6:1 R:R)
⚠️ Worst Trade: AAPL -$234.56 (earnings gap)

Trades: 8 (5W/3L) | Turnover: 24.3% | Slippage: 8.2 bps
```

### Weekly Review (Sunday Evening)

```
📈 Weekly Review - Week of Jan 8-14, 2024

Return: +3.45% | Sharpe: 1.85 | Sortino: 2.41
Trades: 42 | Win Rate: 61.9% | Profit Factor: 2.14

Strategy Attribution:
  trend_breakout: +1.8% (18 trades, 66.7% win rate)
  scalper_sigma: +1.2% (20 trades, 60% win rate)
  bull_mode: +0.5% (4 trades, 50% win rate)

Risk Metrics:
  Max Drawdown: -2.1%
  Realized Vol: 11.2% (target: 12%)
  Leverage: 1.35x avg (max: 1.78x)

Next Week Adjustments:
  • Increase vol targeting (currently below target)
  • Add NVDA to universe based on momentum
  • Monitor earnings calendar (AAPL, MSFT reporting)
```

## Permissions

Configure command permissions in config:

```yaml
discord:
  admin_role: "Quant Admin"
  allowed_user_ids: [123456789, 987654321]
  
  command_permissions:
    # Public commands
    status: everyone
    health: everyone
    report: everyone
    
    # Trader commands
    universe: trader
    strategy: trader
    backtest: trader
    
    # Admin only
    risk: admin
    paper: admin
    retrain: admin
    model: admin
    kill-switch: admin
    config: admin
```

## Troubleshooting

### Bot Not Responding

1. Check bot is online: `/status`
2. Verify permissions: Bot needs "Send Messages" and "Use Slash Commands"
3. Check logs: `docker logs quant-bot-discord`

### Commands Not Appearing

1. Wait up to 1 hour for global command sync
2. Try guild-specific commands (faster): Add `DISCORD_GUILD_ID` to .env
3. Re-invite bot with updated permissions URL

### Webhook Fallback

If bot gateway is rate-limited, use webhook:

```python
import requests

requests.post(
    os.getenv("DISCORD_WEBHOOK_URL"),
    json={"content": "🚨 Alert message"}
)
```

## See Also

- [OBSERVABILITY.md](OBSERVABILITY.md) - Monitoring and alerts
- [bots/discord_bot.py](bots/discord_bot.py) - Bot implementation
- [ui/discord_bot/](ui/discord_bot/) - Discord UI module
