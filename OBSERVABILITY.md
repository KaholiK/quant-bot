# Observability & Monitoring

This guide covers monitoring, logging, and observability for the quant-bot system.

## Discord Control UI

The Discord bot provides mobile-first monitoring and control.

### Setup

1. **Create Discord Application:**
   - Go to https://discord.com/developers/applications
   - Click "New Application"
   - Go to "Bot" section, click "Add Bot"
   - Copy bot token → `DISCORD_BOT_TOKEN`
   - Enable necessary intents (default intents only, no message content needed)

2. **Get IDs:**
   - Enable Developer Mode: User Settings → Advanced → Developer Mode ON
   - Right-click on server → Copy ID → `DISCORD_GUILD_ID`
   - Right-click on reports channel → Copy ID → `DISCORD_REPORTS_CHANNEL_ID`

3. **Invite Bot:**
   - Go to OAuth2 → URL Generator
   - Select scopes: `bot`, `applications.commands`
   - Select permissions: Send Messages, Embed Links, Attach Files
   - Copy URL and open in browser to invite

4. **Start Bot:**
   ```bash
   python -m ui.discord_bot.main
   ```

### Available Commands

All commands are slash commands (type `/` to see them):

#### `/envcheck`
- **Description:** Check active providers and configuration
- **Output:** Shows which data providers, integrations are enabled
- **Visibility:** Ephemeral (only you see it)

**Example Response:**
```
🔍 Environment Check

Mode: paper
Environment: dev
Log Level: INFO

Data Providers:
✅ Equity: tiingo
✅ Crypto: coingecko

Integrations:
✅ OpenAI (Narration)
❌ W&B (Telemetry)
✅ PostgreSQL
```

#### `/pnl <window>`
- **Description:** Show P&L for time window
- **Parameters:**
  - `window`: 1d, 1w, 1m, or ytd
- **Output:** Equity chart + metrics
- **Visibility:** Ephemeral

**Example Response:**
```
📊 P&L Report (1W)

Initial Equity: $100,000.00
Final Equity: $102,450.00
P&L: +$2,450.00 (+2.45%)
Max Drawdown: -1.23%
Data Points: 168

[Equity curve chart attached]
```

#### `/trades [limit] [symbol]`
- **Description:** Show recent trades
- **Parameters:**
  - `limit`: Number of trades (max 50, default 20)
  - `symbol`: Optional filter by symbol
- **Output:** Table of recent trades
- **Visibility:** Ephemeral

**Example Response:**
```
Recent Trades (Last 20)

🟢 09/25 14:30 SPY BUY 100 @ $450.25 PnL: $0.00
🔴 09/25 15:00 SPY SELL 100 @ $449.80 PnL: -$45.00
🟢 09/25 15:30 AAPL BUY 50 @ $175.20 PnL: $0.00
...
```

#### `/halt`
- **Description:** Stop paper trading loop
- **Output:** Confirmation message
- **Visibility:** Ephemeral

**Effect:** Sets halt flag that paper trading runner checks periodically.

### Boot Message

When bot starts, it posts to reports channel:

```
✅ Bot Started (Paper Mode)
Discord control bot is online

Mode: PAPER
Environment: DEV
Database: PostgreSQL

Data Providers:
Equity: tiingo
Crypto: coingecko

Integrations:
✅ OpenAI | ✅ W&B
```

## Database Logging

All trading activity is logged to PostgreSQL (or SQLite fallback).

### Schema

**Tables:**
- `runs` - Backtest/paper run metadata
- `orders` - Order records
- `fills` - Execution records
- `equity_points` - Equity curve data points
- `metrics` - Custom metrics/KPIs
- `error_logs` - Error tracking

**Indices:**
- All tables indexed on `run_id`
- Timestamps indexed for fast queries
- Symbol indexed for filtering

### Database CLI

```bash
# Initialize schema
python -m storage.db init

# Purge simulation data
python -m storage.db purge-sim

# Dump run KPIs
python -m storage.db dump-kpis <RUN_ID>
```

**Example Output:**
```json
{
  "run_id": "a1b2c3d4-...",
  "mode": "backtest",
  "started_at": "2024-09-25T10:00:00Z",
  "ended_at": "2024-09-25T10:05:00Z",
  "universe": "SPY,AAPL",
  "kpis": {
    "sharpe": 1.85,
    "sortino": 2.10,
    "max_dd": -0.08,
    "total_return": 0.12
  },
  "counts": {
    "orders": 42,
    "fills": 40,
    "equity_points": 500,
    "errors": 0
  }
}
```

### Querying Database

```python
from storage.db import get_session, Run, EquityPoint
from sqlalchemy import select

# Get all backtest runs
with get_session() as session:
    runs = session.execute(
        select(Run).where(Run.mode == "backtest")
    ).scalars().all()
    
    for run in runs:
        print(f"{run.id}: {run.kpis}")

# Get equity curve for run
with get_session() as session:
    points = session.execute(
        select(EquityPoint)
        .where(EquityPoint.run_id == run_id)
        .order_by(EquityPoint.ts)
    ).scalars().all()
    
    for point in points:
        print(f"{point.ts}: ${point.equity:,.2f}")
```

## Weights & Biases (W&B)

Optional experiment tracking integration.

### Setup

1. Sign up at https://wandb.ai/
2. Get API key from Settings
3. Add to `.env`: `WANDB_API_KEY=your_key`

### Usage

W&B automatically logs when configured:

**Backtest:**
- Run configuration (universe, dates, etc.)
- Final KPIs (sharpe, sortino, max_dd, etc.)
- Equity curve chart artifact

**Paper Trading:**
- Run parameters (duration, etc.)
- Summary metrics

### Viewing Results

```bash
# Open W&B dashboard
wandb login
# Visit https://wandb.ai/your-username/quantbot
```

### Disabling W&B

Remove or comment out `WANDB_API_KEY` in `.env`. System will automatically operate in no-op mode.

## Logging

### Log Levels

Controlled by `LOG_LEVEL` environment variable:
- `DEBUG`: Verbose logging for development
- `INFO`: Standard operational logging (default)
- `WARNING`: Only warnings and errors
- `ERROR`: Only errors

### Log Output

**Console:** Standard output with rich formatting (using loguru)

**File:** Not configured by default, but can be added:

```python
from loguru import logger

logger.add("logs/quantbot_{time}.log", rotation="1 day", retention="7 days")
```

### Example Logs

```
2024-09-25 10:00:00 | INFO     | Backtest started: SPY,AAPL
2024-09-25 10:00:01 | INFO     | Loaded 500 bars for SPY
2024-09-25 10:00:02 | INFO     | Loaded 500 bars for AAPL
2024-09-25 10:00:05 | INFO     | Backtest complete. Sharpe: 1.85
2024-09-25 10:00:06 | INFO     | W&B run finished
```

## Performance Metrics

### Key Performance Indicators (KPIs)

Automatically calculated for all runs:

- **Total Return**: (Final - Initial) / Initial
- **Sharpe Ratio**: Risk-adjusted return (annualized)
- **Sortino Ratio**: Downside-adjusted return (annualized)
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Total Trades**: Number of round-trip trades

### Accessing KPIs

**Via Discord:**
```
/pnl 1w
```

**Via Database:**
```bash
python -m storage.db dump-kpis <RUN_ID>
```

**Via Python:**
```python
from storage.db import get_session, Run
from sqlalchemy import select

with get_session() as session:
    run = session.execute(
        select(Run).where(Run.id == run_id)
    ).scalar_one()
    
    print(run.kpis)
```

## Chart Generation

Charts are automatically generated for:

- Equity curves (`/pnl` command)
- Backtest results
- Paper trading summaries

**Format:** PNG images attached to Discord messages or saved to `/tmp/`

**Library:** matplotlib (configured for non-interactive backend)

## Health Checks

### Environment Smoke Test

```bash
make env.smoke
# or
python -m scripts.env_smoke
```

**Checks:**
- Configuration validity
- Provider availability
- Database connectivity
- Integration status

**Output:**
- ✅ Green: All good
- ⚠️  Yellow: Warnings (non-critical)
- ❌ Red: Errors (critical issues)

### Database Health

```bash
# Test connection
python -c "from storage.db import get_engine; get_engine().connect()"

# Check tables exist
python -m storage.db init  # Safe to re-run
```

### Discord Bot Health

**Check bot is online:**
- Look for green status in Discord server member list
- Bot should have "Watching paper markets 📊" status

**Check commands are synced:**
- Type `/` in any channel where bot has access
- Commands should appear in autocomplete

## Alerting & Notifications

### Discord Webhooks (Optional)

For passive notifications without bot commands:

1. Create webhook in channel settings
2. Add to `.env`: `DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...`
3. Use in code:

```python
import requests

requests.post(settings.DISCORD_WEBHOOK_URL, json={
    "content": "🚨 Alert: Drawdown exceeded threshold!"
})
```

### Error Logging to Database

All errors are automatically logged to `error_logs` table:

```python
from storage.db import get_session, ErrorLog
from sqlalchemy import select

with get_session() as session:
    errors = session.execute(
        select(ErrorLog)
        .where(ErrorLog.level == "ERROR")
        .order_by(ErrorLog.ts.desc())
        .limit(10)
    ).scalars().all()
    
    for error in errors:
        print(f"{error.ts}: {error.message}")
```

## Best Practices

### 1. Monitor Discord Bot
- Check boot message appears on startup
- Test commands regularly
- Watch for error embeds

### 2. Review Database Periodically
- Check error logs
- Verify equity curve looks reasonable
- Purge old simulation data

### 3. Use W&B for Experiments
- Tag runs appropriately
- Compare across runs
- Track parameter changes

### 4. Set Up Alerts
- Configure Discord webhooks for critical events
- Monitor drawdown thresholds
- Alert on data provider failures

### 5. Regular Health Checks
- Run `make env.smoke` after configuration changes
- Verify database connectivity
- Check Discord bot status

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed troubleshooting guide.

## Next Steps

- Set up Discord bot for mobile monitoring
- Configure W&B for experiment tracking
- Review database schema for custom queries
- Explore chart generation options
