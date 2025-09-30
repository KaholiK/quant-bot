# Environment Setup Guide

This guide covers environment variable configuration for the quant-bot system.

## Quick Start

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your API keys and settings**

3. **Load environment variables (Linux/Mac):**
   ```bash
   set -a
   source .env
   set +a
   ```

4. **Verify configuration:**
   ```bash
   python -m scripts.env_smoke
   ```

## Environment Variables

### Core Runtime Settings

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `APP_ENV` | No | `dev` | Application environment (`dev` or `prod`) |
| `RUN_MODE` | No | `paper` | Trading mode - **MUST be `paper`** (live trading disabled) |
| `LOG_LEVEL` | No | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `TZ` | No | `Pacific/Honolulu` | Timezone for operations |

### Discord Control UI

| Variable | Required | Description |
|----------|----------|-------------|
| `DISCORD_BOT_TOKEN` | Yes* | Discord bot token from Developer Portal |
| `DISCORD_GUILD_ID` | Yes* | Discord server (guild) ID where bot operates |
| `DISCORD_REPORTS_CHANNEL_ID` | Yes* | Channel ID for automated reports |
| `DISCORD_WEBHOOK_URL` | No | Optional webhook for notifications |
| `DISCORD_APP_ID` | No | Discord application ID |

*Required for Discord bot functionality. Bot will warn but still function without full Discord setup.

**How to get Discord IDs:**
1. Enable Developer Mode in Discord (User Settings → Advanced → Developer Mode)
2. Right-click on server/channel → Copy ID

### Data Providers

At least one equity OR crypto provider is recommended.

**Equity Data:**
| Variable | Provider | Free Tier | Notes |
|----------|----------|-----------|-------|
| `TIINGO_API_KEY` | Tiingo | Yes | Recommended for equities, excellent free tier |
| `POLYGON_API_KEY` | Polygon.io | Limited | High-quality data, rate limits on free tier |
| `ALPHA_VANTAGE_API_KEY` | Alpha Vantage | Limited | 5 calls/minute on free tier |

**Crypto Data:**
| Variable | Provider | Free Tier | Notes |
|----------|----------|-----------|-------|
| `COINGECKO_API_KEY` | CoinGecko | Yes | Public API available, Pro API for higher limits |
| `CRYPTOCOMPARE_API_KEY` | CryptoCompare | Yes | Good free tier |

**Macro/Economic Data:**
| Variable | Provider | Notes |
|----------|----------|-------|
| `FRED_API_KEY` | FRED (Federal Reserve) | Free, excellent for macro data |
| `QUANDL_API_KEY` | Quandl/Nasdaq Data Link | Mixed free/paid |

### Optional Integrations

| Variable | Purpose | Notes |
|----------|---------|-------|
| `OPENAI_API_KEY` | AI-generated trade summaries | Falls back to simple summaries if not set |
| `WANDB_API_KEY` | Weights & Biases experiment tracking | No-op if not set |
| `DATABASE_URL` | PostgreSQL connection | Falls back to SQLite if not set |

**Database URL Format:**
```
postgresql://user:password@host:5432/database?sslmode=require
```

**For Neon.tech users:**
- Use **direct connection URL**, not the pooled URL
- Pooled URLs (containing `-pooler`) are fine for runtime but not for migrations
- Direct URL format: `postgresql://user:password@ep-xxx.region.aws.neon.tech/db?sslmode=require`

### Paper Trading Broker (Optional)

| Variable | Purpose | Default |
|----------|---------|---------|
| `ALPACA_API_KEY` | Alpaca paper trading API key | N/A |
| `ALPACA_API_SECRET` | Alpaca paper trading secret | N/A |
| `ALPACA_BASE_URL` | Alpaca API endpoint | `https://paper-api.alpaca.markets` |

Currently optional - system uses simple simulation by default.

## Security Best Practices

### Never Commit Secrets

```bash
# .env files are gitignored
# Always use .env.example as template
# Never commit actual API keys
```

### Use Strong Tokens

- Generate secure random tokens: `openssl rand -hex 32`
- Don't reuse tokens across environments
- Rotate tokens regularly

### Protect Your .env File

```bash
# Set proper permissions (Linux/Mac)
chmod 600 .env

# Never share .env via chat/email
# Use secure secret management (1Password, etc.) for team sharing
```

## Loading Environment Variables

### Linux/macOS (Bash/Zsh)

```bash
# One-time load
set -a
source .env
set +a

# Or add to .bashrc/.zshrc for permanent loading
echo 'set -a; source /path/to/quant-bot/.env; set +a' >> ~/.bashrc
```

### Windows (PowerShell)

```powershell
# Install dotenv tool
pip install python-dotenv

# Load in Python scripts automatically (already configured)
# Or manually:
Get-Content .env | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
    }
}
```

### Docker/Container

```dockerfile
# Dockerfile
ENV ENV_FILE=.env
```

```bash
# Docker run
docker run --env-file .env your-image
```

## Verification

After setting up your environment:

```bash
# Run smoke test
python -m scripts.env_smoke

# Should show:
# ✅ Green checkmarks for configured providers
# ⚠️  Warnings for optional/missing providers
# ❌ Errors only for critical misconfigurations
```

## Troubleshooting

### "No providers configured" Warning

**Cause:** No data provider API keys set.

**Fix:** Add at least one equity or crypto provider:
```bash
# In .env
TIINGO_API_KEY=your_key_here
# or
COINGECKO_API_KEY=your_key_here
```

### "Discord not fully configured" Warning

**Cause:** Missing Discord bot token or IDs.

**Fix:** Set all three required Discord variables:
```bash
DISCORD_BOT_TOKEN=your_token
DISCORD_GUILD_ID=123456789012345678
DISCORD_REPORTS_CHANNEL_ID=123456789012345678
```

### Database Connection Failed

**PostgreSQL:**
- Check connection string format
- Verify network connectivity
- Ensure SSL mode is correct (`?sslmode=require`)
- For Neon: use direct URL, not pooled

**SQLite:**
- Ensure `data/runtime/` directory is writable
- Check disk space

### API Rate Limits

**Symptoms:** Errors downloading data, "429 Too Many Requests"

**Solutions:**
- Upgrade to paid tier
- Use different provider
- Add delays between requests
- Cache data locally (already implemented)

## Next Steps

After environment setup:

1. Initialize database: `make db.init` or `python -m storage.db init`
2. Download sample data: `make data.crypto` and `make data.equities`
3. Run smoke test: `make env.smoke`
4. Start Discord bot: `python -m ui.discord_bot.main`

See [DATA_SETUP.md](DATA_SETUP.md) for data provider details and [OBSERVABILITY.md](OBSERVABILITY.md) for monitoring setup.
