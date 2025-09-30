# Troubleshooting Guide

Common issues and solutions for the quant-bot system.

## Discord Bot Issues

### Bot Not Responding to Commands

**Symptoms:**
- Type `/` but no commands appear
- Commands don't trigger any response

**Possible Causes & Solutions:**

1. **Bot Not Invited Properly**
   ```
   Solution: Reinvite bot with correct permissions
   - OAuth2 → URL Generator
   - Scopes: bot, applications.commands
   - Permissions: Send Messages, Embed Links, Attach Files
   ```

2. **Commands Not Synced**
   ```
   Check logs for:
   "Synced X command(s)"
   
   If missing, restart bot:
   python -m ui.discord_bot.main
   ```

3. **Wrong Guild**
   ```
   Verify DISCORD_GUILD_ID matches your server
   
   Get correct ID:
   - Enable Developer Mode
   - Right-click server → Copy ID
   ```

4. **Bot Offline**
   ```
   Check bot process is running:
   ps aux | grep discord_bot
   
   Check Discord for green status indicator
   ```

### Commands Show "Interaction Failed"

**Symptoms:**
- Commands appear but fail with "The application did not respond"

**Causes & Solutions:**

1. **Timeout (> 3 seconds)**
   ```python
   # Commands use defer for long operations
   await interaction.response.defer(ephemeral=True)
   # ... long operation ...
   await interaction.followup.send(...)
   ```

2. **Database Connection Failed**
   ```bash
   # Test DB connection
   python -c "from storage.db import get_engine; get_engine().connect()"
   
   # Check DATABASE_URL or SQLite path
   ```

3. **Missing Permissions**
   ```
   Bot needs:
   - Send Messages
   - Embed Links
   - Attach Files
   ```

### Bot Starts But No Boot Message

**Symptoms:**
- Bot shows online in Discord
- No boot message in reports channel

**Causes:**

1. **Wrong Channel ID**
   ```bash
   # Verify in .env:
   DISCORD_REPORTS_CHANNEL_ID=123456789012345678
   
   # Get correct ID:
   Right-click channel → Copy ID
   ```

2. **Bot Can't Access Channel**
   ```
   Check channel permissions:
   - Bot role has "View Channel"
   - Bot role has "Send Messages"
   ```

3. **Channel Not Set**
   ```bash
   # Bot warns but continues
   Check logs for:
   "Reports channel not configured"
   ```

### Intents Error on Startup

**Symptom:**
```
discord.errors.PrivilegedIntentsRequired
```

**Solution:**
```
Go to Discord Developer Portal:
1. Select your application
2. Bot → Privileged Gateway Intents
3. DISABLE "Message Content Intent" (not needed)
4. ENABLE "Server Members Intent" (if needed)
5. Save and restart bot
```

**Note:** Current bot uses minimal intents - no privileged intents needed!

## Database Issues

### SQLite: "Database is Locked"

**Symptoms:**
```
sqlite3.OperationalError: database is locked
```

**Causes:**
- Multiple processes accessing SQLite
- Long-running transactions

**Solutions:**

1. **Use PostgreSQL for Concurrent Access**
   ```bash
   # In .env:
   DATABASE_URL=postgresql://user:pass@host/db?sslmode=require
   ```

2. **Increase Timeout**
   ```python
   # In storage/db.py, timeout already set to 30s
   # If still issues, reduce concurrent processes
   ```

3. **Close Other Connections**
   ```bash
   # Find processes using DB
   lsof | grep quantbot.db
   
   # Kill if necessary
   kill <PID>
   ```

### PostgreSQL: Connection Failed

**Symptoms:**
```
could not connect to server: Connection refused
```

**Solutions:**

1. **Check Connection String**
   ```bash
   # Format:
   postgresql://user:password@host:5432/database?sslmode=require
   
   # Test with psql:
   psql "postgresql://user:password@host:5432/database?sslmode=require"
   ```

2. **SSL Mode Required**
   ```bash
   # Most cloud DBs require SSL
   DATABASE_URL=postgresql://...?sslmode=require
   ```

3. **Firewall/Network**
   ```bash
   # Test connectivity
   nc -zv host 5432
   
   # Check firewall rules allow connection
   ```

### Neon: "Pooler Connection" Warning

**Symptom:**
```
⚠️  Detected Neon pooled connection URL.
For migrations, use direct connection URL instead.
```

**Explanation:**
- Pooled URLs (containing `-pooler`) are fine for runtime
- Direct URLs needed for schema changes

**Solution:**
```bash
# For runtime (queries, inserts):
DATABASE_URL=postgresql://user:pass@ep-xxx-pooler.region.aws.neon.tech/db

# For migrations (schema changes):
DATABASE_URL=postgresql://user:pass@ep-xxx.region.aws.neon.tech/db
```

### Tables Don't Exist

**Symptoms:**
```
sqlalchemy.exc.ProgrammingError: relation "runs" does not exist
```

**Solution:**
```bash
# Initialize database
python -m storage.db init

# Or via Makefile
make db.init
```

## Data Provider Issues

### API Rate Limit Exceeded

**Symptoms:**
```
HTTP 429: Too Many Requests
```

**Solutions:**

1. **Wait for Reset**
   ```
   Free tiers reset:
   - Tiingo: Hourly
   - CoinGecko: Per minute
   - Alpha Vantage: Per minute (5 calls)
   ```

2. **Upgrade API Tier**
   ```
   Consider paid plans for higher limits
   ```

3. **Batch Requests**
   ```bash
   # Download one symbol at a time
   python -m data_tools.download_equities --symbols SPY --start ... --end ...
   python -m data_tools.download_equities --symbols AAPL --start ... --end ...
   ```

4. **Use Cached Data**
   ```python
   # Check cache first
   from data.cache_io import get_cache
   df = cache.load_range(asset, symbol, interval, start, end)
   ```

### Invalid API Key

**Symptoms:**
```
HTTP 401: Unauthorized
HTTP 403: Forbidden
```

**Solutions:**

1. **Verify Key in .env**
   ```bash
   # Check key is set
   python -m scripts.env_smoke
   
   # Should show masked key: ***abc1
   ```

2. **Regenerate Key**
   ```
   Go to provider dashboard
   Revoke old key
   Generate new key
   Update .env
   ```

3. **Check Key Type**
   ```
   Some providers have different keys for:
   - Free vs Paid
   - Test vs Production
   ```

### No Data Returned

**Symptoms:**
```
No data found for SYMBOL
```

**Causes:**

1. **Symbol Not Found**
   ```
   Check symbol format:
   - Tiingo: SPY, AAPL (uppercase)
   - CoinGecko: btc, eth (lowercase, mapped to IDs)
   ```

2. **Date Range Too Old/Recent**
   ```
   Provider limits:
   - CoinGecko 15m: Last 1 day only
   - CoinGecko 1h: Last 90 days
   - Free tiers: Limited history
   ```

3. **Delisted/Invalid Symbol**
   ```
   Try:
   - Different symbol
   - Different provider
   - Verify symbol exists
   ```

### Validation Errors

**Symptoms:**
```
ValueError: High must be >= Low
ValueError: NaN values found in OHLCV data
```

**Causes:**
- Bad data from provider
- Corporate actions (splits, etc.)

**Solutions:**

1. **Try Different Date Range**
   ```bash
   # Avoid dates around corporate actions
   # Skip problematic periods
   ```

2. **Use Different Provider**
   ```bash
   # Tiingo instead of Polygon
   # CoinGecko instead of CryptoCompare
   ```

3. **Report to Provider**
   ```
   Contact provider support
   Report data quality issue
   ```

## Configuration Issues

### Environment Variables Not Loaded

**Symptoms:**
- Settings show None/defaults
- "API key not configured" errors

**Solutions:**

1. **Load .env Manually**
   ```bash
   # Linux/Mac
   set -a
   source .env
   set +a
   
   # Or use direnv
   direnv allow
   ```

2. **Check .env Location**
   ```bash
   # Must be in project root
   ls -la .env
   
   # Or specify path
   export ENV_FILE=/path/to/.env
   ```

3. **Verify Syntax**
   ```bash
   # No spaces around =
   CORRECT=value
   WRONG = value  # This won't work
   
   # No quotes needed (but allowed)
   KEY=abc123
   KEY="abc123"  # Also works
   ```

### Settings Validation Errors

**Symptoms:**
```
pydantic.ValidationError: RUN_MODE must be 'paper' or 'live'
```

**Solution:**
```bash
# Check .env values match expected types
RUN_MODE=paper  # NOT "true" or "1"
DISCORD_GUILD_ID=123456789012345678  # Number, not string with quotes
```

## Runtime Issues

### Backtest Fails: No Data

**Symptoms:**
```
❌ No data available for backtest
```

**Solution:**
```bash
# Download data first
make data.equities
make data.crypto

# Or manually
python -m data_tools.download_equities --symbols SPY --start 2024-01-01 --end 2024-09-01
```

### Paper Trading: "Halt Flag Detected"

**Symptom:**
```
⚠️  Halt flag detected, stopping...
```

**Cause:**
- Someone used `/halt` Discord command
- Or manually set `bot.paper_loop_flag = True`

**Solution:**
```bash
# Restart paper trading
python -m apps.paper.run_paper --hours 1

# Flag is reset on startup
```

### Chart Generation Fails

**Symptoms:**
```
WARNING: Cannot create chart - matplotlib not available
```

**Solution:**
```bash
# Install matplotlib
pip install matplotlib

# Verify
python -c "import matplotlib; print('OK')"
```

### OpenAI Narration Fails

**Symptoms:**
```
WARNING: OpenAI summary failed, using fallback
```

**Causes:**
1. API key invalid
2. Rate limit exceeded
3. Network error

**Solutions:**

1. **Check API Key**
   ```bash
   # Verify in .env
   OPENAI_API_KEY=sk-...
   ```

2. **Fallback is Automatic**
   ```
   System uses simple summary if OpenAI fails
   No action needed
   ```

3. **Install openai Package**
   ```bash
   pip install openai
   ```

## Installation Issues

### Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'pydantic_settings'
ImportError: cannot import name 'BaseSettings'
```

**Solutions:**

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   
   # Or with editable install
   pip install -e .
   ```

2. **Check Python Version**
   ```bash
   python --version
   # Should be >= 3.11
   
   # Use correct version
   python3.11 -m pip install -r requirements.txt
   ```

3. **Virtual Environment**
   ```bash
   # Create venv
   python -m venv venv
   
   # Activate
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   
   # Install
   pip install -r requirements.txt
   ```

### Permission Denied

**Symptoms:**
```
PermissionError: [Errno 13] Permission denied: 'data/runtime/quantbot.db'
```

**Solutions:**

1. **Create Directories**
   ```bash
   mkdir -p data/runtime
   mkdir -p data/cache
   ```

2. **Fix Permissions**
   ```bash
   chmod 755 data/
   chmod 755 data/runtime/
   ```

3. **Run as User (Not Root)**
   ```bash
   # Don't use sudo for Python scripts
   python -m storage.db init  # CORRECT
   sudo python -m storage.db init  # WRONG
   ```

## Performance Issues

### Slow Data Downloads

**Causes:**
- Rate limiting
- Large date range
- Network latency

**Solutions:**

1. **Reduce Batch Size**
   ```bash
   # Fewer symbols per run
   python -m data_tools.download_equities --symbols SPY,AAPL
   # Instead of --symbols SPY,AAPL,MSFT,GOOGL,...
   ```

2. **Use Caching**
   ```
   Downloaded data is cached automatically
   Re-running same request uses cache
   ```

3. **Choose Right Interval**
   ```bash
   # Daily data faster than intraday
   --interval 1d  # Fast
   --interval 15m  # Slow (more data points)
   ```

### Backtest Runs Slowly

**Causes:**
- Large universe
- Long date range
- Complex calculations

**Solutions:**

1. **Reduce Universe**
   ```bash
   # Start with 2-3 symbols
   --universe SPY,QQQ
   ```

2. **Shorter Period**
   ```bash
   # Test with 6 months first
   --start 2024-03-01 --end 2024-09-01
   ```

3. **Use Daily Bars**
   ```bash
   # Daily faster than hourly
   --interval 1d
   ```

## Getting Help

### Check Logs

```bash
# Run with DEBUG level
export LOG_LEVEL=DEBUG
python -m ui.discord_bot.main

# Check for errors
grep ERROR logs/*.log
```

### Verify Environment

```bash
# Full environment check
python -m scripts.env_smoke

# Should show all green or acceptable warnings
```

### Test Components Individually

```bash
# Test DB
python -c "from storage.db import get_engine; get_engine().connect()"

# Test config
python -c "from config.settings import settings; print(settings.masked_dict())"

# Test data provider
python -c "from data_providers.coingecko_client import CoinGeckoClient; print('OK')"
```

### Common Error Messages

| Error | Likely Cause | Quick Fix |
|-------|--------------|-----------|
| "No module named..." | Missing dependency | `pip install -r requirements.txt` |
| "API key not configured" | Missing .env or not loaded | Check `.env` exists and is loaded |
| "Connection refused" | Database not accessible | Check `DATABASE_URL` or SQLite path |
| "Rate limit exceeded" | Too many API calls | Wait or upgrade plan |
| "No data found" | Symbol/date invalid | Check symbol format and date range |

### Still Stuck?

1. **Check existing issues:** https://github.com/KaholiK/quant-bot/issues
2. **Read docs again:** [SETUP_ENV.md](SETUP_ENV.md), [DATA_SETUP.md](DATA_SETUP.md)
3. **Run smoke test:** `make env.smoke`
4. **Enable debug logging:** `export LOG_LEVEL=DEBUG`
5. **Test minimal example:** Try quick backtest with one symbol

## Emergency Recovery

### Reset Everything

```bash
# 1. Stop all processes
pkill -f discord_bot
pkill -f run_paper

# 2. Purge database
python -m storage.db purge-sim

# 3. Reinitialize
python -m storage.db init

# 4. Clear cache (optional)
rm -rf data/cache/*

# 5. Reload environment
set -a; source .env; set +a

# 6. Test
python -m scripts.env_smoke
```

### Start Fresh

```bash
# 1. Backup .env
cp .env .env.backup

# 2. Remove virtual env
rm -rf venv/

# 3. Recreate
python -m venv venv
source venv/bin/activate

# 4. Reinstall
pip install -r requirements.txt

# 5. Restore .env
cp .env.backup .env

# 6. Initialize
make db.init
make env.smoke
```
