# Security Policy

## Overview

This document outlines the security model and practices for the quant-bot trading system. The system is designed for **paper trading and backtesting only** - live trading is explicitly disabled.

## Threat Model

### Assets Protected

1. **API Keys & Secrets:** Data provider keys, Discord tokens, database credentials
2. **Trading Logic:** Proprietary strategies and ML models
3. **Historical Data:** Cached market data
4. **Backtest Results:** Performance metrics and trade history

### Threat Actors

1. **External Attackers:** Unauthorized access to secrets or data
2. **Insider Threats:** Accidental exposure of secrets in logs/commits
3. **Supply Chain:** Vulnerable dependencies
4. **Data Leakage:** Future-looking bias in backtests

### Attack Vectors

1. **Credential Exposure:** Secrets committed to version control
2. **Dependency Vulnerabilities:** Outdated packages with known CVEs
3. **API Key Leakage:** Secrets printed in logs or error messages
4. **SQL Injection:** Database queries with unsanitized input
5. **Path Traversal:** File operations with user-controlled paths

## Security Controls

### 1. Secrets Management ✅

**Controls:**
- `.env` file excluded from version control (`.gitignore`)
- `.env.template` provided as template (no real secrets)
- `Settings.masked_dict()` masks sensitive fields (shows only last 4 chars)
- Never print secrets in logs or error messages

**Usage:**
```python
from config.settings import settings

# ✅ Safe: Uses masked dict
print(settings.masked_dict())

# ❌ Unsafe: Direct access
print(settings.TIINGO_API_KEY)  # Never do this!
```

**Environment Variables:**
- `DISCORD_BOT_TOKEN` - Discord bot authentication
- `*_API_KEY` - Data provider keys
- `*_API_SECRET` - Broker secrets
- `DATABASE_URL` - Database connection string (may contain password)

### 2. Dependency Management ✅

**Controls:**
- `requirements.lock.txt` with pinned versions
- Regular dependency updates via Dependabot (TODO)
- Vulnerability scanning with pip-audit (TODO)

**Current Status:**
- ✅ Lock file generated
- 🔴 No automated vulnerability scanning yet
- 🔴 No Dependabot configuration yet

**TODO:**
```bash
# Add to CI pipeline
pip install pip-audit
pip-audit -r requirements.lock.txt
```

### 3. Input Validation ⚠️

**Controls:**
- Pydantic models for configuration validation
- SQLAlchemy ORM prevents SQL injection
- Path validation in cache_io (uses Path objects)

**Areas for Improvement:**
- Add explicit sanitization for user inputs in Discord commands
- Validate symbol names against allowlists
- Sanitize file paths in data downloads

### 4. Logging & PII 🟡

**Controls:**
- Structured logging with loguru
- `LOG_LEVEL` environment variable
- Masked secrets in Settings

**Areas for Review:**
- Audit all log statements for potential PII
- Ensure no account numbers or sensitive data logged
- Redact symbol-specific P&L if needed

**Best Practices:**
```python
from loguru import logger

# ✅ Safe logging
logger.info(f"Fetched data for {len(symbols)} symbols")

# ❌ Unsafe: Might leak sensitive info
logger.info(f"API key: {api_key}")  # Never do this!
```

### 5. Network Security ✅

**Controls:**
- HTTPS for all external API calls
- Retry logic with exponential backoff (`utils/net.py`)
- Rate limiting to avoid triggering provider bans
- Timeout configuration (default 30s)

**Usage:**
```python
from utils.net import NetworkClient

class MyProvider(NetworkClient):
    def __init__(self):
        super().__init__(
            base_url="https://api.provider.com",
            api_key=settings.PROVIDER_KEY,
            rate_limit=5.0,  # 5 calls/sec
            max_retries=3
        )
```

### 6. Database Security 🟡

**Controls:**
- SQLAlchemy ORM (parameterized queries)
- Optional PostgreSQL with encrypted connections
- SQLite fallback for development

**Areas for Improvement:**
- Add encryption at rest for sensitive data
- Implement access control for production deployments
- Regular backup strategy

## Security Checklist

### For Developers

- [ ] Never commit `.env` file
- [ ] Use `Settings.masked_dict()` when logging config
- [ ] Validate all user inputs (Discord commands, API parameters)
- [ ] Use parameterized queries (via ORM)
- [ ] Handle exceptions without exposing secrets
- [ ] Use `utils/net.py` for network calls with retry logic
- [ ] Keep dependencies up to date
- [ ] Review code for hardcoded secrets before commit

### For Deployment

- [ ] Generate unique `.env` from `.env.template`
- [ ] Set restrictive file permissions on `.env` (600)
- [ ] Use environment-specific secrets (dev vs prod)
- [ ] Enable LOG_LEVEL=WARNING or ERROR in production
- [ ] Configure database with encryption and backups
- [ ] Set up monitoring and alerting
- [ ] Regular security patches and updates

### Pre-Commit Checklist

```bash
# 1. Check for secrets
git diff --cached | grep -i "api.*key\|token\|secret\|password"

# 2. Install gitleaks (optional but recommended)
# brew install gitleaks  # macOS
# apt install gitleaks   # Ubuntu/Debian
gitleaks detect --no-git

# 3. Run linters
make lint

# 4. Run tests
make test
```

## Vulnerability Disclosure

### Reporting Security Issues

**Do NOT create public GitHub issues for security vulnerabilities.**

Please report security issues privately via:
- Email: [maintainer email - TODO]
- GitHub Security Advisories: [Use "Report a vulnerability" button]

### Response Timeline

- **Acknowledgment:** Within 48 hours
- **Initial Assessment:** Within 7 days
- **Fix & Disclosure:** Coordinated with reporter

### Scope

**In Scope:**
- Credential exposure in code or logs
- SQL injection vulnerabilities
- Path traversal issues
- Dependency vulnerabilities (if not already known)
- Data leakage in backtesting logic

**Out of Scope:**
- Issues requiring physical access
- Social engineering attacks
- DDoS attacks
- Issues in third-party services (report directly to them)

## Known Limitations

### Current Gaps (To Be Addressed)

1. **No Automated Vulnerability Scanning:** Need pip-audit in CI
2. **No Secret Scanning:** Need gitleaks or similar tool
3. **No Dependency Updates:** Need Dependabot configuration
4. **Limited Input Validation:** Discord commands need sanitization
5. **No Encryption at Rest:** SQLite files are not encrypted

### Accepted Risks

1. **Paper Trading Only:** System is not designed for live trading
2. **Local SQLite:** Development mode uses unencrypted SQLite
3. **No Authentication:** Discord bot relies on Discord's auth

## Compliance

### Data Privacy

- **User Data:** Discord usernames and command history (ephemeral)
- **Market Data:** Cached from providers (check provider ToS)
- **No PII:** System does not collect personal identifying information

### API Terms of Service

Users must comply with:
- Tiingo API Terms
- CoinGecko API Terms
- Discord Developer Terms
- Alpaca API Terms (if using paper trading)

**Responsibility:** Users are responsible for their own API key usage and rate limits.

## Security Updates

| Date | Version | Change |
|------|---------|--------|
| 2025-09-30 | Pre-v0.1.0 | Initial security policy |
| TBD | v0.1.0 | Add automated scanning |

## Additional Resources

- [SETUP_ENV.md](SETUP_ENV.md) - Environment configuration guide
- [AUDIT.md](AUDIT.md) - Code audit report
- [README.md](README.md) - General documentation

---

**Last Updated:** September 30, 2025  
**Next Review:** Before v0.1.0 release

## GitHub Actions Security

### Secrets Management

The repository uses GitHub Secrets for sensitive credentials in CI/CD workflows:

**Required secrets (add in repository Settings → Secrets and variables → Actions):**

- `DISCORD_BOT_TOKEN` - Discord bot authentication
- `DISCORD_GUILD_ID` - Discord server ID
- `DISCORD_REPORTS_CHANNEL_ID` - Channel for automated reports
- `DISCORD_WEBHOOK_URL` - Webhook for notifications
- `ALPACA_API_KEY_ID` - Alpaca paper trading key ID
- `ALPACA_API_SECRET_KEY` - Alpaca paper trading secret
- `POLYGON_API_KEY` - Polygon market data
- `ALPHAVANTAGE_API_KEY` - Alpha Vantage data (optional)
- `FRED_API_KEY` - FRED economic data (optional)
- `COINGECKO_API_KEY` - CoinGecko crypto data (optional)
- `TIINGO_API_KEY` - Tiingo market data (optional)
- `OPENAI_API_KEY` - OpenAI narration (optional)
- `WANDB_API_KEY` - Weights & Biases tracking (optional)
- `DATABASE_URL` - PostgreSQL connection string (optional)

**Security best practices:**

1. **Never log secrets** - Workflows create `.env` files from secrets without echoing values
2. **Use read-only permissions** - Workflows use `permissions: contents: read` by default
3. **Secrets rotation** - Rotate API keys quarterly or when compromised
4. **Minimal scope** - Use paper trading credentials, not production keys
5. **Audit access** - Review Actions logs regularly for unusual activity

### Workflow Security

**CI Workflow (`ci.yml`):**
- Runs on pull requests and pushes to main
- Read-only repository access
- Secrets only used for optional services (Discord notifications)
- All dependencies installed from pinned `requirements.lock.txt`
- Security audit via `pip-audit` on every run

**Retrain Workflow (`retrain.yml`):**
- Runs daily at 03:00 UTC or on manual dispatch
- Read-only repository access
- Secrets used for data providers and model storage
- Models validated before promotion
- Artifacts retained for 30 days with access control

**Concurrency controls:**
- Both workflows use `concurrency: group: workflow-${{ github.ref }}` to prevent overlapping runs
- `cancel-in-progress: true` stops duplicate runs

**Artifact security:**
- Coverage reports and model artifacts stored with GitHub's encryption
- 30-day retention period
- Access restricted to repository collaborators

### Local Development vs CI

**Local `.env` file:**
- Use `.env.template` as starting point
- Keep `.env` in `.gitignore` (already configured)
- Never commit credentials to version control

**CI environment:**
- Secrets injected at runtime via GitHub Actions
- `.env` file created dynamically in each workflow run
- File destroyed when workflow completes

### Incident Response

If credentials are compromised:

1. **Immediately rotate affected keys** in provider dashboards
2. **Update GitHub secrets** with new credentials
3. **Review Actions logs** for unauthorized access
4. **Check artifact downloads** in repository insights
5. **Audit recent workflow runs** for suspicious activity

Report security issues to: [SECURITY_CONTACT_EMAIL]

See [SETUP_ENV.md](SETUP_ENV.md) for detailed environment configuration.
