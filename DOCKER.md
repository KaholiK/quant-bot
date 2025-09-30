# Docker Guide

This guide covers running the quant-bot in Docker containers for development and production.

## Overview

The project provides three Docker configurations:

1. **Dockerfile** - Production runtime (slim, non-root user)
2. **Dockerfile.dev** - Development environment (with build tools)
3. **docker-compose.yml** - Multi-container setup (app + PostgreSQL)

## Quick Start

### Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- `.env` file (copy from `.env.template`)

### Start Containers

```bash
# 1. Create .env file
cp .env.template .env
# Edit .env with your API keys

# 2. Build images
make build

# 3. Start containers (app + database)
make up

# 4. View logs
make logs

# 5. Stop containers
make down
```

## Services

### App Service

**Purpose:** Runs the main application (smoke test by default)

**Configuration:**
- Image: Built from `Dockerfile`
- User: Non-root (`botuser`)
- Volumes:
  - `./data:/app/data` - Persistent data cache
  - `./logs:/app/logs` - Application logs
- Environment: Loaded from `.env`
- Database: PostgreSQL via `DATABASE_URL`

**Default Command:** `python -m scripts.env_smoke`

### Database Service

**Purpose:** PostgreSQL 15 for persistent storage

**Configuration:**
- Image: `postgres:15-alpine`
- Port: `5432` (exposed to host)
- Volume: `postgres_data` (persistent)
- Credentials:
  - User: `quantbot`
  - Password: From `POSTGRES_PASSWORD` env var (default: `quantbot123`)
  - Database: `quantbot`
- Health Check: Every 10s with `pg_isready`

### Dev Service (Optional)

**Purpose:** Interactive development environment

**Configuration:**
- Image: Built from `Dockerfile.dev`
- Profile: `dev` (must be explicitly enabled)
- Volumes:
  - `.:/app` - Live code mounting
  - `dev_cache:/home/botuser/.cache` - Persistent cache
- Features:
  - Build tools (gcc, make, git)
  - Interactive shell
  - Jupyter, ipython, ipdb

**Usage:**
```bash
# Start dev shell
make shell

# Or manually
docker compose --profile dev run --rm dev
```

## Common Tasks

### Run Environment Check

```bash
docker compose up app
# Or
docker compose run --rm app python -m scripts.env_smoke
```

### Initialize Database

```bash
docker compose run --rm app python -m storage.db init
```

### Download Data

```bash
# Crypto data
docker compose run --rm app python -m data_tools.download_crypto \
    --provider coingecko \
    --symbols btc,eth \
    --start 2024-01-01 \
    --end 2024-09-01 \
    --interval 1h

# Equity data
docker compose run --rm app python -m data_tools.download_equities \
    --provider tiingo \
    --symbols SPY,AAPL \
    --start 2023-01-01 \
    --end 2024-09-01 \
    --interval 1d
```

### Run Backtest

```bash
docker compose run --rm app python -m apps.backtest.run_backtest \
    --start 2024-01-01 \
    --end 2024-06-01 \
    --universe SPY,AAPL \
    --interval 1d
```

### Run Paper Trading

```bash
docker compose run --rm app python -m apps.paper.run_paper --hours 24
```

### Run Tests

```bash
docker compose run --rm app pytest -v
```

### Access Database

```bash
# Using psql
docker compose exec db psql -U quantbot -d quantbot

# Or from host (if port 5432 is exposed)
psql postgresql://quantbot:quantbot123@localhost:5432/quantbot
```

### View Logs

```bash
# Follow all logs
make logs

# Follow specific service
docker compose logs -f app
docker compose logs -f db

# View last N lines
docker compose logs --tail 100 app
```

## Configuration

### Environment Variables

Set in `.env` file:

```bash
# Core settings
APP_ENV=dev
RUN_MODE=paper
LOG_LEVEL=INFO

# Database (overrides default SQLite)
DATABASE_URL=postgresql://quantbot:password@db:5432/quantbot

# Data providers
TIINGO_API_KEY=your_key_here
COINGECKO_API_KEY=your_key_here

# Discord
DISCORD_BOT_TOKEN=your_token_here
DISCORD_GUILD_ID=123456789
DISCORD_REPORTS_CHANNEL_ID=987654321
```

### Volumes

**Application Data:**
- `./data` - Market data cache (parquet files)
- `./logs` - Application logs

**Database:**
- `postgres_data` - PostgreSQL data directory (persistent)

**Development:**
- `.:/app` - Live code mounting (dev profile only)
- `dev_cache` - Python cache (dev profile only)

## Production Deployment

### Security Checklist

- [ ] Use strong `POSTGRES_PASSWORD`
- [ ] Don't expose PostgreSQL port (remove `ports:` in compose file)
- [ ] Use secrets management (Docker secrets or vault)
- [ ] Run behind reverse proxy (nginx, traefik)
- [ ] Enable TLS for database connections
- [ ] Set `LOG_LEVEL=WARNING` or `ERROR`
- [ ] Use read-only volumes where possible
- [ ] Regular backups of `postgres_data` volume

### Example Production Compose

```yaml
version: '3.8'

services:
  app:
    image: ghcr.io/your-org/quant-bot:latest
    restart: always
    environment:
      - APP_ENV=prod
      - LOG_LEVEL=WARNING
    env_file:
      - /secure/path/to/.env
    volumes:
      - /mnt/data/quant-bot:/app/data:ro
      - /var/log/quant-bot:/app/logs
    depends_on:
      - db
    networks:
      - internal

  db:
    image: postgres:15-alpine
    restart: always
    environment:
      - POSTGRES_PASSWORD_FILE=/run/secrets/db_password
    secrets:
      - db_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - internal

secrets:
  db_password:
    file: /secure/path/to/db_password.txt

networks:
  internal:
    internal: true
```

### Backup & Restore

**Backup Database:**
```bash
# Backup to file
docker compose exec db pg_dump -U quantbot quantbot > backup.sql

# Or with compression
docker compose exec db pg_dump -U quantbot quantbot | gzip > backup.sql.gz
```

**Restore Database:**
```bash
# From backup file
docker compose exec -T db psql -U quantbot quantbot < backup.sql

# Or from compressed backup
gunzip -c backup.sql.gz | docker compose exec -T db psql -U quantbot quantbot
```

**Backup Data Cache:**
```bash
# Tar the data directory
tar czf data_backup.tar.gz ./data
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker compose logs app

# Check container status
docker compose ps

# Inspect container
docker compose exec app /bin/bash
```

### Database Connection Failed

```bash
# Check database is healthy
docker compose ps db

# Check database logs
docker compose logs db

# Test connection manually
docker compose exec app python -c "from config.settings import settings; print(settings.db_url_or_sqlite())"
```

### Permission Issues

```bash
# Check file ownership
ls -la data/ logs/

# Fix permissions (run as root)
docker compose run --user root app chown -R botuser:botuser /app/data /app/logs
```

### Clean Slate Restart

```bash
# Stop and remove everything
docker compose down -v

# Rebuild images
docker compose build --no-cache

# Start fresh
docker compose up -d
```

## Development Workflow

### Live Code Changes

When using dev profile with volume mounts:

```bash
# 1. Start dev shell
make shell

# 2. Make code changes on host (in your IDE)

# 3. Run tests in container
pytest -v

# 4. Changes are immediately reflected (no rebuild needed)
```

### Debugging

```bash
# Use ipdb for debugging
docker compose run --rm dev python -m ipdb -c continue your_script.py

# Or add breakpoint in code
import ipdb; ipdb.set_trace()

# Then run
docker compose run --rm dev python your_script.py
```

### Jupyter Notebook

```bash
# Start Jupyter in dev container
docker compose --profile dev run --rm -p 8888:8888 dev jupyter lab --ip 0.0.0.0 --allow-root

# Open browser to http://localhost:8888
```

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [SETUP_ENV.md](SETUP_ENV.md) - Environment configuration
- [SECURITY.md](SECURITY.md) - Security best practices

---

**Last Updated:** September 30, 2025
