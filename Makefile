.PHONY: help env.smoke smoke db.init db.purge data.crypto data.equities backtest.quick paper.quick lint test test.cov clean

# Default target
help:
	@echo "Available targets:"
	@echo "  env.smoke         - Run environment smoke test"
	@echo "  smoke             - Run quick smoke test (imports, settings, 3-bar backtest)"
	@echo "  db.init           - Initialize database schema"
	@echo "  db.purge          - Purge simulation data (interactive)"
	@echo "  data.crypto       - Download sample crypto data (BTC, ETH)"
	@echo "  data.equities     - Download sample equity data (SPY, AAPL)"
	@echo "  backtest.quick    - Run quick backtest (requires data)"
	@echo "  paper.quick       - Run 1-hour paper trading simulation"
	@echo "  lint              - Run linters (ruff, black, mypy)"
	@echo "  test              - Run pytest"
	@echo "  test.cov          - Run pytest with coverage (fails if < 90%)"
	@echo "  clean             - Clean build artifacts and caches"

# Environment validation
env.smoke:
	python -m scripts.env_smoke

# Quick smoke test
smoke:
	@echo "Running smoke test..."
	@python -c "import config.settings; print('✅ Settings loaded')"
	@python -c "import data.cache_io; print('✅ Cache I/O loaded')"
	@python -c "import storage.db; print('✅ Database models loaded')"
	@python -c "import data_tools.validate; print('✅ Validation tools loaded')"
	@echo "✅ All critical modules import successfully"

# Database management
db.init:
	python -m storage.db init

db.purge:
	python -m storage.db purge-sim

# Data downloads (sample)
data.crypto:
	python -m data_tools.download_crypto \
		--provider coingecko \
		--symbols btc,eth \
		--start 2024-01-01 \
		--end 2024-09-01 \
		--interval 1h

data.equities:
	python -m data_tools.download_equities \
		--provider tiingo \
		--symbols SPY,AAPL \
		--start 2023-01-01 \
		--end 2024-09-01 \
		--interval 1d

# Run backtest
backtest.quick:
	python -m apps.backtest.run_backtest \
		--start 2024-01-01 \
		--end 2024-06-01 \
		--universe SPY,AAPL \
		--interval 1d

# Run paper trading
paper.quick:
	python -m apps.paper.run_paper --hours 1

# Linting
lint:
	@echo "Running ruff..."
	ruff check .
	@echo "Running black..."
	black --check .
	@echo "Running mypy (optional)..."
	-mypy config data_providers data_tools storage telemetry reporting || true

# Testing
test:
	pytest -v

# Testing with coverage
test.cov:
	@echo "Running tests with coverage..."
	pytest --cov=. --cov-report=term-missing --cov-report=html --cov-fail-under=90 -v

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Clean complete"
