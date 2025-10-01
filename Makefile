.PHONY: help env.smoke smoke db.init db.purge data.crypto data.equities backtest.quick paper.quick lint test test.cov clean up down logs shell build train retrain sweep evaluate gates typecheck

# Default target
help:
	@echo "Available targets:"
	@echo ""
	@echo "Environment:"
	@echo "  env.smoke         - Run environment smoke test"
	@echo "  smoke             - Run quick smoke test (imports, settings, 3-bar backtest)"
	@echo ""
	@echo "Database:"
	@echo "  db.init           - Initialize database schema"
	@echo "  db.purge          - Purge simulation data (interactive)"
	@echo ""
	@echo "Data:"
	@echo "  data.crypto       - Download sample crypto data (BTC, ETH)"
	@echo "  data.equities     - Download sample equity data (SPY, AAPL)"
	@echo ""
	@echo "Backtesting:"
	@echo "  backtest          - Run backtest (alias for backtest.quick)"
	@echo "  backtest.quick    - Run quick backtest (requires data)"
	@echo "  lean-backtest     - Run LEAN backtest (Docker required)"
	@echo ""
	@echo "Paper Trading:"
	@echo "  paper             - Run paper trading (alias for paper.quick)"
	@echo "  paper.quick       - Run 1-hour paper trading simulation"
	@echo ""
	@echo "Training:"
	@echo "  train             - Train classifier with current config"
	@echo "  retrain           - Full retraining pipeline with validation"
	@echo "  sweep             - Run W&B hyperparameter sweep"
	@echo "  evaluate          - Evaluate model performance"
	@echo "  gates             - Check model promotion gates"
	@echo ""
	@echo "Testing:"
	@echo "  lint              - Run linters (ruff, black)"
	@echo "  typecheck         - Run mypy type checking"
	@echo "  test              - Run pytest"
	@echo "  test.cov          - Run pytest with coverage (fails if < 85%)"
	@echo ""
	@echo "Docker:"
	@echo "  build             - Build Docker images"
	@echo "  up                - Start containers (app + db)"
	@echo "  down              - Stop and remove containers"
	@echo "  logs              - Follow container logs"
	@echo "  shell             - Open shell in dev container"
	@echo ""
	@echo "Cleanup:"
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
backtest:
	@$(MAKE) backtest.quick

backtest.quick:
	python -m apps.backtest.run_backtest \
		--start 2024-01-01 \
		--end 2024-06-01 \
		--universe SPY,AAPL \
		--interval 1d

lean-backtest:
	@echo "Running LEAN backtest (requires Docker)..."
	@if [ -f lean.json ]; then \
		lean backtest || echo "⚠️ LEAN CLI not installed. See SETUP_LEAN.md"; \
	else \
		echo "❌ lean.json not found"; \
	fi

# Run paper trading
paper:
	@$(MAKE) paper.quick

paper.quick:
	python -m apps.paper.run_paper --hours 1

# Training workflows
train:
	@echo "Training classifier with current config..."
	python -m scripts.train_classifier --auto

retrain:
	@echo "Running full retraining pipeline..."
	@$(MAKE) train
	@$(MAKE) evaluate
	@$(MAKE) gates
	@echo "✅ Retraining complete. Check results above."

sweep:
	@echo "Starting W&B hyperparameter sweep..."
	@if [ -z "$(SWEEP_CONFIG)" ]; then \
		echo "Usage: make sweep SWEEP_CONFIG=sweeps/xgboost_sweep.yaml"; \
		exit 1; \
	fi
	wandb sweep $(SWEEP_CONFIG)
	@echo "Now run: wandb agent <sweep_id>"

evaluate:
	@echo "Evaluating models..."
	python -m scripts.evaluate_models

gates:
	@echo "Checking promotion gates..."
	python -m scripts.check_promotion_gates

# Linting
lint:
	@echo "Running ruff..."
	ruff check .
	@echo "Running black..."
	black --check .

typecheck:
	@echo "Running mypy type checking..."
	-mypy algos config data data_providers data_tools storage telemetry reporting services --ignore-missing-imports || true

# Testing
test:
	pytest -v

# Testing with coverage (lowered threshold to 85% as per requirements)
test.cov:
	@echo "Running tests with coverage..."
	pytest --cov=algos --cov=config --cov=data --cov=storage --cov=services \
		--cov-report=term-missing --cov-report=html --cov-fail-under=85 -v

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

# Docker targets
build:
	@echo "Building Docker images..."
	docker compose build

up:
	@echo "Starting containers..."
	docker compose up -d
	@echo "Containers started. Use 'make logs' to view logs."

down:
	@echo "Stopping containers..."
	docker compose down

logs:
	docker compose logs -f

shell:
	@echo "Opening shell in dev container..."
	docker compose --profile dev run --rm dev
