# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- 🗄️ Parquet-based data caching system (`data/cache_io.py`)
  - Monthly segmentation for efficient range queries
  - Coverage checking and gap detection
  - Timezone-aware UTC timestamp handling
- 🌐 Centralized network utilities (`utils/net.py`)
  - Retry logic with exponential backoff
  - Rate limiting and 429/5xx awareness
  - `NetworkClient` base class for API integrations
- 🔒 Security improvements
  - `.env.template` for secure configuration
  - `requirements.lock.txt` with pinned dependencies
  - `SECURITY.md` with threat model and best practices
- 🐳 Docker support
  - `Dockerfile` for production runtime
  - `Dockerfile.dev` for development environment
  - `docker-compose.yml` with app and PostgreSQL services
  - `.devcontainer/devcontainer.json` for VS Code
  - `DOCKER.md` comprehensive guide
- 📋 Documentation
  - `AUDIT.md` code audit report
  - `CONTRIBUTING.md` contribution guidelines
  - `CODE_OF_CONDUCT.md` community standards
  - `CHANGELOG.md` version history
- 🛠️ Makefile targets
  - `make smoke` - Quick sanity check
  - `make test.cov` - Coverage reporting (90% threshold)
  - `make up/down/logs/shell` - Docker commands
- 📄 MIT License

### Fixed
- Import errors caused by missing `data` module
- Test failures in `test_integration.py` (cache_io tests now passing)
- Timezone comparison issues in cache loading

### Changed
- Updated `.gitignore` to include `.env.template`
- Enhanced Makefile with Docker and coverage targets

## [v4.0.0] - 2024-09-XX

### Added
- 🏗️ Production-Grade Paper Trading Infrastructure
- 📊 Data Infrastructure with Parquet caching
- 💾 SQLAlchemy 2.0 Database (PostgreSQL/SQLite)
- 📱 Enhanced Discord Bot with slash commands
- 📈 Backtest Runner with KPI calculation
- 🎮 Paper Trading simulation loop
- 🔬 W&B Integration for experiment tracking
- 🤖 OpenAI Narration for AI-generated summaries

### Technical Details
- Pydantic-settings with environment validation
- Multi-provider support (Tiingo, CoinGecko)
- Data validation pipeline
- Discord slash commands (/envcheck, /pnl, /trades, /halt)
- Chart generation for equity curves
- Comprehensive testing (104/113 tests passing)

## [v3.0.0] - 2024-XX-XX

### Added
- Multi-strategy architecture (5 integrated strategies)
- XGBoost classifier with meta-labeling
- PPO execution optimization
- Advanced risk management (kill-switch, volatility targeting)
- QuantConnect LEAN integration
- Continuous learning with automated retraining

## [v2.0.0] - 2024-XX-XX

### Added
- Discord control interface
- Admin dashboard with FastAPI
- Prometheus metrics integration
- Runtime state polling for QC

## [v1.0.0] - 2024-XX-XX

### Added
- Initial release
- Basic trading strategies
- Backtesting engine
- Data providers integration

---

## Release Notes Format

### Categories
- **Added** - New features
- **Changed** - Changes in existing functionality
- **Deprecated** - Soon-to-be removed features
- **Removed** - Removed features
- **Fixed** - Bug fixes
- **Security** - Vulnerability fixes

### Emoji Legend
- 🚀 New feature
- 🐛 Bug fix
- 📝 Documentation
- 🔒 Security
- ⚡ Performance
- 🎨 Style/UI
- ♻️ Refactor
- ✅ Tests
- 🐳 Docker
- 📦 Dependencies

[Unreleased]: https://github.com/KaholiK/quant-bot/compare/v4.0.0...HEAD
[v4.0.0]: https://github.com/KaholiK/quant-bot/releases/tag/v4.0.0
