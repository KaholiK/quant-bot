# Quant Bot Self-Audit Report

Audit run on repository: /home/runner/work/quant-bot/quant-bot

## Results Summary

| Category | Test | Status |
|----------|------|--------|
| Files | pyproject.toml | PASS |
| Files | README.md | PASS |
| Files | config.yaml | PASS |
| Files | lean.json | PASS |
| Files | MainAlgo.py | PASS |
| Files | algos/core/feature_pipe.py | PASS |
| Files | algos/core/labels.py | PASS |
| Files | algos/core/cv_utils.py | PASS |
| Files | algos/core/risk.py | PASS |
| Files | algos/core/portfolio.py | PASS |
| Files | algos/core/exec_rl.py | PASS |
| Files | algos/strategies/scalper_sigma.py | PASS |
| Files | algos/strategies/trend_breakout.py | PASS |
| Files | algos/strategies/bull_mode.py | PASS |
| Files | algos/strategies/market_neutral.py | PASS |
| Files | algos/strategies/gamma_reversal.py | PASS |
| Files | notebooks/train_classifier.ipynb | PASS |
| Files | notebooks/train_ppo.ipynb | PASS |
| Files | tests/test_triple_barrier.py | PASS |
| Files | tests/test_purged_cv.py | PASS |
| Files | tests/test_risk_engine.py | PASS |
| Files | tests/test_*.py | PASS |
| Config | config.yaml loads with new loader | PASS |
| Config | trading section | PASS |
| Config | trading.universe | PASS |
| Config | trading.risk | PASS |
| Config | risk.per_trade_risk_pct in range | PASS |
| Config | risk.kill_switch_dd in range | PASS |
| Config | trading.models | PASS |
| Config | trading.strategies | PASS |
| Imports | algos.core.feature_pipe | SKIP: Missing dependencies (expected in CI) |
| Imports | algos.core.labels | SKIP: Missing dependencies (expected in CI) |
| Imports | algos.core.cv_utils | SKIP: Missing dependencies (expected in CI) |
| Imports | algos.core.risk | SKIP: Missing dependencies (expected in CI) |
| Imports | algos.core.portfolio | SKIP: Missing dependencies (expected in CI) |
| Imports | algos.core.exec_rl | SKIP: Missing dependencies (expected in CI) |
| Imports | algos.strategies.scalper_sigma | SKIP: Missing dependencies (expected in CI) |
| Imports | algos.strategies.trend_breakout | SKIP: Missing dependencies (expected in CI) |
| Imports | algos.strategies.bull_mode | SKIP: Missing dependencies (expected in CI) |
| Imports | algos.strategies.market_neutral | SKIP: Missing dependencies (expected in CI) |
| Imports | algos.strategies.gamma_reversal | SKIP: Missing dependencies (expected in CI) |

**Total Tests:** 41  
**Passed:** 30  
**Failed:** 11  
**Success Rate:** 73.2%  
