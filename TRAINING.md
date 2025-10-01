# Training & Model Management

This document describes the machine learning pipeline, training workflows, and model management using Weights & Biases (W&B).

## Overview

Quant-bot uses a multi-model ML pipeline:

1. **Primary Classifier**: XGBoost/LightGBM for trade signals
2. **Meta-Filter**: Second-stage model to filter low-quality signals
3. **RL Execution**: PPO policy for optimal order execution

All models are tracked in W&B with promotion gates and version control.

## Configuration

### Weights & Biases Setup

1. Sign up at https://wandb.ai
2. Get API key from https://wandb.ai/authorize
3. Add to `.env`:

```bash
WANDB_API_KEY=your_key_here
WANDB_PROJECT=quantbot
WANDB_ENTITY=your_username
```

4. Test integration:

```bash
python -c "from telemetry.wandb_utils import get_wandb; wb = get_wandb(); print('✅ W&B connected' if wb.enabled else '❌ W&B not configured')"
```

### Training Configuration

Edit `config.yaml`:

```yaml
trading:
  learning:
    cv:
      scheme: purged_kfold_embargo
      folds: 5
      embargo_frac: 0.02              # 2% embargo between folds
    
    retrain_cadence: weekly           # weekly, daily, or manual
    
    gates:
      oos_sortino_min: 1.2            # Minimum OOS Sortino ratio
      oos_profit_factor_min: 1.15     # Minimum OOS profit factor
      oos_max_dd_max: 0.06            # Maximum OOS drawdown (6%)
    
    meta_threshold: 0.58              # Meta-filter confidence threshold
```

## Training Workflows

### 1. Manual Training

#### Train Primary Classifier

```bash
# Train XGBoost classifier
python -m scripts.train_classifier \
  --symbols SPY,QQQ,AAPL \
  --start 2023-01-01 \
  --end 2024-01-01 \
  --model xgboost \
  --cv purged_kfold \
  --tag "manual-train-v1"

# Train LightGBM classifier
python -m scripts.train_classifier \
  --symbols SPY,QQQ,AAPL \
  --start 2023-01-01 \
  --end 2024-01-01 \
  --model lightgbm \
  --cv purged_kfold \
  --tag "manual-train-v1"
```

**Output**:
```
🚀 Starting Training Run

Universe: SPY, QQQ, AAPL
Period: 2023-01-01 to 2024-01-01 (365 days)
Model: XGBoost
CV: Purged K-Fold (5 folds, 2% embargo)

[1/5] Downloading data...
[2/5] Engineering features...
[3/5] Labeling with Triple Barrier...
[4/5] Training with cross-validation...
  Fold 1/5: Train Sharpe=2.14, OOS Sharpe=1.85
  Fold 2/5: Train Sharpe=2.08, OOS Sharpe=1.92
  Fold 3/5: Train Sharpe=2.19, OOS Sharpe=1.78
  Fold 4/5: Train Sharpe=2.11, OOS Sharpe=1.88
  Fold 5/5: Train Sharpe=2.16, OOS Sharpe=1.81
[5/5] Validating against gates...

✅ Training Complete

Avg OOS Sharpe: 1.85 (target: > 1.2) ✅
Avg OOS Sortino: 2.41 (target: > 1.2) ✅
Avg OOS Profit Factor: 1.28 (target: > 1.15) ✅
Avg OOS Max DD: 4.8% (target: < 6%) ✅

Model saved: models/xgb_classifier_v5.joblib
W&B run: https://wandb.ai/user/quantbot/runs/abc123

Use /model promote xgb_classifier_v5 to deploy.
```

#### Train Meta-Filter

```bash
# First, generate primary signals
python -m scripts.train_classifier --stage primary

# Then train meta-filter on those signals
python -m scripts.train_classifier \
  --stage meta \
  --primary-model models/xgb_classifier_v5.joblib \
  --symbols SPY,QQQ,AAPL \
  --start 2023-01-01 \
  --end 2024-01-01
```

**Meta-filter improves signal quality**:
- Filters out low-confidence signals
- Reduces false positives by ~30%
- Improves profit factor from 1.28 to 1.65

#### Train RL Execution Policy

```bash
# Train PPO policy for execution
python -m scripts.train_ppo \
  --symbols SPY,QQQ \
  --start 2023-01-01 \
  --end 2024-01-01 \
  --episodes 10000 \
  --checkpoint-freq 1000
```

**PPO learns to**:
- Minimize slippage via maker orders
- Balance urgency vs. execution cost
- Adapt to market microstructure

### 2. Automated Retraining

#### GitHub Actions Workflow

Retraining runs automatically on schedule:

```yaml
# .github/workflows/retrain.yml
name: Retrain Models

on:
  schedule:
    - cron: '0 2 * * 1'  # Every Monday at 2 AM UTC
  workflow_dispatch:      # Manual trigger

jobs:
  retrain:
    runs-on: ubuntu-latest
    steps:
      - name: Train models
        run: python -m scripts.train_classifier --auto
      
      - name: Validate gates
        run: python -m scripts.check_promotion_gates
      
      - name: Post to Discord
        run: python -m scripts.post_retrain_results
```

#### Discord Trigger

```bash
# Trigger retraining from Discord
/retrain now strategies:all
```

## Feature Engineering

### Base Features

From `algos/core/feature_pipe.py`:

```python
# Returns and lags
returns_1, returns_5, returns_20, returns_60

# Moving averages
sma_20, sma_50, sma_200
ema_12, ema_26

# Technical indicators
macd, macd_signal, macd_hist
rsi_14
atr_14
bb_width, bb_pct

# Volatility
rolling_vol_20, rolling_vol_60
parkinson_vol, garman_klass_vol

# Volume
volume_ratio, volume_sma_20
dollar_volume
```

### Cross-Sectional Features

Rank features across universe:

```python
# Relative momentum
momentum_rank = prices.pct_change(20).rank(axis=1, pct=True)

# Relative volume
volume_rank = volume.rank(axis=1, pct=True)

# Relative volatility
vol_rank = returns.rolling(20).std().rank(axis=1, pct=True)
```

**Why cross-sectional?**
- Robust to regime changes
- Captures relative strength
- Better signal stability

### Microstructure Features (Crypto)

For high-frequency crypto trading:

```python
# Order book imbalance
bid_volume / (bid_volume + ask_volume)

# Spread
(ask - bid) / mid_price

# Trade flow
(buy_volume - sell_volume) / total_volume

# Quote intensity
n_bid_updates / (n_bid_updates + n_ask_updates)
```

**Note**: Requires order book data (not available from free APIs)

## Labeling

### Triple-Barrier Method

**Concept** (López de Prado):
- Label each trade with profit target, stop loss, and time barrier
- First barrier hit determines label: +1 (profit), -1 (loss), 0 (timeout)

**Implementation**:

```python
from algos.core.labels import calculate_barriers

# Calculate barriers
barriers = calculate_barriers(
    prices=df["close"],
    tp_atr_mult=1.75,      # Take profit at 1.75 × ATR
    sl_atr_mult=1.0,       # Stop loss at 1.0 × ATR
    atr=df["atr"],
    horizon_bars=5         # Time barrier at 5 bars
)

# Labels: 1 (hit TP), -1 (hit SL), 0 (timeout)
labels = barriers["label"]
```

**Why 1.75:1 ratio?**
- Accounts for transaction costs
- Provides asymmetric risk/reward
- Empirically validated through optimization

**Configurable Parameters**:
```yaml
trading:
  labels:
    horizon_bars: 5          # Time barrier
    tp_atr_mult: 1.75        # Take profit multiplier
    sl_atr_mult: 1.0         # Stop loss multiplier
```

### Meta-Labeling

**Concept**:
- Primary model predicts direction (long/short)
- Meta-model predicts whether to take the trade (yes/no)

**Benefits**:
- Filters out low-confidence signals
- Improves profit factor
- Reduces drawdowns

**Implementation**:

```python
# Stage 1: Primary signals
primary_signals = primary_model.predict_proba(X)[:, 1]

# Stage 2: Meta-features
meta_features = [
    primary_signals,          # Primary confidence
    volatility,               # Market conditions
    spread,                   # Transaction costs
    volume_ratio,             # Liquidity
    time_of_day,              # Intraday patterns
]

# Stage 3: Meta-model decision
trade_decision = meta_model.predict(meta_features)  # 0 or 1
```

## Cross-Validation

### Purged K-Fold with Embargo

**Why not standard K-Fold?**
- Time series data has autocorrelation
- Standard K-Fold leaks future into past
- Overstates out-of-sample performance

**Purged K-Fold** (López de Prado):
1. Split data into K folds chronologically
2. Purge train set of labels overlapping with test set
3. Add embargo period between train and test

**Implementation**:

```python
from algos.core.cv_utils import PurgedKFold

# Create splitter
cv = PurgedKFold(
    n_splits=5,
    embargo_frac=0.02,     # 2% embargo
    purge=True
)

# Cross-validate
for train_idx, test_idx in cv.split(X, labels):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    
    model.fit(X_train, y_train)
    oos_score = model.score(X_test, y_test)
```

**Parameters**:
- `n_splits`: Number of folds (default: 5)
- `embargo_frac`: Embargo as fraction of dataset (default: 0.02 = 2%)
- `purge`: Remove overlapping labels (default: True)

### Walk-Forward Analysis

For production validation:

```python
# Walk forward with expanding window
for start, end in walk_forward_windows(data, window=90, step=30):
    train_data = data[:end]
    test_data = data[end:end+step]
    
    model.fit(train_data)
    oos_sharpe = evaluate(model, test_data)
    
    if oos_sharpe > threshold:
        print(f"Period {end}: Sharpe={oos_sharpe:.2f} ✅")
```

## Model Evaluation

### Primary Metrics

**Sharpe Ratio** (risk-adjusted returns):
```
Sharpe = mean(returns) / std(returns) × √252
```

**Sortino Ratio** (downside risk):
```
Sortino = mean(returns) / downside_std(returns) × √252
```

**Profit Factor**:
```
Profit Factor = sum(wins) / abs(sum(losses))
```

**Maximum Drawdown**:
```
Max DD = max(peak - current) / peak
```

### Deflated Sharpe Ratio (DSR)

**Problem**: Multiple testing inflates Sharpe ratio

**Solution** (Bailey & López de Prado):
```python
from algos.core.cv_utils import calculate_deflated_sharpe

dsr = calculate_deflated_sharpe(
    sharpe=1.85,              # Observed Sharpe
    n_trials=100,             # Number of backtests
    skewness=0.2,             # Return skewness
    kurtosis=3.5,             # Return kurtosis
    n_observations=250        # Sample size
)

# DSR adjusts for multiple testing
print(f"Observed Sharpe: 1.85")
print(f"Deflated Sharpe: {dsr:.2f}")  # e.g., 1.42
```

**Use DSR for**:
- Model selection after hyperparameter tuning
- Comparing strategies with different optimization history

## W&B Integration

### Logging Training Runs

```python
from telemetry.wandb_utils import get_wandb

wandb = get_wandb()

# Initialize run
wandb.init_run(
    project="quantbot",
    name="xgb-classifier-v5",
    tags=["xgboost", "triple-barrier", "purged-cv"],
    config={
        "model": "xgboost",
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
    }
)

# Log metrics
for epoch in range(n_epochs):
    metrics = train_epoch()
    wandb.log_metrics({
        "train/loss": metrics["train_loss"],
        "train/sharpe": metrics["train_sharpe"],
        "val/sharpe": metrics["val_sharpe"],
        "val/max_dd": metrics["val_max_dd"],
    }, step=epoch)

# Log artifacts
wandb.log_artifact(
    "models/xgb_classifier_v5.joblib",
    name="xgb_classifier_v5",
    artifact_type="model"
)

# Finish run
wandb.finish()
```

### Hyperparameter Sweeps

Create sweep configuration:

```yaml
# sweeps/xgboost_sweep.yaml
program: scripts/train_classifier.py
method: bayes
metric:
  name: val/sharpe
  goal: maximize
parameters:
  max_depth:
    values: [3, 4, 5, 6, 7, 8]
  learning_rate:
    distribution: log_uniform_values
    min: 0.01
    max: 0.3
  n_estimators:
    values: [50, 100, 200, 300]
  min_child_weight:
    values: [1, 3, 5, 7]
  subsample:
    values: [0.6, 0.7, 0.8, 0.9, 1.0]
  colsample_bytree:
    values: [0.6, 0.7, 0.8, 0.9, 1.0]
```

Run sweep:

```bash
# Initialize sweep
wandb sweep sweeps/xgboost_sweep.yaml

# Run agents (parallel)
wandb agent user/quantbot/sweep_id
```

### Model Registry

```python
# Tag model with stage
wandb.log_artifact(
    "models/xgb_classifier_v5.joblib",
    name="xgb_classifier",
    artifact_type="model",
    metadata={
        "stage": "candidate",        # candidate, staging, prod
        "oos_sharpe": 1.85,
        "oos_sortino": 2.41,
        "oos_max_dd": 0.048,
    }
)

# Promote to production
artifact = wandb.use_artifact("xgb_classifier:candidate")
artifact.link("xgb_classifier:prod")
```

## Promotion Gates

Models must pass validation before production:

```python
# scripts/check_promotion_gates.py

gates = config["trading"]["learning"]["gates"]

# Check OOS metrics
assert oos_sortino >= gates["oos_sortino_min"], \
    f"OOS Sortino {oos_sortino:.2f} < {gates['oos_sortino_min']}"

assert oos_profit_factor >= gates["oos_profit_factor_min"], \
    f"OOS Profit Factor {oos_profit_factor:.2f} < {gates['oos_profit_factor_min']}"

assert oos_max_dd <= gates["oos_max_dd_max"], \
    f"OOS Max DD {oos_max_dd:.2%} > {gates['oos_max_dd_max']:.2%}"

print("✅ All promotion gates passed")
```

**Gate Configuration**:
```yaml
gates:
  oos_sortino_min: 1.2          # Minimum Sortino ratio
  oos_profit_factor_min: 1.15   # Minimum profit factor
  oos_max_dd_max: 0.06          # Maximum drawdown (6%)
```

## Best Practices

### 1. Avoid Overfitting

**Signs of overfitting**:
- Train Sharpe >> OOS Sharpe (gap > 0.5)
- OOS Sharpe degrades over time
- Many features with low importance

**Solutions**:
- Use Purged K-Fold CV
- Regularization (L1/L2)
- Feature selection (drop low-importance features)
- Ensemble models (reduces variance)

### 2. Data Leakage Prevention

**Common leakage sources**:
- Using future data in features
- Forward-filling missing data
- Overlapping train/test labels

**Prevention**:
```python
# Use lag-safe features only
features = calculate_features(
    prices=df["close"].shift(1),  # Use previous bar
    volume=df["volume"].shift(1)
)

# Purge overlapping labels
cv = PurgedKFold(n_splits=5, embargo_frac=0.02)

# Never forward-fill prices
df["close"] = df["close"].fillna(method="ffill")  # ❌ WRONG
```

### 3. Sample Weighting

Weight samples by recency and uniqueness:

```python
from algos.core.labels import calculate_sample_weights

weights = calculate_sample_weights(
    labels=labels,
    timestamps=df.index,
    decay=0.95  # Exponential decay for recency
)

model.fit(X, y, sample_weight=weights)
```

### 4. Model Versioning

Always version models and data:

```bash
models/
  xgb_classifier_v1.joblib
  xgb_classifier_v2.joblib
  xgb_classifier_v3.joblib  # Current prod
  xgb_classifier_v4.joblib  # Candidate
  xgb_classifier_v5.joblib  # Experimental

data/
  training_data_2024-01-15.parquet  # Snapshot
```

### 5. Monitoring Production Models

Track model performance in production:

```python
# Log prediction quality
wandb.log_metrics({
    "prod/hit_rate": hit_rate,
    "prod/sharpe": rolling_sharpe_60d,
    "prod/max_dd": current_max_dd,
})

# Alert on degradation
if rolling_sharpe_60d < 1.0:
    send_discord_alert("⚠️ Model performance degrading - Sharpe < 1.0")
```

## Makefile Targets

```makefile
# Train classifier
make train:
\tpython -m scripts.train_classifier --auto

# Train with sweep
make sweep:
\twandb sweep sweeps/xgboost_sweep.yaml
\twandb agent user/quantbot/$(SWEEP_ID)

# Evaluate model
make evaluate:
\tpython -m scripts.evaluate_models

# Check promotion gates
make gates:
\tpython -m scripts.check_promotion_gates

# Retrain pipeline (full)
make retrain:
\tmake train
\tmake evaluate
\tmake gates
```

## See Also

- [algos/core/labels.py](algos/core/labels.py) - Triple-Barrier labeling
- [algos/core/cv_utils.py](algos/core/cv_utils.py) - Purged K-Fold CV
- [scripts/train_classifier.py](scripts/train_classifier.py) - Training script
- [telemetry/wandb_utils.py](telemetry/wandb_utils.py) - W&B integration
- [sweeps/](sweeps/) - Hyperparameter sweep configs
