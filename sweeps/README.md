# W&B Hyperparameter Sweeps

This directory contains Weights & Biases sweep configurations for hyperparameter optimization.

## Available Sweeps

### 1. XGBoost Classifier (`xgboost_sweep.yaml`)

Optimizes XGBoost hyperparameters for primary signal classifier.

**Tuned Parameters**:
- `max_depth`: Tree depth (3-8)
- `learning_rate`: Learning rate (0.01-0.3)
- `n_estimators`: Number of trees (50-500)
- `min_child_weight`: Minimum sum of instance weight in child
- `subsample`: Subsample ratio of training instances
- `colsample_bytree`: Subsample ratio of features
- `gamma`: Minimum loss reduction for split
- `reg_alpha`: L1 regularization
- `reg_lambda`: L2 regularization

**Labeling Parameters**:
- `tp_atr_mult`: Take profit multiplier (1.5-2.0)
- `sl_atr_mult`: Stop loss multiplier (0.8-1.2)
- `horizon_bars`: Time barrier (3-10 bars)

**Usage**:
```bash
# Initialize sweep
wandb sweep sweeps/xgboost_sweep.yaml

# Run agents (can run multiple in parallel)
wandb agent <sweep_id>

# Or run multiple agents
for i in {1..4}; do
  wandb agent <sweep_id> &
done
```

**Expected Results**:
- 50-100 runs for convergence
- Target: OOS Sharpe > 1.8
- Runtime: 2-4 hours (4 parallel agents)

---

### 2. LightGBM Classifier (`lightgbm_sweep.yaml`)

Optimizes LightGBM hyperparameters as alternative to XGBoost.

**Tuned Parameters**:
- `num_leaves`: Max number of leaves (15-255)
- `max_depth`: Tree depth (-1 for unlimited, 3-10)
- `learning_rate`: Learning rate (0.01-0.3)
- `n_estimators`: Number of trees (50-500)
- `min_child_samples`: Minimum data in leaf
- `subsample`: Subsample ratio
- `colsample_bytree`: Feature subsample ratio
- `reg_alpha`: L1 regularization
- `reg_lambda`: L2 regularization
- `min_split_gain`: Minimum gain for split

**Usage**:
```bash
wandb sweep sweeps/lightgbm_sweep.yaml
wandb agent <sweep_id>
```

**Expected Results**:
- Similar performance to XGBoost
- Often faster training
- Better for large datasets

---

### 3. PPO Execution Policy (`ppo_sweep.yaml`)

Optimizes PPO hyperparameters for RL-based execution optimization.

**Tuned Parameters**:
- `learning_rate`: Actor/Critic learning rate (1e-5 to 1e-3)
- `n_steps`: Steps per update (512-4096)
- `batch_size`: Minibatch size (64-512)
- `n_epochs`: Epochs per update (3-20)
- `gamma`: Discount factor (0.95-0.999)
- `gae_lambda`: GAE lambda (0.9-0.99)
- `clip_range`: PPO clip parameter (0.1-0.3)
- `ent_coef`: Entropy coefficient (1e-5 to 1e-2)
- `vf_coef`: Value function coefficient (0.1-1.0)
- `max_grad_norm`: Max gradient norm (0.3-1.0)
- `net_arch`: Network architecture ([64,64] to [256,256])

**Usage**:
```bash
wandb sweep sweeps/ppo_sweep.yaml
wandb agent <sweep_id>
```

**Expected Results**:
- 100+ runs for convergence
- Target: Mean reward > 0.8
- Runtime: 6-12 hours
- Reduces slippage by 20-40%

---

## Sweep Workflow

### 1. Initialize Sweep

```bash
# Create sweep on W&B server
wandb sweep sweeps/xgboost_sweep.yaml

# Output:
# wandb: Created sweep with ID: abc123def
# wandb: View sweep at: https://wandb.ai/user/quantbot/sweeps/abc123def
```

### 2. Run Agents

```bash
# Single agent
wandb agent user/quantbot/abc123def

# Multiple agents (parallel)
for i in {1..4}; do
  wandb agent user/quantbot/abc123def &
done
```

### 3. Monitor Progress

View sweep progress at: https://wandb.ai/user/quantbot/sweeps/abc123def

**Metrics to Watch**:
- `val/sharpe`: Out-of-sample Sharpe ratio
- `val/sortino`: Out-of-sample Sortino ratio  
- `val/max_dd`: Maximum drawdown
- `val/profit_factor`: Gross profits / gross losses

### 4. Select Best Model

```bash
# Best run is automatically highlighted in W&B UI
# Download best model
wandb artifact get user/quantbot/xgb_classifier:best -v

# Or use in code
import wandb
api = wandb.Api()
sweep = api.sweep("user/quantbot/abc123def")
best_run = sweep.best_run()
print(f"Best config: {best_run.config}")
```

### 5. Promote to Production

```bash
# Tag as candidate
wandb artifact link xgb_classifier:v5 candidate

# Test in paper trading
python -m apps.paper.run_paper --model xgb_classifier:candidate

# Promote to production
wandb artifact link xgb_classifier:v5 prod

# Or use Discord
/model promote xgb_classifier_v5
```

---

## Bayesian Optimization

All sweeps use Bayesian optimization for efficient search:

**How it Works**:
1. Random exploration (first 10-20 runs)
2. Build surrogate model of objective function
3. Sample next point with highest expected improvement
4. Update surrogate model
5. Repeat until budget exhausted

**Benefits**:
- More efficient than grid/random search
- Focuses on promising regions
- Typically finds good hyperparameters in 50-100 runs

**Alternatives**:
- `method: grid` - Exhaustive search (use for small spaces)
- `method: random` - Random sampling (good baseline)

---

## Early Termination

All sweeps use Hyperband for early stopping:

```yaml
early_terminate:
  type: hyperband
  min_iter: 3    # Minimum iterations before termination
  s: 2           # Reduction factor
  eta: 3         # Downsampling rate
```

**How it Works**:
- Kill runs that perform poorly early
- Allocates more resources to promising runs
- Speeds up sweep by 2-4x

**Example**:
- Run 1: Sharpe = 0.8 after 3 folds → Terminated early
- Run 2: Sharpe = 1.5 after 3 folds → Continue to 5 folds

---

## Parallel Sweeps

Run multiple agents for faster convergence:

```bash
# On single machine
for i in {1..4}; do
  CUDA_VISIBLE_DEVICES=$i wandb agent <sweep_id> &
done

# On multiple machines
# Machine 1
wandb agent <sweep_id>

# Machine 2
wandb agent <sweep_id>

# Both agents pull from same sweep queue
```

**Scaling Guidelines**:
- 1 agent: 10-15 runs/hour
- 4 agents: 40-60 runs/hour
- 10 agents: 100-150 runs/hour

---

## Custom Sweep

Create custom sweep for specific needs:

```yaml
program: scripts/train_custom.py
method: grid
metric:
  name: custom/metric
  goal: maximize

parameters:
  param1:
    values: [a, b, c]
  param2:
    min: 0
    max: 10
    distribution: uniform
```

**Advanced Options**:
- `command`: Override default command
- `controller`: Custom sweep controller
- `resume`: Resume previous sweep
- `count`: Max runs (default: unlimited)

---

## Troubleshooting

### Agent Not Starting

```bash
# Check W&B login
wandb login

# Verify sweep exists
wandb sweep --info <sweep_id>

# Check logs
wandb agent <sweep_id> --verbose
```

### Sweep Stuck

```bash
# Check queue
wandb sweep --queue <sweep_id>

# Cancel bad runs
wandb run cancel <run_id>

# Pause sweep
wandb sweep pause <sweep_id>

# Resume sweep
wandb sweep resume <sweep_id>
```

### Out of Memory

Reduce batch size or model complexity:

```yaml
parameters:
  batch_size:
    values: [32, 64]  # Reduce from [64, 128, 256]
  
  net_arch:
    values: ["[64, 64]"]  # Reduce from larger networks
```

---

## Best Practices

### 1. Start Small

Begin with narrow range:

```yaml
max_depth:
  values: [3, 5, 7]  # Narrow range
```

Then expand if needed:

```yaml
max_depth:
  values: [2, 3, 4, 5, 6, 7, 8, 10]  # Wider range
```

### 2. Use Log Scale

For learning rate and regularization:

```yaml
learning_rate:
  distribution: log_uniform_values
  min: 0.001
  max: 1.0
```

Not:

```yaml
learning_rate:
  distribution: uniform  # Bad for learning rate
  min: 0.001
  max: 1.0
```

### 3. Monitor Overfitting

Watch train vs. validation gap:

```python
wandb.log({
    "train/sharpe": train_sharpe,
    "val/sharpe": val_sharpe,
    "gap": train_sharpe - val_sharpe  # Should be small
})
```

### 4. Save Artifacts

Always save models:

```python
# Save model artifact
wandb.log_artifact(
    model_path,
    name=f"{model_type}_sweep",
    type="model",
    metadata=config
)
```

### 5. Document Results

Add notes to successful runs:

```python
# In W&B UI, add notes:
# "Best run so far. Good generalization. Try similar configs."
```

---

## See Also

- [TRAINING.md](../TRAINING.md) - Training workflows
- [telemetry/wandb_utils.py](../telemetry/wandb_utils.py) - W&B integration
- [scripts/train_classifier.py](../scripts/train_classifier.py) - Training script
- [W&B Sweeps Docs](https://docs.wandb.ai/guides/sweeps)
