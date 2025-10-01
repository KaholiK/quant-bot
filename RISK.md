# Risk Management Documentation

This document describes the risk management framework implemented in quant-bot, following industry best practices and López de Prado principles.

## Core Principles

1. **1% Rule**: Risk no more than 1% of equity per trade (configurable)
2. **Volatility Targeting**: Maintain 10-12% annualized portfolio volatility
3. **Leverage Caps**: Maximum 2x gross leverage
4. **Concentration Limits**: No single position > 10% of portfolio
5. **Circuit Breakers**: Automatic halt at 20% drawdown
6. **Fractional Kelly**: Use 0.25x Kelly for position sizing (conservative)

## Configuration

Default risk parameters in `config.yaml`:

```yaml
trading:
  risk:
    # Position sizing
    per_trade_risk_pct: 0.01        # 1% of equity per trade
    max_leverage: 2.0                # 2x gross leverage cap
    single_name_max_pct: 0.10        # 10% max per symbol
    sector_max_pct: 0.30             # 30% max per sector
    
    # Volatility targeting
    vol_target_ann: 0.12             # 12% annualized target
    
    # Kill switch
    kill_switch_dd: 0.20             # 20% equity drawdown halt
    day_stop_dd: 0.06                # 6% daily stop
    week_stop_dd: 0.10               # 10% weekly stop
    
    # Asset class limits
    asset_class_caps:
      crypto_max_gross_pct: 0.50     # 50% max crypto exposure
    
    # Blackout windows
    blackout:
      earnings: true                 # Skip earnings days
      fomc: true                     # Skip FOMC days
```

## Risk Formulas

### 1. Position Sizing (ATR-Based)

**Formula**:
```
Position Size = (Account Equity × Risk %) / (ATR × ATR Multiplier)
```

**Example**:
```
Equity: $100,000
Risk %: 1%
ATR: $2.50
Stop Multiplier: 1.0

Position Size = ($100,000 × 0.01) / ($2.50 × 1.0) = 400 shares
Risk Amount: $1,000
```

**Implementation**:
```python
from algos.core.risk import RiskManager

risk = RiskManager(config)
size = risk.calculate_position_size(
    symbol="SPY",
    current_price=450.0,
    atr=2.50,
    stop_loss_atr=1.0
)
# Returns: 400 shares (risking $1,000 = 1% of $100k)
```

### 2. Fractional Kelly Sizing

**Formula**:
```
Kelly % = (p × b - q) / b
where:
  p = win probability
  b = win amount / loss amount
  q = 1 - p

Position Size = Kelly % × Fraction (default: 0.25)
```

**Example**:
```
Win Rate: 55%
Win/Loss Ratio: 1.5
Kelly: (0.55 × 1.5 - 0.45) / 1.5 = 0.25 = 25%
Fractional Kelly (0.25x): 6.25%
```

**Why Fractional?**
- Full Kelly is aggressive and assumes perfect parameter estimation
- 0.25x Kelly (quarter-Kelly) provides 3/4 of growth with 1/16 of variance
- Recommended by Edward Thorp and empirical studies
- More robust to estimation errors

**Implementation**:
```python
kelly_fraction = risk.calculate_kelly_fraction(
    win_rate=0.55,
    avg_win=0.015,
    avg_loss=0.010,
    fraction=0.25  # Use quarter-Kelly
)
# Returns: 0.0625 (6.25% position size)
```

### 3. Volatility Targeting

**Formula**:
```
Target Position = (Target Vol / Realized Vol) × Base Position

where:
  Target Vol = 0.12 (12% annualized)
  Realized Vol = Rolling 20-day standard deviation × √252
```

**Example**:
```
Target Vol: 12%
Current Vol: 18%
Base Position: 10%

Scaled Position = (0.12 / 0.18) × 0.10 = 6.67%
```

**Implementation**:
```python
vol_scalar = risk.calculate_volatility_scalar(
    returns=portfolio_returns,  # Recent returns
    window=20,                  # Rolling window
    target_vol=0.12            # 12% annualized
)
adjusted_size = base_size * vol_scalar
```

### 4. Stop Loss Levels

**Triple-Barrier Approach** (López de Prado):

```
Take Profit:  Entry + (1.75 × ATR)
Stop Loss:    Entry - (1.0 × ATR)
Time Exit:    5 bars (configurable)
```

**Why 1.75:1 ratio?**
- Asymmetric risk/reward favors profitability
- Accounts for transaction costs
- Validated through backtest optimization

**Trailing Stop**:
```
Initial Stop: Entry - 1.0 × ATR
After +1 ATR profit: Trailing stop at Entry + 0.5 × ATR
After +2 ATR profit: Trailing stop at Entry + 1.5 × ATR
```

**Implementation**:
```python
from algos.core.labels import calculate_barriers

barriers = calculate_barriers(
    prices=df["close"],
    tp_atr_mult=1.75,  # Take profit at 1.75 × ATR
    sl_atr_mult=1.0,   # Stop loss at 1.0 × ATR
    atr=df["atr"],
    horizon_bars=5
)
```

## Risk Limits

### 1. Portfolio-Level Limits

**Gross Leverage**:
```python
gross_exposure = sum(abs(position_value) for position in portfolio)
leverage = gross_exposure / portfolio_value
assert leverage <= 2.0, "Leverage limit exceeded"
```

**Net Exposure** (for long-short strategies):
```python
net_exposure = sum(position_value for position in portfolio)
net_leverage = abs(net_exposure) / portfolio_value
# Target: -0.2 to +0.8 depending on regime
```

**Volatility Check**:
```python
realized_vol = portfolio_returns.rolling(20).std() * np.sqrt(252)
assert realized_vol <= 0.15, "Volatility target exceeded"
```

### 2. Position-Level Limits

**Single Name Concentration**:
```python
max_position_value = portfolio_value × 0.10  # 10% max
assert abs(position_value) <= max_position_value
```

**Sector Concentration**:
```python
sector_exposure = sum(position_value for pos in sector_positions)
assert sector_exposure <= portfolio_value × 0.30  # 30% max
```

**Crypto Concentration** (higher volatility):
```python
crypto_exposure = sum(abs(pos_value) for pos in crypto_positions)
assert crypto_exposure <= portfolio_value × 0.50  # 50% max
```

### 3. Trade-Level Limits

**Minimum Position Size**:
```python
min_trade_value = 100.0  # $100 minimum (avoid tiny positions)
```

**Maximum Order Size** (market impact):
```python
max_order_size = adv × 0.05  # Max 5% of average daily volume
```

**Slippage Budget**:
```python
expected_slippage = 0.0010  # 10 bps for liquid stocks
max_slippage = 0.0050       # 50 bps for illiquid stocks
```

## Circuit Breakers

### 1. Kill Switch (Global Halt)

**Trigger Conditions**:
- Portfolio drawdown exceeds 20% from peak
- Requires manual resume via Discord `/resume`
- All open positions remain (no panic liquidation)
- New orders blocked until reset

**Implementation**:
```python
risk = RiskManager(config)

# Check before each trade
if risk.is_kill_switch_active():
    logger.error("🛑 Kill switch active - no trading allowed")
    return

# Update equity curve
risk.update_equity(current_equity)

# Kill switch activates automatically at 20% DD
if risk.check_drawdown_limits():
    # Send Discord alert
    await discord_bot.send_alert(
        "🚨 KILL SWITCH ACTIVATED - 20% drawdown exceeded"
    )
```

**Resume Process**:
1. Analyze what went wrong
2. Fix issues (bad strategy, data error, etc.)
3. Run `/resume` in Discord (admin only)
4. Confirm resume intent

### 2. Daily Stop Loss

**Trigger**: 6% intraday drawdown from day's start

```python
# Check at each trade
daily_pnl_pct = (current_equity - start_of_day_equity) / start_of_day_equity

if daily_pnl_pct <= -0.06:
    logger.warning("⚠️ Daily stop loss triggered (-6%)")
    # Flatten all positions
    # Wait until next day
```

### 3. Weekly Stop Loss

**Trigger**: 10% drawdown from week's start

```python
weekly_pnl_pct = (current_equity - start_of_week_equity) / start_of_week_equity

if weekly_pnl_pct <= -0.10:
    logger.warning("⚠️ Weekly stop loss triggered (-10%)")
    # Flatten all positions
    # Review strategy performance
```

## Blackout Windows

### 1. Earnings Blackout

**Rationale**: Avoid unpredictable gaps around earnings

**Implementation**:
- Skip trades 1 day before and after earnings
- Use earnings calendar API or hardcoded dates

```python
if risk.is_blackout_period(symbol, "earnings"):
    logger.info(f"Skipping {symbol} - earnings blackout")
    return
```

### 2. FOMC Blackout

**Rationale**: Avoid high volatility around Fed announcements

**Implementation**:
- Skip trades on FOMC meeting days
- Optionally skip day before/after

```python
fomc_dates = ["2024-01-31", "2024-03-20", "2024-05-01", ...]

if datetime.now().date() in fomc_dates:
    logger.info("Skipping - FOMC blackout day")
    return
```

### 3. PDT (Pattern Day Trader) Throttle

**Rule**: < $25k accounts limited to 3 day trades per 5 business days

**Implementation**:
```python
if account_value < 25000:
    day_trades_in_window = risk.count_recent_day_trades(days=5)
    
    if day_trades_in_window >= 3:
        logger.warning("PDT limit reached - no new day trades")
        # Only allow swing trades (hold overnight)
        return
```

## HRP (Hierarchical Risk Parity)

Portfolio allocation based on correlation structure.

**Why HRP?**
- More robust than mean-variance optimization
- No need to invert covariance matrix
- Better out-of-sample performance
- Accounts for hierarchical structure of returns

**Formula**:
```
1. Calculate correlation matrix
2. Cluster assets by correlation (hierarchical)
3. Allocate inversely to cluster variance
4. Bisect tree to allocate within clusters
```

**Implementation**:
```python
from algos.core.portfolio import HRPOptimizer

hrp = HRPOptimizer()
weights = hrp.allocate(returns_df)

# Returns: {"SPY": 0.15, "QQQ": 0.12, "TLT": 0.25, ...}
```

**Constraints**:
- Min weight: 2% (avoid tiny positions)
- Max weight: 10% (single name cap)
- Rebalance: Weekly or when drift > 20%

## Risk Monitoring

### Real-Time Metrics

Monitor these in Discord daily report:

1. **Current Drawdown**: `(peak - current) / peak`
2. **Realized Volatility**: 20-day rolling std × √252
3. **Gross Leverage**: Total exposure / equity
4. **Largest Position**: Max single position as % of portfolio
5. **Sharpe Ratio**: Rolling 60-day annualized
6. **Sortino Ratio**: Rolling 60-day (downside deviation only)
7. **Hit Rate**: % of winning trades
8. **Profit Factor**: Gross profits / gross losses

### Discord Commands

```bash
# Check risk status
/risk status

# View current positions
/risk positions

# Update risk parameters
/risk set per_trade_risk_pct 0.015

# Force kill switch
/kill-switch halt

# Resume trading
/kill-switch resume
```

## Validation & Testing

### Unit Tests

```python
def test_position_sizing():
    """Test 1% risk rule."""
    risk = RiskManager(config)
    
    size = risk.calculate_position_size(
        symbol="SPY",
        current_price=450.0,
        atr=2.50,
        stop_loss_atr=1.0
    )
    
    # With $100k equity, 1% risk = $1,000
    # Stop at 1 ATR = $2.50
    # Size = $1,000 / $2.50 = 400 shares
    assert size == 400

def test_kill_switch():
    """Test circuit breaker activation."""
    risk = RiskManager(config)
    
    # Start at $100k
    risk.update_equity(100000)
    assert not risk.is_kill_switch_active()
    
    # Drop to $79k (-21%)
    risk.update_equity(79000)
    assert risk.is_kill_switch_active()
```

See `tests/test_risk_engine.py` for full test suite.

## Best Practices

### 1. Start Conservative

Begin with tighter risk limits:
```yaml
per_trade_risk_pct: 0.005  # 0.5% (half of default)
max_leverage: 1.5           # 1.5x instead of 2x
vol_target_ann: 0.10        # 10% instead of 12%
```

### 2. Monitor Continuously

Set up alerts for:
- Drawdown > 10%
- Volatility > 15%
- Single position > 8%
- Leverage > 1.8x

### 3. Review Regularly

Weekly review checklist:
- [ ] Risk-adjusted returns (Sharpe, Sortino)
- [ ] Maximum drawdown
- [ ] Win rate and profit factor
- [ ] Correlation changes
- [ ] Strategy performance attribution

### 4. Backtest Risk Rules

Validate risk rules reduce maximum drawdown:
```python
# Backtest with and without risk rules
results_no_risk = backtest(strategy, use_risk_limits=False)
results_with_risk = backtest(strategy, use_risk_limits=True)

assert results_with_risk["max_dd"] < results_no_risk["max_dd"]
```

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*
  - Chapter 3: Labeling (Triple Barrier)
  - Chapter 4: Sample Weights
  - Chapter 5: Fractional Differentiation
  - Chapter 6: Feature Engineering

- Thorp, E. (2008). *The Kelly Criterion in Blackjack Sports Betting and the Stock Market*
  - Fractional Kelly for robustness

- Bailey, D. H., & López de Prado, M. (2014). *The Deflated Sharpe Ratio*
  - Adjusting for multiple testing bias

- Raffinot, T. (2017). *Hierarchical Clustering-Based Asset Allocation*
  - HRP implementation details

## See Also

- [README.md](README.md#risk-management) - Overview
- [algos/core/risk.py](algos/core/risk.py) - Implementation
- [algos/core/portfolio.py](algos/core/portfolio.py) - HRP optimizer
- [tests/test_risk_engine.py](tests/test_risk_engine.py) - Test suite
