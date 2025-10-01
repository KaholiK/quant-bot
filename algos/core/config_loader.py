"""
Configuration loader with pydantic validation for the quant trading bot.
Provides type-safe configuration loading with backward compatibility.
"""

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class UniverseConfig(BaseModel):
    """Configuration for trading universe."""
    equities: str = Field(default="SP100", description="Equity universe identifier")
    crypto: list[str] = Field(default=["BTCUSD", "ETHUSD"], description="Crypto pairs to trade")


class BarsConfig(BaseModel):
    """Configuration for bar data timeframes."""
    equities: str = Field(default="30m", description="Equity bar timeframe")
    crypto: str = Field(default="15m", description="Crypto bar timeframe")


class BlackoutConfig(BaseModel):
    """Blackout period configuration."""
    earnings: bool = Field(default=True, description="Block trades around earnings")
    fomc: bool = Field(default=True, description="Block trades on FOMC days")


class RiskConfig(BaseModel):
    """Risk management configuration."""
    per_trade_risk_pct: float = Field(default=0.01, ge=0.0, le=0.05, description="Risk per trade (0-5%)")
    max_leverage: float = Field(default=2.0, ge=1.0, le=10.0, description="Maximum leverage")
    single_name_max_pct: float = Field(default=0.10, ge=0.01, le=0.50, description="Max position per symbol")
    sector_max_pct: float = Field(default=0.30, ge=0.05, le=1.0, description="Max exposure per sector")
    crypto_max_gross_pct: float = Field(default=0.50, ge=0.1, le=1.0, description="Max crypto gross exposure")
    vol_target_ann: float = Field(default=0.12, gt=0.01, le=1.0, description="Volatility target (annualized)")
    kill_switch_dd: float = Field(default=0.20, gt=0.0, le=0.5, description="Kill switch drawdown threshold")
    day_stop_dd: float = Field(default=0.06, gt=0.0, le=0.5, description="Daily stop loss drawdown")
    week_stop_dd: float = Field(default=0.10, gt=0.0, le=0.5, description="Weekly stop loss drawdown")
    blackout: BlackoutConfig = BlackoutConfig()


class ModelsConfig(BaseModel):
    """Model file paths configuration."""
    classifier_path: str = Field(default="models/xgb_classifier.joblib", description="Classifier model path")
    meta_model_path: str = Field(default="models/meta_filter.joblib", description="Meta model path")
    rl_policy_path: str = Field(default="policies/ppo_policy.zip", description="RL policy path")


class StrategyConfig(BaseModel):
    """Individual strategy configuration."""
    enabled: bool = Field(default=True, description="Whether strategy is enabled")


class StrategiesConfig(BaseModel):
    """All strategies configuration."""
    scalper_sigma: StrategyConfig = StrategyConfig()
    trend_breakout: StrategyConfig = StrategyConfig()
    bull_mode: StrategyConfig = StrategyConfig()
    market_neutral: StrategyConfig = StrategyConfig(enabled=False)
    gamma_reversal: StrategyConfig = StrategyConfig(enabled=False)


class FeaturesConfig(BaseModel):
    """Feature engineering configuration."""
    returns_periods: list[int] = Field(default=[1, 5, 20, 60], description="Return calculation periods")
    sma_periods: list[int] = Field(default=[20, 50, 200], description="Simple moving average periods")
    ema_periods: list[int] = Field(default=[12, 26], description="Exponential moving average periods")
    macd_params: dict[str, int] = Field(
        default={"fast": 12, "slow": 26, "signal": 9},
        description="MACD parameters"
    )
    atr_period: int = Field(default=14, description="ATR calculation period")
    bollinger_period: int = Field(default=20, description="Bollinger bands period")
    vol_window: int = Field(default=20, description="Volatility calculation window")


class LabelsConfig(BaseModel):
    """Label generation configuration."""
    horizon_bars: int = Field(default=5, description="Label horizon in bars")
    tp_atr_mult: float = Field(default=1.75, description="Take profit ATR multiplier")
    sl_atr_mult: float = Field(default=1.00, description="Stop loss ATR multiplier")


class LearningConfig(BaseModel):
    """Machine learning configuration."""
    cv: dict[str, Any] = Field(
        default={
            "scheme": "purged_kfold_embargo",
            "folds": 5,
            "embargo_frac": 0.02
        },
        description="Cross-validation configuration"
    )
    retrain_cadence: str = Field(default="weekly", description="Model retraining frequency")
    gates: dict[str, float] = Field(
        default={
            "oos_sortino_min": 1.2,
            "oos_profit_factor_min": 1.15,
            "oos_max_dd_max": 0.06
        },
        description="Model promotion gates"
    )
    meta_threshold: float = Field(default=0.55, description="Meta-model probability threshold")


class LabelsConfig(BaseModel):
    """Label generation configuration."""
    horizon_bars: int = Field(default=5, description="Label horizon in bars")
    tp_atr_mult: float = Field(default=1.75, description="Take profit ATR multiplier")
    sl_atr_mult: float = Field(default=1.00, description="Stop loss ATR multiplier")


class RegimeConfig(BaseModel):
    """Regime detection configuration."""
    enabled: bool = Field(default=True, description="Enable regime detection")
    smooth_bars: int = Field(default=5, description="Smoothing window for regime")


class CapacityConfig(BaseModel):
    """Capacity management configuration."""
    adv_cap_pct: float = Field(default=0.05, ge=0.01, le=0.2, description="ADV capacity percentage")


class ExecutionConfig(BaseModel):
    """Execution configuration."""
    maker_ladder_offsets_atr: list[float] = Field(
        default=[0.10, 0.20, 0.30],
        description="Maker ladder offsets in ATR units"
    )
    min_ms_between_orders: int = Field(default=300, ge=50, description="Minimum milliseconds between orders")
    min_hold_secs: int = Field(default=60, ge=1, description="Minimum hold time in seconds")
    use_vector_env: bool = Field(default=True, description="Use vectorized environment for RL training")


class DiscordConfig(BaseModel):
    """Discord bot configuration."""
    enabled: bool = Field(default=True, description="Enable Discord bot")
    guild_id: int = Field(default=0, description="Discord guild ID")


class AdminApiConfig(BaseModel):
    """Admin API configuration."""
    enabled: bool = Field(default=True, description="Enable admin API")
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8080, ge=1, le=65535, description="API port")
    polling_interval_sec: int = Field(default=120, ge=30, description="Polling interval for QC integration")


class UIConfig(BaseModel):
    """UI configuration."""
    discord: DiscordConfig = DiscordConfig()
    admin_api: AdminApiConfig = AdminApiConfig()


class TradingConfig(BaseModel):
    """Main trading configuration."""
    universe: UniverseConfig = UniverseConfig()
    bars: BarsConfig = BarsConfig()
    features: FeaturesConfig = FeaturesConfig()
    labels: LabelsConfig = LabelsConfig()
    regime: RegimeConfig = RegimeConfig()
    capacity: CapacityConfig = CapacityConfig()
    risk: RiskConfig = RiskConfig()
    models: ModelsConfig = ModelsConfig()
    strategies: StrategiesConfig = StrategiesConfig()
    learning: LearningConfig = LearningConfig()
    execution: ExecutionConfig = ExecutionConfig()
    strategy_priority_by_regime: dict[str, list[str]] = Field(
        default={
            "trend": ["trend_breakout", "bull_mode", "scalper_sigma", "market_neutral", "gamma_reversal"],
            "chop": ["scalper_sigma", "market_neutral", "gamma_reversal", "trend_breakout", "bull_mode"],
            "high-vol": ["scalper_sigma", "gamma_reversal", "trend_breakout", "market_neutral", "bull_mode"],
            "low-vol": ["trend_breakout", "bull_mode", "market_neutral", "scalper_sigma", "gamma_reversal"]
        },
        description="Strategy priority by market regime"
    )
    ui: UIConfig = UIConfig()


class QuantBotConfig(BaseModel):
    """Root configuration model."""
    trading: TradingConfig = TradingConfig()

    model_config = {"extra": "allow"}  # Allow additional fields for backward compatibility


def _normalize_legacy_config(config_data: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy flat configuration to new nested structure."""

    # If already has trading key, return as-is
    if "trading" in config_data:
        return config_data

    logger.info("Normalizing legacy configuration format")

    normalized = {"trading": {}}
    trading = normalized["trading"]

    # Map legacy keys to new structure
    legacy_mappings = {
        # Universe mappings
        "universe": "universe",

        # Risk mappings
        "risk": "risk",
        "max_leverage": ("risk", "max_leverage"),
        "max_position_pct": ("risk", "single_name_max_pct"),
        "max_sector_pct": ("risk", "sector_max_pct"),
        "risk_pct_per_trade": ("risk", "per_trade_risk_pct"),
        "kill_switch_dd": ("risk", "kill_switch_dd"),

        # Features, labels, strategies - pass through
        "features": "features",
        "labels": "labels",
        "strategies": "strategies",
        "learning": "learning",
        "execution": "execution",

        # Models
        "models": "models"
    }

    # Process legacy mappings
    for legacy_key, target in legacy_mappings.items():
        if legacy_key in config_data:
            if isinstance(target, tuple):
                # Nested mapping like ("risk", "max_leverage")
                section, key = target
                if section not in trading:
                    trading[section] = {}
                trading[section][key] = config_data[legacy_key]
            else:
                # Direct mapping
                trading[target] = config_data[legacy_key]

    # Handle special cases for bars configuration
    if "universe" in config_data:
        universe_data = config_data["universe"]
        bars_config = {}

        if "equities" in universe_data and "resolution" in universe_data["equities"]:
            resolution = universe_data["equities"]["resolution"]
            # Convert LEAN resolution to our format
            resolution_map = {
                "ThirtyMinute": "30m",
                "FifteenMinute": "15m",
                "OneMinute": "1m"
            }
            bars_config["equities"] = resolution_map.get(resolution, "30m")

        if "crypto" in universe_data and "resolution" in universe_data["crypto"]:
            resolution = universe_data["crypto"]["resolution"]
            bars_config["crypto"] = resolution_map.get(resolution, "15m")

        if bars_config:
            trading["bars"] = bars_config

    # Copy any remaining top-level keys (models, execution, etc.)
    for key, value in config_data.items():
        if key not in legacy_mappings and key != "trading":
            normalized[key] = value

    return normalized


def load_config(path: str = "config.yaml") -> QuantBotConfig:
    """
    Load and validate configuration from YAML file.
    
    Args:
        path: Path to configuration file
        
    Returns:
        Validated configuration object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    config_path = Path(path)

    if not config_path.exists():
        logger.warning(f"Config file not found at {path}, using defaults")
        return QuantBotConfig()

    try:
        with open(config_path) as f:
            raw_config = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Failed to parse YAML config: {e}")

    if not raw_config:
        logger.warning("Empty config file, using defaults")
        return QuantBotConfig()

    # Normalize legacy configuration
    normalized_config = _normalize_legacy_config(raw_config)

    try:
        config = QuantBotConfig.model_validate(normalized_config)
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}")

    # Log redacted configuration (remove sensitive fields)
    redacted_dict = config.model_dump(exclude={"trading": {"models": {"classifier_path", "meta_model_path", "rl_policy_path"}}})
    logger.info(f"Loaded configuration: {redacted_dict}")

    return config


def get_legacy_dict(config: QuantBotConfig) -> dict[str, Any]:
    """
    Convert configuration back to legacy dictionary format for backward compatibility.
    
    Args:
        config: Validated configuration object
        
    Returns:
        Legacy format dictionary
    """
    config_dict = config.model_dump()

    # Flatten trading section for legacy compatibility
    result = {}

    # Copy trading section contents to root
    trading = config_dict.get("trading", {})
    for key, value in trading.items():
        result[key] = value

    # Copy other top-level sections
    for key, value in config_dict.items():
        if key != "trading":
            result[key] = value

    return result
