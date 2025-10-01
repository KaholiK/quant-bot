"""
Runtime state management for quant trading bot.
Thread-safe in-memory state that serves as single source of truth for UI → Algo control.
"""

import json
import threading
from datetime import datetime
from typing import Any

from loguru import logger

from .config_loader import QuantBotConfig


class RuntimeState:
    """Thread-safe runtime state management for trading bot."""

    def __init__(self):
        self._lock = threading.RLock()  # Reentrant lock for nested calls

        # Core state
        self._trading_paused = False
        self._kill_switch_active = False
        self._last_kill_reason = ""
        self._last_regime = "unknown"

        # Strategy toggles
        self._strategy_enabled: dict[str, bool] = {}

        # Risk parameters (can be updated at runtime)
        self._risk_params: dict[str, float] = {}

        # Metadata
        self._last_update = datetime.utcnow()
        self._last_heartbeat = datetime.utcnow()

    def load_from_config(self, config: QuantBotConfig) -> None:
        """Load initial state from configuration."""
        with self._lock:
            # Initialize strategy states
            self._strategy_enabled = {
                "scalper_sigma": config.trading.strategies.scalper_sigma.enabled,
                "trend_breakout": config.trading.strategies.trend_breakout.enabled,
                "bull_mode": config.trading.strategies.bull_mode.enabled,
                "market_neutral": config.trading.strategies.market_neutral.enabled,
                "gamma_reversal": config.trading.strategies.gamma_reversal.enabled,
            }

            # Initialize risk parameters
            self._risk_params = {
                "per_trade_risk_pct": config.trading.risk.per_trade_risk_pct,
                "max_leverage": config.trading.risk.max_leverage,
                "single_name_max_pct": config.trading.risk.single_name_max_pct,
                "sector_max_pct": config.trading.risk.sector_max_pct,
                "crypto_max_gross_pct": config.trading.risk.crypto_max_gross_pct,
                "vol_target_ann": config.trading.risk.vol_target_ann,
                "kill_switch_dd": config.trading.risk.kill_switch_dd,
                "day_stop_dd": config.trading.risk.day_stop_dd,
                "week_stop_dd": config.trading.risk.week_stop_dd,
            }

            self._last_update = datetime.utcnow()
            logger.info("Runtime state loaded from configuration")

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to dictionary for JSON export."""
        with self._lock:
            return {
                "trading_paused": self._trading_paused,
                "kill_switch_active": self._kill_switch_active,
                "last_kill_reason": self._last_kill_reason,
                "last_regime": self._last_regime,
                "strategy_enabled": self._strategy_enabled.copy(),
                "risk_params": self._risk_params.copy(),
                "last_update": self._last_update.isoformat(),
                "last_heartbeat": self._last_heartbeat.isoformat(),
                "version": "1.0",
            }

    def to_json(self) -> str:
        """Serialize state to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def apply_patch(self, patch: dict[str, Any]) -> bool:
        """
        Apply state changes from external UI.
        
        Args:
            patch: Dictionary with state changes
            
        Returns:
            True if patch was applied successfully
        """
        with self._lock:
            try:
                applied_changes = []

                # Handle pause/resume
                if "trading_paused" in patch:
                    old_value = self._trading_paused
                    self._trading_paused = bool(patch["trading_paused"])
                    if old_value != self._trading_paused:
                        status = "paused" if self._trading_paused else "resumed"
                        applied_changes.append(f"trading {status}")

                # Handle strategy toggles
                if "strategy_enabled" in patch:
                    strategy_patch = patch["strategy_enabled"]
                    for strategy, enabled in strategy_patch.items():
                        if strategy in self._strategy_enabled:
                            old_value = self._strategy_enabled[strategy]
                            self._strategy_enabled[strategy] = bool(enabled)
                            if old_value != self._strategy_enabled[strategy]:
                                status = "enabled" if enabled else "disabled"
                                applied_changes.append(f"{strategy} {status}")

                # Handle risk parameter updates
                if "risk_params" in patch:
                    risk_patch = patch["risk_params"]
                    for param, value in risk_patch.items():
                        if param in self._risk_params:
                            old_value = self._risk_params[param]
                            new_value = float(value)

                            # Validate ranges
                            if self._validate_risk_param(param, new_value):
                                self._risk_params[param] = new_value
                                if abs(old_value - new_value) > 1e-6:
                                    applied_changes.append(f"{param}: {old_value:.4f} → {new_value:.4f}")
                            else:
                                logger.warning(f"Invalid risk parameter value: {param}={new_value}")

                # Handle kill switch
                if "kill_switch_active" in patch:
                    if patch["kill_switch_active"]:
                        reason = patch.get("kill_reason", "Manual kill switch")
                        self.mark_kill(reason)
                        applied_changes.append(f"kill switch activated: {reason}")
                    else:
                        self.resume()
                        applied_changes.append("kill switch deactivated")

                # Update metadata
                if applied_changes:
                    self._last_update = datetime.utcnow()
                    logger.info(f"Applied runtime state changes: {', '.join(applied_changes)}")

                return True

            except Exception as e:
                logger.error(f"Failed to apply runtime state patch: {e}")
                return False

    def _validate_risk_param(self, param: str, value: float) -> bool:
        """Validate risk parameter value is within acceptable range."""
        validations = {
            "per_trade_risk_pct": lambda x: 0.0 <= x <= 0.05,
            "max_leverage": lambda x: 1.0 <= x <= 10.0,
            "single_name_max_pct": lambda x: 0.01 <= x <= 0.50,
            "sector_max_pct": lambda x: 0.05 <= x <= 1.0,
            "crypto_max_gross_pct": lambda x: 0.1 <= x <= 1.0,
            "vol_target_ann": lambda x: 0.01 <= x <= 1.0,
            "kill_switch_dd": lambda x: 0.0 < x <= 0.5,
            "day_stop_dd": lambda x: 0.0 < x <= 0.5,
            "week_stop_dd": lambda x: 0.0 < x <= 0.5,
        }

        validator = validations.get(param)
        return validator(value) if validator else False

    def mark_kill(self, reason: str) -> None:
        """Activate kill switch with reason."""
        with self._lock:
            self._kill_switch_active = True
            self._trading_paused = True
            self._last_kill_reason = reason
            self._last_update = datetime.utcnow()
            logger.warning(f"Kill switch activated: {reason}")

    def resume(self) -> None:
        """Resume trading (deactivate kill switch and unpause)."""
        with self._lock:
            self._kill_switch_active = False
            self._trading_paused = False
            self._last_kill_reason = ""
            self._last_update = datetime.utcnow()
            logger.info("Trading resumed - kill switch deactivated")

    def heartbeat(self) -> None:
        """Update heartbeat timestamp."""
        with self._lock:
            self._last_heartbeat = datetime.utcnow()

    def update_regime(self, regime: str) -> None:
        """Update current market regime."""
        with self._lock:
            if self._last_regime != regime:
                self._last_regime = regime
                self._last_update = datetime.utcnow()
                logger.info(f"Market regime updated to: {regime}")

    # Read-only properties
    @property
    def trading_paused(self) -> bool:
        """Get trading pause status."""
        with self._lock:
            return self._trading_paused

    @property
    def kill_switch_active(self) -> bool:
        """Get kill switch status."""
        with self._lock:
            return self._kill_switch_active

    @property
    def last_kill_reason(self) -> str:
        """Get last kill switch reason."""
        with self._lock:
            return self._last_kill_reason

    @property
    def strategy_enabled(self) -> dict[str, bool]:
        """Get strategy enabled states."""
        with self._lock:
            return self._strategy_enabled.copy()

    @property
    def risk_params(self) -> dict[str, float]:
        """Get current risk parameters."""
        with self._lock:
            return self._risk_params.copy()

    @property
    def last_regime(self) -> str:
        """Get last known market regime."""
        with self._lock:
            return self._last_regime

    def get_strategy_enabled(self, strategy: str) -> bool:
        """Get enabled status for specific strategy."""
        with self._lock:
            return self._strategy_enabled.get(strategy, False)

    def get_risk_param(self, param: str) -> float | None:
        """Get specific risk parameter value."""
        with self._lock:
            return self._risk_params.get(param)


# Global instance for singleton access
_runtime_state: RuntimeState | None = None


def get_runtime_state() -> RuntimeState:
    """Get global runtime state instance."""
    global _runtime_state
    if _runtime_state is None:
        _runtime_state = RuntimeState()
    return _runtime_state


def initialize_runtime_state(config: QuantBotConfig) -> RuntimeState:
    """Initialize global runtime state with configuration."""
    global _runtime_state
    _runtime_state = RuntimeState()
    _runtime_state.load_from_config(config)
    return _runtime_state
