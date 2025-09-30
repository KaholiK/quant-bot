"""
Regime detection engine for quantitative trading.
Uses LightGBM classifier to detect market regimes and provides fallback rules.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RegimeType = Literal["trend", "chop", "high-vol", "low-vol"]


class RegimeEngine:
    """
    Market regime detection engine using LightGBM classifier with fallback rules.
    """

    def __init__(self, config: dict[str, Any], model_path: str | None = None):
        """Initialize regime engine."""
        self.config = config
        self.regime_config = config.get("regime", {})
        self.enabled = self.regime_config.get("enabled", True)
        self.smooth_bars = self.regime_config.get("smooth_bars", 5)

        # Model and cache
        self.model = None
        self.model_path = model_path or "models/regime_classifier.joblib"
        self._regime_cache = {}
        self._feature_cache = {}

        # Load model if available
        self._load_model()

        logger.info(f"Regime engine initialized: enabled={self.enabled}, smooth_bars={self.smooth_bars}")

    def _load_model(self) -> None:
        """Load regime classification model if available."""
        if not Path(self.model_path).exists():
            logger.warning(f"Regime model not found at {self.model_path}, using fallback rules")
            return

        try:
            import joblib
            self.model = joblib.load(self.model_path)
            logger.info(f"Regime model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load regime model: {e}")
            self.model = None

    def get_regime(self,
                   data: pd.DataFrame,
                   now: datetime | None = None,
                   symbol: str = "SPY") -> tuple[RegimeType, float]:
        """
        Get current market regime with confidence score.
        
        Args:
            data: OHLCV data with recent history
            now: Current timestamp (defaults to latest in data)
            symbol: Symbol for caching key
            
        Returns:
            Tuple of (regime, confidence)
        """
        if not self.enabled:
            return "trend", 1.0

        if now is None:
            now = data.index[-1] if not data.empty else datetime.now()

        # Check cache
        cache_key = f"{symbol}_{now.isoformat()}"
        if cache_key in self._regime_cache:
            return self._regime_cache[cache_key]

        try:
            # Compute regime features
            features = self._compute_regime_features(data)

            # Get regime prediction
            if self.model is not None and not features.empty:
                regime, confidence = self._predict_regime(features.iloc[-1])
            else:
                regime, confidence = self._fallback_regime(features.iloc[-1] if not features.empty else None)

            # Apply smoothing if configured
            if self.smooth_bars > 1:
                regime = self._smooth_regime(regime, symbol, now)

            # Cache result
            self._regime_cache[cache_key] = (regime, confidence)

            # Cleanup old cache entries
            self._cleanup_cache()

            return regime, confidence

        except Exception as e:
            logger.error(f"Error computing regime: {e}")
            return "trend", 0.5  # Safe default

    def _compute_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute features for regime classification."""
        if data.empty or len(data) < 50:
            return pd.DataFrame()

        df = data.copy()

        # Price-based features
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"]/df["close"].shift(1))

        # Trend features
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()
        df["sma_200"] = df["close"].rolling(200).mean()
        df["trend_slope"] = (df["sma_20"] / df["sma_50"] - 1).rolling(10).mean()

        # Volatility features
        df["volatility"] = df["returns"].rolling(20).std()
        df["parkinson_vol"] = np.sqrt(252 * np.log(df["high"]/df["low"]).rolling(20).var())

        # ATR and relative measures
        df["tr"] = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                abs(df["high"] - df["close"].shift(1)),
                abs(df["low"] - df["close"].shift(1))
            )
        )
        df["atr"] = df["tr"].rolling(14).mean()
        df["atr_ratio"] = df["atr"] / df["close"]

        # Autocorrelation (trend persistence)
        def rolling_autocorr(series, window=20):
            return series.rolling(window).apply(lambda x: np.corrcoef(x[:-1], x[1:])[0,1] if len(x) > 1 else 0)

        df["autocorr"] = rolling_autocorr(df["returns"])

        # Volume-based breadth proxy (if volume available)
        if "volume" in df.columns:
            df["volume_ma"] = df["volume"].rolling(20).mean()
            df["volume_ratio"] = df["volume"] / df["volume_ma"]
        else:
            df["volume_ratio"] = 1.0

        # Regime features (final)
        features = pd.DataFrame(index=df.index)
        features["trend_slope"] = df["trend_slope"]
        features["volatility"] = df["volatility"]
        features["atr_ratio"] = df["atr_ratio"]
        features["autocorr"] = df["autocorr"]
        features["volume_ratio"] = df["volume_ratio"]
        features["sma_cross"] = (df["close"] > df["sma_200"]).astype(int)
        features["vol_quantile"] = df["volatility"].rolling(252).rank(pct=True)

        return features.dropna()

    def _predict_regime(self, features: pd.Series) -> tuple[RegimeType, float]:
        """Predict regime using trained model."""
        try:
            # Convert to array for prediction
            feature_array = features.values.reshape(1, -1)

            # Get prediction and probabilities
            prediction = self.model.predict(feature_array)[0]
            probabilities = self.model.predict_proba(feature_array)[0]

            # Map prediction to regime
            regime_map = {0: "trend", 1: "chop", 2: "high-vol", 3: "low-vol"}
            regime = regime_map.get(prediction, "trend")
            confidence = float(probabilities.max())

            return regime, confidence

        except Exception as e:
            logger.error(f"Error in regime prediction: {e}")
            return self._fallback_regime(features)

    def _fallback_regime(self, features: pd.Series | None) -> tuple[RegimeType, float]:
        """Fallback regime detection using simple rules."""
        if features is None or features.isna().any():
            return "trend", 0.5

        try:
            trend_slope = features.get("trend_slope", 0)
            volatility = features.get("volatility", 0)
            vol_quantile = features.get("vol_quantile", 0.5)
            autocorr = features.get("autocorr", 0)
            sma_cross = features.get("sma_cross", 1)

            # High/low volatility regimes
            if vol_quantile > 0.8:
                return "high-vol", 0.7
            if vol_quantile < 0.2:
                return "low-vol", 0.7

            # Trend vs chop
            if abs(trend_slope) > 0.01 and sma_cross and autocorr > 0.1:
                return "trend", 0.6
            return "chop", 0.6

        except Exception as e:
            logger.error(f"Error in fallback regime: {e}")
            return "trend", 0.5

    def _smooth_regime(self, current_regime: RegimeType, symbol: str, now: datetime) -> RegimeType:
        """Apply temporal smoothing to regime detection."""
        # Get recent regime history
        history_key = f"{symbol}_history"
        if history_key not in self._regime_cache:
            self._regime_cache[history_key] = []

        history = self._regime_cache[history_key]
        history.append((now, current_regime))

        # Keep only recent history
        cutoff = now - timedelta(minutes=self.smooth_bars * 30)  # Assuming 30min bars
        history = [(t, r) for t, r in history if t >= cutoff]
        self._regime_cache[history_key] = history

        # Use majority vote for smoothing
        if len(history) >= self.smooth_bars:
            recent_regimes = [r for _, r in history[-self.smooth_bars:]]
            regime_counts = {r: recent_regimes.count(r) for r in set(recent_regimes)}
            return max(regime_counts.items(), key=lambda x: x[1])[0]

        return current_regime

    def _cleanup_cache(self) -> None:
        """Cleanup old cache entries to prevent memory leak."""
        if len(self._regime_cache) > 1000:
            # Keep only most recent 500 entries
            keys = list(self._regime_cache.keys())
            for key in keys[:-500]:
                if not key.endswith("_history"):  # Keep history entries
                    del self._regime_cache[key]

    def get_regime_features(self, data: pd.DataFrame) -> dict[str, float]:
        """Get regime features for strategy conditioning."""
        if not self.enabled or data.empty:
            return {}

        try:
            features = self._compute_regime_features(data)
            if features.empty:
                return {}

            latest = features.iloc[-1]
            return {
                "trend_slope": float(latest.get("trend_slope", 0)),
                "volatility": float(latest.get("volatility", 0)),
                "vol_quantile": float(latest.get("vol_quantile", 0.5)),
                "autocorr": float(latest.get("autocorr", 0)),
                "atr_ratio": float(latest.get("atr_ratio", 0)),
            }
        except Exception as e:
            logger.error(f"Error computing regime features: {e}")
            return {}

    def is_regime_transition(self,
                           data: pd.DataFrame,
                           symbol: str = "SPY",
                           lookback_bars: int = 5) -> bool:
        """Check if we're in a regime transition period."""
        if not self.enabled:
            return False

        try:
            # Get recent regime history
            recent_regimes = []
            now = data.index[-1] if not data.empty else datetime.now()

            for i in range(lookback_bars):
                hist_data = data.iloc[:-i] if i > 0 else data
                if not hist_data.empty:
                    regime, _ = self.get_regime(hist_data, symbol=symbol)
                    recent_regimes.append(regime)

            # Check for regime changes
            unique_regimes = set(recent_regimes)
            return len(unique_regimes) > 1

        except Exception as e:
            logger.error(f"Error checking regime transition: {e}")
            return False
