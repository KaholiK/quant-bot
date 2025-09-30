"""
Test suite for Triple Barrier Labeling functionality.
Tests barrier touch detection, exact hit scenarios, timeout conditions, and NaN handling.
"""

import os

# Import modules to test
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from algos.core.labels import (
    TripleBarrierLabeler,
    apply_triple_barrier,
    create_meta_labels,
    fractional_differentiation,
    validate_labels,
)


class TestTripleBarrier:
    """Test cases for triple barrier labeling."""

    def setup_method(self):
        """Setup test data."""
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=1000, freq="1H")

        # Generate realistic price series
        returns = np.random.normal(0, 0.01, 1000)
        prices = 100 * np.exp(np.cumsum(returns))

        self.prices = pd.Series(prices, index=dates)

        # Generate ATR series (realistic values)
        self.atr = pd.Series(np.random.uniform(0.5, 2.0, 1000), index=dates)

        # Configuration for testing
        self.config = {
            "labels": {
                "profit_take_mult": 1.75,
                "stop_loss_mult": 1.0,
                "horizon_bars": 5
            }
        }

    def test_basic_triple_barrier(self):
        """Test basic triple barrier functionality."""
        labels, touch_times, barriers_df = apply_triple_barrier(
            prices=self.prices,
            atr=self.atr,
            profit_take_mult=1.75,
            stop_loss_mult=1.0,
            horizon_bars=5
        )

        # Basic checks
        assert len(labels) > 0, "Should generate some labels"
        assert len(touch_times) == len(labels), "Touch times should match labels"
        assert len(barriers_df) >= len(labels), "Barriers DataFrame should have entries"

        # Check label values
        unique_labels = set(labels.unique())
        assert unique_labels.issubset({-1, 0, 1}), f"Invalid labels: {unique_labels}"

        # Check that we have a reasonable distribution
        label_counts = labels.value_counts()
        assert len(label_counts) >= 2, "Should have at least 2 different label types"

        print(f"Generated {len(labels)} labels with distribution: {label_counts.to_dict()}")

    def test_profit_take_scenario(self):
        """Test specific profit take scenario."""
        # Create a scenario where price moves up significantly
        rising_prices = pd.Series([100, 101, 102, 105, 108, 110, 112],
                                index=pd.date_range("2023-01-01", periods=7, freq="1H"))
        atr_values = pd.Series([1.0] * 7, index=rising_prices.index)

        labels, touch_times, barriers_df = apply_triple_barrier(
            prices=rising_prices,
            atr=atr_values,
            profit_take_mult=2.0,
            stop_loss_mult=1.0,
            horizon_bars=5
        )

        # Should detect profit take on the upward move
        profit_labels = labels[labels == 1]
        assert len(profit_labels) > 0, "Should detect profit take in rising market"

        print(f"Profit take test: {len(profit_labels)} profit signals detected")

    def test_stop_loss_scenario(self):
        """Test specific stop loss scenario."""
        # Create a scenario where price drops significantly
        falling_prices = pd.Series([100, 99, 98, 95, 92, 90, 88],
                                 index=pd.date_range("2023-01-01", periods=7, freq="1H"))
        atr_values = pd.Series([1.0] * 7, index=falling_prices.index)

        labels, touch_times, barriers_df = apply_triple_barrier(
            prices=falling_prices,
            atr=atr_values,
            profit_take_mult=1.5,
            stop_loss_mult=1.0,
            horizon_bars=5
        )

        # Should detect stop loss on the downward move
        loss_labels = labels[labels == -1]
        assert len(loss_labels) > 0, "Should detect stop loss in falling market"

        print(f"Stop loss test: {len(loss_labels)} stop loss signals detected")

    def test_timeout_scenario(self):
        """Test timeout when no barriers are hit."""
        # Create flat prices that don't hit barriers
        flat_prices = pd.Series([100 + 0.1 * np.sin(i/10) for i in range(50)],
                               index=pd.date_range("2023-01-01", periods=50, freq="1H"))
        atr_values = pd.Series([5.0] * 50, index=flat_prices.index)  # High ATR = wide barriers

        labels, touch_times, barriers_df = apply_triple_barrier(
            prices=flat_prices,
            atr=atr_values,
            profit_take_mult=3.0,  # Very wide barriers
            stop_loss_mult=3.0,
            horizon_bars=10
        )

        # Should have timeout labels
        timeout_labels = labels[labels == 0]
        assert len(timeout_labels) > 0, "Should detect timeouts with wide barriers"

        print(f"Timeout test: {len(timeout_labels)} timeout signals detected")

    def test_nan_handling(self):
        """Test handling of NaN values."""
        # Introduce NaN values
        prices_with_nan = self.prices.copy()
        prices_with_nan.iloc[50:55] = np.nan

        atr_with_nan = self.atr.copy()
        atr_with_nan.iloc[100:105] = np.nan

        labels, touch_times, barriers_df = apply_triple_barrier(
            prices=prices_with_nan,
            atr=atr_with_nan,
            profit_take_mult=1.75,
            stop_loss_mult=1.0,
            horizon_bars=5
        )

        # Should handle NaN gracefully
        assert not labels.isna().any(), "Labels should not contain NaN"
        assert not touch_times.isna().any(), "Touch times should not contain NaN"

        print(f"NaN handling test: Generated {len(labels)} valid labels despite NaN inputs")

    def test_exact_barrier_hit(self):
        """Test exact barrier hit detection."""
        # Create price series with exact barrier hit
        entry_price = 100.0
        atr_val = 2.0
        profit_barrier = entry_price * (1 + 1.5 * atr_val / entry_price)  # 103.0

        exact_prices = pd.Series([100.0, 101.0, 102.0, 103.0, 102.5],
                                index=pd.date_range("2023-01-01", periods=5, freq="1H"))
        atr_values = pd.Series([atr_val] * 5, index=exact_prices.index)

        labels, touch_times, barriers_df = apply_triple_barrier(
            prices=exact_prices,
            atr=atr_values,
            profit_take_mult=1.5,
            stop_loss_mult=1.0,
            horizon_bars=10
        )

        # Should detect the exact hit
        assert len(labels) > 0, "Should detect exact barrier hit"
        profit_hits = (labels == 1).sum()
        assert profit_hits > 0, "Should detect profit take on exact hit"

        print("Exact hit test: Detected exact barrier hit")

    def test_labeler_class(self):
        """Test TripleBarrierLabeler class."""
        labeler = TripleBarrierLabeler(self.config)

        labels, touch_times, barriers_df = labeler.fit_transform(self.prices, self.atr)

        assert len(labels) > 0, "Labeler should generate labels"
        assert isinstance(labels, pd.Series), "Should return pandas Series"

        # Test get_label_info
        info = labeler.get_label_info(labels)
        assert "total_labels" in info, "Should provide label info"
        assert info["total_labels"] == len(labels), "Total labels should match"

        print(f"Labeler class test: {info}")

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        # Very short series
        short_prices = self.prices.head(3)
        short_atr = self.atr.head(3)

        labels, touch_times, barriers_df = apply_triple_barrier(
            prices=short_prices,
            atr=short_atr,
            profit_take_mult=1.75,
            stop_loss_mult=1.0,
            horizon_bars=5
        )

        # Should handle gracefully
        assert isinstance(labels, pd.Series), "Should return Series even with insufficient data"
        assert isinstance(barriers_df, pd.DataFrame), "Should return DataFrame"

        print(f"Insufficient data test: Handled {len(short_prices)} data points")

    def test_misaligned_data(self):
        """Test handling of misaligned price and ATR data."""
        # Create misaligned data
        shorter_atr = self.atr.head(500)  # ATR shorter than prices

        with pytest.raises(ValueError, match="same length"):
            apply_triple_barrier(
                prices=self.prices,
                atr=shorter_atr,
                profit_take_mult=1.75,
                stop_loss_mult=1.0,
                horizon_bars=5
            )

        print("Misaligned data test: Correctly raised ValueError")


class TestMetaLabeling:
    """Test meta-labeling functionality."""

    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        n_samples = 1000

        # Generate synthetic predictions
        self.primary_predictions = pd.Series(
            np.random.beta(2, 2, n_samples),  # Beta distribution for probabilities
            index=pd.date_range("2023-01-01", periods=n_samples, freq="1H")
        )

        # Generate true labels (with some correlation to predictions)
        noise = np.random.normal(0, 0.3, n_samples)
        true_probs = np.clip(self.primary_predictions + noise, 0, 1)
        self.true_labels = pd.Series(
            (true_probs > 0.5).astype(int),
            index=self.primary_predictions.index
        )

    def test_basic_meta_labeling(self):
        """Test basic meta-labeling functionality."""
        meta_labels = create_meta_labels(
            self.primary_predictions,
            self.true_labels,
            threshold=0.5
        )

        assert len(meta_labels) > 0, "Should generate meta-labels"
        assert len(meta_labels) <= len(self.primary_predictions), "Meta-labels should not exceed input"
        assert set(meta_labels.unique()).issubset({0, 1}), "Meta-labels should be binary"

        acceptance_rate = meta_labels.mean()
        print(f"Meta-labeling test: Acceptance rate = {acceptance_rate:.1%}")

        # Acceptance rate should be reasonable (not too high or low)
        assert 0.1 <= acceptance_rate <= 0.9, f"Acceptance rate {acceptance_rate:.1%} seems unreasonable"

    def test_meta_labeling_threshold(self):
        """Test meta-labeling with different thresholds."""
        thresholds = [0.3, 0.5, 0.7]
        acceptance_rates = []

        for threshold in thresholds:
            meta_labels = create_meta_labels(
                self.primary_predictions,
                self.true_labels,
                threshold=threshold
            )
            acceptance_rates.append(meta_labels.mean())

        print(f"Threshold test: {dict(zip(thresholds, acceptance_rates, strict=False))}")

        # Different thresholds should give different acceptance rates
        assert len(set(acceptance_rates)) > 1, "Different thresholds should give different results"

    def test_meta_labeling_empty_input(self):
        """Test meta-labeling with empty input."""
        empty_predictions = pd.Series(dtype=float)
        empty_labels = pd.Series(dtype=int)

        meta_labels = create_meta_labels(empty_predictions, empty_labels)

        assert len(meta_labels) == 0, "Empty input should return empty output"
        assert isinstance(meta_labels, pd.Series), "Should return Series type"

        print("Empty input test: Handled correctly")


class TestLabelValidation:
    """Test label validation functionality."""

    def setup_method(self):
        """Setup test data."""
        dates = pd.date_range("2023-01-01", periods=100, freq="1H")
        self.valid_labels = pd.Series([1, -1, 0] * 33 + [1], index=dates)
        self.prices = pd.Series(np.random.randn(100).cumsum() + 100, index=dates)

    def test_valid_labels(self):
        """Test validation of valid labels."""
        is_valid, issues = validate_labels(self.valid_labels, self.prices)

        assert is_valid, f"Valid labels failed validation: {issues}"
        assert len(issues) == 0, f"Valid labels should have no issues: {issues}"

        print("Valid labels test: Passed validation")

    def test_invalid_label_values(self):
        """Test validation with invalid label values."""
        invalid_labels = self.valid_labels.copy()
        invalid_labels.iloc[0] = 5  # Invalid value

        is_valid, issues = validate_labels(invalid_labels, self.prices)

        assert not is_valid, "Should detect invalid label values"
        assert any("Invalid label values" in issue for issue in issues), f"Should report invalid values: {issues}"

        print(f"Invalid values test: Correctly detected issues: {issues}")

    def test_empty_labels(self):
        """Test validation with empty labels."""
        empty_labels = pd.Series(dtype=int)

        is_valid, issues = validate_labels(empty_labels, self.prices)

        assert not is_valid, "Should detect empty labels"
        assert any("Empty label series" in issue for issue in issues), f"Should report empty series: {issues}"

        print(f"Empty labels test: Correctly detected issues: {issues}")

    def test_constant_labels(self):
        """Test validation with constant labels."""
        constant_labels = pd.Series([1] * 100, index=self.valid_labels.index)

        is_valid, issues = validate_labels(constant_labels, self.prices)

        assert not is_valid, "Should detect constant labels"
        assert any("same value" in issue for issue in issues), f"Should report constant values: {issues}"

        print(f"Constant labels test: Correctly detected issues: {issues}")


class TestFractionalDifferentiation:
    """Test fractional differentiation functionality."""

    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        # Create integrated series (non-stationary)
        self.series = pd.Series(
            np.cumsum(np.random.randn(1000)),
            index=pd.date_range("2023-01-01", periods=1000, freq="1H")
        )

    def test_basic_fractional_diff(self):
        """Test basic fractional differentiation."""
        frac_diff = fractional_differentiation(self.series, d=0.5)

        assert len(frac_diff) > 0, "Should generate fractionally differentiated series"
        assert len(frac_diff) < len(self.series), "Output should be shorter due to lookback"
        assert not frac_diff.isna().all(), "Should not be all NaN"

        # Should reduce autocorrelation
        original_autocorr = self.series.autocorr(lag=1)
        frac_diff_autocorr = frac_diff.autocorr(lag=1)

        print(f"Fractional diff test: Original autocorr = {original_autocorr:.3f}, "
              f"Frac diff autocorr = {frac_diff_autocorr:.3f}")

        # Typically should reduce autocorrelation
        assert abs(frac_diff_autocorr) < abs(original_autocorr) + 0.1, "Should reduce autocorrelation"

    def test_fractional_diff_parameters(self):
        """Test fractional differentiation with different parameters."""
        # Test different d values
        for d in [0.3, 0.5, 0.7]:
            frac_diff = fractional_differentiation(self.series, d=d)
            assert len(frac_diff) > 0, f"Should work with d={d}"

        # Test invalid d values
        with pytest.raises(ValueError):
            fractional_differentiation(self.series, d=0)  # d must be > 0

        with pytest.raises(ValueError):
            fractional_differentiation(self.series, d=1)  # d must be < 1

        print("Parameter test: Correctly handled valid and invalid parameters")


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
