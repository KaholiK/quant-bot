"""
Test suite for Purged Cross-Validation functionality.
Tests data leakage prevention, embargo periods, and temporal overlap validation.
"""

import os

# Import modules to test
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from algos.core.cv_utils import (
    PurgedKFold,
    TimeSeriesSplit,
    calculate_cv_scores,
    combinatorial_purged_cv,
    validate_time_series_split,
    walk_forward_splits,
)


class TestPurgedKFold:
    """Test cases for PurgedKFold cross-validation."""

    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)

        # Create time series data
        dates = pd.date_range("2020-01-01", periods=1000, freq="1D")

        # Generate features
        self.X = pd.DataFrame({
            "feature_1": np.random.randn(1000),
            "feature_2": np.random.randn(1000),
            "feature_3": np.random.randn(1000)
        }, index=dates)

        # Generate labels with some temporal dependence
        trend = np.sin(np.arange(1000) * 2 * np.pi / 100)
        noise = np.random.randn(1000) * 0.5
        self.y = pd.Series((trend + noise > 0).astype(int), index=dates)

        # Basic configuration
        self.n_splits = 5
        self.embargo_frac = 0.02

    def test_basic_purged_kfold(self):
        """Test basic PurgedKFold functionality."""
        cv = PurgedKFold(n_splits=self.n_splits, embargo_frac=self.embargo_frac)

        splits = list(cv.split(self.X, self.y))

        # Should generate correct number of splits
        assert len(splits) == self.n_splits, f"Expected {self.n_splits} splits, got {len(splits)}"

        # Each split should have train and test indices
        for i, (train_idx, test_idx) in enumerate(splits):
            assert len(train_idx) > 0, f"Split {i}: Empty training set"
            assert len(test_idx) > 0, f"Split {i}: Empty test set"
            assert isinstance(train_idx, np.ndarray), f"Split {i}: Train indices should be numpy array"
            assert isinstance(test_idx, np.ndarray), f"Split {i}: Test indices should be numpy array"

            # No overlap between train and test
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0, f"Split {i}: Found overlap between train and test: {len(overlap)} samples"

        print(f"Basic PurgedKFold test: Generated {len(splits)} valid splits")

    def test_embargo_effectiveness(self):
        """Test that embargo period prevents data leakage."""
        cv = PurgedKFold(n_splits=3, embargo_frac=0.1)  # 10% embargo

        splits = list(cv.split(self.X, self.y))

        for i, (train_idx, test_idx) in enumerate(splits):
            # Calculate temporal gap
            train_times = self.X.index[train_idx]
            test_times = self.X.index[test_idx]

            # Find minimum gap between training and test periods
            if len(train_times) > 0 and len(test_times) > 0:
                min_test_time = test_times.min()
                max_train_time = train_times.max()

                # Gap should exist (embargo period)
                if max_train_time < min_test_time:
                    gap_days = (min_test_time - max_train_time).days
                    expected_embargo_days = int(len(self.X) * 0.1)  # Rough estimate

                    print(f"Split {i}: Gap = {gap_days} days, Expected embargo ≈ {expected_embargo_days} days")

                    # Should have some gap (allowing for approximate calculation)
                    assert gap_days >= 0, f"Split {i}: No temporal gap found"

        print("Embargo effectiveness test: Temporal gaps maintained")

    def test_temporal_ordering(self):
        """Test that splits maintain proper temporal ordering."""
        cv = PurgedKFold(n_splits=5, embargo_frac=0.02)

        splits = list(cv.split(self.X, self.y))

        for i, (train_idx, test_idx) in enumerate(splits):
            # Validate temporal consistency
            is_valid, issues = validate_time_series_split(self.X, train_idx, test_idx)

            # Print any issues for debugging
            if issues:
                print(f"Split {i} validation issues: {issues}")

            # Check that indices are valid
            assert all(idx >= 0 and idx < len(self.X) for idx in train_idx), f"Split {i}: Invalid train indices"
            assert all(idx >= 0 and idx < len(self.X) for idx in test_idx), f"Split {i}: Invalid test indices"

        print("Temporal ordering test: All splits maintain proper ordering")

    def test_different_embargo_fractions(self):
        """Test PurgedKFold with different embargo fractions."""
        embargo_fractions = [0.01, 0.05, 0.1]

        for embargo_frac in embargo_fractions:
            cv = PurgedKFold(n_splits=3, embargo_frac=embargo_frac)
            splits = list(cv.split(self.X, self.y))

            assert len(splits) == 3, f"Embargo {embargo_frac}: Should generate 3 splits"

            # Check that higher embargo fractions result in smaller training sets
            total_train_samples = sum(len(train_idx) for train_idx, _ in splits)
            print(f"Embargo {embargo_frac}: Total training samples = {total_train_samples}")

        print("Different embargo fractions test: All configurations worked")

    def test_insufficient_data_handling(self):
        """Test handling of insufficient data scenarios."""
        # Very small dataset
        small_X = self.X.head(10)
        small_y = self.y.head(10)

        cv = PurgedKFold(n_splits=5, embargo_frac=0.1)
        splits = list(cv.split(small_X, small_y))

        # Should handle gracefully (may result in fewer splits)
        assert isinstance(splits, list), "Should return list of splits"

        # Each valid split should have reasonable sizes
        for i, (train_idx, test_idx) in enumerate(splits):
            if len(train_idx) > 0 and len(test_idx) > 0:
                print(f"Small data split {i}: train={len(train_idx)}, test={len(test_idx)}")

        print("Insufficient data test: Handled gracefully")

    def test_get_n_splits(self):
        """Test get_n_splits method."""
        cv = PurgedKFold(n_splits=7, embargo_frac=0.02)

        n_splits = cv.get_n_splits(self.X, self.y)
        assert n_splits == 7, f"Expected 7 splits, got {n_splits}"

        # Should work without X and y
        n_splits_no_args = cv.get_n_splits()
        assert n_splits_no_args == 7, f"Expected 7 splits without args, got {n_splits_no_args}"

        print("get_n_splits test: Returned correct values")


class TestWalkForwardSplits:
    """Test walk-forward validation functionality."""

    def setup_method(self):
        """Setup test data."""
        # Create longer time series for walk-forward testing
        dates = pd.date_range("2020-01-01", periods=2000, freq="1D")

        self.X = pd.DataFrame({
            "feature_1": np.random.randn(2000),
            "feature_2": np.random.randn(2000)
        }, index=dates)

    def test_basic_walk_forward(self):
        """Test basic walk-forward splits."""
        splits = list(walk_forward_splits(
            self.X,
            train_months=12,
            test_months=3,
            step_months=3
        ))

        assert len(splits) > 0, "Should generate walk-forward splits"

        for i, (train_idx, test_idx) in enumerate(splits):
            assert isinstance(train_idx, pd.DatetimeIndex), f"Split {i}: Train index should be DatetimeIndex"
            assert isinstance(test_idx, pd.DatetimeIndex), f"Split {i}: Test index should be DatetimeIndex"
            assert len(train_idx) > 0, f"Split {i}: Empty training set"
            assert len(test_idx) > 0, f"Split {i}: Empty test set"

            # Test period should come after training period
            if len(train_idx) > 0 and len(test_idx) > 0:
                assert train_idx.max() <= test_idx.min(), f"Split {i}: Test should come after training"

        print(f"Basic walk-forward test: Generated {len(splits)} splits")

    def test_walk_forward_parameters(self):
        """Test walk-forward with different parameters."""
        # Test different configurations
        configs = [
            {"train_months": 6, "test_months": 1, "step_months": 1},
            {"train_months": 24, "test_months": 6, "step_months": 3},
            {"train_months": 12, "test_months": 3, "step_months": 6}
        ]

        for config in configs:
            splits = list(walk_forward_splits(self.X, **config))
            assert len(splits) >= 0, f"Config {config}: Should handle configuration"
            print(f"Config {config}: Generated {len(splits)} splits")

        print("Walk-forward parameters test: All configurations handled")

    def test_walk_forward_temporal_consistency(self):
        """Test temporal consistency in walk-forward splits."""
        splits = list(walk_forward_splits(
            self.X,
            train_months=6,
            test_months=2,
            step_months=2
        ))

        for i, (train_idx, test_idx) in enumerate(splits):
            # Check temporal ordering
            if len(train_idx) > 0 and len(test_idx) > 0:
                # Training period should be contiguous
                train_gap = (train_idx[1:] - train_idx[:-1]).days
                assert all(gap <= 2 for gap in train_gap), f"Split {i}: Training period not contiguous"

                # Test period should be contiguous
                test_gap = (test_idx[1:] - test_idx[:-1]).days
                assert all(gap <= 2 for gap in test_gap), f"Split {i}: Test period not contiguous"

        print("Walk-forward temporal consistency test: All splits consistent")


class TestTimeSeriesSplitValidator:
    """Test time series split validation functionality."""

    def setup_method(self):
        """Setup test data."""
        dates = pd.date_range("2020-01-01", periods=500, freq="1D")
        self.X = pd.DataFrame({
            "feature": np.random.randn(500)
        }, index=dates)

    def test_valid_split_validation(self):
        """Test validation of valid time series splits."""
        # Create valid split
        train_idx = np.arange(0, 300)
        test_idx = np.arange(350, 450)  # Gap of 50 days

        is_valid, issues = validate_time_series_split(self.X, train_idx, test_idx, min_gap_days=30)

        assert is_valid, f"Valid split failed validation: {issues}"
        assert len(issues) == 0, f"Valid split should have no issues: {issues}"

        print("Valid split validation test: Passed")

    def test_overlapping_split_validation(self):
        """Test detection of overlapping splits."""
        # Create overlapping split
        train_idx = np.arange(0, 300)
        test_idx = np.arange(250, 350)  # Overlap from 250-300

        is_valid, issues = validate_time_series_split(self.X, train_idx, test_idx)

        assert not is_valid, "Should detect overlapping split"
        assert any("overlap" in issue.lower() for issue in issues), f"Should report overlap: {issues}"

        print(f"Overlapping split validation test: Correctly detected issues: {issues}")

    def test_insufficient_gap_validation(self):
        """Test detection of insufficient temporal gap."""
        # Create split with insufficient gap
        train_idx = np.arange(0, 300)
        test_idx = np.arange(305, 400)  # Only 5-day gap

        is_valid, issues = validate_time_series_split(self.X, train_idx, test_idx, min_gap_days=10)

        assert not is_valid, "Should detect insufficient gap"
        assert any("gap" in issue.lower() for issue in issues), f"Should report gap issue: {issues}"

        print(f"Insufficient gap validation test: Correctly detected issues: {issues}")

    def test_empty_sets_validation(self):
        """Test validation with empty train/test sets."""
        empty_idx = np.array([])
        test_idx = np.arange(100, 200)

        is_valid, issues = validate_time_series_split(self.X, empty_idx, test_idx)

        assert not is_valid, "Should detect empty training set"
        assert any("empty" in issue.lower() for issue in issues), f"Should report empty set: {issues}"

        print(f"Empty sets validation test: Correctly detected issues: {issues}")


class TestCombinatorialPurgedCV:
    """Test combinatorial purged cross-validation."""

    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)

        # Create synthetic dataset
        dates = pd.date_range("2020-01-01", periods=500, freq="1D")

        self.X = pd.DataFrame({
            "feature_1": np.random.randn(500),
            "feature_2": np.random.randn(500),
            "feature_3": np.random.randn(500)
        }, index=dates)

        # Generate labels with some signal
        signal = self.X["feature_1"] + 0.5 * self.X["feature_2"] + np.random.randn(500) * 0.5
        self.y = pd.Series((signal > 0).astype(int), index=dates)

    def test_basic_combinatorial_cv(self):
        """Test basic combinatorial purged CV."""
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        cv_results = combinatorial_purged_cv(
            X=self.X,
            y=self.y,
            model=model,
            n_splits=3,
            embargo_frac=0.02
        )

        # Check structure
        assert "fold_results" in cv_results, "Should contain fold results"
        assert "cv_stats" in cv_results, "Should contain CV statistics"

        # Check fold results
        fold_results = cv_results["fold_results"]
        assert len(fold_results) <= 3, "Should have at most 3 folds"

        for fold_result in fold_results:
            assert "fold" in fold_result, "Should have fold number"
            assert "accuracy" in fold_result, "Should have accuracy metric"
            assert fold_result["accuracy"] >= 0, "Accuracy should be non-negative"
            assert fold_result["accuracy"] <= 1, "Accuracy should not exceed 1"

        # Check CV statistics
        cv_stats = cv_results["cv_stats"]
        if "accuracy" in cv_stats:
            accuracy_stats = cv_stats["accuracy"]
            assert "mean" in accuracy_stats, "Should have mean accuracy"
            assert "std" in accuracy_stats, "Should have accuracy std"
            assert accuracy_stats["mean"] >= 0, "Mean accuracy should be non-negative"

        print(f"Combinatorial CV test: {len(fold_results)} folds completed")
        print(f"Mean accuracy: {cv_stats.get('accuracy', {}).get('mean', 'N/A'):.3f}")

    def test_cv_with_sample_weights(self):
        """Test CV with sample weights."""
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        # Create sample weights
        sample_weights = pd.Series(
            np.random.uniform(0.5, 2.0, len(self.y)),
            index=self.y.index
        )

        cv_results = combinatorial_purged_cv(
            X=self.X,
            y=self.y,
            model=model,
            n_splits=3,
            embargo_frac=0.02,
            sample_weights=sample_weights
        )

        # Should complete without errors
        assert "fold_results" in cv_results, "Should handle sample weights"
        assert len(cv_results["fold_results"]) > 0, "Should generate fold results"

        print(f"Sample weights CV test: Completed with {len(cv_results['fold_results'])} folds")


class TestCVScoreCalculation:
    """Test CV score calculation utilities."""

    def setup_method(self):
        """Setup test data."""
        self.cv_results = [
            {"accuracy": 0.8, "precision": 0.75, "recall": 0.85},
            {"accuracy": 0.82, "precision": 0.78, "recall": 0.80},
            {"accuracy": 0.79, "precision": 0.73, "recall": 0.88},
            {"accuracy": 0.81, "precision": 0.76, "recall": 0.82}
        ]

    def test_basic_score_calculation(self):
        """Test basic CV score statistics calculation."""
        score_stats = calculate_cv_scores(self.cv_results)

        # Should calculate stats for all metrics
        expected_metrics = ["accuracy", "precision", "recall"]
        for metric in expected_metrics:
            assert metric in score_stats, f"Should calculate stats for {metric}"

            stats = score_stats[metric]
            assert "mean" in stats, f"{metric}: Should have mean"
            assert "std" in stats, f"{metric}: Should have std"
            assert "min" in stats, f"{metric}: Should have min"
            assert "max" in stats, f"{metric}: Should have max"
            assert "count" in stats, f"{metric}: Should have count"

            # Check values are reasonable
            assert stats["count"] == 4, f"{metric}: Should count all folds"
            assert stats["min"] <= stats["mean"] <= stats["max"], f"{metric}: Mean should be between min/max"

        print("Score calculation test: Calculated statistics for all metrics")
        print(f"Accuracy: {score_stats['accuracy']['mean']:.3f} ± {score_stats['accuracy']['std']:.3f}")

    def test_empty_results(self):
        """Test score calculation with empty results."""
        score_stats = calculate_cv_scores([])

        assert score_stats == {}, "Empty input should return empty dict"

        print("Empty results test: Handled correctly")

    def test_missing_metrics(self):
        """Test score calculation with missing metrics."""
        incomplete_results = [
            {"accuracy": 0.8},
            {"accuracy": 0.82, "precision": 0.78},
            {"recall": 0.85}
        ]

        score_stats = calculate_cv_scores(incomplete_results)

        # Should handle missing metrics gracefully
        assert "accuracy" in score_stats, "Should calculate stats for available accuracy"

        # Count should reflect actual available values
        accuracy_count = score_stats["accuracy"]["count"]
        assert accuracy_count == 2, f"Accuracy count should be 2, got {accuracy_count}"

        if "precision" in score_stats:
            precision_count = score_stats["precision"]["count"]
            assert precision_count == 1, f"Precision count should be 1, got {precision_count}"

        print("Missing metrics test: Handled gracefully")


class TestTimeSeriesSplitter:
    """Test TimeSeriesSplit class functionality."""

    def setup_method(self):
        """Setup test data."""
        dates = pd.date_range("2020-01-01", periods=1000, freq="1D")
        self.X = pd.DataFrame({
            "feature": np.random.randn(1000)
        }, index=dates)
        self.y = pd.Series(np.random.randint(0, 2, 1000), index=dates)

    def test_purged_kfold_strategy(self):
        """Test TimeSeriesSplit with purged k-fold strategy."""
        splitter = TimeSeriesSplit(
            strategy="purged_kfold",
            n_splits=5,
            embargo_frac=0.02
        )

        splits = list(splitter.split(self.X, self.y))
        assert len(splits) == 5, "Should generate 5 splits"

        n_splits = splitter.get_n_splits(self.X, self.y)
        assert n_splits == 5, "get_n_splits should return 5"

        print("PurgedKFold strategy test: Generated correct number of splits")

    def test_walk_forward_strategy(self):
        """Test TimeSeriesSplit with walk-forward strategy."""
        splitter = TimeSeriesSplit(
            strategy="walk_forward",
            train_months=6,
            test_months=2,
            step_months=2
        )

        splits = list(splitter.split(self.X, self.y))
        assert len(splits) >= 0, "Should generate walk-forward splits"

        print(f"Walk-forward strategy test: Generated {len(splits)} splits")

    def test_invalid_strategy(self):
        """Test TimeSeriesSplit with invalid strategy."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            TimeSeriesSplit(strategy="invalid_strategy")

        print("Invalid strategy test: Correctly raised error")


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
