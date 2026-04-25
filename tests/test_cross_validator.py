"""
Tests for cross-validation utilities.
"""

import numpy as np
import pandas as pd
import pytest

from src.validation.cross_validator import CrossValidator, create_cross_validator


class TestCrossValidator:
    """Test cross-validation split logic."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with multiple groups and classes."""
        return pd.DataFrame(
            {
                "fixture_id": [1, 1, 1, 2, 2, 3, 3, 4, 4, 5],
                "target": [0, 0, 1, 0, 0, 1, 0, 0, 1, 1],
            }
        )

    def test_group_kfold_no_overlap(self, sample_data):
        """Test that groups don't overlap between train and validation."""
        cv = CrossValidator(n_splits=2, stratify=False, shuffle=False)

        X = sample_data.drop(columns=["fixture_id", "target"])
        y = sample_data["target"]
        groups = sample_data["fixture_id"]

        for train_idx, val_idx in cv.split(X, y, groups):
            train_groups = set(groups.iloc[train_idx])
            val_groups = set(groups.iloc[val_idx])

            # No overlap
            assert len(train_groups & val_groups) == 0

    def test_all_samples_used_once(self, sample_data):
        """Test that each sample appears in validation exactly once."""
        cv = CrossValidator(n_splits=3, stratify=False, shuffle=False)

        X = sample_data.drop(columns=["fixture_id", "target"])
        y = sample_data["target"]
        groups = sample_data["fixture_id"]

        all_val_indices = []

        for train_idx, val_idx in cv.split(X, y, groups):
            all_val_indices.extend(val_idx)

        # Each index should appear exactly once
        assert len(all_val_indices) == len(sample_data)
        assert len(set(all_val_indices)) == len(sample_data)

    def test_stratified_split_balances_classes(self):
        """Test stratified split maintains class balance."""
        # Create imbalanced data
        data = pd.DataFrame(
            {
                "fixture_id": list(range(1, 21)),  # 20 fixtures
                "target": [0] * 18 + [1] * 2,  # 10% positive class
            }
        )

        cv = CrossValidator(n_splits=5, stratify=True, shuffle=True, random_state=42)

        X = pd.DataFrame({"dummy": range(len(data))})
        y = data["target"]
        groups = data["fixture_id"]

        positive_ratios = []

        for train_idx, val_idx in cv.split(X, y, groups):
            val_positive = y.iloc[val_idx].sum()
            val_total = len(val_idx)
            positive_ratios.append(val_positive / val_total)

        # All folds should have similar positive ratios (within tolerance)
        assert all(0.0 <= ratio <= 0.3 for ratio in positive_ratios)

    def test_create_cross_validator_default(self):
        """Test default cross-validator creation."""
        cv = create_cross_validator()

        assert cv.n_splits == 5
        assert cv.stratify == True
        assert cv.shuffle == True

    def test_create_cross_validator_custom_params(self):
        """Test custom cross-validator parameters."""
        cv = create_cross_validator(n_splits=3, stratify=False, shuffle=False)

        assert cv.n_splits == 3
        assert cv.stratify == False
        assert cv.shuffle == False

    def test_reproducible_splits_with_random_state(self):
        """Test that splits are reproducible with same random_state."""
        data = pd.DataFrame(
            {
                "fixture_id": list(range(1, 11)),
                "target": [0, 1] * 5,
            }
        )

        X = pd.DataFrame({"dummy": range(len(data))})
        y = data["target"]
        groups = data["fixture_id"]

        cv1 = CrossValidator(n_splits=3, stratify=True, shuffle=True, random_state=42)
        cv2 = CrossValidator(n_splits=3, stratify=True, shuffle=True, random_state=42)

        splits1 = list(cv1.split(X, y, groups))
        splits2 = list(cv2.split(X, y, groups))

        # Should produce identical splits
        for (train1, val1), (train2, val2) in zip(splits1, splits2):
            assert np.array_equal(train1, train2)
            assert np.array_equal(val1, val2)

    def test_different_random_states_different_splits(self):
        """Test that different random states produce different splits."""
        data = pd.DataFrame(
            {
                "fixture_id": list(range(1, 11)),
                "target": [0, 1] * 5,
            }
        )

        X = pd.DataFrame({"dummy": range(len(data))})
        y = data["target"]
        groups = data["fixture_id"]

        cv1 = CrossValidator(n_splits=3, stratify=True, shuffle=True, random_state=42)
        cv2 = CrossValidator(n_splits=3, stratify=True, shuffle=True, random_state=123)

        splits1 = list(cv1.split(X, y, groups))
        splits2 = list(cv2.split(X, y, groups))

        # At least one split should be different
        any_different = False
        for (train1, val1), (train2, val2) in zip(splits1, splits2):
            if not np.array_equal(val1, val2):
                any_different = True
                break

        assert any_different

    def test_minimum_samples_per_fold(self):
        """Test that each fold has minimum number of samples."""
        data = pd.DataFrame(
            {
                "fixture_id": list(range(1, 11)),
                "target": [0, 1] * 5,
            }
        )

        cv = CrossValidator(n_splits=3, stratify=False, shuffle=False)

        X = pd.DataFrame({"dummy": range(len(data))})
        y = data["target"]
        groups = data["fixture_id"]

        for train_idx, val_idx in cv.split(X, y, groups):
            # Each fold should have at least 1 sample
            assert len(train_idx) > 0
            assert len(val_idx) > 0

            # Train set should be larger than validation
            assert len(train_idx) > len(val_idx)
