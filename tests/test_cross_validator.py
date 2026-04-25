"""
Tests for cross-validation utilities.
"""

import pandas as pd

from src.validation.cross_validator import FixtureGroupKFold


class TestCrossValidator:
    """Test cross-validation split logic."""

    def test_group_kfold_no_overlap(self):
        """Test that groups don't overlap between train and validation."""
        data = pd.DataFrame(
            {
                "fixture_id": [1, 1, 1, 2, 2, 3, 3, 4, 4, 5],
                "target": [0, 0, 1, 0, 0, 1, 0, 0, 1, 1],
            }
        )

        cv = FixtureGroupKFold(n_splits=2, stratify=False, shuffle=False)

        X = data.drop(columns=["fixture_id", "target"])
        y = data["target"]
        groups = data["fixture_id"]

        for train_idx, val_idx in cv.split(X, y, groups):
            train_groups = set(groups.iloc[train_idx])
            val_groups = set(groups.iloc[val_idx])

            assert len(train_groups & val_groups) == 0
