"""
Tests for preprocessing utilities.
"""

import numpy as np
import pandas as pd

from run_experiment import FoldArtifacts, feature_matrix
from scripts.analyze_ensemble_explainability import encoded_fold_matrices
from src.preprocessing import CategoricalEncoder


class TestCategoricalEncoder:
    """Test categorical feature encoding."""

    def test_consistent_encoding_across_folds(self):
        """Test that same categories get same codes across transformations."""
        train_data = pd.DataFrame(
            {
                "position": ["F", "M", "D", "F", "M"],
                "formation": ["4-4-2", "4-3-3", "4-4-2", "3-5-2", "4-3-3"],
            }
        )

        encoder = CategoricalEncoder()
        encoder.fit(train_data)

        result1 = encoder.transform(train_data.copy())
        result2 = encoder.transform(train_data.copy())

        pd.testing.assert_frame_equal(result1, result2)

    def test_unseen_categories_mapped_to_minus_one(self):
        """Test that unseen categories are mapped to -1."""
        train_data = pd.DataFrame({"position": ["F", "M", "D"]})
        test_data = pd.DataFrame({"position": ["F", "G"]})  # G is unseen

        encoder = CategoricalEncoder()
        encoder.fit(train_data)
        result = encoder.transform(test_data)

        assert result.loc[result["position"] == -1].shape[0] == 1


def test_feature_matrix_buckets_rare_categories_and_adds_train_only_encodings():
    train = pd.DataFrame(
        {
            "formation": ["common"] * 12 + ["rare_train"],
            "numeric": list(range(13)),
            "scored_after": [0] * 11 + [1, 1],
        }
    )
    val = pd.DataFrame(
        {
            "formation": ["common", "unseen"],
            "numeric": [1, None],
            "scored_after": [0, 1],
        }
    )

    x_train, x_val, _ = feature_matrix(
        train,
        val,
        ["formation", "numeric"],
        y_train=train["scored_after"],
    )

    assert "formation_rare" in x_train.columns
    assert "formation_frequency" in x_train.columns
    assert "formation_target_rate" in x_train.columns
    assert x_val.loc[1, "formation_rare"] == 1
    assert x_val["numeric"].isna().sum() == 0


def test_explainability_fold_matrices_include_target_encoded_columns():
    train = pd.DataFrame(
        {
            "formation": ["common"] * 12 + ["rare_train"],
            "numeric": list(range(13)),
            "scored_after": [0] * 11 + [1, 1],
        }
    )
    val = pd.DataFrame(
        {
            "formation": ["common", "unseen"],
            "numeric": [1, None],
            "scored_after": [0, 1],
        }
    )
    fold = FoldArtifacts(
        train_features=train,
        val_features=val,
        train_idx=np.arange(len(train)),
        val_idx=np.arange(len(val)),
        builder=object(),
    )

    x_train, x_val = encoded_fold_matrices([fold], ["formation", "numeric"])[0]

    assert "formation_target_rate" in x_train.columns
    assert x_train.columns.tolist() == x_val.columns.tolist()
