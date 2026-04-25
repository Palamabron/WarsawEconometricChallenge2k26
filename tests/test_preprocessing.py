"""
Tests for preprocessing utilities.
"""

import numpy as np
import pandas as pd
import pytest

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
