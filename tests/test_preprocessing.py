"""
Tests for preprocessing utilities.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import CategoricalEncoder, safe_fillna, safe_clip_infinite


class TestCategoricalEncoder:
    """Test categorical feature encoding."""

    @pytest.fixture
    def train_data(self):
        """Create training data with categorical features."""
        return pd.DataFrame(
            {
                "position": ["F", "M", "D", "F", "M"],
                "formation": ["4-4-2", "4-3-3", "4-4-2", "3-5-2", "4-3-3"],
                "numeric_col": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )

    @pytest.fixture
    def test_data(self):
        """Create test data with some unseen categories."""
        return pd.DataFrame(
            {
                "position": ["F", "G", "D"],  # G is unseen
                "formation": ["4-4-2", "5-3-2", "4-3-3"],  # 5-3-2 is unseen
                "numeric_col": [6.0, 7.0, 8.0],
            }
        )

    def test_fit_transform(self, train_data):
        """Test fitting and transforming in one step."""
        encoder = CategoricalEncoder()
        result = encoder.fit_transform(train_data)

        # Check categorical columns are encoded
        assert result["position"].dtype in [np.int64, np.int32]
        assert result["formation"].dtype in [np.int64, np.int32]

        # Numeric column unchanged
        assert result["numeric_col"].dtype == np.float64

    def test_consistent_encoding_across_folds(self, train_data):
        """Test that same categories get same codes across transformations."""
        encoder = CategoricalEncoder()
        encoder.fit(train_data)

        # Transform same data twice
        result1 = encoder.transform(train_data.copy())
        result2 = encoder.transform(train_data.copy())

        # Should be identical
        pd.testing.assert_frame_equal(result1, result2)

    def test_unseen_categories_mapped_to_minus_one(self, train_data, test_data):
        """Test that unseen categories are mapped to -1."""
        encoder = CategoricalEncoder()
        encoder.fit(train_data)

        result = encoder.transform(test_data)

        # G (unseen position) should be -1
        assert result.loc[result["position"] == -1].shape[0] == 1

        # 5-3-2 (unseen formation) should be -1
        assert result.loc[result["formation"] == -1].shape[0] == 1

    def test_auto_detect_categorical(self, train_data):
        """Test automatic detection of categorical columns."""
        encoder = CategoricalEncoder()
        encoder.fit(train_data)  # No categorical_cols specified

        # Should detect position and formation as categorical
        assert "position" in encoder.encoders
        assert "formation" in encoder.encoders
        assert "numeric_col" not in encoder.encoders

    def test_explicit_categorical_list(self, train_data):
        """Test specifying categorical columns explicitly."""
        encoder = CategoricalEncoder()
        encoder.fit(train_data, categorical_cols=["position"])

        # Only position should be encoded
        assert "position" in encoder.encoders
        assert "formation" not in encoder.encoders

    def test_transform_without_fit_raises(self, train_data):
        """Test that transform without fit raises error."""
        encoder = CategoricalEncoder()

        with pytest.raises(ValueError, match="must be fitted"):
            encoder.transform(train_data)

    def test_get_feature_names(self, train_data):
        """Test getting encoded feature names."""
        encoder = CategoricalEncoder()
        encoder.fit(train_data)

        feature_names = encoder.get_feature_names()

        assert set(feature_names) == {"position", "formation"}

    def test_preserves_dataframe_index(self, train_data):
        """Test that DataFrame index is preserved."""
        train_data.index = [10, 20, 30, 40, 50]

        encoder = CategoricalEncoder()
        result = encoder.fit_transform(train_data)

        assert list(result.index) == [10, 20, 30, 40, 50]


class TestSafeFillNA:
    """Test safe NaN filling."""

    def test_fill_numeric_nan(self):
        """Test filling NaN in numeric columns."""
        df = pd.DataFrame(
            {
                "a": [1.0, np.nan, 3.0],
                "b": [4.0, 5.0, np.nan],
            }
        )

        result = safe_fillna(df, value=0)

        assert result["a"].isna().sum() == 0
        assert result["b"].isna().sum() == 0
        assert result.loc[1, "a"] == 0
        assert result.loc[2, "b"] == 0

    def test_fill_with_custom_value(self):
        """Test filling with custom value."""
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})

        result = safe_fillna(df, value=-999)

        assert result.loc[1, "a"] == -999


class TestSafeClipInfinite:
    """Test infinite value clipping."""

    def test_clip_positive_infinity(self):
        """Test clipping positive infinity."""
        df = pd.DataFrame(
            {
                "a": [1.0, np.inf, 3.0],
                "b": [4.0, 5.0, 6.0],
            }
        )

        result = safe_clip_infinite(df, lower=-1e6, upper=1e6)

        assert result["a"].iloc[1] == 1e6
        assert not np.isinf(result["a"]).any()

    def test_clip_negative_infinity(self):
        """Test clipping negative infinity."""
        df = pd.DataFrame({"a": [1.0, -np.inf, 3.0]})

        result = safe_clip_infinite(df, lower=-1e6, upper=1e6)

        assert result["a"].iloc[1] == -1e6
        assert not np.isinf(result["a"]).any()

    def test_numeric_columns_only(self):
        """Test that only numeric columns are processed."""
        df = pd.DataFrame(
            {
                "numeric": [1.0, np.inf, 3.0],
                "text": ["a", "b", "c"],
            }
        )

        result = safe_clip_infinite(df)

        # Numeric column clipped
        assert not np.isinf(result["numeric"]).any()

        # Text column unchanged
        assert result["text"].tolist() == ["a", "b", "c"]
