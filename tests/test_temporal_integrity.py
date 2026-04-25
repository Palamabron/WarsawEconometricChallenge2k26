"""
Tests for temporal integrity and data leakage prevention
"""

import numpy as np
import pandas as pd
import pytest

from src.validation.leakage_checks import (
    LeakageValidator,
    check_feature_leakage_correlation,
    safe_temporal_merge,
)


def test_temporal_boundary_validation():
    """Test that events after checkpoint are flagged"""

    # Create mock checkpoint data
    checkpoint_df = pd.DataFrame(
        {
            "player_appearance_id": [1, 1, 2],
            "checkpoint_min": [15, 30, 15],
            "minute_in": [1, 1, 1],
            "minute_out": [90, 90, 90],
            "scored_after": [0, 1, 0],
        }
    )

    # Create mock event data (some events after checkpoint)
    event_df = pd.DataFrame(
        {
            "player_appearance_id": [1, 1, 1, 2, 2],
            "minute": [10, 20, 35, 12, 18],  # 35 is after checkpoint 30, 18 after 15
            "value": [1, 2, 3, 4, 5],
        }
    )

    # Valid merge should exclude future events
    merged = safe_temporal_merge(
        checkpoint_df,
        event_df,
        on=["player_appearance_id"],
        checkpoint_col="checkpoint_min",
        event_time_col="minute",
    )

    # Check no events after checkpoint
    assert (merged["minute"] <= merged["checkpoint_min"]).all()


def test_player_on_pitch_validation():
    """Test substitution boundary enforcement"""

    df = pd.DataFrame(
        {
            "player_appearance_id": [1, 2, 3],
            "checkpoint_min": [15, 50, 80],
            "minute_in": [1, 46, 1],  # Player 2 comes in at 46
            "minute_out": [90, 90, 70],  # Player 3 leaves at 70
            "scored_after": [0, 1, 0],
        }
    )

    filtered = LeakageValidator.validate_player_on_pitch(df)

    # Player 3's checkpoint at 80 should be removed (out at 70)
    assert len(filtered) == 2
    assert 3 not in filtered["player_appearance_id"].values


def test_cv_split_validation():
    """Test no match overlap between folds"""

    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    groups = np.repeat([1, 2, 3, 4, 5], 20)  # 5 matches, 20 obs each

    train_idx = np.arange(80)  # First 4 matches
    val_idx = np.arange(80, 100)  # Last match

    # Should pass - no overlap
    assert LeakageValidator.validate_cv_splits(X, y, groups, train_idx, val_idx)

    # Create overlap
    train_idx_bad = np.arange(90)  # Includes part of match 5
    val_idx_bad = np.arange(80, 100)  # Also includes match 5

    # Should raise error
    with pytest.raises(ValueError, match="overlap"):
        LeakageValidator.validate_cv_splits(X, y, groups, train_idx_bad, val_idx_bad)


def test_feature_leakage_correlation():
    """Test detection of suspiciously high correlations"""

    n = 1000
    X = pd.DataFrame(
        {
            "normal_feature": np.random.randn(n),
            "suspicious_feature": np.random.randn(n),
        }
    )

    y = np.random.randint(0, 2, n)

    # Make suspicious feature highly correlated with target
    X["suspicious_feature"] = y + np.random.randn(n) * 0.1

    suspicious = check_feature_leakage_correlation(X, pd.Series(y), threshold=0.9)

    # Should flag the suspicious feature
    assert "suspicious_feature" in suspicious
    assert abs(suspicious["suspicious_feature"]) > 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
