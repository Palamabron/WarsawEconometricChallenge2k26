"""
Tests for Expected Threat (xT) feature engineering.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.expected_threat import aggregate_xt_features


class TestAggregateXTFeatures:
    """Test xT feature aggregation per player checkpoint."""

    def test_temporal_filtering(self):
        """Test that only passes before checkpoint are included."""
        checkpoint_data = pd.DataFrame(
            {
                "player_appearance_id": [1],
                "checkpoint_min": [15],
                "checkpoint_period": ["half_1"],
                "minute_in": [0],
            }
        )

        pass_data = pd.DataFrame(
            {
                "player_appearance_id": [1, 1, 1],
                "period": ["half_1", "half_1", "half_1"],
                "minute": [5, 12, 25],
                "xt_added": [0.1, 0.2, -0.05],
            }
        )

        result = aggregate_xt_features(checkpoint_data, pass_data)

        # Should include passes at minute 5, 12 (not 25)
        assert result["cumul_xt_count"].iloc[0] == 2
        assert np.isclose(result["cumul_xt_added"].iloc[0], 0.3)
