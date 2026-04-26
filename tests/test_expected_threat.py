"""
Tests for Expected Threat (xT) feature engineering.
"""

from typing import ClassVar

import numpy as np
import pandas as pd

from src.features.expected_threat import aggregate_xt_features


class PrecomputedThreatCalculator:
    """Test double for aggregation with already prepared threat values."""

    zones: ClassVar[list[str]] = ["bottom", "middle", "top"]
    zone_threat_values: ClassVar[np.ndarray] = np.array([0.0, 0.1, 0.3])


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
                "id": [1, 2, 3],
                "player_appearance_id": [1, 1, 1],
                "period": ["half_1", "half_1", "half_1"],
                "minute": [5, 12, 25],
                "stage": ["middle", "top", "bottom"],
            }
        )

        result = aggregate_xt_features(
            pass_data,
            checkpoint_data,
            PrecomputedThreatCalculator(),  # type: ignore[arg-type]
        )

        # The transition 5->12 is valid. The pass at 25 must not become the
        # destination for minute 12 because it is after the checkpoint.
        assert result["cumul_xt_count"].iloc[0] == 1
        assert np.isclose(result["cumul_xt_added"].iloc[0], 0.2)
