"""
Tests for Press Resistance feature engineering.
"""

import numpy as np
import pandas as pd

from src.features.press_resistance import _calculate_single_checkpoint_features


class TestPressResistanceFeatures:
    """Test press resistance feature calculations."""

    def test_division_by_zero_protection(self):
        """Test that division by zero is handled gracefully."""
        pressure = pd.DataFrame(
            {
                "player_appearance_id": [1],
                "period": ["half_1"],
                "minute": [5],
                "stage": ["middle"],
                "press_induced_outcome": ["turnover"],
                "pass_angle": [np.nan],
            }
        )

        features = _calculate_single_checkpoint_features(
            pressure, ["forward_pass"], ["turnover"], {"middle": 0.3}
        )

        # Should not raise division by zero error
        assert features["retention"] == 0.0
        assert features["angle_std"] == 0.0
