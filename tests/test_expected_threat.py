"""
Tests for Expected Threat (xT) feature engineering.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.expected_threat import (
    ExpectedThreatCalculator,
    aggregate_xt_features,
)


class TestExpectedThreatCalculator:
    """Test xT calculation and convergence."""

    @pytest.fixture
    def synthetic_passes(self):
        """Create synthetic pass data with known transition probabilities."""
        # Create deterministic transitions: bottom -> middle -> top
        passes = pd.DataFrame(
            {
                "player_appearance_id": [1] * 9,
                "period": ["half_1"] * 9,
                "minute": range(1, 10),
                "stage": ["bottom", "middle", "top"] * 3,
                "addressee_player_appearance_id": [2] * 9,
                "accurate": [True] * 9,
            }
        )
        return passes

    @pytest.fixture
    def calculator(self):
        """Create xT calculator instance."""
        return ExpectedThreatCalculator()

    def test_transition_matrix_shape(self, calculator, synthetic_passes):
        """Test that transition matrix has correct shape."""
        # Add zone columns
        stage_map = {"bottom": "defensive", "middle": "central", "top": "attacking"}
        synthetic_passes["zone_origin"] = synthetic_passes["stage"].map(stage_map)
        synthetic_passes["zone_dest"] = synthetic_passes["zone_origin"]

        matrix = calculator.compute_transition_matrix(synthetic_passes)

        assert matrix.shape == (3, 3), "Transition matrix should be 3x3"
        assert set(matrix.index) == {"defensive", "central", "attacking"}
        assert set(matrix.columns) == {"defensive", "central", "attacking"}

    def test_transition_probabilities_sum_to_one(self, calculator, synthetic_passes):
        """Test that each row of transition matrix sums to ~1.0."""
        stage_map = {"bottom": "defensive", "middle": "central", "top": "attacking"}
        synthetic_passes["zone_origin"] = synthetic_passes["stage"].map(stage_map)
        synthetic_passes["zone_dest"] = synthetic_passes["zone_origin"]

        matrix = calculator.compute_transition_matrix(synthetic_passes)

        for zone in matrix.index:
            row_sum = matrix.loc[zone].sum()
            assert np.isclose(row_sum, 1.0, atol=0.01), f"Row {zone} should sum to 1.0"

    def test_xt_convergence(self, calculator):
        """Test that xT values converge after iterations."""
        # Simple transition matrix: can only move forward
        transition_matrix = pd.DataFrame(
            {
                "defensive": [0.0, 1.0, 0.0],  # Always goes to central
                "central": [0.0, 0.0, 1.0],  # Always goes to attacking
                "attacking": [0.0, 0.0, 1.0],  # Stays in attacking
            },
            index=["defensive", "central", "attacking"],
        )

        # Shot probability higher in attacking third
        shot_probs = {"defensive": 0.01, "central": 0.05, "attacking": 0.20}

        xt_values = calculator.compute_expected_threat(transition_matrix, shot_probs)

        # Attacking zone should have highest xT
        assert xt_values["attacking"] > xt_values["central"]
        assert xt_values["central"] > xt_values["defensive"]

        # Values should be reasonable (0-1 range)
        for zone, value in xt_values.items():
            assert 0.0 <= value <= 1.0, f"xT for {zone} should be in [0, 1]"

    def test_xt_values_increase_toward_goal(self, calculator):
        """Test that xT increases as you move toward goal."""
        # Realistic transition matrix
        transition_matrix = pd.DataFrame(
            {
                "defensive": [0.5, 0.4, 0.1],
                "central": [0.2, 0.5, 0.3],
                "attacking": [0.1, 0.3, 0.6],
            },
            index=["defensive", "central", "attacking"],
        )

        shot_probs = {"defensive": 0.02, "central": 0.08, "attacking": 0.25}

        xt_values = calculator.compute_expected_threat(transition_matrix, shot_probs)

        # Should monotonically increase
        assert xt_values["attacking"] > xt_values["central"] > xt_values["defensive"]


class TestAggregateXTFeatures:
    """Test xT feature aggregation per player checkpoint."""

    @pytest.fixture
    def checkpoint_data(self):
        """Create checkpoint data."""
        return pd.DataFrame(
            {
                "player_appearance_id": [1, 1, 2],
                "checkpoint_min": [15, 30, 15],
                "checkpoint_period": ["half_1", "half_1", "half_1"],
                "minute_in": [0, 0, 0],
            }
        )

    @pytest.fixture
    def pass_data_with_xt(self):
        """Create pass data with xT values."""
        return pd.DataFrame(
            {
                "player_appearance_id": [1, 1, 1, 2],
                "period": ["half_1", "half_1", "half_1", "half_1"],
                "minute": [5, 12, 25, 10],
                "xt_added": [0.1, 0.2, -0.05, 0.15],
            }
        )

    def test_temporal_filtering(self, checkpoint_data, pass_data_with_xt):
        """Test that only passes before checkpoint are included."""
        result = aggregate_xt_features(checkpoint_data, pass_data_with_xt)

        # Player 1, checkpoint 15: should include passes at minute 5, 12 (not 25)
        player1_ck15 = result[
            (result["player_appearance_id"] == 1) & (result["checkpoint_min"] == 15)
        ].iloc[0]

        assert player1_ck15["cumul_xt_count"] == 2
        assert np.isclose(player1_ck15["cumul_xt_added"], 0.3)  # 0.1 + 0.2

    def test_rolling_window(self, checkpoint_data, pass_data_with_xt):
        """Test 15-minute rolling window calculation."""
        result = aggregate_xt_features(checkpoint_data, pass_data_with_xt)

        # Player 1, checkpoint 30: last15 should only include minute 25 (>15, <=30)
        player1_ck30 = result[
            (result["player_appearance_id"] == 1) & (result["checkpoint_min"] == 30)
        ].iloc[0]

        assert player1_ck30["last15_xt_count"] == 1
        assert np.isclose(player1_ck30["last15_xt_added"], -0.05)

    def test_cumulative_from_minute_in(self, pass_data_with_xt):
        """Test cumulative features respect player substitution."""
        checkpoint_data = pd.DataFrame(
            {
                "player_appearance_id": [1],
                "checkpoint_min": [30],
                "checkpoint_period": ["half_1"],
                "minute_in": [10],  # Player entered at minute 10
            }
        )

        result = aggregate_xt_features(checkpoint_data, pass_data_with_xt)

        # Should only include passes at minute >= 10 (12, 25), not minute 5
        assert result["cumul_xt_count"].iloc[0] == 2
        assert np.isclose(result["cumul_xt_added"].iloc[0], 0.15)  # 0.2 + (-0.05)

    def test_no_passes_returns_zeros(self, checkpoint_data):
        """Test that players with no passes get zero features."""
        empty_passes = pd.DataFrame(
            {
                "player_appearance_id": [],
                "period": [],
                "minute": [],
                "xt_added": [],
            }
        )

        result = aggregate_xt_features(checkpoint_data, empty_passes)

        assert (result["cumul_xt_count"] == 0).all()
        assert (result["cumul_xt_added"] == 0.0).all()
        assert (result["last15_xt_count"] == 0).all()
        assert (result["last15_xt_added"] == 0.0).all()

    def test_period_filtering(self):
        """Test that passes from different periods are excluded."""
        checkpoint_data = pd.DataFrame(
            {
                "player_appearance_id": [1],
                "checkpoint_min": [15],
                "checkpoint_period": ["half_2"],  # Second half
                "minute_in": [0],
            }
        )

        pass_data = pd.DataFrame(
            {
                "player_appearance_id": [1, 1],
                "period": ["half_1", "half_2"],  # Different periods
                "minute": [10, 10],
                "xt_added": [0.5, 0.3],
            }
        )

        result = aggregate_xt_features(checkpoint_data, pass_data)

        # Should only include half_2 pass
        assert result["cumul_xt_count"].iloc[0] == 1
        assert np.isclose(result["cumul_xt_added"].iloc[0], 0.3)
