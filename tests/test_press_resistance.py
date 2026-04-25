"""
Tests for Press Resistance feature engineering.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.press_resistance import (
    calculate_press_resistance_features,
    _calculate_single_checkpoint_features,
    _calculate_baseline_turnover_rates,
)


class TestPressResistanceFeatures:
    """Test press resistance feature calculations."""

    @pytest.fixture
    def checkpoint_data(self):
        """Create checkpoint data."""
        return pd.DataFrame(
            {
                "player_appearance_id": [1, 2],
                "checkpoint_min": [15, 15],
                "checkpoint_period": ["half_1", "half_1"],
                "minute_in": [0, 0],
            }
        )

    @pytest.fixture
    def pressure_data(self):
        """Create pressure event data."""
        return pd.DataFrame(
            {
                "player_appearance_id": [1, 1, 1, 2],
                "period": ["half_1"] * 4,
                "minute": [5, 10, 12, 8],
                "stage": ["middle", "middle", "top", "bottom"],
                "press_induced_outcome": [
                    "forward_pass",
                    "turnover",
                    "ball_carry",
                    "forward_pass",
                ],
                "pass_angle": [45.0, np.nan, 30.0, 60.0],
            }
        )

    @pytest.fixture
    def pass_data(self):
        """Create regular pass data (for baselines)."""
        return pd.DataFrame(
            {
                "player_appearance_id": [1] * 5,
                "period": ["half_1"] * 5,
                "minute": range(1, 6),
                "accurate": [True, True, False, True, True],
            }
        )

    def test_retention_rate_calculation(self, pressure_data):
        """Test press retention rate calculation."""
        positive_outcomes = ["forward_pass", "backward_pass", "ball_carry"]
        negative_outcomes = ["turnover"]
        baseline_rates = {"bottom": 0.3, "middle": 0.3, "top": 0.3}

        # Player 1 pressure events
        player1_pressure = pressure_data[pressure_data["player_appearance_id"] == 1]

        features = _calculate_single_checkpoint_features(
            player1_pressure, positive_outcomes, negative_outcomes, baseline_rates
        )

        # Player 1: 2 positive (forward_pass, ball_carry) out of 3 total
        expected_retention = 2 / 3
        assert np.isclose(features["retention"], expected_retention)

    def test_progressive_press_count(self, pressure_data):
        """Test progressive press evasion count."""
        positive_outcomes = ["forward_pass", "backward_pass", "ball_carry"]
        negative_outcomes = ["turnover"]
        baseline_rates = {"bottom": 0.3, "middle": 0.3, "top": 0.3}

        player1_pressure = pressure_data[pressure_data["player_appearance_id"] == 1]

        features = _calculate_single_checkpoint_features(
            player1_pressure, positive_outcomes, negative_outcomes, baseline_rates
        )

        # Player 1: 1 forward_pass + 1 ball_carry = 2 progressive
        assert features["progressive"] == 2

    def test_angle_std_calculation(self, pressure_data):
        """Test pass angle standard deviation."""
        positive_outcomes = ["forward_pass", "backward_pass", "ball_carry"]
        negative_outcomes = ["turnover"]
        baseline_rates = {"bottom": 0.3, "middle": 0.3, "top": 0.3}

        player1_pressure = pressure_data[pressure_data["player_appearance_id"] == 1]

        features = _calculate_single_checkpoint_features(
            player1_pressure, positive_outcomes, negative_outcomes, baseline_rates
        )

        # Player 1: angles are 45.0, NaN, 30.0 -> std of [45.0, 30.0]
        expected_std = np.std([45.0, 30.0])
        assert np.isclose(features["angle_std"], expected_std)

    def test_press_quality_score(self, pressure_data):
        """Test press quality (expected - actual turnovers)."""
        positive_outcomes = ["forward_pass", "backward_pass", "ball_carry"]
        negative_outcomes = ["turnover"]

        # Set baseline: 50% turnover rate in middle, 20% in top
        baseline_rates = {"bottom": 0.3, "middle": 0.5, "top": 0.2}

        player1_pressure = pressure_data[pressure_data["player_appearance_id"] == 1]

        features = _calculate_single_checkpoint_features(
            player1_pressure, positive_outcomes, negative_outcomes, baseline_rates
        )

        # Player 1: 2 events in middle (expect 1.0 turnovers), 1 in top (expect 0.2)
        # Expected total: 1.2, Actual: 1
        # Quality: 1.2 - 1 = 0.2
        expected_turnovers = 2 * 0.5 + 1 * 0.2
        actual_turnovers = 1
        expected_quality = expected_turnovers - actual_turnovers

        assert np.isclose(features["quality"], expected_quality)

    def test_empty_pressure_returns_zeros(self):
        """Test handling of players with no pressure events."""
        empty_pressure = pd.DataFrame(
            {
                "player_appearance_id": [],
                "period": [],
                "minute": [],
                "stage": [],
                "press_induced_outcome": [],
                "pass_angle": [],
            }
        )

        features = _calculate_single_checkpoint_features(
            empty_pressure, ["forward_pass"], ["turnover"], {"middle": 0.3}
        )

        assert features["retention"] == 0.0
        assert features["progressive"] == 0
        assert features["angle_std"] == 0.0
        assert features["quality"] == 0.0
        assert features["count"] == 0

    def test_division_by_zero_protection(self):
        """Test that division by zero is handled gracefully."""
        # Single event to test edge cases
        pressure = pd.DataFrame(
            {
                "player_appearance_id": [1],
                "period": ["half_1"],
                "minute": [5],
                "stage": ["middle"],
                "press_induced_outcome": ["turnover"],
                "pass_angle": [np.nan],  # No valid angles
            }
        )

        features = _calculate_single_checkpoint_features(
            pressure, ["forward_pass"], ["turnover"], {"middle": 0.3}
        )

        # Should not raise, and retention should be 0/1 = 0
        assert features["retention"] == 0.0
        assert features["angle_std"] == 0.0  # No valid angles

    def test_baseline_turnover_rates(self):
        """Test baseline turnover rate calculation."""
        pressure = pd.DataFrame(
            {
                "stage": ["bottom", "bottom", "middle", "middle", "top"],
                "press_induced_outcome": [
                    "turnover",
                    "forward_pass",
                    "turnover",
                    "turnover",
                    "ball_carry",
                ],
            }
        )

        baseline_rates = _calculate_baseline_turnover_rates(pressure, ["turnover"])

        # bottom: 1/2 = 0.5, middle: 2/2 = 1.0, top: 0/1 = 0.0
        assert np.isclose(baseline_rates["bottom"], 0.5)
        assert np.isclose(baseline_rates["middle"], 1.0)
        assert np.isclose(baseline_rates["top"], 0.0)

    def test_temporal_boundaries(self, checkpoint_data, pressure_data, pass_data):
        """Test that features respect temporal boundaries."""
        result = calculate_press_resistance_features(
            pressure_data, pass_data, checkpoint_data
        )

        # All pressure events are before minute 15, so should be included
        player1 = result[result["player_appearance_id"] == 1].iloc[0]

        # Player 1 has 3 pressure events before minute 15
        assert player1["cumul_press_count"] == 3

        # Also check last15 includes all (all events between 0-15)
        assert player1["last15_press_count"] == 3

    def test_different_periods_excluded(self):
        """Test that pressure events from different periods are excluded."""
        checkpoint = pd.DataFrame(
            {
                "player_appearance_id": [1],
                "checkpoint_min": [15],
                "checkpoint_period": ["half_2"],  # Second half
                "minute_in": [0],
            }
        )

        pressure = pd.DataFrame(
            {
                "player_appearance_id": [1, 1],
                "period": ["half_1", "half_2"],  # Different periods
                "minute": [5, 10],
                "stage": ["middle", "middle"],
                "press_induced_outcome": ["forward_pass", "turnover"],
                "pass_angle": [45.0, 30.0],
            }
        )

        pass_df = pd.DataFrame(columns=["player_appearance_id", "period", "minute", "accurate"])

        result = calculate_press_resistance_features(pressure, pass_df, checkpoint)

        # Should only include half_2 event
        assert result["cumul_press_count"].iloc[0] == 1
