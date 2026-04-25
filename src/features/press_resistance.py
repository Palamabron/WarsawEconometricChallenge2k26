"""
Press Resistance feature engineering

Analyzes player behavior under defensive pressure to quantify composure,
decision-making quality, and ability to progress the ball under duress.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union

try:
    import cudf
    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False
    cudf = None


def calculate_press_resistance_features(
    pressure_df: Union[pd.DataFrame, 'cudf.DataFrame'],
    pass_df: Union[pd.DataFrame, 'cudf.DataFrame'],
    checkpoint_df: Union[pd.DataFrame, 'cudf.DataFrame'],
    positive_outcomes: List[str] = None,
    negative_outcomes: List[str] = None,
    baseline_turnover_rates: Optional[Dict[str, float]] = None,
    use_gpu: bool = False
) -> Union[pd.DataFrame, 'cudf.DataFrame']:
    """
    Calculate press resistance features per player per checkpoint.

    Features:
    1. Press Retention Rate: % of pressure events with positive outcome
    2. Progressive Press Evasion: Count of forward passes/carries under pressure
    3. Pass Angle Distortion: std(angles under pressure) vs baseline
    4. Expected Turnovers vs Actual: Over/under performance vs zone baseline
    5. Press Count: total pressure events (used as denominator elsewhere)

    Args:
        pressure_df: Behavior under pressure events
        pass_df: Regular pass events (for baseline calculations)
        checkpoint_df: Checkpoint observations
        positive_outcomes: List of positive press outcomes
        negative_outcomes: List of negative press outcomes
        baseline_turnover_rates: Pre-computed zone turnover rates from training data.
            If None, rates are computed from pressure_df (use only outside CV loops to
            avoid leakage; pass training-fold rates when called inside a CV loop).
        use_gpu: Use cuDF backend

    Returns:
        DataFrame with press resistance features
    """
    # Default outcomes
    if positive_outcomes is None:
        positive_outcomes = ['forward_pass', 'backward_pass', 'ball_carry']
    if negative_outcomes is None:
        negative_outcomes = ['turnover']

    # Convert to pandas for complex operations
    if use_gpu and CUDF_AVAILABLE:
        pressure_pd = pressure_df.to_pandas()
        pass_pd = pass_df.to_pandas()
        checkpoint_pd = checkpoint_df.to_pandas()
    else:
        pressure_pd = pressure_df.copy()
        pass_pd = pass_df.copy()
        checkpoint_pd = checkpoint_df.copy()

    # Compute baseline rates from the provided data only if not supplied by caller.
    # When called inside a CV fold, the caller should pass rates from the training
    # split to avoid leakage from validation examples.
    if baseline_turnover_rates is None:
        baseline_turnover_rates = _calculate_baseline_turnover_rates(
            pressure_pd, negative_outcomes
        )

    print(f"Baseline turnover rates by zone: {baseline_turnover_rates}")

    # Initialize feature columns
    checkpoint_pd['last15_press_retention'] = 0.0
    checkpoint_pd['cumul_press_retention'] = 0.0
    checkpoint_pd['last15_progressive_press'] = 0
    checkpoint_pd['cumul_progressive_press'] = 0
    checkpoint_pd['last15_press_angle_std'] = 0.0
    checkpoint_pd['cumul_press_angle_std'] = 0.0
    checkpoint_pd['last15_press_quality'] = 0.0
    checkpoint_pd['cumul_press_quality'] = 0.0
    checkpoint_pd['last15_press_count'] = 0
    checkpoint_pd['cumul_press_count'] = 0

    # Process each player
    for player_id in checkpoint_pd['player_appearance_id'].unique():
        # Get player's pressure events
        player_pressure = pressure_pd[
            pressure_pd['player_appearance_id'] == player_id
        ]

        # Get player's checkpoints
        player_checkpoints = checkpoint_pd[
            checkpoint_pd['player_appearance_id'] == player_id
        ]

        for idx, row in player_checkpoints.iterrows():
            checkpoint_min = row['checkpoint_min']
            checkpoint_period = row['checkpoint_period']
            minute_in = row['minute_in']

            # Filter pressure events for this checkpoint
            valid_pressure = player_pressure[
                (player_pressure['period'] == checkpoint_period) &
                (player_pressure['minute'] <= checkpoint_min)
            ]

            if len(valid_pressure) == 0:
                continue

            # Cumulative features
            cumul_pressure = valid_pressure[valid_pressure['minute'] >= minute_in]
            cumul_features = _calculate_single_checkpoint_features(
                cumul_pressure, positive_outcomes, negative_outcomes,
                baseline_turnover_rates
            )

            checkpoint_pd.loc[idx, 'cumul_press_retention'] = cumul_features['retention']
            checkpoint_pd.loc[idx, 'cumul_progressive_press'] = cumul_features['progressive']
            checkpoint_pd.loc[idx, 'cumul_press_angle_std'] = cumul_features['angle_std']
            checkpoint_pd.loc[idx, 'cumul_press_quality'] = cumul_features['quality']
            checkpoint_pd.loc[idx, 'cumul_press_count'] = cumul_features['count']

            # Rolling 15-minute features
            rolling_pressure = valid_pressure[
                valid_pressure['minute'] > (checkpoint_min - 15)
            ]

            if len(rolling_pressure) > 0:
                rolling_features = _calculate_single_checkpoint_features(
                    rolling_pressure, positive_outcomes, negative_outcomes,
                    baseline_turnover_rates
                )

                checkpoint_pd.loc[idx, 'last15_press_retention'] = rolling_features['retention']
                checkpoint_pd.loc[idx, 'last15_progressive_press'] = rolling_features['progressive']
                checkpoint_pd.loc[idx, 'last15_press_angle_std'] = rolling_features['angle_std']
                checkpoint_pd.loc[idx, 'last15_press_quality'] = rolling_features['quality']
                checkpoint_pd.loc[idx, 'last15_press_count'] = rolling_features['count']

    # Convert back to cuDF if needed
    if use_gpu and CUDF_AVAILABLE:
        return cudf.from_pandas(checkpoint_pd)

    return checkpoint_pd


def _calculate_single_checkpoint_features(
    pressure_events: pd.DataFrame,
    positive_outcomes: List[str],
    negative_outcomes: List[str],
    baseline_turnover_rates: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate press resistance features for a single checkpoint

    Args:
        pressure_events: Pressure events for this checkpoint
        positive_outcomes: List of positive outcomes
        negative_outcomes: List of negative outcomes
        baseline_turnover_rates: Expected turnover rates by zone

    Returns:
        Dictionary of feature values
    """
    if len(pressure_events) == 0:
        return {
            'retention': 0.0,
            'progressive': 0,
            'angle_std': 0.0,
            'quality': 0.0,
            'count': 0,
        }

    # 1. Press Retention Rate
    positive_count = pressure_events[
        pressure_events['press_induced_outcome'].isin(positive_outcomes)
    ].shape[0]
    retention_rate = positive_count / len(pressure_events)

    # 2. Progressive Press Evasion
    progressive_count = pressure_events[
        pressure_events['press_induced_outcome'].isin(['forward_pass', 'ball_carry'])
    ].shape[0]

    # 3. Pass Angle Distortion (std of angles)
    angles = pressure_events['pass_angle'].dropna()
    angle_std = angles.std() if len(angles) > 0 else 0.0

    # 4. Expected Turnovers vs Actual (press quality)
    # Calculate expected turnovers based on zone baseline rates
    expected_turnovers = 0.0
    for zone, rate in baseline_turnover_rates.items():
        zone_events = pressure_events[pressure_events['stage'] == zone]
        expected_turnovers += len(zone_events) * rate

    actual_turnovers = pressure_events[
        pressure_events['press_induced_outcome'].isin(negative_outcomes)
    ].shape[0]

    # Positive quality = fewer turnovers than expected
    press_quality = expected_turnovers - actual_turnovers

    return {
        'retention': retention_rate,
        'progressive': progressive_count,
        'angle_std': angle_std,
        'quality': press_quality,
        'count': len(pressure_events),
    }


def _calculate_baseline_turnover_rates(
    pressure_df: pd.DataFrame,
    negative_outcomes: List[str]
) -> Dict[str, float]:
    """
    Calculate baseline turnover rates by zone across all players

    Args:
        pressure_df: All pressure events
        negative_outcomes: List of turnover outcomes

    Returns:
        Dictionary mapping zone to turnover rate
    """
    baseline_rates = {}

    for zone in pressure_df['stage'].unique():
        zone_events = pressure_df[pressure_df['stage'] == zone]
        if len(zone_events) == 0:
            baseline_rates[zone] = 0.0
            continue

        turnovers = zone_events[
            zone_events['press_induced_outcome'].isin(negative_outcomes)
        ].shape[0]

        baseline_rates[zone] = turnovers / len(zone_events)

    return baseline_rates


def calculate_baseline_pass_angles(
    pass_df: Union[pd.DataFrame, 'cudf.DataFrame'],
    use_gpu: bool = False
) -> Dict[int, float]:
    """
    Calculate baseline pass angle std per player from regular passes

    This provides a comparison for angle distortion under pressure.

    Args:
        pass_df: Regular pass events
        use_gpu: Use cuDF backend

    Returns:
        Dictionary mapping player_appearance_id to baseline angle std
    """
    if use_gpu and CUDF_AVAILABLE:
        pass_df = pass_df.to_pandas()

    # Note: Regular pass data doesn't have angles in the provided schema
    # This is a placeholder for potential future enhancement
    # For now, we'll use pressure-specific angle std as the primary metric

    baseline_angles = {}

    # If angle data becomes available, calculate per-player baseline:
    # for player_id in pass_df['player_appearance_id'].unique():
    #     player_passes = pass_df[pass_df['player_appearance_id'] == player_id]
    #     angles = player_passes['angle'].dropna()
    #     baseline_angles[player_id] = angles.std() if len(angles) > 0 else 0.0

    return baseline_angles
