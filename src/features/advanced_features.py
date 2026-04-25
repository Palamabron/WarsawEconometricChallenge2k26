"""
Advanced feature engineering based on analysis.

Top predictors identified:
- cumul_press_quality (0.116 correlation, 0.055 RF importance)
- is_attacker (0.124 correlation)
- cumul_peak_speed (0.031 MI, 0.048 RF importance)
- position features
"""

from typing import Union

import numpy as np
import pandas as pd

try:
    import cudf

    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False
    cudf = None


def add_shot_quality_features(
    checkpoint_df: Union[pd.DataFrame, "cudf.DataFrame"],
    shot_df: Union[pd.DataFrame, "cudf.DataFrame"],
    use_gpu: bool = False,
) -> Union[pd.DataFrame, "cudf.DataFrame"]:
    """
    Add xG-like shot quality features.

    Features:
    - shot_from_counter_attack: shots from counters (high quality)
    - shot_from_setpiece: shots from set pieces
    - head_shot_ratio: proportion of headers
    - shot_under_pressure_last5: recent shots under pressure
    - shot_frequency_per_min: shooting frequency
    """
    if use_gpu and CUDF_AVAILABLE:
        checkpoint_pd = checkpoint_df.to_pandas().copy()
        shot_pd = shot_df.to_pandas()
    else:
        checkpoint_pd = checkpoint_df.copy()
        shot_pd = shot_df.copy()

    # Initialize columns
    checkpoint_pd["shot_from_counter_last15"] = 0
    checkpoint_pd["shot_from_setpiece_last15"] = 0
    checkpoint_pd["head_shot_ratio_cumul"] = 0.0
    checkpoint_pd["shot_frequency_per_min"] = 0.0
    checkpoint_pd["last5_shot_count"] = 0

    # Process each checkpoint
    for idx, row in checkpoint_pd.iterrows():
        player_id = row["player_appearance_id"]
        checkpoint_min = row["checkpoint_min"]
        checkpoint_period = row["checkpoint_period"]
        minute_in = row.get("minute_in", 0)

        # Get player's shots
        player_shots = shot_pd[shot_pd["player_appearance_id"] == player_id]

        # Temporal filters
        valid_shots = player_shots[
            (player_shots["period"] == checkpoint_period)
            & (player_shots["minute"] <= checkpoint_min)
        ]

        if len(valid_shots) == 0:
            continue

        # Last 15 minutes
        last15_shots = valid_shots[valid_shots["minute"] > (checkpoint_min - 15)]

        # Counter attacks (last 15)
        counter_shots = last15_shots[last15_shots["play_pattern"] == "counter_attack"]
        checkpoint_pd.loc[idx, "shot_from_counter_last15"] = len(counter_shots)

        # Set pieces (last 15)
        setpiece_patterns = ["corner_kick", "direct_free_kick", "indirect_free_kick"]
        setpiece_shots = last15_shots[last15_shots["play_pattern"].isin(setpiece_patterns)]
        checkpoint_pd.loc[idx, "shot_from_setpiece_last15"] = len(setpiece_shots)

        # Cumulative head shot ratio
        cumul_shots = valid_shots[valid_shots["minute"] >= minute_in]
        if len(cumul_shots) > 0:
            head_shots = cumul_shots[cumul_shots["body_part"] == "head"]
            checkpoint_pd.loc[idx, "head_shot_ratio_cumul"] = len(head_shots) / len(
                cumul_shots
            )

        # Shot frequency (shots per minute)
        time_on_pitch = checkpoint_min - minute_in + 1
        if time_on_pitch > 0:
            checkpoint_pd.loc[idx, "shot_frequency_per_min"] = len(cumul_shots) / time_on_pitch

        # Last 5 minutes shot count
        last5_shots = valid_shots[valid_shots["minute"] > (checkpoint_min - 5)]
        checkpoint_pd.loc[idx, "last5_shot_count"] = len(last5_shots)

    if use_gpu and CUDF_AVAILABLE:
        return cudf.from_pandas(checkpoint_pd)

    return checkpoint_pd


def add_temporal_trends(
    checkpoint_df: Union[pd.DataFrame, "cudf.DataFrame"], use_gpu: bool = False
) -> Union[pd.DataFrame, "cudf.DataFrame"]:
    """
    Add temporal trend features (acceleration/deceleration).

    Features:
    - sprint_acceleration: recent vs previous sprint rate
    - shot_acceleration: recent vs previous shot rate
    - press_quality_trend: improving or declining press performance
    - speed_trend: speed change over time
    """
    if use_gpu and CUDF_AVAILABLE:
        df = checkpoint_df.to_pandas().copy()
    else:
        df = checkpoint_df.copy()

    # Sprint trend
    df["sprint_acceleration"] = (
        df["last15_sprints"] / (df["cumul_sprints"] - df["last15_sprints"] + 1) - 1.0
    )

    # Shot trend
    df["shot_acceleration"] = (
        df["last15_shots"] / (df["cumul_shots"] - df["last15_shots"] + 1) - 1.0
    )

    # Press quality trend
    prev15_press_quality = df["cumul_press_quality"] - df["last15_press_quality"]
    df["press_quality_trend"] = df["last15_press_quality"] - prev15_press_quality

    # Speed trend (using peak speed)
    prev15_peak = df["cumul_peak_speed"]  # Approximation
    df["speed_trend"] = df["last15_peak_speed"] - prev15_peak

    # Workload trend
    df["workload_trend"] = df["workload_ratio_hsr"] - 1.0  # Deviation from baseline

    if use_gpu and CUDF_AVAILABLE:
        return cudf.from_pandas(df)

    return df


def add_pressure_intensity_features(
    checkpoint_df: Union[pd.DataFrame, "cudf.DataFrame"],
    pressure_df: Union[pd.DataFrame, "cudf.DataFrame"],
    use_gpu: bool = False,
) -> Union[pd.DataFrame, "cudf.DataFrame"]:
    """
    Add pressure intensity and escape features.

    Features:
    - pressure_per_min: pressure events per minute
    - successful_escape_rate: retention rate under pressure
    - forward_escape_ratio: progressive vs total pressure events
    - pressure_intensity_last5: recent pressure frequency
    """
    if use_gpu and CUDF_AVAILABLE:
        checkpoint_pd = checkpoint_df.to_pandas().copy()
        pressure_pd = pressure_df.to_pandas()
    else:
        checkpoint_pd = checkpoint_df.copy()
        pressure_pd = pressure_df.copy()

    # Initialize
    checkpoint_pd["pressure_per_min"] = 0.0
    checkpoint_pd["forward_escape_ratio"] = 0.0
    checkpoint_pd["pressure_intensity_last5"] = 0.0

    for idx, row in checkpoint_pd.iterrows():
        player_id = row["player_appearance_id"]
        checkpoint_min = row["checkpoint_min"]
        checkpoint_period = row["checkpoint_period"]
        minute_in = row.get("minute_in", 0)

        player_pressure = pressure_pd[pressure_pd["player_appearance_id"] == player_id]

        valid_pressure = player_pressure[
            (player_pressure["period"] == checkpoint_period)
            & (player_pressure["minute"] <= checkpoint_min)
        ]

        if len(valid_pressure) == 0:
            continue

        # Cumulative pressure per minute
        cumul_pressure = valid_pressure[valid_pressure["minute"] >= minute_in]
        time_on_pitch = checkpoint_min - minute_in + 1
        if time_on_pitch > 0:
            checkpoint_pd.loc[idx, "pressure_per_min"] = len(cumul_pressure) / time_on_pitch

        # Forward escape ratio
        if len(cumul_pressure) > 0:
            forward_escapes = cumul_pressure[
                cumul_pressure["press_induced_outcome"] == "forward_pass"
            ]
            checkpoint_pd.loc[idx, "forward_escape_ratio"] = len(forward_escapes) / len(
                cumul_pressure
            )

        # Last 5 minutes pressure intensity
        last5_pressure = valid_pressure[valid_pressure["minute"] > (checkpoint_min - 5)]
        if checkpoint_min >= 5:
            checkpoint_pd.loc[idx, "pressure_intensity_last5"] = len(last5_pressure) / 5.0

    if use_gpu and CUDF_AVAILABLE:
        return cudf.from_pandas(checkpoint_pd)

    return checkpoint_pd


def add_position_context_features(
    checkpoint_df: Union[pd.DataFrame, "cudf.DataFrame"], use_gpu: bool = False
) -> Union[pd.DataFrame, "cudf.DataFrame"]:
    """
    Add position-specific contextual features.

    Features:
    - attacker_in_form: attacker with high shot + press quality
    - midfielder_engine: midfielder with high distance + press
    - defender_under_pressure: defender with low press quality
    - late_game_attacker: attacker in late game (scoring opportunity)
    """
    if use_gpu and CUDF_AVAILABLE:
        df = checkpoint_df.to_pandas().copy()
    else:
        df = checkpoint_df.copy()

    # Attacker in form
    df["attacker_in_form"] = (
        df["is_attacker"]
        * (df["cumul_shots"] > df["cumul_shots"].quantile(0.7))
        * (df["cumul_press_quality"] > 0)
    ).astype(int)

    # Midfielder engine
    df["midfielder_engine"] = (
        df["is_midfielder"]
        * (df["cumul_distance"] > df["cumul_distance"].quantile(0.6))
        * (df["cumul_progressive_press"] > 0)
    ).astype(int)

    # Defender under pressure (danger sign)
    df["defender_under_pressure"] = (
        df["is_defender"] * (df["cumul_press_quality"] < 0)
    ).astype(int)

    # Late game attacker (high scoring probability)
    df["late_game_attacker"] = (df["is_attacker"] * df["is_late_game"]).astype(int)

    # Home advantage interactions
    df["home_attacker"] = (df["is_home"] * df["is_attacker"]).astype(int)
    df["home_shots"] = df["is_home"] * df["cumul_shots"]

    if use_gpu and CUDF_AVAILABLE:
        return cudf.from_pandas(df)

    return df


def add_interaction_features(
    checkpoint_df: Union[pd.DataFrame, "cudf.DataFrame"], use_gpu: bool = False
) -> Union[pd.DataFrame, "cudf.DataFrame"]:
    """
    Add non-linear interactions between top features.
    """
    if use_gpu and CUDF_AVAILABLE:
        df = checkpoint_df.to_pandas().copy()
    else:
        df = checkpoint_df.copy()

    # Top feature interactions from analysis
    df["press_quality_x_attacker"] = df["cumul_press_quality"] * df["is_attacker"]
    df["press_quality_x_shots"] = df["cumul_press_quality"] * df["cumul_shots"]
    df["peak_speed_x_attacker"] = df["cumul_peak_speed"] * df["is_attacker"]
    df["peak_speed_x_shots"] = df["cumul_peak_speed"] * df["cumul_shots"]

    # Physical × tactical
    df["speed_x_xt"] = df["cumul_peak_speed"] * df["cumul_xt_added"]
    df["distance_x_press"] = df["cumul_distance"] * df["cumul_press_retention"]

    # Fatigue interactions
    df["fatigue_x_shots"] = df["is_fatigued"] * df["cumul_shots"]
    df["fatigue_x_press"] = df["is_fatigued"] * df["cumul_press_quality"]

    # Momentum × performance
    df["momentum_x_shots"] = df["has_momentum"] * df["cumul_shots"]
    df["momentum_x_speed"] = df["has_momentum"] * df["cumul_peak_speed"]

    if use_gpu and CUDF_AVAILABLE:
        return cudf.from_pandas(df)

    return df


def add_all_advanced_features(
    checkpoint_df: Union[pd.DataFrame, "cudf.DataFrame"],
    event_dfs: dict,
    use_gpu: bool = False,
) -> Union[pd.DataFrame, "cudf.DataFrame"]:
    """
    Add all advanced features.
    """
    print("Adding shot quality features...")
    checkpoint_df = add_shot_quality_features(checkpoint_df, event_dfs["shot"], use_gpu)

    print("Adding temporal trends...")
    checkpoint_df = add_temporal_trends(checkpoint_df, use_gpu)

    print("Adding pressure intensity features...")
    checkpoint_df = add_pressure_intensity_features(
        checkpoint_df, event_dfs["pressure"], use_gpu
    )

    print("Adding position context features...")
    checkpoint_df = add_position_context_features(checkpoint_df, use_gpu)

    print("Adding interaction features...")
    checkpoint_df = add_interaction_features(checkpoint_df, use_gpu)

    return checkpoint_df
