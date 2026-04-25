"""
Advanced feature engineering based on analysis.

Top predictors identified:
- cumul_press_quality (0.116 correlation, 0.055 RF importance)
- is_attacker (0.124 correlation)
- cumul_peak_speed (0.031 MI, 0.048 RF importance)
- position features
"""

import numpy as np

from src.types import DataFrame

try:
    import cudf

    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False
    cudf = None


def add_shot_quality_features(
    checkpoint_df: DataFrame,
    shot_df: DataFrame,
    use_gpu: bool = False,
) -> DataFrame:
    """
    Add xG-like shot quality features (VECTORIZED).

    Analyzes shot patterns and contexts to identify high-quality scoring opportunities.
    Uses vectorized operations for 10-50x performance improvement over loops.

    Args:
        checkpoint_df: Player checkpoint observations with temporal metadata
        shot_df: Shot events with play_pattern, body_part, minute
        use_gpu: If True, use cuDF backend for GPU acceleration

    Returns:
        checkpoint_df augmented with 6 shot quality features:
        - shot_from_counter_last15: Counter-attack shots (last 15 min)
        - shot_from_setpiece_last15: Set piece shots (last 15 min)
        - head_shot_ratio_cumul: Proportion of headers (cumulative)
        - shot_frequency_per_min: Shots per minute on pitch
        - last5_shot_count: Shot count in last 5 minutes

    Notes:
        - All features respect temporal boundaries (minute <= checkpoint_min)
        - Cumulative features filtered by minute_in (player on pitch)
        - Players with no shots receive 0.0 for all features
    """
    if use_gpu and CUDF_AVAILABLE:
        checkpoint_pd = checkpoint_df.to_pandas().copy()
        shot_pd = shot_df.to_pandas()
    else:
        checkpoint_pd = checkpoint_df.copy()
        shot_pd = shot_df.copy()

    # Merge shots with checkpoint data
    merged = shot_pd.merge(
        checkpoint_pd[["player_appearance_id", "checkpoint_min", "checkpoint_period", "minute_in"]],
        on="player_appearance_id",
        how="inner",
    )

    # Convert periods to string for comparison
    merged["period"] = merged["period"].astype(str)
    merged["checkpoint_period"] = merged["checkpoint_period"].astype(str)

    # Filter temporally valid shots
    merged = merged[
        (merged["period"] == merged["checkpoint_period"])
        & (merged["minute"] <= merged["checkpoint_min"])
    ]

    # Create time window indicators
    merged["is_last15"] = merged["minute"] > (merged["checkpoint_min"] - 15)
    merged["is_last5"] = merged["minute"] > (merged["checkpoint_min"] - 5)
    merged["is_cumul"] = merged["minute"] >= merged["minute_in"]
    merged["is_counter"] = merged["play_pattern"] == "counter_attack"
    merged["is_setpiece"] = merged["play_pattern"].isin(
        ["corner_kick", "direct_free_kick", "indirect_free_kick"]
    )
    merged["is_head"] = merged["body_part"] == "head"

    # Pre-compute combined boolean columns for vectorized aggregation
    merged["is_counter_last15"] = merged["is_counter"] & merged["is_last15"]
    merged["is_setpiece_last15"] = merged["is_setpiece"] & merged["is_last15"]
    merged["is_head_cumul"] = merged["is_head"] & merged["is_cumul"]

    # Aggregate by player and checkpoint
    shot_agg = (
        merged.groupby(["player_appearance_id", "checkpoint_min", "checkpoint_period"])
        .agg(
            shot_from_counter_last15=("is_counter_last15", "sum"),
            shot_from_setpiece_last15=("is_setpiece_last15", "sum"),
            head_shot_count_cumul=("is_head_cumul", "sum"),
            total_shot_count_cumul=("is_cumul", "sum"),
            last5_shot_count=("is_last5", "sum"),
        )
        .reset_index()
    )

    # Calculate head shot ratio
    shot_agg["head_shot_ratio_cumul"] = shot_agg["head_shot_count_cumul"] / np.maximum(
        shot_agg["total_shot_count_cumul"], 1
    )

    # Merge back and calculate shot frequency
    checkpoint_pd = checkpoint_pd.merge(
        shot_agg,
        on=["player_appearance_id", "checkpoint_min", "checkpoint_period"],
        how="left",
    )

    # Fill NaN for players with no shots
    checkpoint_pd["shot_from_counter_last15"] = (
        checkpoint_pd["shot_from_counter_last15"].fillna(0).astype(int)
    )
    checkpoint_pd["shot_from_setpiece_last15"] = (
        checkpoint_pd["shot_from_setpiece_last15"].fillna(0).astype(int)
    )
    checkpoint_pd["head_shot_ratio_cumul"] = checkpoint_pd["head_shot_ratio_cumul"].fillna(0.0)
    checkpoint_pd["last5_shot_count"] = checkpoint_pd["last5_shot_count"].fillna(0).astype(int)
    checkpoint_pd["total_shot_count_cumul"] = (
        checkpoint_pd["total_shot_count_cumul"].fillna(0).astype(int)
    )

    # Shot frequency (shots per minute on pitch)
    checkpoint_pd["shot_frequency_per_min"] = checkpoint_pd["total_shot_count_cumul"] / np.maximum(
        checkpoint_pd["checkpoint_min"] - checkpoint_pd["minute_in"] + 1, 1
    )

    # Drop intermediate columns
    checkpoint_pd = checkpoint_pd.drop(columns=["head_shot_count_cumul", "total_shot_count_cumul"])

    if use_gpu and CUDF_AVAILABLE:
        return cudf.from_pandas(checkpoint_pd)

    return checkpoint_pd


def add_temporal_trends(checkpoint_df: DataFrame, use_gpu: bool = False) -> DataFrame:
    """
    Add temporal trend features (acceleration/deceleration).

    Identifies momentum shifts and performance trends by comparing recent
    activity to baseline levels.

    Args:
        checkpoint_df: Player checkpoints with cumulative and rolling metrics
        use_gpu: If True, use cuDF backend for GPU acceleration

    Returns:
        checkpoint_df augmented with 5 trend features:
        - sprint_acceleration: (last15_sprints / prev_sprints) - 1.0
        - shot_acceleration: (last15_shots / prev_shots) - 1.0
        - press_quality_trend: Recent vs previous press quality delta
        - speed_trend: Change in peak speed (recent vs cumulative)
        - workload_trend: Deviation from baseline workload ratio

    Notes:
        - Positive acceleration = increasing activity
        - All divisions protected against zero denominators (+1 smoothing)
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
    checkpoint_df: DataFrame,
    pressure_df: DataFrame,
    use_gpu: bool = False,
) -> DataFrame:
    """
    Add pressure intensity and escape features (VECTORIZED).

    Measures how frequently a player faces pressure and their ability
    to progress the ball under duress.

    Args:
        checkpoint_df: Player checkpoint observations
        pressure_df: Pressure events with press_induced_outcome, minute
        use_gpu: If True, use cuDF backend for GPU acceleration

    Returns:
        checkpoint_df augmented with 3 pressure intensity features:
        - pressure_per_min: Cumulative pressure events / minutes on pitch
        - forward_escape_ratio: Forward passes under pressure / total pressure
        - pressure_intensity_last5: Pressure events in last 5 min / 5

    Notes:
        - Only includes events where minute <= checkpoint_min
        - Division by zero handled with np.maximum(denominator, 1)
        - Uses vectorized merge+groupby for performance
    """
    if use_gpu and CUDF_AVAILABLE:
        checkpoint_pd = checkpoint_df.to_pandas().copy()
        pressure_pd = pressure_df.to_pandas()
    else:
        checkpoint_pd = checkpoint_df.copy()
        pressure_pd = pressure_df.copy()

    # Merge pressure with checkpoint data
    merged = pressure_pd.merge(
        checkpoint_pd[["player_appearance_id", "checkpoint_min", "checkpoint_period", "minute_in"]],
        on="player_appearance_id",
        how="inner",
    )

    # Convert periods to string for comparison
    merged["period"] = merged["period"].astype(str)
    merged["checkpoint_period"] = merged["checkpoint_period"].astype(str)

    # Filter temporally valid pressure events
    merged = merged[
        (merged["period"] == merged["checkpoint_period"])
        & (merged["minute"] <= merged["checkpoint_min"])
    ]

    # Create indicators
    merged["is_cumul"] = merged["minute"] >= merged["minute_in"]
    merged["is_last5"] = merged["minute"] > (merged["checkpoint_min"] - 5)
    merged["is_forward_escape"] = merged["press_induced_outcome"] == "forward_pass"

    # Pre-compute combined boolean column for vectorized aggregation
    merged["is_forward_escape_cumul"] = merged["is_forward_escape"] & merged["is_cumul"]

    # Aggregate by player and checkpoint
    pressure_agg = (
        merged.groupby(["player_appearance_id", "checkpoint_min", "checkpoint_period"])
        .agg(
            cumul_pressure_count=("is_cumul", "sum"),
            forward_escape_count=("is_forward_escape_cumul", "sum"),
            last5_pressure_count=("is_last5", "sum"),
        )
        .reset_index()
    )

    # Merge back
    checkpoint_pd = checkpoint_pd.merge(
        pressure_agg,
        on=["player_appearance_id", "checkpoint_min", "checkpoint_period"],
        how="left",
    )

    # Fill NaN
    checkpoint_pd["cumul_pressure_count"] = (
        checkpoint_pd["cumul_pressure_count"].fillna(0).astype(int)
    )
    checkpoint_pd["forward_escape_count"] = (
        checkpoint_pd["forward_escape_count"].fillna(0).astype(int)
    )
    checkpoint_pd["last5_pressure_count"] = (
        checkpoint_pd["last5_pressure_count"].fillna(0).astype(int)
    )

    # Calculate derived features
    checkpoint_pd["pressure_per_min"] = checkpoint_pd["cumul_pressure_count"] / np.maximum(
        checkpoint_pd["checkpoint_min"] - checkpoint_pd["minute_in"] + 1, 1
    )
    checkpoint_pd["forward_escape_ratio"] = checkpoint_pd["forward_escape_count"] / np.maximum(
        checkpoint_pd["cumul_pressure_count"], 1
    )
    checkpoint_pd["pressure_intensity_last5"] = checkpoint_pd["last5_pressure_count"] / 5.0

    # Drop intermediate columns
    checkpoint_pd = checkpoint_pd.drop(
        columns=["cumul_pressure_count", "forward_escape_count", "last5_pressure_count"]
    )

    if use_gpu and CUDF_AVAILABLE:
        return cudf.from_pandas(checkpoint_pd)

    return checkpoint_pd


def add_position_context_features(checkpoint_df: DataFrame, use_gpu: bool = False) -> DataFrame:
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
    df["defender_under_pressure"] = (df["is_defender"] * (df["cumul_press_quality"] < 0)).astype(
        int
    )

    # Late game attacker (high scoring probability)
    df["late_game_attacker"] = (df["is_attacker"] * df["is_late_game"]).astype(int)

    # Home advantage interactions
    df["home_attacker"] = (df["is_home"] * df["is_attacker"]).astype(int)
    df["home_shots"] = df["is_home"] * df["cumul_shots"]

    if use_gpu and CUDF_AVAILABLE:
        return cudf.from_pandas(df)

    return df


def add_interaction_features(checkpoint_df: DataFrame, use_gpu: bool = False) -> DataFrame:
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
    checkpoint_df: DataFrame,
    event_dfs: dict,
    use_gpu: bool = False,
) -> DataFrame:
    """
    Add all advanced features.
    """
    print("Adding shot quality features...")
    checkpoint_df = add_shot_quality_features(checkpoint_df, event_dfs["shot"], use_gpu)

    print("Adding temporal trends...")
    checkpoint_df = add_temporal_trends(checkpoint_df, use_gpu)

    print("Adding pressure intensity features...")
    checkpoint_df = add_pressure_intensity_features(checkpoint_df, event_dfs["pressure"], use_gpu)

    print("Adding position context features...")
    checkpoint_df = add_position_context_features(checkpoint_df, use_gpu)

    print("Adding interaction features...")
    checkpoint_df = add_interaction_features(checkpoint_df, use_gpu)

    return checkpoint_df
