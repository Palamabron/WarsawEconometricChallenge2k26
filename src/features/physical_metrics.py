"""
Physical Metrics feature engineering

Calculates relative intensity, fatigue indicators, and positional deviations
from physical tracking data (runs, sprints, speed).
"""

import numpy as np
import pandas as pd
from typing import Union, Dict

try:
    import cudf
    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False
    cudf = None


def calculate_physical_features(
    run_df: Union[pd.DataFrame, 'cudf.DataFrame'],
    checkpoint_df: Union[pd.DataFrame, 'cudf.DataFrame'],
    use_gpu: bool = False
) -> Union[pd.DataFrame, 'cudf.DataFrame']:
    """
    Calculate physical performance features per checkpoint

    Features:
    1. Intra-Match Workload Ratio: acute (last 15min) / chronic (avg per 15min)
    2. Positional Momentum Deviation: player vs position average
    3. Cumulative Speed Decay: current peak vs match peak
    4. Run Type Distribution: diversity of running patterns

    Args:
        run_df: High-speed run events
        checkpoint_df: Checkpoint observations with existing physical metrics
        use_gpu: Use cuDF backend

    Returns:
        DataFrame with additional physical features
    """
    # Convert to pandas for complex operations
    if use_gpu and CUDF_AVAILABLE:
        run_pd = run_df.to_pandas()
        checkpoint_pd = checkpoint_df.to_pandas()
    else:
        run_pd = run_df.copy()
        checkpoint_pd = checkpoint_df.copy()

    # Initialize feature columns
    checkpoint_pd['workload_ratio_hsr'] = 1.0  # Default to balanced
    checkpoint_pd['workload_ratio_sprints'] = 1.0
    checkpoint_pd['speed_decay'] = 1.0  # Default to no decay
    checkpoint_pd['positional_sprint_deviation'] = 0.0
    checkpoint_pd['positional_hsr_deviation'] = 0.0
    checkpoint_pd['run_type_diversity'] = 0.0

    # Calculate positional averages
    position_averages = _calculate_positional_averages(checkpoint_pd)

    # Process each checkpoint
    for idx, row in checkpoint_pd.iterrows():
        # 1. Intra-Match Workload Ratio
        workload_features = _calculate_workload_ratio(row)
        checkpoint_pd.loc[idx, 'workload_ratio_hsr'] = workload_features['hsr_ratio']
        checkpoint_pd.loc[idx, 'workload_ratio_sprints'] = workload_features['sprint_ratio']

        # 2. Speed Decay
        speed_decay = _calculate_speed_decay(row)
        checkpoint_pd.loc[idx, 'speed_decay'] = speed_decay

        # 3. Positional Deviations
        position = row['position']
        if position in position_averages:
            pos_avg = position_averages[position]

            sprint_deviation = row['last15_sprints'] - pos_avg['avg_sprints']
            hsr_deviation = row['last15_hsr'] - pos_avg['avg_hsr']

            checkpoint_pd.loc[idx, 'positional_sprint_deviation'] = sprint_deviation
            checkpoint_pd.loc[idx, 'positional_hsr_deviation'] = hsr_deviation

    # 4. Run Type Diversity (from run events)
    if 'run_type' in run_pd.columns:
        diversity_features = _calculate_run_diversity(run_pd, checkpoint_pd)
        # Drop the initialized column before merging to avoid _x/_y suffix conflicts
        checkpoint_pd = checkpoint_pd.drop(columns=['run_type_diversity']).merge(
            diversity_features,
            on=['player_appearance_id', 'checkpoint_period', 'checkpoint_min'],
            how='left'
        )
        checkpoint_pd['run_type_diversity'] = checkpoint_pd['run_type_diversity'].fillna(0)

    # Convert back to cuDF if needed
    if use_gpu and CUDF_AVAILABLE:
        return cudf.from_pandas(checkpoint_pd)

    return checkpoint_pd


def _calculate_workload_ratio(row: pd.Series) -> Dict[str, float]:
    """
    Calculate acute:chronic workload ratio

    Acute = last 15 minutes
    Chronic = cumulative average per 15-minute period

    Args:
        row: Checkpoint row with physical metrics

    Returns:
        Dictionary with HSR and sprint ratios
    """
    checkpoint_min = row['checkpoint_min']

    # Avoid division by zero for early checkpoints
    if checkpoint_min < 15:
        return {'hsr_ratio': 1.0, 'sprint_ratio': 1.0}

    # Calculate number of 15-minute periods elapsed
    n_periods = checkpoint_min / 15.0

    # Chronic workload = cumulative / number of periods
    chronic_hsr = row['cumul_hsr'] / n_periods if n_periods > 0 else row['last15_hsr']
    chronic_sprints = row['cumul_sprints'] / n_periods if n_periods > 0 else row['last15_sprints']

    # Acute workload = last 15 minutes
    acute_hsr = row['last15_hsr']
    acute_sprints = row['last15_sprints']

    # Calculate ratios (avoid division by zero)
    hsr_ratio = acute_hsr / chronic_hsr if chronic_hsr > 0 else 1.0
    sprint_ratio = acute_sprints / chronic_sprints if chronic_sprints > 0 else 1.0

    # Cap extreme values
    hsr_ratio = np.clip(hsr_ratio, 0.0, 3.0)
    sprint_ratio = np.clip(sprint_ratio, 0.0, 3.0)

    return {
        'hsr_ratio': hsr_ratio,
        'sprint_ratio': sprint_ratio
    }


def _calculate_speed_decay(row: pd.Series) -> float:
    """
    Calculate speed preservation ratio

    speed_decay = last15_peak_speed / cumul_peak_speed

    Values < 0.9 indicate loss of explosive capacity

    Args:
        row: Checkpoint row with speed metrics

    Returns:
        Speed decay ratio
    """
    cumul_peak = row['cumul_peak_speed']
    last15_peak = row['last15_peak_speed']

    if cumul_peak == 0:
        return 1.0

    decay = last15_peak / cumul_peak

    # Cap to reasonable range [0, 1.2]
    # Values > 1.0 can occur if recent peak exceeds previous peaks
    decay = np.clip(decay, 0.0, 1.2)

    return decay


def _calculate_positional_averages(checkpoint_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calculate average physical metrics by position

    Args:
        checkpoint_df: All checkpoint observations

    Returns:
        Dictionary mapping position to average metrics
    """
    position_averages = {}

    for position in checkpoint_df['position'].unique():
        pos_data = checkpoint_df[checkpoint_df['position'] == position]

        position_averages[position] = {
            'avg_sprints': pos_data['last15_sprints'].mean(),
            'avg_hsr': pos_data['last15_hsr'].mean(),
            'avg_distance': pos_data['last15_distance'].mean(),
        }

    return position_averages


def _calculate_run_diversity(
    run_df: pd.DataFrame,
    checkpoint_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate diversity of run types per checkpoint

    Higher diversity indicates varied running patterns (good fitness indicator)

    Args:
        run_df: Run event data with run_type column
        checkpoint_df: Checkpoint observations

    Returns:
        DataFrame with run_type_diversity per checkpoint
    """
    diversity_results = []

    for idx, row in checkpoint_df.iterrows():
        player_id = row['player_appearance_id']
        checkpoint_min = row['checkpoint_min']
        checkpoint_period = row['checkpoint_period']
        minute_in = row['minute_in']

        # Get runs for this player in rolling 15-min window
        player_runs = run_df[
            (run_df['player_appearance_id'] == player_id) &
            (run_df['period'] == checkpoint_period) &
            (run_df['minute'] > (checkpoint_min - 15)) &
            (run_df['minute'] <= checkpoint_min)
        ]

        # Calculate diversity (entropy or unique count)
        if len(player_runs) == 0:
            diversity = 0.0
        else:
            # Count unique run types
            run_type_counts = player_runs['run_type'].value_counts()
            n_types = len(run_type_counts)

            # Simple diversity metric: number of unique run types
            diversity = float(n_types)

            # Alternative: Shannon entropy for more sophisticated measure
            # proportions = run_type_counts / len(player_runs)
            # diversity = -np.sum(proportions * np.log2(proportions + 1e-10))

        diversity_results.append({
            'player_appearance_id': player_id,
            'checkpoint_period': checkpoint_period,
            'checkpoint_min': checkpoint_min,
            'run_type_diversity': diversity
        })

    return pd.DataFrame(diversity_results)


def calculate_fatigue_indicators(
    checkpoint_df: Union[pd.DataFrame, 'cudf.DataFrame'],
    fatigue_threshold: float = 0.8,
    momentum_threshold: float = 1.2
) -> Union[pd.DataFrame, 'cudf.DataFrame']:
    """
    Create binary fatigue and momentum indicators

    Args:
        checkpoint_df: DataFrame with workload_ratio features
        fatigue_threshold: Ratio below which indicates fatigue
        momentum_threshold: Ratio above which indicates momentum surge

    Returns:
        DataFrame with fatigue/momentum indicators
    """
    if isinstance(checkpoint_df, cudf.DataFrame) if CUDF_AVAILABLE else False:
        df = checkpoint_df.to_pandas().copy()
        use_cudf = True
    else:
        df = checkpoint_df.copy()
        use_cudf = False

    # Binary indicators
    df['is_fatigued'] = (df['workload_ratio_hsr'] < fatigue_threshold).astype(int)
    df['has_momentum'] = (df['workload_ratio_hsr'] > momentum_threshold).astype(int)

    # Intensity state (categorical: fatigued, normal, momentum)
    df['intensity_state'] = 'normal'
    df.loc[df['is_fatigued'] == 1, 'intensity_state'] = 'fatigued'
    df.loc[df['has_momentum'] == 1, 'intensity_state'] = 'momentum'

    if use_cudf:
        return cudf.from_pandas(df)

    return df
