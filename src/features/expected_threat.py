"""
Expected Threat (xT) feature engineering using three-zone Markov model

Calculates the value of possession in different pitch zones and aggregates
threat-adding actions per player per checkpoint.
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Tuple

try:
    import cudf
    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False
    cudf = None


class ExpectedThreatCalculator:
    """
    Calculate Expected Threat values for three-zone pitch model

    Zones: bottom (defensive third), middle (central), top (attacking third)
    """

    def __init__(
        self,
        zones: list = None,
        convergence_threshold: float = 0.001,
        max_iterations: int = 10,
        use_gpu: bool = False
    ):
        """
        Initialize xT calculator

        Args:
            zones: List of zone names [defensive, central, attacking]
            convergence_threshold: Convergence threshold for iterative calculation
            max_iterations: Maximum iterations for convergence
            use_gpu: Use cuDF backend
        """
        self.zones = zones or ['bottom', 'middle', 'top']
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.use_gpu = use_gpu and CUDF_AVAILABLE

        self.zone_threat_values = None
        self.transition_matrix = None

    def compute_transition_matrix(
        self,
        pass_df: Union[pd.DataFrame, 'cudf.DataFrame']
    ) -> np.ndarray:
        """
        Compute zone transition probability matrix from pass data

        P[i,j] = probability of transitioning from zone i to zone j

        Args:
            pass_df: Pass events with 'stage' column and accurate=True

        Returns:
            3x3 transition matrix
        """
        # Filter for accurate passes only
        if self.use_gpu:
            accurate_passes = pass_df[pass_df['accurate'] == True].to_pandas()
        else:
            accurate_passes = pass_df[pass_df['accurate'] == True].copy()

        # Count transitions
        # Note: We don't have explicit destination zones, so we'll use sequential analysis
        # Group by player_appearance_id and sort by minute to infer transitions
        accurate_passes = accurate_passes.sort_values(['player_appearance_id', 'minute'])

        # Create destination zone by shifting stage within same player/period
        accurate_passes['dest_zone'] = accurate_passes.groupby(
            ['player_appearance_id', 'period']
        )['stage'].shift(-1)

        # Remove NaN destinations (end of sequences)
        transitions = accurate_passes.dropna(subset=['dest_zone'])

        # Initialize transition matrix
        n_zones = len(self.zones)
        transition_counts = np.zeros((n_zones, n_zones))

        # Count transitions between zones
        for origin_zone in self.zones:
            origin_idx = self.zones.index(origin_zone)

            # Get all passes from this origin zone
            from_zone = transitions[transitions['stage'] == origin_zone]

            # Count destinations
            for dest_zone in self.zones:
                dest_idx = self.zones.index(dest_zone)
                count = (from_zone['dest_zone'] == dest_zone).sum()
                transition_counts[origin_idx, dest_idx] = count

        # Convert counts to probabilities (row-wise normalization)
        # Add small epsilon to avoid division by zero
        row_sums = transition_counts.sum(axis=1, keepdims=True) + 1e-10
        transition_matrix = transition_counts / row_sums

        self.transition_matrix = transition_matrix
        return transition_matrix

    def compute_shot_probabilities(
        self,
        shot_df: Union[pd.DataFrame, 'cudf.DataFrame'],
        pass_df: Union[pd.DataFrame, 'cudf.DataFrame'] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute P(shot|zone) from actual shot/pass data and P(goal|shot, zone) from
        domain-knowledge conversion rates (shot outcomes excluded to prevent leakage).

        P(shot|zone) = shots_from_zone / total_events_from_zone
        Total events approximated as shots + accurate passes per zone.

        P(goal|shot, zone) uses published conversion rates by zone (not from the
        dataset's shot outcome columns, which are excluded to prevent leakage).

        Args:
            shot_df: Shot events with 'stage' column
            pass_df: Pass events used to estimate total possession events per zone.
                     If None, falls back to empirical shot-count proportions.

        Returns:
            Tuple of (shot_prob_by_zone, goal_prob_by_zone)
        """
        if self.use_gpu:
            shot_df = shot_df.to_pandas()
            if pass_df is not None:
                pass_df = pass_df.to_pandas()

        n_zones = len(self.zones)
        shot_probs = np.zeros(n_zones)

        # Derive P(shot|zone) from actual shot counts vs total possessions
        shot_counts = shot_df['stage'].value_counts()

        if pass_df is not None:
            accurate_passes = pass_df[pass_df['accurate'] == True]
            pass_counts = accurate_passes['stage'].value_counts()
        else:
            pass_counts = pd.Series(dtype=int)

        for i, zone in enumerate(self.zones):
            n_shots = shot_counts.get(zone, 0)
            n_passes = pass_counts.get(zone, 0)
            total_events = n_shots + n_passes
            shot_probs[i] = n_shots / total_events if total_events > 0 else 0.05

        # P(goal|shot, zone): domain-knowledge conversion rates.
        # Shot outcome columns are excluded from the feature dataset to prevent leakage,
        # so we use published football statistics here.
        zone_conversion_rates = {
            'bottom': 0.02,   # Almost never score from own half
            'middle': 0.05,   # Low conversion from distance
            'top': 0.12,      # Reasonable conversion in attacking third
        }
        goal_probs = np.array([
            zone_conversion_rates.get(zone, 0.05) for zone in self.zones
        ])

        return shot_probs, goal_probs

    def iterate_threat_values(
        self,
        transition_matrix: np.ndarray,
        shot_probs: np.ndarray,
        goal_probs: np.ndarray
    ) -> np.ndarray:
        """
        Iteratively compute Expected Threat values until convergence

        xT[zone] = P(shot|zone) * P(goal|shot, zone) +
                   P(move|zone) * sum(P(dest|zone) * xT[dest])

        Args:
            transition_matrix: Zone transition probabilities
            shot_probs: P(shot|zone)
            goal_probs: P(goal|shot, zone)

        Returns:
            Array of threat values per zone
        """
        n_zones = len(self.zones)
        threat_values = np.zeros(n_zones)

        for iteration in range(self.max_iterations):
            old_values = threat_values.copy()

            for i in range(n_zones):
                # Immediate shot threat
                shot_threat = shot_probs[i] * goal_probs[i]

                # Move threat (pass to other zones)
                move_prob = 1 - shot_probs[i]
                move_threat = move_prob * np.dot(transition_matrix[i], threat_values)

                threat_values[i] = shot_threat + move_threat

            # Check convergence
            delta = np.abs(threat_values - old_values).max()
            if delta < self.convergence_threshold:
                print(f"xT converged after {iteration + 1} iterations (delta={delta:.6f})")
                break
        else:
            print(f"xT did not converge after {self.max_iterations} iterations")

        self.zone_threat_values = threat_values
        return threat_values

    def fit(
        self,
        pass_df: Union[pd.DataFrame, 'cudf.DataFrame'],
        shot_df: Union[pd.DataFrame, 'cudf.DataFrame']
    ) -> 'ExpectedThreatCalculator':
        """
        Fit Expected Threat model

        Args:
            pass_df: Pass event data
            shot_df: Shot event data

        Returns:
            Self
        """
        print("Computing Expected Threat model...")

        # Compute transition matrix
        transition_matrix = self.compute_transition_matrix(pass_df)
        print(f"Transition matrix:\n{transition_matrix}")

        # Compute shot probabilities (pass_df used to estimate total possessions per zone)
        shot_probs, goal_probs = self.compute_shot_probabilities(shot_df, pass_df)

        # Iterate to convergence
        threat_values = self.iterate_threat_values(transition_matrix, shot_probs, goal_probs)

        print("\nExpected Threat values by zone:")
        for zone, value in zip(self.zones, threat_values):
            print(f"  {zone:10s}: {value:.6f}")

        return self

    def calculate_threat_added(
        self,
        pass_df: Union[pd.DataFrame, 'cudf.DataFrame']
    ) -> Union[pd.DataFrame, 'cudf.DataFrame']:
        """
        Calculate threat added for each pass

        threat_added = xT[destination] - xT[origin]

        Args:
            pass_df: Pass events with origin zones

        Returns:
            DataFrame with threat_added column
        """
        if self.zone_threat_values is None:
            raise ValueError("Model must be fit before calculating threat added")

        # Convert to pandas if needed
        use_cudf = isinstance(pass_df, cudf.DataFrame) if CUDF_AVAILABLE else False
        if use_cudf:
            df = pass_df.to_pandas().copy()
        else:
            df = pass_df.copy()

        # Map zones to threat values
        zone_to_threat = dict(zip(self.zones, self.zone_threat_values))

        # Get origin threat
        df['origin_threat'] = df['stage'].map(zone_to_threat)

        # Get destination threat (from sequential analysis)
        df = df.sort_values(['player_appearance_id', 'period', 'minute'])
        df['dest_zone'] = df.groupby(['player_appearance_id', 'period'])['stage'].shift(-1)
        df['dest_threat'] = df['dest_zone'].map(zone_to_threat)

        # Calculate threat added
        df['threat_added'] = df['dest_threat'] - df['origin_threat']

        # Only keep rows with valid threat calculations
        df = df.dropna(subset=['threat_added'])

        # Convert back to cuDF if needed
        if use_cudf:
            return cudf.from_pandas(df)
        return df


def aggregate_xt_features(
    pass_df: Union[pd.DataFrame, 'cudf.DataFrame'],
    checkpoint_df: Union[pd.DataFrame, 'cudf.DataFrame'],
    xt_calculator: ExpectedThreatCalculator,
    use_gpu: bool = False
) -> Union[pd.DataFrame, 'cudf.DataFrame']:
    """
    Aggregate Expected Threat features per player per checkpoint.

    Creates rolling 15-min and cumulative xT-added features using a vectorized
    merge+groupby approach instead of nested loops.

    Args:
        pass_df: Pass events with temporal information
        checkpoint_df: Checkpoint observations
        xt_calculator: Fitted ExpectedThreatCalculator
        use_gpu: Use cuDF backend

    Returns:
        DataFrame with xT features per checkpoint
    """
    # Calculate threat added for all passes
    passes_with_threat = xt_calculator.calculate_threat_added(pass_df)

    # Convert to pandas for operations
    if use_gpu and CUDF_AVAILABLE:
        passes_pd = passes_with_threat.to_pandas()
        checkpoint_pd = checkpoint_df.to_pandas()
    else:
        passes_pd = passes_with_threat
        checkpoint_pd = checkpoint_df.copy()

    group_keys = ['player_appearance_id', 'checkpoint_min', 'checkpoint_period']

    # Cross-join passes with checkpoints per player, then apply temporal filters.
    # Each pass row is duplicated for every checkpoint of the same player.
    merged = passes_pd.merge(
        checkpoint_pd[['player_appearance_id', 'checkpoint_min', 'checkpoint_period', 'minute_in']],
        on='player_appearance_id',
        how='inner'
    )

    # --- Cumulative window: minute_in <= minute <= checkpoint_min, same period ---
    cumul_mask = (
        (merged['period'] == merged['checkpoint_period']) &
        (merged['minute'] >= merged['minute_in']) &
        (merged['minute'] <= merged['checkpoint_min'])
    )
    cumul_agg = (
        merged[cumul_mask]
        .groupby(group_keys, sort=False)
        .agg(cumul_xt_added=('threat_added', 'sum'), cumul_xt_count=('threat_added', 'count'))
        .reset_index()
    )

    # --- Rolling 15-min window: checkpoint_min-15 < minute <= checkpoint_min, same period ---
    last15_mask = (
        (merged['period'] == merged['checkpoint_period']) &
        (merged['minute'] > merged['checkpoint_min'] - 15) &
        (merged['minute'] <= merged['checkpoint_min'])
    )
    last15_agg = (
        merged[last15_mask]
        .groupby(group_keys, sort=False)
        .agg(last15_xt_added=('threat_added', 'sum'), last15_xt_count=('threat_added', 'count'))
        .reset_index()
    )

    # Merge aggregated features back to checkpoint_pd; fill 0 where no passes found
    checkpoint_pd = checkpoint_pd.merge(cumul_agg, on=group_keys, how='left')
    checkpoint_pd = checkpoint_pd.merge(last15_agg, on=group_keys, how='left')

    checkpoint_pd['cumul_xt_added'] = checkpoint_pd['cumul_xt_added'].fillna(0.0)
    checkpoint_pd['cumul_xt_count'] = checkpoint_pd['cumul_xt_count'].fillna(0).astype(int)
    checkpoint_pd['last15_xt_added'] = checkpoint_pd['last15_xt_added'].fillna(0.0)
    checkpoint_pd['last15_xt_count'] = checkpoint_pd['last15_xt_count'].fillna(0).astype(int)

    # Convert back to cuDF if needed
    if use_gpu and CUDF_AVAILABLE:
        return cudf.from_pandas(checkpoint_pd)

    return checkpoint_pd
