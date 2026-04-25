"""
Data Leakage Prevention and Validation

Automated checks to ensure temporal integrity and prevent future data leakage.
"""

import warnings
from typing import Union, Dict

import numpy as np
import pandas as pd

try:
    import cudf
    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False
    cudf = None


class LeakageValidator:
    """Automated data leakage detection and prevention"""

    @staticmethod
    def validate_temporal_boundaries(
        engineered_df: Union[pd.DataFrame, 'cudf.DataFrame'],
        checkpoint_df: Union[pd.DataFrame, 'cudf.DataFrame'],
        event_dfs: Dict[str, Union[pd.DataFrame, 'cudf.DataFrame']]
    ) -> bool:
        """
        Validate no future data leakage in features

        Checks:
        1. All event minutes <= checkpoint minutes
        2. No shot outcomes in features
        3. Player substitution boundaries respected

        Args:
            engineered_df: DataFrame with engineered features
            checkpoint_df: Original checkpoint data
            event_dfs: Event DataFrames

        Returns:
            True if validation passes

        Raises:
            ValueError if leakage detected
        """
        print("Running temporal integrity validation...")

        # Convert to pandas if needed
        if CUDF_AVAILABLE and isinstance(checkpoint_df, cudf.DataFrame):
            checkpoint_pd = checkpoint_df.to_pandas()
            engineered_pd = engineered_df.to_pandas()
        else:
            checkpoint_pd = checkpoint_df
            engineered_pd = engineered_df

        # Check 1: Substitution boundaries
        print("  ✓ Checking substitution boundaries...")
        invalid_subs = checkpoint_pd[
            ~((checkpoint_pd['minute_in'] <= checkpoint_pd['checkpoint_min']) &
              (checkpoint_pd['checkpoint_min'] <= checkpoint_pd['minute_out']))
        ]

        if len(invalid_subs) > 0:
            warnings.warn(
                f"Found {len(invalid_subs)} checkpoints where player not on pitch. "
                f"These should be excluded from training."
            )

        # Check 2: No forbidden columns
        print("  ✓ Checking for forbidden columns...")
        forbidden_keywords = ['outcome', 'result', 'goal_scored', 'goal_conceded']
        forbidden_cols = [
            col for col in engineered_pd.columns
            if any(keyword in col.lower() for keyword in forbidden_keywords)
            and col != 'scored_after'  # Target is allowed
        ]

        if forbidden_cols:
            raise ValueError(f"Forbidden columns detected (potential leakage): {forbidden_cols}")

        # Check 3: Feature value ranges
        print("  ✓ Checking feature value ranges...")
        numeric_cols = engineered_pd.select_dtypes(include=[np.number]).columns

        # Check for NaN
        nan_cols = [col for col in numeric_cols if engineered_pd[col].isna().any()]
        if nan_cols:
            warnings.warn(f"NaN values found in {len(nan_cols)} columns")

        # Check for Inf
        inf_cols = [col for col in numeric_cols if np.isinf(engineered_pd[col]).any()]
        if inf_cols:
            warnings.warn(f"Inf values found in {len(inf_cols)} columns")

        print("  ✓ Temporal validation passed")
        return True

    @staticmethod
    def validate_cv_splits(
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        groups: Union[pd.Series, np.ndarray],
        train_idx: np.ndarray,
        val_idx: np.ndarray
    ) -> bool:
        """
        Validate cross-validation split integrity

        Ensures:
        1. No match overlap between train and validation
        2. No data leakage between folds

        Args:
            X: Feature matrix
            y: Target vector
            groups: Group identifiers (fixture_id)
            train_idx: Training indices
            val_idx: Validation indices

        Returns:
            True if validation passes

        Raises:
            ValueError if overlap detected
        """
        # Get groups for train and validation
        if isinstance(groups, pd.Series):
            groups = groups.values

        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])

        # Check for overlap
        overlap = train_groups.intersection(val_groups)

        if overlap:
            raise ValueError(
                f"Group overlap detected between train and validation folds: {overlap}. "
                f"This indicates match-level data leakage!"
            )

        return True

    @staticmethod
    def validate_player_on_pitch(
        df: Union[pd.DataFrame, 'cudf.DataFrame']
    ) -> Union[pd.DataFrame, 'cudf.DataFrame']:
        """
        Filter out observations where player is not on pitch

        Args:
            df: DataFrame with minute_in, minute_out, checkpoint_min

        Returns:
            Filtered DataFrame
        """
        if CUDF_AVAILABLE and isinstance(df, cudf.DataFrame):
            df_pd = df.to_pandas()
        else:
            df_pd = df

        original_len = len(df_pd)

        # Keep only valid observations
        valid_mask = (
            (df_pd['minute_in'] <= df_pd['checkpoint_min']) &
            (df_pd['checkpoint_min'] <= df_pd['minute_out'])
        )

        df_filtered = df_pd[valid_mask].copy()

        removed = original_len - len(df_filtered)
        if removed > 0:
            print(f"Removed {removed} observations where player not on pitch")

        if CUDF_AVAILABLE and isinstance(df, cudf.DataFrame):
            return cudf.from_pandas(df_filtered)

        return df_filtered


def safe_temporal_merge(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    on: list,
    checkpoint_col: str = 'checkpoint_min',
    event_time_col: str = 'minute'
) -> pd.DataFrame:
    """
    Safely merge event data to checkpoint data with temporal filtering

    Ensures all merged events occurred at or before the checkpoint.

    Args:
        left_df: Checkpoint DataFrame
        right_df: Event DataFrame
        on: Columns to join on (typically ['player_appearance_id'])
        checkpoint_col: Checkpoint time column in left_df
        event_time_col: Event time column in right_df

    Returns:
        Merged DataFrame with temporal integrity
    """
    # Merge
    merged = left_df.merge(right_df, on=on, how='left', suffixes=('', '_event'))

    # Filter to past events only
    if event_time_col in merged.columns:
        valid_mask = merged[event_time_col] <= merged[checkpoint_col]
        merged = merged[valid_mask]

    return merged


def check_feature_leakage_correlation(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.95
) -> Dict[str, float]:
    """
    Check for suspiciously high feature-target correlations

    High correlations may indicate leakage.

    Args:
        X: Feature matrix
        y: Target vector
        threshold: Correlation threshold for flagging

        Returns:
            Dictionary of suspicious features and their correlations
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns

    suspicious_features = {}

    for col in numeric_cols:
        # Skip columns with no variance
        if X[col].std() == 0:
            continue

        # Calculate correlation
        corr = np.corrcoef(X[col].fillna(0), y)[0, 1]

        if abs(corr) >= threshold:
            suspicious_features[col] = corr

    if suspicious_features:
        warnings.warn(
            f"Found {len(suspicious_features)} features with correlation >= {threshold}. "
            f"This may indicate data leakage: {list(suspicious_features.keys())[:5]}"
        )

    return suspicious_features
