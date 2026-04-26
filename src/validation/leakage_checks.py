"""
Data Leakage Prevention and Validation

Automated checks to ensure temporal integrity and prevent future data leakage.
"""

import logging
import warnings
from typing import Union

import numpy as np
import pandas as pd

PERIOD_OFFSETS = {
    "half_1": 0,
    "half_2": 45,
    "extra_time_1": 90,
    "extra_time_2": 105,
}

logger = logging.getLogger(__name__)

try:
    import cudf

    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False
    cudf = None


class LeakageValidator:
    """Automated data leakage detection and prevention"""

    @staticmethod
    def _with_absolute_time(
        df: pd.DataFrame, period_col: str, minute_col: str, out_col: str
    ) -> pd.DataFrame:
        """Add absolute match time from period-relative minutes."""
        out = df.copy()
        offsets = out[period_col].astype(str).map(PERIOD_OFFSETS).fillna(0)
        out[out_col] = offsets + pd.to_numeric(out[minute_col], errors="coerce")
        return out

    @staticmethod
    def validate_temporal_boundaries(
        engineered_df: Union[pd.DataFrame, "cudf.DataFrame"],
        checkpoint_df: Union[pd.DataFrame, "cudf.DataFrame"],
        event_dfs: dict[str, Union[pd.DataFrame, "cudf.DataFrame"]] | None = None,
    ) -> bool:
        """
        Validate no future data leakage in features

        Checks:
        1. No shot outcomes in features
        2. Player substitution boundaries respected
        3. Optional raw event schemas contain time columns needed by temporal features

        Args:
            engineered_df: DataFrame with engineered features
            checkpoint_df: Original checkpoint data
            event_dfs: Optional event DataFrames to schema-check for temporal columns

        Returns:
            True if validation passes

        Raises:
            ValueError if leakage detected
        """
        logger.info("Running temporal integrity validation")

        # Convert to pandas if needed
        if CUDF_AVAILABLE and isinstance(checkpoint_df, cudf.DataFrame):
            checkpoint_pd = checkpoint_df.to_pandas()
            engineered_pd = engineered_df.to_pandas()
        else:
            checkpoint_pd = checkpoint_df
            engineered_pd = engineered_df

        if "checkpoint_period" in checkpoint_pd.columns:
            checkpoint_time = LeakageValidator._with_absolute_time(
                checkpoint_pd, "checkpoint_period", "checkpoint_min", "_checkpoint_abs_min"
            )
            checkpoint_compare_col = "_checkpoint_abs_min"
        else:
            checkpoint_time = checkpoint_pd.copy()
            checkpoint_time["_checkpoint_abs_min"] = pd.to_numeric(
                checkpoint_time["checkpoint_min"], errors="coerce"
            )
            checkpoint_compare_col = "checkpoint_min"

        # Check 1: Substitution boundaries
        logger.info("Checking substitution boundaries")
        invalid_subs = checkpoint_time[
            ~(
                (checkpoint_time["minute_in"] <= checkpoint_time[checkpoint_compare_col])
                & (checkpoint_time[checkpoint_compare_col] <= checkpoint_time["minute_out"])
            )
        ]

        if len(invalid_subs) > 0:
            warnings.warn(
                f"Found {len(invalid_subs)} checkpoints where player not on pitch. "
                f"These should be excluded from training."
            )

        if event_dfs:
            # Check 2: Raw event schemas expose time boundaries for downstream feature builders
            logger.info("Checking raw event time schemas")
            missing_time_cols = [
                name
                for name, event_df in event_dfs.items()
                if "minute" not in event_df.columns or "period" not in event_df.columns
            ]
            if missing_time_cols:
                raise ValueError(
                    "Event DataFrames missing required temporal columns: "
                    f"{missing_time_cols}. Expected both 'period' and 'minute'."
                )

            # Check 3: Event period+minute <= max checkpoint period+minute for that player
            logger.info("Checking event temporal boundaries with period+minute")
            max_checkpoint_per_player = checkpoint_time.groupby("player_appearance_id")[
                "_checkpoint_abs_min"
            ].max()
            for event_name, event_df in event_dfs.items():
                if CUDF_AVAILABLE and isinstance(event_df, cudf.DataFrame):
                    event_pd = event_df.to_pandas()
                else:
                    event_pd = event_df
                if "player_appearance_id" not in event_pd.columns:
                    continue
                event_time = LeakageValidator._with_absolute_time(
                    event_pd, "period", "minute", "_event_abs_min"
                )
                merged_check = event_time.merge(
                    max_checkpoint_per_player.rename("max_checkpoint"),
                    on="player_appearance_id",
                    how="inner",
                )
                future_events = merged_check[
                    merged_check["_event_abs_min"] > merged_check["max_checkpoint"]
                ]
                if len(future_events) > 0:
                    warnings.warn(
                        f"Event dataset '{event_name}': found {len(future_events)} events "
                        f"occurring after the player's last checkpoint period+minute."
                    )

        # Check 4: No forbidden columns
        logger.info("Checking for forbidden columns")
        forbidden_keywords = ["outcome", "result", "goal_scored", "goal_conceded"]
        forbidden_cols = [
            col
            for col in engineered_pd.columns
            if any(keyword in col.lower() for keyword in forbidden_keywords)
            and col != "scored_after"  # Target is allowed
        ]

        if forbidden_cols:
            raise ValueError(f"Forbidden columns detected (potential leakage): {forbidden_cols}")

        # Check 4: Feature value ranges
        logger.info("Checking feature value ranges")
        numeric_cols = engineered_pd.select_dtypes(include=[np.number]).columns

        # Check for NaN
        nan_cols = [col for col in numeric_cols if engineered_pd[col].isna().any()]
        if nan_cols:
            warnings.warn(f"NaN values found in {len(nan_cols)} columns")

        # Check for Inf
        inf_cols = [col for col in numeric_cols if np.isinf(engineered_pd[col]).any()]
        if inf_cols:
            warnings.warn(f"Inf values found in {len(inf_cols)} columns")

        logger.info("Temporal validation passed")
        return True

    @staticmethod
    def validate_cv_splits(
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        groups: pd.Series | np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
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
            groups = groups.to_numpy()

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
        df: Union[pd.DataFrame, "cudf.DataFrame"],
    ) -> Union[pd.DataFrame, "cudf.DataFrame"]:
        """
        Filter out observations where player is not on pitch

        Args:
            df: DataFrame with minute_in, minute_out, checkpoint_min, and
                optionally checkpoint_period for period-aware filtering.

        Returns:
            Filtered DataFrame
        """
        if CUDF_AVAILABLE and isinstance(df, cudf.DataFrame):
            df_pd = df.to_pandas()
        else:
            df_pd = df

        original_len = len(df_pd)

        # Keep only valid observations. Checkpoint minutes are period-relative
        # in WEC data, while substitutions are absolute match minutes.
        if "checkpoint_period" in df_pd.columns:
            checkpoint_time = LeakageValidator._with_absolute_time(
                df_pd, "checkpoint_period", "checkpoint_min", "_checkpoint_abs_min"
            )
            checkpoint_compare = checkpoint_time["_checkpoint_abs_min"]
        else:
            checkpoint_compare = pd.to_numeric(df_pd["checkpoint_min"], errors="coerce")

        valid_mask = (df_pd["minute_in"] <= checkpoint_compare) & (
            checkpoint_compare <= df_pd["minute_out"]
        )

        df_filtered = df_pd[valid_mask].copy()

        removed = original_len - len(df_filtered)
        if removed > 0:
            logger.info("Removed %s observations where player not on pitch", removed)

        if CUDF_AVAILABLE and isinstance(df, cudf.DataFrame):
            return cudf.from_pandas(df_filtered)

        return df_filtered


def safe_temporal_merge(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    on: list,
    checkpoint_col: str = "checkpoint_min",
    event_time_col: str = "minute",
    checkpoint_period_col: str | None = None,
    event_period_col: str | None = None,
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
        checkpoint_period_col: Optional period column in left_df. When provided
            with event_period_col, filtering uses period+minute absolute time.
        event_period_col: Optional period column in right_df.

    Returns:
        Merged DataFrame with temporal integrity
    """
    left_marker = "__safe_temporal_merge_left_row_id"
    while left_marker in left_df.columns or left_marker in right_df.columns:
        left_marker = f"_{left_marker}"

    left_with_marker = left_df.copy()
    left_with_marker[left_marker] = np.arange(len(left_with_marker))

    # Merge
    merged = left_with_marker.merge(right_df, on=on, how="left", suffixes=("", "_event"))

    # Filter to past events only while preserving unmatched left rows.
    # Period-aware filtering is required because event minutes restart each period
    # and stoppage time can create first-half minutes above 45.
    if event_time_col in merged.columns:
        if checkpoint_period_col and event_period_col:
            checkpoint_abs = merged[checkpoint_period_col].astype(str).map(PERIOD_OFFSETS).fillna(
                0
            ) + pd.to_numeric(merged[checkpoint_col], errors="coerce")
            event_abs = merged[event_period_col].astype(str).map(PERIOD_OFFSETS).fillna(
                0
            ) + pd.to_numeric(merged[event_time_col], errors="coerce")
            valid_mask = merged[event_time_col].isna() | (event_abs <= checkpoint_abs)
        else:
            valid_mask = merged[event_time_col].isna() | (
                merged[event_time_col] <= merged[checkpoint_col]
            )
        merged = merged[valid_mask]

    missing_left_ids = left_with_marker.loc[
        ~left_with_marker[left_marker].isin(merged[left_marker]), left_marker
    ]
    if not missing_left_ids.empty:
        empty_event_rows = left_with_marker[
            left_with_marker[left_marker].isin(missing_left_ids)
        ].copy()
        for col in merged.columns:
            if col not in empty_event_rows.columns:
                empty_event_rows[col] = np.nan
        merged = pd.concat([merged, empty_event_rows[merged.columns]], ignore_index=True)

    return merged.sort_values(left_marker).drop(columns=[left_marker]).reset_index(drop=True)


def check_feature_leakage_correlation(
    X: pd.DataFrame, y: pd.Series, threshold: float = 0.95
) -> dict[str, float]:
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
