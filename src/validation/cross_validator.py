"""
Cross-Validation Strategy with Group-based splitting

Implements fixture-level GroupKFold to prevent match-level data leakage.
"""

import logging
from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from src.config import get_config

logger = logging.getLogger(__name__)


class FixtureGroupKFold:
    """
    Cross-validation iterator with fixture-level grouping

    Ensures entire matches stay within single folds to prevent
    tactical/environmental leakage.
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
        stratify: bool = False,
    ):
        """
        Initialize fixture-grouped cross-validator

        Args:
            n_splits: Number of folds
            shuffle: Shuffle groups before splitting
            random_state: Random seed
            stratify: Attempt to balance target distribution across folds
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.stratify = stratify

        # Use stratified grouping only when requested. For non-stratified shuffling,
        # create shuffled group holdouts manually while preserving fixture isolation.
        if stratify:
            self.splitter = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=shuffle, random_state=random_state if shuffle else None
            )
        elif shuffle:
            self.splitter = None
        else:
            # GroupKFold does not natively support shuffle, so when shuffle is requested
            # without stratification, we permute unique groups via random_state before
            # splitting (handled in the split() method) and use plain GroupKFold.
            self.splitter = GroupKFold(n_splits=n_splits)

    def split(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        groups: pd.Series | np.ndarray,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/validation indices

        Args:
            X: Feature matrix
            y: Target vector
            groups: Group identifiers (fixture_id)

        Yields:
            Tuple of (train_indices, validation_indices)
        """
        logger.info(
            "Generating %s-fold cross-validation splits: group_by=fixture_id stratified=%s",
            self.n_splits,
            self.stratify,
        )

        # Convert to numpy arrays if needed
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        if isinstance(groups, pd.Series):
            groups = groups.to_numpy()

        # Generate splits
        fold_num = 1
        splits = (
            self._shuffled_group_splits(groups)
            if self.splitter is None
            else self.splitter.split(X, y, groups=groups)
        )
        for train_idx, val_idx in splits:
            # Validate split
            train_groups = set(groups[train_idx])
            val_groups = set(groups[val_idx])

            overlap = train_groups.intersection(val_groups)
            if overlap:
                raise ValueError(f"Fold {fold_num}: Group overlap detected! {overlap}")

            # Calculate class distribution
            train_pos = y[train_idx].sum()
            val_pos = y[val_idx].sum()
            train_pct = (train_pos / len(train_idx)) * 100
            val_pct = (val_pos / len(val_idx)) * 100

            logger.info(
                "Fold %s: train_samples=%s train_positive=%s train_positive_rate=%.2f%% "
                "val_samples=%s val_positive=%s val_positive_rate=%.2f%% "
                "train_matches=%s val_matches=%s",
                fold_num,
                len(train_idx),
                train_pos,
                train_pct,
                len(val_idx),
                val_pos,
                val_pct,
                len(train_groups),
                len(val_groups),
            )

            yield train_idx, val_idx
            fold_num += 1

    def _shuffled_group_splits(self, groups: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate deterministic shuffled group holdouts without stratifying by y."""
        unique_groups = np.unique(groups)
        rng = np.random.default_rng(self.random_state)
        shuffled_groups = rng.permutation(unique_groups)
        group_folds = np.array_split(shuffled_groups, self.n_splits)
        all_indices = np.arange(len(groups))

        for val_groups in group_folds:
            val_mask = np.isin(groups, val_groups)
            yield all_indices[~val_mask], all_indices[val_mask]

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Get number of splits"""
        return self.n_splits


def create_cross_validator(config_path: str = "config.yaml") -> FixtureGroupKFold:
    """
    Create cross-validator from config

    Args:
        config_path: Path to configuration file

    Returns:
        Configured FixtureGroupKFold instance
    """
    config = get_config(config_path)

    return FixtureGroupKFold(
        n_splits=config.n_folds,
        shuffle=True,
        random_state=config.random_state,
        stratify=config.get("cross_validation.stratify", False),
    )


def get_oof_predictions(
    model, X: pd.DataFrame, y: pd.Series, groups: pd.Series, cv: FixtureGroupKFold
) -> np.ndarray:
    """
    Generate out-of-fold predictions for stacking

    Args:
        model: Model instance with fit/predict_proba methods
        X: Feature matrix
        y: Target vector
        groups: Group identifiers
        cv: Cross-validator

    Returns:
        Out-of-fold predictions (same length as X)
    """
    oof_preds = np.zeros(len(X))

    for train_idx, val_idx in cv.split(X, y, groups):
        # Train on fold
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train = y.iloc[train_idx]

        model.fit(X_train, y_train)

        # Predict on validation
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

    return oof_preds
