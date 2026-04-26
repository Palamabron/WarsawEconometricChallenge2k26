"""Regression tests for the report-grade training entrypoint."""

import pandas as pd
import pytest

from run_experiment import make_cv, validate_training_checkpoints, write_report


def _event_dfs() -> dict[str, pd.DataFrame]:
    event_frame = pd.DataFrame(
        {
            "player_appearance_id": pd.Series(dtype="int64"),
            "period": pd.Series(dtype="object"),
            "minute": pd.Series(dtype="float64"),
        }
    )
    return {
        "pass": event_frame.copy(),
        "run": event_frame.copy(),
        "shot": event_frame.copy(),
        "pressure": event_frame.copy(),
    }


def test_validate_training_checkpoints_filters_off_pitch_rows():
    checkpoint = pd.DataFrame(
        {
            "player_appearance_id": [1, 2],
            "player_id": [10, 20],
            "fixture_id": [100, 100],
            "date": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "checkpoint": [1, 1],
            "checkpoint_period": ["half_2", "half_2"],
            "checkpoint_min": [15, 15],
            "minute_in": [46, 70],
            "minute_out": [90, 90],
            "is_home": ["TRUE", "FALSE"],
            "subbed": ["FALSE", "FALSE"],
            "position": ["M", "F"],
            "formation": ["4-3-3", "4-3-3"],
            "scored_after": [0, 1],
        }
    )

    filtered = validate_training_checkpoints(checkpoint, _event_dfs())

    assert filtered["player_appearance_id"].tolist() == [1]


def test_make_cv_rejects_single_fixture():
    y = pd.Series([0, 1, 0, 1])
    groups = pd.Series([1, 1, 1, 1])

    with pytest.raises(ValueError, match="at least 2 distinct fixture_id"):
        make_cv(y, groups, n_splits=5)


def test_make_cv_rejects_single_target_class():
    y = pd.Series([0, 0, 0, 0])
    groups = pd.Series([1, 1, 2, 2])

    with pytest.raises(ValueError, match="both target classes"):
        make_cv(y, groups, n_splits=2)


def test_write_report_requires_grouped_cv_metrics():
    empty_metrics = pd.DataFrame(columns=["model", "feature_set", "pr_auc", "roc_auc"])

    with pytest.raises(ValueError, match="no grouped-CV model metrics"):
        write_report(
            empty_metrics,
            pd.DataFrame(),
            pd.DataFrame(columns=["feature", "importance_pr_auc_drop"]),
            {
                "n_rows": 0,
                "n_positives": 0,
                "positive_rate": 0.0,
                "n_fixtures": 0,
                "n_appearances": 0,
            },
            skipped_models=["xgboost: unavailable"],
        )
