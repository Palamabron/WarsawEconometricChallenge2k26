"""Exploratory data analysis figures for WEC2026 (no model training).

Run standalone:
    python scripts/wec_eda.py

Outputs under ``reports/figures/``: target rates, event stage quality, feature
correlations, and ``eda_summary.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.temporal_features import TARGET_COL, TemporalFeatureBuilder, modeling_columns  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
EDA_SUMMARY_PATH = FIGURE_DIR / "eda_summary.json"


def load_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    checkpoint = pd.read_csv(DATA_DIR / "players_quarters_final.csv", parse_dates=["date"])
    event_dfs = {
        "pass": pd.read_csv(DATA_DIR / "player_appearance_pass.csv"),
        "run": pd.read_csv(DATA_DIR / "player_appearance_run.csv"),
        "shot": pd.read_csv(DATA_DIR / "player_appearance_shot_limited.csv"),
        "pressure": pd.read_csv(DATA_DIR / "player_appearance_behaviour_under_pressure.csv"),
    }
    return checkpoint.reset_index(drop=True), event_dfs


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def make_target_plots(checkpoint: pd.DataFrame) -> dict[str, Any]:
    y = checkpoint[TARGET_COL].astype(int)
    order = ["H1_15", "H1_30", "H1_45", "H2_15", "H2_30", "H2_45", "ET1_15"]
    checkpoint_rates = (
        checkpoint.assign(target=y)
        .groupby("checkpoint", observed=True)["target"]
        .agg(["count", "sum", "mean"])
        .reindex(order)
        .dropna()
    )
    position_rates = (
        checkpoint.assign(target=y)
        .groupby("position", observed=True)["target"]
        .agg(["count", "sum", "mean"])
        .reindex(["A", "M", "D", "G"])
    )

    _fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(checkpoint_rates.index, checkpoint_rates["mean"] * 100, color="#4C78A8")
    axes[0].set_title("Scoring Rate by Checkpoint")
    axes[0].set_ylabel("Positive rate (%)")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(position_rates.index, position_rates["mean"] * 100, color="#F58518")
    axes[1].set_title("Scoring Rate by Position")
    axes[1].set_ylabel("Positive rate (%)")
    axes[1].grid(axis="y", alpha=0.25)
    savefig(FIGURE_DIR / "target_rates.pdf")

    return {
        "checkpoint_rates": {
            str(k): float(v) for k, v in checkpoint_rates["mean"].dropna().items()
        },
        "position_rates": {str(k): float(v) for k, v in position_rates["mean"].dropna().items()},
    }


def make_event_quality_plot(event_dfs: dict[str, pd.DataFrame]) -> dict[str, Any]:
    passes = event_dfs["pass"].copy()
    pressure = event_dfs["pressure"].copy()
    passes["accurate_bool"] = passes["accurate"].astype(str).str.upper().isin(["TRUE", "1"])
    passes["stage"] = passes["stage"].fillna("missing")
    pressure["accurate_bool"] = pressure["accurate"].astype(str).str.upper().isin(["TRUE", "1"])
    pressure["is_turnover"] = pressure["press_induced_outcome"].eq("turnover")
    pressure["stage"] = pressure["stage"].fillna("missing")

    pass_acc = passes.groupby("stage", observed=True)["accurate_bool"].mean().reindex(
        ["bottom", "middle", "top"]
    )
    pressure_stage = (
        pressure.groupby("stage", observed=True)
        .agg(
            accuracy=("accurate_bool", "mean"),
            turnover=("is_turnover", "mean"),
        )
        .reindex(["bottom", "middle", "top"])
    )

    stages = np.arange(3)
    width = 0.28
    _fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(stages - width, pass_acc.to_numpy() * 100, width, label="Pass accuracy")
    ax.bar(stages, pressure_stage["accuracy"].to_numpy() * 100, width, label="Pressure accuracy")
    ax.bar(stages + width, pressure_stage["turnover"].to_numpy() * 100, width, label="Pressure turnover")
    ax.set_xticks(stages)
    ax.set_xticklabels(["Bottom", "Middle", "Top"])
    ax.set_ylabel("Rate (%)")
    ax.set_title("Stage-Specific Technical Quality")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    savefig(FIGURE_DIR / "event_stage_quality.pdf")

    return {
        "pass_accuracy": {str(k): float(v) for k, v in pass_acc.dropna().items()},
        "pressure_turnover": {
            str(k): float(v) for k, v in pressure_stage["turnover"].dropna().items()
        },
    }


def build_fold_safe_features(
    checkpoint: pd.DataFrame,
    event_dfs: dict[str, pd.DataFrame],
    n_splits: int = 5,
) -> tuple[pd.DataFrame, list[str], list[tuple[np.ndarray, np.ndarray]]]:
    y = checkpoint[TARGET_COL].astype(int)
    groups = checkpoint["fixture_id"]
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(cv.split(checkpoint, y, groups))
    fold_features = []
    feature_sets = None

    for train_idx, val_idx in splits:
        builder = TemporalFeatureBuilder().fit(checkpoint.iloc[train_idx].copy(), event_dfs)
        val_features = builder.transform(checkpoint.iloc[val_idx].copy(), event_dfs)
        if feature_sets is None:
            feature_sets = builder.get_feature_sets(modeling_columns(val_features))
        val_features = val_features.assign(_row_idx=val_idx)
        fold_features.append(val_features)

    features = pd.concat(fold_features, axis=0).sort_values("_row_idx").reset_index(drop=True)
    assert feature_sets is not None
    all_features = [col for col in feature_sets["all_features"] if col != "_row_idx"]
    return features, all_features, splits


def make_correlation_plot(features: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    numeric_cols = [
        c
        for c in feature_cols
        if c in features.columns and pd.api.types.is_numeric_dtype(features[c])
    ]
    corr = (
        features[numeric_cols]
        .corrwith(features[TARGET_COL].astype(int))
        .dropna()
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .head(18)
    )
    corr_df = corr.rename("correlation").reset_index().rename(columns={"index": "feature"})
    corr_df.to_csv(FIGURE_DIR / "feature_correlations.csv", index=False)

    colors = np.where(corr_df["correlation"] >= 0, "#4C78A8", "#E45756")
    _fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(corr_df["feature"][::-1], corr_df["correlation"][::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Top Absolute Feature Correlations with Target")
    ax.set_xlabel("Point-biserial correlation")
    ax.grid(axis="x", alpha=0.25)
    savefig(FIGURE_DIR / "feature_correlations.pdf")
    return corr_df


def run_eda(
    checkpoint: pd.DataFrame,
    event_dfs: dict[str, pd.DataFrame],
    n_splits: int = 5,
) -> tuple[dict[str, Any], pd.DataFrame, list[str], list[tuple[np.ndarray, np.ndarray]]]:
    summary: dict[str, Any] = {}
    summary.update(make_target_plots(checkpoint))
    summary.update(make_event_quality_plot(event_dfs))
    features, feature_cols, splits = build_fold_safe_features(checkpoint, event_dfs, n_splits)
    corr_df = make_correlation_plot(features, feature_cols)
    summary["top_correlations"] = corr_df.to_dict(orient="records")
    return summary, features, feature_cols, splits


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint, event_dfs = load_data()
    summary, _, _, _ = run_eda(checkpoint, event_dfs)
    EDA_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote EDA figures and {EDA_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
