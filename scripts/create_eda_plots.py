"""Create EDA and modeling figures for the WEC2026 LaTeX report.

The script intentionally keeps all model evaluation fixture-grouped. Feature
correlations are descriptive EDA statistics, while model importances are averaged
over validation folds from the best grouped-CV models in this run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.temporal_features import TARGET_COL, TemporalFeatureBuilder, modeling_columns

DATA_DIR = PROJECT_ROOT / "data"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
SUMMARY_PATH = FIGURE_DIR / "eda_plot_summary.json"


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

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
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
    pressure_stage = pressure.groupby("stage", observed=True).agg(
        accuracy=("accurate_bool", "mean"),
        turnover=("is_turnover", "mean"),
    ).reindex(["bottom", "middle", "top"])

    stages = np.arange(3)
    width = 0.28
    fig, ax = plt.subplots(figsize=(8, 4.5))
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


def encode_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train = train[cols].copy()
    x_val = val[cols].copy()
    cat_cols = x_train.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    num_cols = [c for c in cols if c not in cat_cols]

    train_num = x_train[num_cols].apply(pd.to_numeric, errors="coerce")
    val_num = x_val[num_cols].apply(pd.to_numeric, errors="coerce")
    imputer = SimpleImputer(strategy="median")
    train_num = pd.DataFrame(imputer.fit_transform(train_num), columns=num_cols, index=train.index)
    val_num = pd.DataFrame(imputer.transform(val_num), columns=num_cols, index=val.index)

    if cat_cols:
        train_cat = pd.get_dummies(x_train[cat_cols].fillna("missing").astype(str), prefix=cat_cols)
        val_cat = pd.get_dummies(x_val[cat_cols].fillna("missing").astype(str), prefix=cat_cols)
        val_cat = val_cat.reindex(columns=train_cat.columns, fill_value=0)
    else:
        train_cat = pd.DataFrame(index=train.index)
        val_cat = pd.DataFrame(index=val.index)

    return (
        pd.concat([train_num, train_cat], axis=1).astype("float32"),
        pd.concat([val_num, val_cat], axis=1).astype("float32"),
    )


def model_factory(name: str, y_train: pd.Series):
    pos_weight = (len(y_train) - y_train.sum()) / max(float(y_train.sum()), 1.0)
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=350,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=4,
            reg_alpha=0.05,
            reg_lambda=4.0,
            scale_pos_weight=pos_weight,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    if name == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", C=0.25, max_iter=2000, random_state=42),
        )
    raise ValueError(f"Unknown model {name}")


def evaluate_models(
    features: pd.DataFrame,
    feature_cols: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = features[TARGET_COL].astype(int)
    rows: list[dict[str, Any]] = []
    importance_rows: list[pd.Series] = []
    models = ["xgboost", "extra_trees", "random_forest", "logistic"]

    for model_name in models:
        oof = np.zeros(len(features), dtype=float)
        fold_importances = []
        for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):
            train = features.iloc[train_idx]
            val = features.iloc[val_idx]
            x_train, x_val = encode_features(train, val, feature_cols)
            y_train = train[TARGET_COL].astype(int)
            model = model_factory(model_name, y_train)
            model.fit(x_train, y_train)
            oof[val_idx] = model.predict_proba(x_val)[:, 1]

            estimator = model[-1] if model_name == "logistic" else model
            if hasattr(estimator, "feature_importances_"):
                fold_importances.append(
                    pd.Series(estimator.feature_importances_, index=x_train.columns, name=fold_id)
                )
            elif hasattr(estimator, "coef_"):
                fold_importances.append(
                    pd.Series(np.abs(estimator.coef_[0]), index=x_train.columns, name=fold_id)
                )

        rows.append(
            {
                "model": model_name,
                "roc_auc": roc_auc_score(y, oof),
                "pr_auc": average_precision_score(y, oof),
                "brier": brier_score_loss(y, oof),
            }
        )
        if fold_importances:
            imp = pd.concat(fold_importances, axis=1).fillna(0).mean(axis=1)
            imp = (imp / imp.sum()).sort_values(ascending=False)
            for feature, value in imp.head(25).items():
                importance_rows.append(
                    pd.Series({"model": model_name, "feature": feature, "importance": float(value)})
                )

    metrics = pd.DataFrame(rows).sort_values(["pr_auc", "roc_auc"], ascending=False)
    importances = pd.DataFrame(importance_rows)
    return metrics, importances


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
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(corr_df["feature"][::-1], corr_df["correlation"][::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Top Absolute Feature Correlations with Target")
    ax.set_xlabel("Point-biserial correlation")
    ax.grid(axis="x", alpha=0.25)
    savefig(FIGURE_DIR / "feature_correlations.pdf")
    return corr_df


def make_model_plots(metrics: pd.DataFrame, importances: pd.DataFrame) -> dict[str, Any]:
    metrics.to_csv(FIGURE_DIR / "model_metrics.csv", index=False)
    importances.to_csv(FIGURE_DIR / "model_feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width / 2, metrics["pr_auc"], width, label="PR-AUC")
    ax.bar(x + width / 2, metrics["roc_auc"], width, label="ROC-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics["model"], rotation=25, ha="right")
    ax.set_ylim(0, max(0.75, metrics["roc_auc"].max() + 0.05))
    ax.set_title("Grouped Cross-Validated Model Ranking")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    savefig(FIGURE_DIR / "model_comparison.pdf")

    best_models = metrics.head(2)["model"].tolist()
    top_importance = (
        importances[importances["model"].isin(best_models)]
        .sort_values(["model", "importance"], ascending=[True, False])
        .groupby("model", observed=True)
        .head(12)
    )
    n = len(best_models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    for ax, model_name in zip(axes[0], best_models):
        subset = top_importance[top_importance["model"] == model_name].sort_values("importance")
        ax.barh(subset["feature"], subset["importance"], color="#4C78A8")
        ax.set_title(f"{model_name}: top importances")
        ax.set_xlabel("Normalized importance")
        ax.grid(axis="x", alpha=0.25)
    savefig(FIGURE_DIR / "model_feature_importance.pdf")

    return {
        "best_models": best_models,
        "metrics": metrics.to_dict(orient="records"),
        "top_importances": top_importance.to_dict(orient="records"),
    }


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint, event_dfs = load_data()
    summary: dict[str, Any] = {}
    summary.update(make_target_plots(checkpoint))
    summary.update(make_event_quality_plot(event_dfs))
    features, feature_cols, splits = build_fold_safe_features(checkpoint, event_dfs)
    corr_df = make_correlation_plot(features, feature_cols)
    metrics, importances = evaluate_models(features, feature_cols, splits)
    summary.update(make_model_plots(metrics, importances))
    summary["top_correlations"] = corr_df.to_dict(orient="records")
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote figures and summaries to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
