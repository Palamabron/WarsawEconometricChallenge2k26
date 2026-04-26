"""Create EDA and modeling figures for the WEC2026 LaTeX report.

EDA-only figures (targets, events, correlations) can be regenerated with
``python scripts/wec_eda.py``. This script adds grouped-CV model benchmarks and
importance plots on top of the same fold-safe feature build.
"""

from __future__ import annotations

import argparse
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import wec_eda  # noqa: E402

from src.temporal_features import TARGET_COL  # noqa: E402

FIGURE_DIR = wec_eda.FIGURE_DIR
SUMMARY_PATH = FIGURE_DIR / "eda_plot_summary.json"

DEFAULT_EXPERIMENT_DIR = PROJECT_ROOT / "outputs" / "experiments" / "wec_modeling_upgrade"


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


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


def make_model_plots(metrics: pd.DataFrame, importances: pd.DataFrame) -> dict[str, Any]:
    metrics.to_csv(FIGURE_DIR / "model_metrics.csv", index=False)
    importances.to_csv(FIGURE_DIR / "model_feature_importance.csv", index=False)

    _fig, ax = plt.subplots(figsize=(8, 4.5))
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
    _fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    for ax, model_name in zip(axes[0], best_models, strict=True):
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


def make_experiment_artifact_plots(
    experiment_dir: Path, top_importance: int
) -> dict[str, Any]:
    """Use metrics/feature importance produced by run_experiment.py (no extra CV training)."""
    metrics_path = experiment_dir / "metrics.csv"
    importance_path = experiment_dir / "feature_importance.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics export: {metrics_path}")
    if not importance_path.exists():
        raise FileNotFoundError(f"Missing feature importance export: {importance_path}")

    metrics = pd.read_csv(metrics_path)
    ranked = metrics.sort_values(["pr_auc", "roc_auc"], ascending=False)
    oof_only = ranked[ranked["feature_set"].eq("all_features")].copy()
    if oof_only.empty:
        oof_only = ranked.copy()

    _fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(oof_only))
    width = 0.35
    ax.bar(x - width / 2, oof_only["pr_auc"], width, label="PR-AUC")
    ax.bar(x + width / 2, oof_only["roc_auc"], width, label="ROC-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(oof_only["model"], rotation=25, ha="right")
    ax.set_ylim(0, max(0.25, float(oof_only["roc_auc"].max()) + 0.05))
    ax.set_title("Experiment models (OOF, all_features)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    savefig(FIGURE_DIR / "model_comparison.pdf")
    oof_only.to_csv(FIGURE_DIR / "model_metrics.csv", index=False)

    importance = pd.read_csv(importance_path)
    if not importance.empty and importance["importance_pr_auc_drop"].abs().sum() > 0:
        top = importance.sort_values("importance_pr_auc_drop", ascending=False).head(top_importance)
        plt.figure(figsize=(8, max(3.2, 0.28 * len(top))))
        plt.barh(top["feature"][::-1], top["importance_pr_auc_drop"][::-1], color="#4C78A8")
        plt.title("Top OOF permutation importance (from run_experiment.py)")
        plt.xlabel("PR-AUC drop after permutation")
        plt.grid(axis="x", alpha=0.25)
        savefig(FIGURE_DIR / "model_feature_importance.pdf")
        top.to_csv(FIGURE_DIR / "model_feature_importance.csv", index=False)
        importance_payload = top.to_dict(orient="records")
    else:
        importance_payload = []

    return {
        "source": "run_experiment_exports",
        "experiment_dir": str(experiment_dir),
        "metrics": oof_only.to_dict(orient="records"),
        "importance": importance_payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate EDA and modeling report figures")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip the internal grouped-CV retraining benchmark; plot from run_experiment outputs",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
        help="Directory containing metrics.csv and feature_importance.csv from run_experiment.py",
    )
    parser.add_argument(
        "--top-importance",
        type=int,
        default=15,
        help="Number of top features to plot in fast mode",
    )
    args = parser.parse_args()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint, event_dfs = wec_eda.load_data()
    summary, features, feature_cols, splits = wec_eda.run_eda(checkpoint, event_dfs)
    if args.fast:
        summary.update(make_experiment_artifact_plots(args.experiment_dir, args.top_importance))
    else:
        metrics, importances = evaluate_models(features, feature_cols, splits)
        summary.update(make_model_plots(metrics, importances))
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote figures and summaries to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
