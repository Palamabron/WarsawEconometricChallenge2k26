"""Post-run explainability analysis for the WEC2026 ensemble experiment.

This script is intentionally separate from ``run_experiment.py`` so a long HPO
run can finish first. It consumes saved OOF predictions and fold models, rebuilds
the same fold-safe feature matrices, and writes richer model/ensemble diagnostics.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tyro
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import run_experiment as exp  # noqa: E402

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiments" / "wec_modeling_upgrade"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "outputs" / "models" / "wec_modeling_upgrade"
DEFAULT_ANALYSIS_DIR = (
    PROJECT_ROOT / "outputs" / "experiments" / "wec_modeling_upgrade_explainability"
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExplainabilityArgs:
    """Command-line options for post-run explainability analysis."""

    mode: Literal["full", "scan"] = "full"
    output_dir: Path = DEFAULT_OUTPUT_DIR
    model_dir: Path = DEFAULT_MODEL_DIR
    analysis_dir: Path = DEFAULT_ANALYSIS_DIR
    folds: int | None = None
    top_n: int = 12
    permutation_repeats: int = 2
    shap_sample: int = 200
    max_diagnostic_cols: int = 4
    max_shap_summary_plots: int = 4
    log_level: str = "INFO"


def configure_logging(level: str) -> None:
    """Configure process-wide logging for command-line analysis runs."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def load_run_metadata(output_dir: Path, folds_arg: int | None) -> tuple[pd.DataFrame, int]:
    summary_path = output_dir / "run_summary.json"
    if summary_path.exists():
        with summary_path.open() as f:
            summary = json.load(f)
        folds = int(folds_arg or summary.get("folds", 5))
    else:
        folds = int(folds_arg or 5)

    oof = pd.read_csv(output_dir / "oof_predictions.csv")
    return oof, folds


def encoded_fold_matrices(
    folds: list[exp.FoldArtifacts], feature_cols: list[str]
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    matrices = []
    for fold in folds:
        y_train = fold.train_features[exp.TARGET_COL].astype(int)
        x_train, x_val, _ = exp.feature_matrix(
            fold.train_features,
            fold.val_features,
            feature_cols,
            y_train=y_train,
        )
        matrices.append((x_train, x_val))
    return matrices


def predict_saved_model(model: Any, x: pd.DataFrame) -> np.ndarray:
    return exp.predict_model_proba(model, x)


def native_importance_for_model(model_name: str, model: Any, columns: list[str]) -> pd.DataFrame:
    if isinstance(model, tuple) and hasattr(model[0], "transform"):
        _, estimator = model
        if hasattr(estimator, "coef_"):
            values = np.abs(estimator.coef_).ravel()
            return pd.DataFrame({"feature": columns, "importance": values})
        return pd.DataFrame(columns=["feature", "importance"])

    if hasattr(model, "feature_importances_"):
        return pd.DataFrame(
            {"feature": columns, "importance": np.asarray(model.feature_importances_)}
        )

    if hasattr(model, "coef_"):
        return pd.DataFrame({"feature": columns, "importance": np.abs(model.coef_).ravel()})

    return pd.DataFrame(columns=["feature", "importance"])


def compute_native_importances(
    saved_models: dict[str, list[Any]],
    matrices: list[tuple[pd.DataFrame, pd.DataFrame]],
) -> pd.DataFrame:
    rows = []
    for model_name, fold_models in saved_models.items():
        for fold_num, model in enumerate(fold_models, start=1):
            columns = matrices[fold_num - 1][0].columns.tolist()
            imp = native_importance_for_model(model_name, model, columns)
            if imp.empty:
                continue
            imp["model"] = model_name
            imp["fold"] = fold_num
            rows.append(imp)
    if not rows:
        return pd.DataFrame(columns=["model", "feature", "importance_mean", "importance_std"])
    out = pd.concat(rows, ignore_index=True)
    return (
        out.groupby(["model", "feature"], as_index=False)["importance"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "importance_mean", "std": "importance_std"})
        .sort_values(["model", "importance_mean"], ascending=[True, False])
    )


def compute_permutation_importances(
    saved_models: dict[str, list[Any]],
    folds: list[exp.FoldArtifacts],
    matrices: list[tuple[pd.DataFrame, pd.DataFrame]],
    feature_cols: list[str],
    top_n: int,
    repeats: int,
) -> pd.DataFrame:
    rows = []
    native = compute_native_importances(saved_models, matrices)
    rng = np.random.default_rng(42)

    for model_name, fold_models in saved_models.items():
        candidates = (
            native[native["model"].eq(model_name)]
            .head(max(top_n, 5))["feature"]
            .astype(str)
            .tolist()
        )
        if not candidates:
            candidates = feature_cols[:top_n]

        for fold_num, model in enumerate(fold_models, start=1):
            _, x_val_full = matrices[fold_num - 1]
            y_val = folds[fold_num - 1].val_features[exp.TARGET_COL].astype(int)
            cols = [c for c in candidates if c in x_val_full.columns]
            if not cols:
                continue
            baseline_pred = predict_saved_model(model, x_val_full)
            baseline_score = average_precision_score(y_val, baseline_pred)
            fold_rows = []
            for col in cols:
                drops = []
                for _ in range(repeats):
                    shuffled = x_val_full.copy()
                    shuffled[col] = rng.permutation(shuffled[col].to_numpy())
                    pred = predict_saved_model(model, shuffled)
                    drops.append(baseline_score - average_precision_score(y_val, pred))
                fold_rows.append(
                    {
                        "model": model_name,
                        "fold": fold_num,
                        "feature": col,
                        "pr_auc_drop_mean": float(np.mean(drops)),
                        "pr_auc_drop_std": float(np.std(drops)),
                    }
                )
            rows.append(pd.DataFrame(fold_rows))

    if not rows:
        return pd.DataFrame(columns=["model", "feature", "pr_auc_drop_mean", "pr_auc_drop_std"])
    out = pd.concat(rows, ignore_index=True)
    return (
        out.groupby(["model", "feature"], as_index=False)
        .agg(
            pr_auc_drop_mean=("pr_auc_drop_mean", "mean"),
            pr_auc_drop_std=("pr_auc_drop_mean", "std"),
        )
        .sort_values(["model", "pr_auc_drop_mean"], ascending=[True, False])
    )


def compute_shap_importances(
    saved_models: dict[str, list[Any]],
    matrices: list[tuple[pd.DataFrame, pd.DataFrame]],
    sample_size: int,
) -> pd.DataFrame:
    try:
        import shap
    except Exception:
        return pd.DataFrame(columns=["model", "feature", "mean_abs_shap"])

    supported = {"xgboost", "catboost", "lightgbm", "extra_trees", "random_forest"}
    rows = []
    rng = np.random.default_rng(42)
    for model_name, fold_models in saved_models.items():
        if model_name not in supported:
            continue
        for fold_num, model in enumerate(fold_models, start=1):
            if isinstance(model, tuple):
                continue
            _, x_val = matrices[fold_num - 1]
            if len(x_val) == 0:
                continue
            sample_idx = rng.choice(len(x_val), size=min(sample_size, len(x_val)), replace=False)
            x_sample = x_val.iloc[sample_idx]
            try:
                explainer = shap.TreeExplainer(model)
                values = explainer.shap_values(x_sample)
                if isinstance(values, list):
                    values = values[-1]
                mean_abs = np.abs(values).mean(axis=0)
            except Exception:
                continue
            rows.append(
                pd.DataFrame(
                    {
                        "model": model_name,
                        "fold": fold_num,
                        "feature": x_sample.columns,
                        "mean_abs_shap": mean_abs,
                    }
                )
            )

    if not rows:
        return pd.DataFrame(columns=["model", "feature", "mean_abs_shap"])
    out = pd.concat(rows, ignore_index=True)
    grouped = (
        out.groupby(["model", "feature"], as_index=False)
        .agg(mean_abs_shap=("mean_abs_shap", "mean"))
        .sort_values(["model", "mean_abs_shap"], ascending=[True, False])
    )
    return grouped


def analyze_ensemble(oof_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    oof = pd.read_csv(oof_path)
    y = oof["true_label"].astype(int)
    pred_cols = [
        c for c in oof.columns if c.endswith("_all_features") and c != "stacked_all_features"
    ]
    if not pred_cols:
        return pd.DataFrame(), pd.DataFrame()

    x = oof[pred_cols].copy()
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    meta = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    meta.fit(x_scaled, y)
    weights = pd.DataFrame(
        {
            "base_model": pred_cols,
            "standardized_meta_weight": meta.coef_.ravel(),
            "abs_weight": np.abs(meta.coef_.ravel()),
        }
    ).sort_values("abs_weight", ascending=False)

    perf_rows = []
    for col in pred_cols + (["stacked_all_features"] if "stacked_all_features" in oof else []):
        pred = oof[col].astype(float)
        perf_rows.append(
            {
                "prediction": col,
                "pr_auc": average_precision_score(y, pred),
                "roc_auc": roc_auc_score(y, pred) if y.nunique() == 2 else np.nan,
                "correlation_with_target": pred.corr(y.astype(float)),
            }
        )
    return weights, pd.DataFrame(perf_rows).sort_values("pr_auc", ascending=False)


def write_bar_plot(df: pd.DataFrame, value_col: str, title: str, path: Path, top_n: int) -> None:
    if df.empty:
        return
    plot_df = df.head(top_n).iloc[::-1]
    plt.figure(figsize=(9, max(4, 0.25 * len(plot_df))))
    plt.barh(plot_df["feature"], plot_df[value_col])
    plt.title(title)
    plt.xlabel(value_col)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _short_label(name: str, max_len: int = 42) -> str:
    if len(name) <= max_len:
        return name
    return name[: max_len - 1] + "…"


def select_oof_diagnostic_columns(
    oof: pd.DataFrame,
    ensemble_perf: pd.DataFrame,
    max_cols: int,
) -> list[str]:
    """Pick OOF probability columns for calibration / PR / confusion plots."""
    numeric_excl = {"true_label"}
    candidates = [
        c
        for c in oof.columns
        if c not in numeric_excl and pd.api.types.is_numeric_dtype(oof[c])
    ]
    if not candidates:
        return []

    ordered: list[str] = []
    if "stacked_all_features" in candidates:
        ordered.append("stacked_all_features")

    if not ensemble_perf.empty and "prediction" in ensemble_perf.columns:
        perf_sorted = ensemble_perf.sort_values("pr_auc", ascending=False)
        for col in perf_sorted["prediction"].astype(str):
            if col in candidates and col not in ordered:
                ordered.append(col)
    for col in candidates:
        if col not in ordered:
            ordered.append(col)
    return ordered[:max_cols]


def plot_oof_calibration(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    path: Path,
) -> None:
    if not predictions:
        return
    n = len(predictions)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for ax, (name, proba) in zip(axes.ravel(), predictions.items(), strict=True):
        try:
            prob_true, prob_pred = calibration_curve(
                y_true, proba, n_bins=8, strategy="quantile"
            )
        except ValueError:
            ax.text(0.5, 0.5, "insufficient data", ha="center", va="center")
            ax.set_axis_off()
            continue
        ax.plot(prob_pred, prob_true, marker="o", label="model")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="perfect")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(_short_label(name))
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.25)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    for j in range(len(predictions), nrows * ncols):
        axes.ravel()[j].set_visible(False)
    fig.suptitle("OOF reliability (calibration)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_oof_precision_recall(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    path: Path,
) -> None:
    if not predictions:
        return
    _fig, ax = plt.subplots(figsize=(9.2, 5.4))
    base = float(np.mean(y_true)) if len(y_true) else 0.0
    ax.axhline(base, color="gray", linestyle=":", linewidth=1, label=f"baseline ({base:.4f})")
    for name, proba in predictions.items():
        precision, recall, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        ax.plot(
            recall,
            precision,
            label=f"{_short_label(name, 36)} (PR-AUC={ap:.4f})",
            linewidth=1.2,
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("OOF precision-recall curves")
    ax.legend(
        frameon=False,
        fontsize=7,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
    )
    ax.grid(alpha=0.25)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    plt.tight_layout(rect=(0.0, 0.0, 0.8, 1.0))
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_oof_confusion_matrices(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    path: Path,
) -> None:
    if not predictions or len(np.unique(y_true)) < 2:
        return
    n = len(predictions)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), squeeze=False)
    for ax, (name, proba) in zip(axes.ravel(), predictions.items(), strict=True):
        thr = exp.threshold_for_objective(y_true, proba, objective="f1")
        y_pred = (proba >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1"])
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(f"{_short_label(name, 32)}\nF1-opt. thr={thr:.3f}")
    for j in range(len(predictions), nrows * ncols):
        axes.ravel()[j].set_visible(False)
    fig.suptitle("OOF confusion matrices (F1-optimal threshold)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def write_shap_summary_figures(
    saved_models: dict[str, list[Any]],
    matrices: list[tuple[pd.DataFrame, pd.DataFrame]],
    sample_size: int,
    analysis_dir: Path,
    max_models: int,
) -> None:
    try:
        import shap
    except Exception:
        logger.warning("shap not available; skipping SHAP summary plots")
        return

    supported = {"xgboost", "catboost", "lightgbm", "extra_trees", "random_forest"}
    rng = np.random.default_rng(42)
    n_written = 0
    for model_name, fold_models in saved_models.items():
        if model_name not in supported or n_written >= max_models:
            continue
        model = fold_models[0] if fold_models else None
        if model is None or isinstance(model, tuple):
            continue
        _, x_val = matrices[0]
        if len(x_val) == 0:
            continue
        sample_idx = rng.choice(len(x_val), size=min(sample_size, len(x_val)), replace=False)
        x_sample = x_val.iloc[sample_idx]
        try:
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(x_sample)
        except Exception as exc:
            logger.warning("SHAP summary skipped for %s: %s", model_name, exc)
            continue
        if isinstance(values, list):
            values = values[-1]
        max_display = min(25, x_sample.shape[1])
        shap.summary_plot(
            values,
            x_sample,
            show=False,
            max_display=max_display,
        )
        fig = plt.gcf()
        fig.suptitle(f"SHAP summary (fold 1 sample): {model_name}", y=1.02)
        fig.tight_layout()
        out = analysis_dir / f"shap_summary_{model_name}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        n_written += 1


def main() -> None:
    args = tyro.cli(ExplainabilityArgs)
    configure_logging(args.log_level)
    args.analysis_dir.mkdir(parents=True, exist_ok=True)

    oof_df, n_splits = load_run_metadata(args.output_dir, args.folds)
    if args.mode == "scan":
        logger.info(
            "Explainability scan mode: skipping fold rebuild, permutation, SHAP, and native importances"
        )
        weights, ensemble_perf = analyze_ensemble(args.output_dir / "oof_predictions.csv")

        y_true = oof_df["true_label"].astype(int).to_numpy()
        diag_cols = select_oof_diagnostic_columns(oof_df, ensemble_perf, args.max_diagnostic_cols)
        oof_preds = {c: oof_df[c].astype(float).to_numpy() for c in diag_cols if c in oof_df.columns}
        plot_oof_precision_recall(y_true, oof_preds, args.analysis_dir / "oof_precision_recall.pdf")

        weights.to_csv(args.analysis_dir / "stacked_meta_weights.csv", index=False)
        ensemble_perf.to_csv(args.analysis_dir / "ensemble_oof_performance.csv", index=False)

        if not weights.empty:
            plot_weights = weights.sort_values("standardized_meta_weight").copy()
            plt.figure(figsize=(8, max(3, 0.35 * len(plot_weights))))
            plt.barh(plot_weights["base_model"], plot_weights["standardized_meta_weight"])
            plt.axvline(0, color="black", linewidth=0.8)
            plt.title("Stacked Ensemble Meta-Model Weights")
            plt.xlabel("standardized logistic coefficient")
            plt.tight_layout()
            plt.savefig(args.analysis_dir / "stacked_meta_weights.pdf", bbox_inches="tight")
            plt.close()

        logger.info("Explainability scan artifacts written to %s", args.analysis_dir)
        return

    checkpoint_df, event_dfs = exp.load_data()
    folds = exp.build_fold_artifacts(
        checkpoint_df.reset_index(drop=True), event_dfs, n_splits=n_splits
    )
    first_full = pd.concat([folds[0].train_features, folds[0].val_features], axis=0)
    feature_sets = folds[0].builder.get_feature_sets(exp.modeling_columns(first_full))
    all_cols = feature_sets["all_features"]
    matrices = encoded_fold_matrices(folds, all_cols)

    models_path = args.model_dir / "fold_models.joblib"
    saved_models = joblib.load(models_path)

    native = compute_native_importances(saved_models, matrices)
    permutation = compute_permutation_importances(
        saved_models,
        folds,
        matrices,
        all_cols,
        top_n=args.top_n,
        repeats=args.permutation_repeats,
    )
    shap_importance = compute_shap_importances(saved_models, matrices, args.shap_sample)
    weights, ensemble_perf = analyze_ensemble(args.output_dir / "oof_predictions.csv")

    y_true = oof_df["true_label"].astype(int).to_numpy()
    diag_cols = select_oof_diagnostic_columns(oof_df, ensemble_perf, args.max_diagnostic_cols)
    oof_preds = {c: oof_df[c].astype(float).to_numpy() for c in diag_cols if c in oof_df.columns}
    plot_oof_calibration(y_true, oof_preds, args.analysis_dir / "oof_calibration.pdf")
    plot_oof_precision_recall(y_true, oof_preds, args.analysis_dir / "oof_precision_recall.pdf")
    plot_oof_confusion_matrices(y_true, oof_preds, args.analysis_dir / "oof_confusion_matrices.pdf")

    native.to_csv(args.analysis_dir / "native_feature_importance_by_model.csv", index=False)
    permutation.to_csv(args.analysis_dir / "permutation_importance_by_model.csv", index=False)
    shap_importance.to_csv(args.analysis_dir / "shap_importance_by_model.csv", index=False)
    weights.to_csv(args.analysis_dir / "stacked_meta_weights.csv", index=False)
    ensemble_perf.to_csv(args.analysis_dir / "ensemble_oof_performance.csv", index=False)

    for model_name in permutation["model"].drop_duplicates().tolist():
        subset = permutation[permutation["model"].eq(model_name)].sort_values(
            "pr_auc_drop_mean", ascending=False
        )
        write_bar_plot(
            subset,
            "pr_auc_drop_mean",
            f"Permutation Importance: {model_name}",
            args.analysis_dir / f"permutation_importance_{model_name}.pdf",
            args.top_n,
        )

    if not shap_importance.empty:
        for model_name in shap_importance["model"].drop_duplicates().tolist():
            subset = shap_importance[shap_importance["model"].eq(model_name)].sort_values(
                "mean_abs_shap", ascending=False
            )
            write_bar_plot(
                subset,
                "mean_abs_shap",
                f"SHAP mean |value|: {model_name}",
                args.analysis_dir / f"shap_bar_{model_name}.pdf",
                args.top_n,
            )

    write_shap_summary_figures(
        saved_models,
        matrices,
        args.shap_sample,
        args.analysis_dir,
        args.max_shap_summary_plots,
    )

    if not weights.empty:
        plot_weights = weights.sort_values("standardized_meta_weight").copy()
        plt.figure(figsize=(8, max(3, 0.35 * len(plot_weights))))
        plt.barh(plot_weights["base_model"], plot_weights["standardized_meta_weight"])
        plt.axvline(0, color="black", linewidth=0.8)
        plt.title("Stacked Ensemble Meta-Model Weights")
        plt.xlabel("standardized logistic coefficient")
        plt.tight_layout()
        plt.savefig(args.analysis_dir / "stacked_meta_weights.pdf", bbox_inches="tight")
        plt.close()

    logger.info("Explainability artifacts written to %s", args.analysis_dir)


if __name__ == "__main__":
    main()
