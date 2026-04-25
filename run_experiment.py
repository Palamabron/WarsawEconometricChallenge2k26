"""Run the WEC2026 modeling experiment and generate the LaTeX report.

Examples:
    python run_experiment.py --preset smoke
    python run_experiment.py --preset practical --hpo-trials 20 --include-autogluon
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from src.temporal_features import TARGET_COL, TemporalFeatureBuilder, modeling_columns

warnings.filterwarnings("ignore", category=FutureWarning)


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiments" / "wec_modeling_upgrade"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models" / "wec_modeling_upgrade"
REPORT_PATH = PROJECT_ROOT / "reports" / "wec2026_modeling_report.tex"
CACHE_DIR = PROJECT_ROOT / ".cache" / "ml"

os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("HF_HOME", str(CACHE_DIR / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(CACHE_DIR / "torch"))
os.environ.setdefault("TABPFN_CACHE_DIR", str(CACHE_DIR / "tabpfn"))


@dataclass
class FoldArtifacts:
    train_features: pd.DataFrame
    val_features: pd.DataFrame
    train_idx: np.ndarray
    val_idx: np.ndarray
    builder: TemporalFeatureBuilder


def detect_cuda() -> bool:
    """Best-effort CUDA detection without importing heavy frameworks."""
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0 and "GPU" in result.stdout
    except Exception:
        return False


def load_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Load all raw CSV files."""
    checkpoint = pd.read_csv(DATA_DIR / "players_quarters_final.csv", parse_dates=["date"])
    event_dfs = {
        "pass": pd.read_csv(DATA_DIR / "player_appearance_pass.csv"),
        "run": pd.read_csv(DATA_DIR / "player_appearance_run.csv"),
        "shot": pd.read_csv(DATA_DIR / "player_appearance_shot_limited.csv"),
        "pressure": pd.read_csv(DATA_DIR / "player_appearance_behaviour_under_pressure.csv"),
    }
    return checkpoint, event_dfs


def make_cv(y: pd.Series, groups: pd.Series, n_splits: int = 5) -> StratifiedGroupKFold:
    if groups.nunique() < n_splits:
        n_splits = int(groups.nunique())
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)


def build_fold_artifacts(
    checkpoint_df: pd.DataFrame,
    event_dfs: dict[str, pd.DataFrame],
    n_splits: int,
) -> list[FoldArtifacts]:
    """Build feature matrices inside each CV fold."""
    y = checkpoint_df[TARGET_COL].astype(int)
    groups = checkpoint_df["fixture_id"]
    cv = make_cv(y, groups, n_splits=n_splits)
    folds: list[FoldArtifacts] = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(checkpoint_df, y, groups), start=1):
        print(f"\nBuilding fold-safe features for fold {fold}...")
        train_checkpoint = checkpoint_df.iloc[train_idx].copy()
        val_checkpoint = checkpoint_df.iloc[val_idx].copy()
        builder = TemporalFeatureBuilder().fit(train_checkpoint, event_dfs)
        train_features = builder.transform(train_checkpoint, event_dfs)
        val_features = builder.transform(val_checkpoint, event_dfs)
        folds.append(
            FoldArtifacts(
                train_features=train_features,
                val_features=val_features,
                train_idx=train_idx,
                val_idx=val_idx,
                builder=builder,
            )
        )
    return folds


def feature_matrix(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, SimpleImputer]:
    """Encode categorical columns and impute using training data only."""
    x_train = train_df[feature_cols].copy()
    x_val = val_df[feature_cols].copy()

    cat_cols = x_train.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    num_cols = [c for c in feature_cols if c not in cat_cols]

    for col in num_cols:
        x_train[col] = pd.to_numeric(x_train[col], errors="coerce")
        x_val[col] = pd.to_numeric(x_val[col], errors="coerce")

    if cat_cols:
        train_cat = pd.get_dummies(x_train[cat_cols].fillna("missing").astype(str), prefix=cat_cols)
        val_cat = pd.get_dummies(x_val[cat_cols].fillna("missing").astype(str), prefix=cat_cols)
        val_cat = val_cat.reindex(columns=train_cat.columns, fill_value=0)
    else:
        train_cat = pd.DataFrame(index=x_train.index)
        val_cat = pd.DataFrame(index=x_val.index)

    train_num = x_train[num_cols]
    val_num = x_val[num_cols]
    imputer = SimpleImputer(strategy="median")
    train_num = pd.DataFrame(
        imputer.fit_transform(train_num), columns=num_cols, index=x_train.index
    )
    val_num = pd.DataFrame(imputer.transform(val_num), columns=num_cols, index=x_val.index)

    train_encoded = pd.concat([train_num, train_cat], axis=1).astype("float32")
    val_encoded = pd.concat([val_num, val_cat], axis=1).astype("float32")
    return train_encoded, val_encoded, imputer


def threshold_for_balanced_accuracy(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    del precision, recall
    if len(thresholds) == 0:
        return 0.5
    best_threshold = 0.5
    best_score = -math.inf
    for threshold in thresholds:
        pred = (y_proba >= threshold).astype(int)
        score = balanced_accuracy_score(y_true, pred)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def evaluate_predictions(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model: str,
    feature_set: str,
    threshold: float | None = None,
    y_pred: np.ndarray | None = None,
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    if y_pred is None:
        if threshold is None:
            threshold = 0.5
        y_pred = (y_proba >= threshold).astype(int)
    elif threshold is None:
        threshold = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "model": model,
        "feature_set": feature_set,
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else np.nan,
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, np.clip(y_proba, 1e-6, 1 - 1e-6))),
        "log_loss": float(log_loss(y_true, np.clip(y_proba, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def import_optional(name: str):
    try:
        return __import__(name)
    except Exception:
        return None


def tune_xgboost(x_train, y_train, x_val, y_val, use_gpu: bool, n_trials: int) -> dict[str, Any]:
    optuna = import_optional("optuna")
    xgboost = import_optional("xgboost")
    if optuna is None or xgboost is None or n_trials <= 0:
        return {}

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 250, 1200),
            "max_depth": trial.suggest_int("max_depth", 2, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 12.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 3.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 12.0, log=True),
            "scale_pos_weight": (len(y_train) - y_train.sum()) / max(y_train.sum(), 1),
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "device": "cuda" if use_gpu else "cpu",
            "random_state": 42,
            "n_jobs": -1,
        }
        model = xgboost.XGBClassifier(**params)
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        pred = model.predict_proba(x_val)[:, 1]
        return average_precision_score(y_val, pred)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def fit_predict_model(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    use_gpu: bool,
    hpo_params: dict[str, Any] | None = None,
) -> tuple[np.ndarray, Any]:
    """Fit one model and return validation probabilities."""
    hpo_params = hpo_params or {}

    if model_name == "logistic":
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            C=0.3,
            solver="lbfgs",
            random_state=42,
        )
        model.fit(x_train_scaled, y_train)
        return model.predict_proba(x_val_scaled)[:, 1], (scaler, model)

    if model_name == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=600,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        return model.predict_proba(x_val)[:, 1], model

    if model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=700,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        return model.predict_proba(x_val)[:, 1], model

    if model_name == "xgboost":
        xgboost = import_optional("xgboost")
        if xgboost is None:
            raise ImportError("xgboost is not installed")
        params = {
            "n_estimators": 700,
            "max_depth": 4,
            "learning_rate": 0.035,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "min_child_weight": 4.0,
            "reg_alpha": 0.05,
            "reg_lambda": 4.0,
            "scale_pos_weight": (len(y_train) - y_train.sum()) / max(y_train.sum(), 1),
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "device": "cuda" if use_gpu else "cpu",
            "random_state": 42,
            "n_jobs": -1,
        }
        params.update(hpo_params)
        model = xgboost.XGBClassifier(**params)
        try:
            model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        except Exception as exc:
            if use_gpu:
                print(f"XGBoost GPU failed ({exc}); retrying on CPU.")
                params["device"] = "cpu"
                model = xgboost.XGBClassifier(**params)
                model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
            else:
                raise
        return model.predict_proba(x_val)[:, 1], model

    if model_name == "catboost":
        try:
            from catboost import CatBoostClassifier
        except Exception as exc:
            raise ImportError("catboost is not installed") from exc
        params = {
            "iterations": 900,
            "depth": 5,
            "learning_rate": 0.035,
            "l2_leaf_reg": 8.0,
            "loss_function": "Logloss",
            "eval_metric": "PRAUC",
            "auto_class_weights": "Balanced",
            "random_seed": 42,
            "verbose": False,
            "early_stopping_rounds": 80,
            "task_type": "GPU" if use_gpu else "CPU",
        }
        model = CatBoostClassifier(**params)
        try:
            model.fit(x_train, y_train, eval_set=(x_val, y_val), verbose=False)
        except Exception as exc:
            if use_gpu:
                print(f"CatBoost GPU failed ({exc}); retrying on CPU.")
                params["task_type"] = "CPU"
                model = CatBoostClassifier(**params)
                model.fit(x_train, y_train, eval_set=(x_val, y_val), verbose=False)
            else:
                raise
        return model.predict_proba(x_val)[:, 1], model

    if model_name == "lightgbm":
        lightgbm = import_optional("lightgbm")
        if lightgbm is None:
            raise ImportError("lightgbm is not installed")
        params = {
            "n_estimators": 800,
            "learning_rate": 0.035,
            "num_leaves": 15,
            "max_depth": 5,
            "min_child_samples": 15,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.05,
            "reg_lambda": 5.0,
            "class_weight": "balanced",
            "objective": "binary",
            "random_state": 42,
            "n_jobs": -1,
            "device_type": "gpu" if use_gpu else "cpu",
        }
        model = lightgbm.LGBMClassifier(**params)
        try:
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_val, y_val)],
                eval_metric="average_precision",
            )
        except Exception as exc:
            if use_gpu:
                print(f"LightGBM GPU failed ({exc}); retrying on CPU.")
                params["device_type"] = "cpu"
                model = lightgbm.LGBMClassifier(**params)
                model.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric="average_precision")
            else:
                raise
        return model.predict_proba(x_val)[:, 1], model

    if model_name == "tabpfn":
        try:
            from tabpfn import TabPFNClassifier
        except Exception as exc:
            raise ImportError("tabpfn is not installed") from exc
        max_features = min(500, x_train.shape[1])
        selected = x_train.var().sort_values(ascending=False).head(max_features).index.tolist()
        model = TabPFNClassifier(device="cuda" if use_gpu else "cpu")
        model.fit(x_train[selected], y_train)
        return model.predict_proba(x_val[selected])[:, 1], (selected, model)

    raise ValueError(f"Unknown model: {model_name}")


def predict_model_proba(model: Any, x: pd.DataFrame) -> np.ndarray:
    """Predict probabilities from a model artifact returned by fit_predict_model."""
    if isinstance(model, tuple) and hasattr(model[0], "transform"):
        scaler, estimator = model
        return estimator.predict_proba(scaler.transform(x))[:, 1]
    if isinstance(model, tuple) and isinstance(model[0], list):
        selected, estimator = model
        return estimator.predict_proba(x[selected])[:, 1]
    return model.predict_proba(x)[:, 1]


def run_oof_for_model(
    folds: list[FoldArtifacts],
    feature_set_name: str,
    feature_cols: list[str],
    model_name: str,
    y_all: pd.Series,
    use_gpu: bool,
    hpo_trials: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[Any]]:
    """Run a model across folds and collect OOF probabilities."""
    oof = np.zeros(len(y_all), dtype=float)
    oof_labels = np.zeros(len(y_all), dtype=int)
    fold_metrics: list[dict[str, Any]] = []
    models = []
    for fold_num, fold in enumerate(folds, start=1):
        x_train, x_val, _ = feature_matrix(fold.train_features, fold.val_features, feature_cols)
        y_train = fold.train_features[TARGET_COL].astype(int)
        y_val = fold.val_features[TARGET_COL].astype(int)
        hpo_params = {}
        if model_name == "xgboost" and hpo_trials > 0:
            hpo_params = tune_xgboost(x_train, y_train, x_val, y_val, use_gpu, hpo_trials)
        pred, model = fit_predict_model(
            model_name, x_train, y_train, x_val, y_val, use_gpu, hpo_params=hpo_params
        )
        train_pred = predict_model_proba(model, x_train)
        threshold = threshold_for_balanced_accuracy(y_train.to_numpy(), train_pred)
        pred_labels = (pred >= threshold).astype(int)
        oof[fold.val_idx] = pred
        oof_labels[fold.val_idx] = pred_labels
        fold_result = evaluate_predictions(
            y_val.to_numpy(),
            pred,
            model=model_name,
            feature_set=feature_set_name,
            threshold=threshold,
            y_pred=pred_labels,
        )
        fold_result["fold"] = fold_num
        fold_metrics.append(fold_result)
        models.append(model)
        print(
            f"{model_name}/{feature_set_name} fold {fold_num}: "
            f"PR-AUC={fold_result['pr_auc']:.4f}, ROC-AUC={fold_result['roc_auc']:.4f}, "
            f"BalAcc={fold_result['balanced_accuracy']:.4f}"
        )
    return oof, oof_labels, fold_metrics, models


def run_stacking(
    base_oof: dict[str, np.ndarray],
    y: pd.Series,
    groups: pd.Series,
    n_splits: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Train a lightweight meta-learner with grouped OOF predictions."""
    if len(base_oof) < 2:
        proba = next(iter(base_oof.values()))
        return proba, (proba >= 0.5).astype(int)
    x_meta = pd.DataFrame(base_oof)
    oof = np.zeros(len(y), dtype=float)
    labels = np.zeros(len(y), dtype=int)
    cv = make_cv(y, groups, n_splits=n_splits)
    for train_idx, val_idx in cv.split(x_meta, y, groups):
        x_train, x_val = x_meta.iloc[train_idx], x_meta.iloc[val_idx]
        y_train = y.iloc[train_idx]
        meta = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        try:
            calibrated = CalibratedClassifierCV(meta, cv=3, method="sigmoid")
            calibrated.fit(x_train, y_train)
            train_pred = calibrated.predict_proba(x_train)[:, 1]
            val_pred = calibrated.predict_proba(x_val)[:, 1]
        except (ValueError, NotFittedError):
            meta.fit(x_train, y_train)
            train_pred = meta.predict_proba(x_train)[:, 1]
            val_pred = meta.predict_proba(x_val)[:, 1]
        threshold = threshold_for_balanced_accuracy(y_train.to_numpy(), train_pred)
        oof[val_idx] = val_pred
        labels[val_idx] = (val_pred >= threshold).astype(int)
    return oof, labels


def permutation_importance_oof(
    folds: list[FoldArtifacts],
    feature_cols: list[str],
    y_all: pd.Series,
    baseline_pred: np.ndarray,
    model_name: str,
    use_gpu: bool,
    max_features: int = 40,
) -> pd.DataFrame:
    """Compute validation-fold permutation importance for a manageable feature subset."""
    base_pr = average_precision_score(y_all, baseline_pred)
    candidates = []
    all_features = pd.concat([fold.train_features[feature_cols] for fold in folds], axis=0)
    numeric = all_features.select_dtypes(include=[np.number])
    if not numeric.empty:
        corrs = numeric.corrwith(
            pd.concat([fold.train_features[TARGET_COL] for fold in folds], axis=0).astype(int)
        )
        candidates = corrs.abs().sort_values(ascending=False).head(max_features).index.tolist()
    if not candidates:
        candidates = feature_cols[:max_features]

    rng = np.random.default_rng(42)
    rows = []
    for col in candidates:
        perm_pred = np.zeros(len(y_all), dtype=float)
        for fold in folds:
            cols = [c for c in feature_cols if c in fold.train_features.columns]
            x_train, x_val, _ = feature_matrix(fold.train_features, fold.val_features, cols)
            if col not in x_val.columns:
                # Categorical dummy expansion means raw categoricals may not appear directly.
                continue
            shuffled = x_val.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            y_train = fold.train_features[TARGET_COL].astype(int)
            y_val = fold.val_features[TARGET_COL].astype(int)
            _, model = fit_predict_model(model_name, x_train, y_train, x_val, y_val, use_gpu)
            if isinstance(model, tuple) and hasattr(model[0], "transform"):
                scaler, estimator = model
                pred = estimator.predict_proba(scaler.transform(shuffled))[:, 1]
            elif isinstance(model, tuple) and isinstance(model[0], list):
                selected, estimator = model
                pred = estimator.predict_proba(shuffled[selected])[:, 1]
            else:
                pred = model.predict_proba(shuffled)[:, 1]
            perm_pred[fold.val_idx] = pred
        if perm_pred.sum() > 0:
            rows.append(
                {
                    "feature": col,
                    "baseline_pr_auc": base_pr,
                    "permuted_pr_auc": average_precision_score(y_all, perm_pred),
                    "importance_pr_auc_drop": base_pr - average_precision_score(y_all, perm_pred),
                }
            )
    return pd.DataFrame(rows).sort_values("importance_pr_auc_drop", ascending=False)


def run_autogluon_smoke(
    train_features: pd.DataFrame,
    val_features: pd.DataFrame,
    feature_cols: list[str],
    time_limit: int,
) -> dict[str, Any] | None:
    """Optional AutoGluon benchmark on one holdout fold."""
    try:
        from autogluon.tabular import TabularPredictor
    except Exception:
        return None

    train = train_features[feature_cols + [TARGET_COL]].copy()
    val = val_features[feature_cols + [TARGET_COL]].copy()
    out_dir = MODEL_DIR / "autogluon_holdout"
    predictor = TabularPredictor(
        label=TARGET_COL,
        problem_type="binary",
        eval_metric="average_precision",
        path=str(out_dir),
    )
    predictor.fit(
        train,
        presets="medium_quality",
        time_limit=time_limit,
        verbosity=2,
    )
    pred = predictor.predict_proba(val)[1].to_numpy()
    result = evaluate_predictions(
        val[TARGET_COL].to_numpy(),
        pred,
        model="autogluon_holdout",
        feature_set="all_features",
        threshold=0.5,
    )
    leaderboard = predictor.leaderboard(val, silent=True)
    leaderboard.to_csv(OUTPUT_DIR / "autogluon_leaderboard.csv", index=False)
    return result


def write_report(
    metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    importances: pd.DataFrame,
    data_summary: dict[str, Any],
    skipped_models: list[str],
) -> None:
    """Write the final LaTeX report."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv_metrics = metrics[~metrics["model"].str.contains("autogluon", case=False, na=False)].copy()
    best = cv_metrics.sort_values(["pr_auc", "roc_auc", "balanced_accuracy"], ascending=False).iloc[0]
    autogluon = metrics[metrics["model"].str.contains("autogluon", case=False, na=False)].copy()
    ablations = metrics[metrics["model"].eq("xgboost")].copy()

    def latex_escape(value: Any) -> str:
        text = str(value)
        replacements = {
            "\\": r"\textbackslash{}",
            "_": r"\_",
            "%": r"\%",
            "&": r"\&",
            "#": r"\#",
            "{": r"\{",
            "}": r"\}",
            "<": r"\textless{}",
            ">": r"\textgreater{}",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def table_rows(frame: pd.DataFrame, cols: list[str]) -> str:
        lines = []
        for _, row in frame.iterrows():
            vals = []
            for col in cols:
                val = row[col]
                vals.append(f"{val:.3f}" if isinstance(val, float | np.floating) else latex_escape(val))
            lines.append(" & ".join(vals) + r" \\")
        return "\n".join(lines)

    stable_importance = importances[importances["importance_pr_auc_drop"] > 0].copy()
    top_importance = stable_importance.head(12) if not stable_importance.empty else pd.DataFrame()
    importance_rows = (
        table_rows(top_importance, ["feature", "importance_pr_auc_drop"])
        if not top_importance.empty
        else r"Permutation importance did not show stable positive PR-AUC drops in this run. \\"
    )
    skipped_text = "; ".join(str(item).splitlines()[0] for item in skipped_models) if skipped_models else "None"
    ag_text = (
        f"AutoGluon holdout benchmark reached ROC-AUC {autogluon.iloc[0]['roc_auc']:.3f}, "
        f"PR-AUC {autogluon.iloc[0]['pr_auc']:.3f}, and balanced accuracy "
        f"{autogluon.iloc[0]['balanced_accuracy']:.3f} at threshold 0.5 on its validation fold."
        if not autogluon.empty
        else "AutoGluon was not available in this run."
    )
    autogluon_rows = (
        table_rows(
            autogluon.sort_values("pr_auc", ascending=False),
            ["model", "feature_set", "roc_auc", "pr_auc", "balanced_accuracy", "f1", "brier"],
        )
        if not autogluon.empty
        else r"No AutoGluon holdout result was recorded. \\"
    )

    report = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{hyperref}}
\usepackage{{amsmath}}
\title{{WEC2026 Goal-Scoring Prediction: Data, Preprocessing, and Modeling Report}}
\author{{Automated experiment report}}
\date{{\today}}

\begin{{document}}
\maketitle

\section{{Executive Summary}}
The task is a rare-event binary classification problem: predicting whether a player scores after a match checkpoint.
The modeling table contains {data_summary['n_rows']:,} checkpoint observations, {data_summary['n_positives']:,} positives ({data_summary['positive_rate']:.2%}), {data_summary['n_fixtures']} fixtures, and {data_summary['n_appearances']} player appearances.
The best grouped cross-validated model in this run was \textbf{{{latex_escape(best['model'])}}} on \textbf{{{latex_escape(best['feature_set'])}}} features with ROC-AUC {best['roc_auc']:.3f}, PR-AUC {best['pr_auc']:.3f}, and balanced accuracy {best['balanced_accuracy']:.3f}.
{latex_escape(ag_text)}

\section{{Data Audit}}
The checkpoint file provides player state every 15 minutes, including position, home status, formation, substitution times, physical load, shots, and the target \texttt{{scored\_after}}.
Four event files were joined using \texttt{{player\_appearance\_id}} and absolute match time: passes, high-speed runs/sprints, shots, and actions under defensive pressure.
The target is highly imbalanced; therefore ROC-AUC, PR-AUC, balanced accuracy, Brier score, and log loss are more informative than raw accuracy.

\section{{Preprocessing}}
All event-derived features were computed with temporal filters:
\[
  \texttt{{minute\_in}} \leq \texttt{{event\_abs\_minute}} \leq \texttt{{checkpoint\_abs\_minute}}.
\]
The implementation converts period-relative minutes to absolute match minutes so second-half and extra-time checkpoints are handled correctly.
Reference quantities such as expected-threat zone values, pressure turnover baselines, and positional baselines were fitted inside each cross-validation training fold before being applied to validation checkpoints.
Categorical variables were one-hot encoded from training folds only, and numerical missing values were median-imputed from training folds only.

\section{{Feature Families}}
The all-feature model uses signals from every available file:
physical intensity and shots from \texttt{{players\_quarters\_final.csv}}, run event intensity from \texttt{{player\_appearance\_run.csv}}, pass volume and xT proxies from \texttt{{player\_appearance\_pass.csv}}, shot context from \texttt{{player\_appearance\_shot\_limited.csv}}, and pressure resistance plus pressing-applied metrics from \texttt{{player\_appearance\_behaviour\_under\_pressure.csv}}.
Recent-vs-cumulative ratios were included to test short-term intensity surges.

\section{{Model Selection and HPO}}
The benchmark used fixture-grouped stratified cross-validation to prevent match-level leakage.
Models attempted in this practical run included logistic regression, ExtraTrees, RandomForest, XGBoost, CatBoost, LightGBM, TabPFN, stacking, and optional AutoGluon.
TabPFN probabilities were included as a base learner in the stacked ensemble whenever the package was available, rather than being evaluated only as a standalone benchmark.
Skipped or unavailable models: {latex_escape(skipped_text)}.
XGBoost used Optuna HPO when the requested trial budget was nonzero; other model parameters were conservative regularized defaults appropriate for the small sample size.

\section{{Grouped Cross-Validated Results}}
\begin{{longtable}}{{llrrrrr}}
\toprule
Model & Feature set & ROC-AUC & PR-AUC & Bal. Acc. & F1 & Brier \\
\midrule
{table_rows(cv_metrics.sort_values('pr_auc', ascending=False).head(20), ['model', 'feature_set', 'roc_auc', 'pr_auc', 'balanced_accuracy', 'f1', 'brier'])}
\bottomrule
\end{{longtable}}

\section{{AutoGluon Holdout Result}}
\begin{{longtable}}{{llrrrrr}}
\toprule
Model & Feature set & ROC-AUC & PR-AUC & Bal. Acc. & F1 & Brier \\
\midrule
{autogluon_rows}
\bottomrule
\end{{longtable}}

\section{{Ablation Results}}
\begin{{longtable}}{{lrrrr}}
\toprule
Feature set & ROC-AUC & PR-AUC & Bal. Acc. & F1 \\
\midrule
{table_rows(ablations.sort_values('pr_auc', ascending=False), ['feature_set', 'roc_auc', 'pr_auc', 'balanced_accuracy', 'f1'])}
\bottomrule
\end{{longtable}}

\section{{Feature Importance}}
\begin{{longtable}}{{lr}}
\toprule
Feature & PR-AUC drop after permutation \\
\midrule
{importance_rows}
\bottomrule
\end{{longtable}}

\section{{Research Questions}}
\paragraph{{RQ1.}}
Player behaviour during a match is predictive but noisy because goals are rare. In this run the best grouped-CV model reached ROC-AUC {best['roc_auc']:.3f}, PR-AUC {best['pr_auc']:.3f}, and balanced accuracy {best['balanced_accuracy']:.3f}.
\paragraph{{RQ2.}}
The strongest determinants are identified by a combination of ablation deltas and permutation importance. The leading stable signals in this run were: {', '.join(latex_escape(x) for x in top_importance['feature'].head(6).astype(str)) if not top_importance.empty else 'not stable under permutation in this small sample'}.
\paragraph{{RQ3.}}
The sprint-and-shot-only ablation should be compared against the all-feature model. Its result in this run was {metrics[(metrics['feature_set']=='sprints_shots')]['pr_auc'].max() if (metrics['feature_set']=='sprints_shots').any() else float('nan'):.3f} PR-AUC, so it is not assumed sufficient unless it matches the all-feature score.
\paragraph{{RQ4.}}
Passing and pressure data are useful when their ablation improves over the base checkpoint model. The report tables show whether \texttt{{base\_plus\_passing}} and \texttt{{base\_plus\_pressure}} improve ROC-AUC/PR-AUC/balanced accuracy.
\paragraph{{RQ5.}}
The \texttt{{recent\_only}} and \texttt{{cumulative\_only}} ablations compare short-term and accumulated performance. The stronger family is the one with higher grouped-CV PR-AUC and ROC-AUC.
\paragraph{{RQ6.}}
Recent-vs-cumulative ratios and intensity-surge interactions directly test whether short-term intensity above a player's own baseline increases scoring probability.
\paragraph{{RQ7.}}
External factors were modeled through home status, period, checkpoint time, extra time, formation, position, and substitution context. Their influence is assessed by the \texttt{{external\_context}} ablation and permutation importance.

\section{{Limitations}}
Only 203 positive examples are available, so fold-level variance is expected. Some event locations are coarse three-zone labels, which limits xT precision. The stacked model is evaluated with grouped OOF meta-predictions, but it should still be validated on a hidden competition set before final claims.

\end{{document}}
"""
    REPORT_PATH.write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["smoke", "practical"], default="practical")
    parser.add_argument("--hpo-trials", type=int, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--include-autogluon", action="store_true")
    parser.add_argument("--autogluon-time-limit", type=int, default=900)
    parser.add_argument("--importance-model", choices=["xgboost", "extra_trees"], default="xgboost")
    args = parser.parse_args()

    start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    use_gpu = detect_cuda()
    hpo_trials = args.hpo_trials
    if hpo_trials is None:
        hpo_trials = 1 if args.preset == "smoke" else 12
    if args.preset == "smoke":
        args.folds = min(args.folds, 3)

    checkpoint_df, event_dfs = load_data()
    y = checkpoint_df[TARGET_COL].astype(int).reset_index(drop=True)
    data_summary = {
        "n_rows": int(len(checkpoint_df)),
        "n_positives": int(y.sum()),
        "positive_rate": float(y.mean()),
        "n_fixtures": int(checkpoint_df["fixture_id"].nunique()),
        "n_appearances": int(checkpoint_df["player_appearance_id"].nunique()),
    }

    folds = build_fold_artifacts(checkpoint_df.reset_index(drop=True), event_dfs, n_splits=args.folds)
    first_full = pd.concat([folds[0].train_features, folds[0].val_features], axis=0)
    feature_sets = folds[0].builder.get_feature_sets(modeling_columns(first_full))

    skipped_models: list[str] = []
    metrics_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    oof_columns: dict[str, np.ndarray] = {"true_label": y.to_numpy()}

    # Ablations: use XGBoost as the main nonlinear reference, falling back if unavailable.
    ablation_model = "xgboost" if import_optional("xgboost") is not None else "extra_trees"
    for feature_set_name in [
        "base_checkpoint",
        "sprints_shots",
        "base_plus_passing",
        "base_plus_pressure",
        "recent_only",
        "cumulative_only",
        "external_context",
        "all_features",
    ]:
        cols = feature_sets[feature_set_name]
        print(f"\nRunning ablation {feature_set_name} with {len(cols)} features...")
        try:
            oof, oof_labels, fold_metrics, _ = run_oof_for_model(
                folds,
                feature_set_name,
                cols,
                ablation_model,
                y,
                use_gpu=use_gpu,
                hpo_trials=hpo_trials if feature_set_name == "all_features" else 0,
            )
        except Exception as exc:
            skipped_models.append(f"{ablation_model}/{feature_set_name}: {exc}")
            continue
        oof_columns[f"{ablation_model}_{feature_set_name}"] = oof
        metrics_rows.append(
            evaluate_predictions(y, oof, ablation_model, feature_set_name, y_pred=oof_labels)
        )
        fold_rows.extend(fold_metrics)

    # Main all-feature benchmark.
    base_models = ["logistic", "extra_trees", "random_forest", "xgboost", "catboost", "lightgbm", "tabpfn"]
    if args.preset == "smoke":
        base_models = ["logistic", "extra_trees", "xgboost", "catboost", "tabpfn"]
    all_cols = feature_sets["all_features"]
    all_oof_for_stack: dict[str, np.ndarray] = {}
    saved_models: dict[str, Any] = {}
    for model_name in base_models:
        print(f"\nRunning all-features model: {model_name}")
        try:
            oof, oof_labels, fold_metrics, models = run_oof_for_model(
                folds,
                "all_features",
                all_cols,
                model_name,
                y,
                use_gpu=use_gpu,
                hpo_trials=hpo_trials if model_name == "xgboost" else 0,
            )
        except Exception as exc:
            print(f"Skipping {model_name}: {exc}")
            skipped_models.append(f"{model_name}: {exc}")
            continue
        oof_columns[f"{model_name}_all_features"] = oof
        all_oof_for_stack[model_name] = oof
        metrics_rows.append(evaluate_predictions(y, oof, model_name, "all_features", y_pred=oof_labels))
        fold_rows.extend(fold_metrics)
        saved_models[model_name] = models

    if all_oof_for_stack:
        stacked, stacked_labels = run_stacking(
            all_oof_for_stack,
            y,
            checkpoint_df["fixture_id"].reset_index(drop=True),
            n_splits=args.folds,
        )
        oof_columns["stacked_all_features"] = stacked
        metrics_rows.append(
            evaluate_predictions(y, stacked, "stacked", "all_features", y_pred=stacked_labels)
        )

    if args.include_autogluon:
        print("\nRunning optional AutoGluon holdout benchmark...")
        ag_result = run_autogluon_smoke(
            folds[0].train_features,
            folds[0].val_features,
            all_cols,
            time_limit=args.autogluon_time_limit if args.preset != "smoke" else 120,
        )
        if ag_result is None:
            skipped_models.append("autogluon: package unavailable or failed to import")
        else:
            metrics_rows.append(ag_result)

    metrics = (
        pd.DataFrame(metrics_rows)
        .drop_duplicates(subset=["model", "feature_set"], keep="first")
        .sort_values(["pr_auc", "roc_auc"], ascending=False)
    )
    fold_metrics = pd.DataFrame(fold_rows)
    oof_predictions = pd.DataFrame(oof_columns)
    oof_predictions.insert(0, "checkpoint", checkpoint_df["checkpoint"].values)
    oof_predictions.insert(0, "fixture_id", checkpoint_df["fixture_id"].values)
    oof_predictions.insert(0, "player_appearance_id", checkpoint_df["player_appearance_id"].values)

    importance_model = args.importance_model if args.importance_model in all_oof_for_stack else "extra_trees"
    importance_key = f"{importance_model}_all_features"
    baseline_pred = oof_columns.get(
        importance_key,
        oof_columns.get("stacked_all_features", next(iter(all_oof_for_stack.values()), np.zeros(len(y)))),
    )
    try:
        importances = permutation_importance_oof(
            folds,
            all_cols,
            y,
            baseline_pred,
            model_name=importance_model,
            use_gpu=use_gpu,
            max_features=20 if args.preset == "smoke" else 40,
        )
    except Exception as exc:
        skipped_models.append(f"permutation_importance: {exc}")
        importances = pd.DataFrame(columns=["feature", "baseline_pr_auc", "permuted_pr_auc", "importance_pr_auc_drop"])

    metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    fold_metrics.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)
    oof_predictions.to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)
    importances.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    with (OUTPUT_DIR / "run_summary.json").open("w") as f:
        json.dump(
            {
                "data_summary": data_summary,
                "use_gpu": use_gpu,
                "preset": args.preset,
                "hpo_trials": hpo_trials,
                "skipped_models": skipped_models,
                "elapsed_seconds": time.time() - start,
            },
            f,
            indent=2,
        )
    joblib.dump(saved_models, MODEL_DIR / "fold_models.joblib")
    write_report(metrics, fold_metrics, importances, data_summary, skipped_models)

    print("\nExperiment complete.")
    print(metrics.head(10).to_string(index=False))
    print(f"\nArtifacts: {OUTPUT_DIR}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
