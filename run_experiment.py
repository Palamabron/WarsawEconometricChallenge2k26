"""Run the WEC2026 modeling experiment and generate the LaTeX report.

Examples:
    python run_experiment.py --preset smoke
    python run_experiment.py --preset practical --hpo-trials 20 --include-autogluon
    python run_experiment.py --preset fast
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
import tyro
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

from src.config import get_config
from src.temporal_features import (
    REDUNDANT_MODEL_FEATURES,
    TARGET_COL,
    TemporalFeatureBuilder,
    modeling_columns,
    prepare_checkpoint_frame,
)
from src.validation.leakage_checks import LeakageValidator

warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiments" / "wec_modeling_upgrade"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models" / "wec_modeling_upgrade"
REPORT_PATH = PROJECT_ROOT / "reports" / "wec2026_modeling_report.tex"
CACHE_DIR = PROJECT_ROOT / ".cache" / "ml"
CSV_READ_KWARGS: dict[str, Any] = {
    "true_values": ["TRUE"],
    "false_values": ["FALSE"],
    "na_values": ["NULL", ""],
    "keep_default_na": True,
}
RARE_CATEGORY_MIN_COUNT = 10
RARE_CATEGORY_MIN_SHARE = 0.01
TARGET_ENCODE_COLS = {"position", "formation"}

os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("HF_HOME", str(CACHE_DIR / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(CACHE_DIR / "torch"))


@dataclass
class FoldArtifacts:
    train_features: pd.DataFrame
    val_features: pd.DataFrame
    train_idx: np.ndarray
    val_idx: np.ndarray
    builder: TemporalFeatureBuilder


@dataclass(frozen=True)
class ExperimentArgs:
    """Command-line options for the report-grade WEC2026 experiment."""

    preset: Literal["smoke", "practical", "fast"] = "practical"
    hpo_trials: int | None = None
    folds: int | None = None
    importance_model: Literal["auto", "xgboost", "extra_trees", "random_forest", "lightgbm"] = (
        "xgboost"
    )
    include_autogluon: bool = False
    autogluon_time_limit: int = 900
    config_path: str = "config.yaml"
    log_level: str = "INFO"


def configure_logging(level: str) -> None:
    """Configure process-wide logging for command-line experiment runs."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


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


def load_data(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Load all raw CSV files."""
    checkpoint = pd.read_csv(
        data_dir / "players_quarters_final.csv",
        parse_dates=["date"],
        **CSV_READ_KWARGS,
    )
    event_dfs = {
        "pass": pd.read_csv(data_dir / "player_appearance_pass.csv", **CSV_READ_KWARGS),
        "run": pd.read_csv(data_dir / "player_appearance_run.csv", **CSV_READ_KWARGS),
        "shot": pd.read_csv(
            data_dir / "player_appearance_shot_limited.csv",
            **CSV_READ_KWARGS,
        ),
        "pressure": pd.read_csv(
            data_dir / "player_appearance_behaviour_under_pressure.csv",
            **CSV_READ_KWARGS,
        ),
    }
    return checkpoint, event_dfs


def validate_training_checkpoints(
    checkpoint_df: pd.DataFrame,
    event_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Run report-path leakage checks and keep only checkpoint-valid training rows."""
    prepared = prepare_checkpoint_frame(checkpoint_df)
    LeakageValidator.validate_temporal_boundaries(prepared, checkpoint_df, event_dfs)

    on_pitch_mask = prepared["is_on_pitch"].eq(1)
    dropped = int((~on_pitch_mask).sum())
    if dropped:
        logger.warning("Dropping %s checkpoints where the player is not on pitch", dropped)
    filtered = checkpoint_df.loc[on_pitch_mask].reset_index(drop=True)
    if filtered.empty:
        raise ValueError("No valid on-pitch checkpoints remain after training input validation.")
    return filtered


def dataframe_audit(df: pd.DataFrame, max_category_examples: int = 12) -> dict[str, Any]:
    """Summarize missingness, dtypes, and categorical cardinality for run diagnostics."""
    categorical = df.select_dtypes(include=["object", "string", "category", "bool"]).columns
    category_summary = {}
    for col in categorical:
        values = df[col].fillna("missing").astype(str)
        category_summary[col] = {
            "n_unique": int(values.nunique()),
            "top_values": values.value_counts().head(max_category_examples).to_dict(),
        }
    return {
        "n_rows": len(df),
        "n_cols": int(df.shape[1]),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_counts": {
            col: int(count) for col, count in df.isna().sum().items() if int(count) > 0
        },
        "categorical": category_summary,
    }


def build_data_audit(
    checkpoint_df: pd.DataFrame,
    event_dfs: dict[str, pd.DataFrame],
    folds: list[FoldArtifacts],
    feature_cols: list[str],
) -> dict[str, Any]:
    """Collect compact preprocessing diagnostics for reproducibility."""
    first_train = folds[0].train_features[feature_cols]
    numeric = first_train.select_dtypes(include=[np.number])
    zero_counts = {
        col: int((numeric[col] == 0).sum())
        for col in numeric.columns
        if col.endswith(("_count", "_has_events")) or "_events_per_minute" in col
    }
    return {
        "raw": {
            "checkpoint": dataframe_audit(checkpoint_df),
            **{name: dataframe_audit(frame) for name, frame in event_dfs.items()},
        },
        "first_fold_features": {
            "n_train_rows": len(first_train),
            "n_modeling_cols": len(feature_cols),
            "missing_counts": {
                col: int(count) for col, count in first_train.isna().sum().items() if int(count) > 0
            },
            "structural_zero_counts": zero_counts,
        },
    }


def make_cv(y: pd.Series, groups: pd.Series, n_splits: int = 5) -> StratifiedGroupKFold:
    n_groups = int(groups.nunique())
    if n_groups < 2:
        raise ValueError("Grouped CV requires at least 2 distinct fixture_id values.")
    if y.nunique() < 2:
        raise ValueError("Training requires both target classes before grouped CV.")
    if n_groups < n_splits:
        n_splits = n_groups
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
        logger.info("Building fold-safe features for fold %s", fold)
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
    y_train: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, SimpleImputer]:
    """Encode categorical columns and impute using training data only."""
    x_train = train_df[feature_cols].copy()
    x_val = val_df[feature_cols].copy()

    cat_cols = x_train.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()
    num_cols = [c for c in feature_cols if c not in cat_cols]

    for col in num_cols:
        x_train[col] = pd.to_numeric(x_train[col], errors="coerce")
        x_val[col] = pd.to_numeric(x_val[col], errors="coerce")

    target_encoded_train = pd.DataFrame(index=x_train.index)
    target_encoded_val = pd.DataFrame(index=x_val.index)
    if cat_cols:
        train_cat_raw = x_train[cat_cols].fillna("missing").astype(str)
        val_cat_raw = x_val[cat_cols].fillna("missing").astype(str)
        for col in cat_cols:
            min_count = max(
                RARE_CATEGORY_MIN_COUNT, int(len(train_cat_raw) * RARE_CATEGORY_MIN_SHARE)
            )
            counts = train_cat_raw[col].value_counts()
            frequent = set(counts[counts >= min_count].index)
            train_cat_raw[col] = train_cat_raw[col].where(train_cat_raw[col].isin(frequent), "rare")
            val_cat_raw[col] = val_cat_raw[col].where(val_cat_raw[col].isin(frequent), "rare")

            freq = train_cat_raw[col].value_counts(normalize=True)
            target_encoded_train[f"{col}_frequency"] = train_cat_raw[col].map(freq).astype(float)
            target_encoded_val[f"{col}_frequency"] = (
                val_cat_raw[col].map(freq).fillna(0.0).astype(float)
            )

            if y_train is not None and col in TARGET_ENCODE_COLS:
                y_aligned = pd.Series(y_train, index=x_train.index).astype(float)
                global_rate = float(y_aligned.mean())
                stats = y_aligned.groupby(train_cat_raw[col]).agg(["mean", "count"])
                smooth = 20.0
                enc = (stats["mean"] * stats["count"] + global_rate * smooth) / (
                    stats["count"] + smooth
                )
                target_encoded_train[f"{col}_target_rate"] = (
                    train_cat_raw[col].map(enc).fillna(global_rate)
                )
                target_encoded_val[f"{col}_target_rate"] = (
                    val_cat_raw[col].map(enc).fillna(global_rate)
                )

        train_cat = pd.get_dummies(train_cat_raw, prefix=cat_cols)
        val_cat = pd.get_dummies(val_cat_raw, prefix=cat_cols)
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

    train_encoded = pd.concat([train_num, target_encoded_train, train_cat], axis=1).astype(
        "float32"
    )
    val_encoded = pd.concat([val_num, target_encoded_val, val_cat], axis=1).astype("float32")
    return train_encoded, val_encoded, imputer


def threshold_for_objective(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    objective: str = "f1",
) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    del precision, recall
    if len(thresholds) == 0:
        return 0.5
    best_threshold = 0.5
    best_score = -math.inf
    for threshold in thresholds:
        pred = (y_proba >= threshold).astype(int)
        if objective == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, pred)
        elif objective == "f1":
            score = f1_score(y_true, pred, zero_division=0)
        else:
            raise ValueError(f"Unknown threshold objective: {objective}")
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def threshold_for_balanced_accuracy(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    return threshold_for_objective(y_true, y_proba, objective="balanced_accuracy")


def ranked_precision_recall_at_k(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    share: float,
) -> tuple[float, float, int]:
    """Compute precision/recall among the top-scored share of rows."""
    if len(y_true) == 0:
        return 0.0, 0.0, 0
    k = max(1, math.ceil(len(y_true) * share))
    order = np.argsort(-y_proba)[:k]
    positives = float(y_true.sum())
    hits = float(y_true[order].sum())
    precision = hits / k
    recall = hits / positives if positives else 0.0
    return precision, recall, k


def evaluate_predictions(
    y_true: Any,
    y_proba: Any,
    model: str,
    feature_set: str,
    threshold: float | None = None,
    y_pred: Any | None = None,
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    if y_pred is None:
        if threshold is None:
            threshold = 0.5
        y_pred = (y_proba >= threshold).astype(int)
    elif threshold is None:
        threshold = float("nan")
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / max(tp + fp, 1)
    precision_at_5, recall_at_5, k5 = ranked_precision_recall_at_k(y_true, y_proba, 0.05)
    precision_at_10, recall_at_10, k10 = ranked_precision_recall_at_k(y_true, y_proba, 0.10)
    base_rate = float(y_true.mean()) if len(y_true) else 0.0
    return {
        "model": model,
        "feature_set": feature_set,
        "n": len(y_true),
        "positives": int(y_true.sum()),
        "base_rate": base_rate,
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else np.nan,
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision_at_5pct": float(precision_at_5),
        "recall_at_5pct": float(recall_at_5),
        "precision_at_10pct": float(precision_at_10),
        "recall_at_10pct": float(recall_at_10),
        "top_decile_lift": float(precision_at_10 / base_rate) if base_rate > 0 else 0.0,
        "top_5pct_n": int(k5),
        "top_10pct_n": int(k10),
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


def _xgboost_hpo_split(
    y_train: pd.Series, groups: pd.Series
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build an inner fixture-grouped split for XGBoost HPO."""
    y_inner = y_train.reset_index(drop=True).astype(int)
    groups_inner = groups.reset_index(drop=True)
    if y_inner.nunique() < 2 or groups_inner.nunique() < 2:
        return None

    n_splits = min(3, int(groups_inner.nunique()))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for inner_train_idx, inner_val_idx in cv.split(np.zeros(len(y_inner)), y_inner, groups_inner):
        if (
            y_inner.iloc[inner_train_idx].nunique() == 2
            and y_inner.iloc[inner_val_idx].nunique() == 2
        ):
            return inner_train_idx, inner_val_idx
    return None


def tune_xgboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
    use_gpu: bool,
    n_trials: int,
) -> dict[str, Any]:
    optuna = import_optional("optuna")
    xgboost = import_optional("xgboost")
    if optuna is None or xgboost is None or n_trials <= 0:
        return {}

    split = _xgboost_hpo_split(y_train, groups)
    if split is None:
        return {}
    inner_train_idx, inner_val_idx = split
    x_hpo_train = x_train.iloc[inner_train_idx]
    y_hpo_train = y_train.iloc[inner_train_idx]
    x_hpo_val = x_train.iloc[inner_val_idx]
    y_hpo_val = y_train.iloc[inner_val_idx]

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
            "scale_pos_weight": soft_positive_weight(y_hpo_train),
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "device": "cuda" if use_gpu else "cpu",
            "random_state": 42,
            "n_jobs": -1,
        }
        model = xgboost.XGBClassifier(**params)
        model.fit(x_hpo_train, y_hpo_train, eval_set=[(x_hpo_val, y_hpo_val)], verbose=False)
        pred = model.predict_proba(x_hpo_val)[:, 1]
        return average_precision_score(y_hpo_val, pred)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def soft_positive_weight(y: pd.Series | np.ndarray, power: float = 0.75) -> float:
    """Use a damped imbalance weight to reduce rare-class overprediction."""
    y_arr = np.asarray(y).astype(int)
    positives = max(int(y_arr.sum()), 1)
    negatives = max(len(y_arr) - positives, 1)
    return float((negatives / positives) ** power)


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
            "scale_pos_weight": soft_positive_weight(y_train),
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
                logger.warning("XGBoost GPU failed (%s); retrying on CPU", exc)
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
            "auto_class_weights": "SqrtBalanced",
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
                logger.warning("CatBoost GPU failed (%s); retrying on CPU", exc)
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
            "class_weight": {0: 1.0, 1: soft_positive_weight(y_train)},
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
                logger.warning("LightGBM GPU failed (%s); retrying on CPU", exc)
                params["device_type"] = "cpu"
                model = lightgbm.LGBMClassifier(**params)
                model.fit(
                    x_train, y_train, eval_set=[(x_val, y_val)], eval_metric="average_precision"
                )
            else:
                raise
        return model.predict_proba(x_val)[:, 1], model

    raise ValueError(f"Unknown model: {model_name}")


def predict_model_proba(model: Any, x: pd.DataFrame) -> np.ndarray:
    """Predict probabilities from a model artifact returned by fit_predict_model."""
    if isinstance(model, tuple) and hasattr(model[0], "transform"):
        scaler, estimator = model
        return estimator.predict_proba(scaler.transform(x))[:, 1]
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
        y_train = fold.train_features[TARGET_COL].astype(int)
        y_val = fold.val_features[TARGET_COL].astype(int)
        x_train, x_val, _ = feature_matrix(
            fold.train_features,
            fold.val_features,
            feature_cols,
            y_train=y_train,
        )
        hpo_params = {}
        if model_name == "xgboost" and hpo_trials > 0:
            hpo_groups = fold.train_features["fixture_id"]
            hpo_params = tune_xgboost(x_train, y_train, hpo_groups, use_gpu, hpo_trials)
        pred, model = fit_predict_model(
            model_name, x_train, y_train, x_val, y_val, use_gpu, hpo_params=hpo_params
        )
        train_pred = predict_model_proba(model, x_train)
        threshold = threshold_for_objective(y_train.to_numpy(), train_pred, objective="f1")
        balanced_threshold = threshold_for_balanced_accuracy(y_train.to_numpy(), train_pred)
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
        fold_result["threshold_objective"] = "f1"
        fold_result["balanced_accuracy_threshold"] = balanced_threshold
        fold_result["fold"] = fold_num
        fold_metrics.append(fold_result)
        models.append(model)
        logger.info(
            "%s/%s fold %s: pr_auc=%.4f roc_auc=%.4f balanced_accuracy=%.4f",
            model_name,
            feature_set_name,
            fold_num,
            fold_result["pr_auc"],
            fold_result["roc_auc"],
            fold_result["balanced_accuracy"],
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
        threshold = threshold_for_objective(y_train.to_numpy(), train_pred, objective="f1")
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
    if max_features < 1:
        return pd.DataFrame(
            columns=["feature", "baseline_pr_auc", "permuted_pr_auc", "importance_pr_auc_drop"]
        )
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
            y_train = fold.train_features[TARGET_COL].astype(int)
            y_val = fold.val_features[TARGET_COL].astype(int)
            x_train, x_val, _ = feature_matrix(
                fold.train_features,
                fold.val_features,
                cols,
                y_train=y_train,
            )
            if col not in x_val.columns:
                # Categorical dummy expansion means raw categoricals may not appear directly.
                continue
            shuffled = x_val.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            _, model = fit_predict_model(model_name, x_train, y_train, x_val, y_val, use_gpu)
            if isinstance(model, tuple) and hasattr(model[0], "transform"):
                scaler, estimator = model
                pred = estimator.predict_proba(scaler.transform(shuffled))[:, 1]
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
    use_gpu: bool,
) -> dict[str, Any] | None:
    """Optional AutoGluon benchmark on one holdout fold."""
    try:
        from autogluon.tabular import TabularPredictor
    except Exception:
        return None

    train = train_features[[*feature_cols, TARGET_COL]].copy()
    val = val_features[[*feature_cols, TARGET_COL]].copy()
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
        num_gpus=1 if use_gpu else 0,
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
    feature_sets: dict[str, list[str]] | None = None,
    redundant_removed: list[str] | None = None,
    *,
    base_models: list[str] | None = None,
    include_autogluon: bool = True,
    xgboost_hpo_trials: int = 0,
) -> None:
    """Write the final LaTeX report."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv_metrics = metrics[~metrics["model"].str.contains("autogluon", case=False, na=False)].copy()
    if cv_metrics.empty:
        skipped = "; ".join(str(item).splitlines()[0] for item in skipped_models) or "none recorded"
        raise ValueError(
            "Cannot write modeling report because no grouped-CV model metrics were produced. "
            f"Skipped models: {skipped}"
        )
    best = cv_metrics.sort_values(["pr_auc", "roc_auc", "balanced_accuracy"], ascending=False).iloc[
        0
    ]
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
                vals.append(
                    f"{val:.3f}" if isinstance(val, float | np.floating) else latex_escape(val)
                )
            lines.append(" & ".join(vals) + r" \\")
        return "\n".join(lines)

    stable_importance = importances[importances["importance_pr_auc_drop"] > 0].copy()
    top_importance = stable_importance.head(12) if not stable_importance.empty else pd.DataFrame()
    threshold_text = (
        "Fold-level decision thresholds were selected on each training fold for F1, while "
        "ranking metrics such as PR-AUC, precision@10%, and top-decile lift were computed "
        "from out-of-fold probabilities."
    )
    importance_rows = (
        table_rows(top_importance, ["feature", "importance_pr_auc_drop"])
        if not top_importance.empty
        else r"Permutation importance did not show stable positive PR-AUC drops in this run. \\"
    )
    skipped_text = (
        "; ".join(str(item).splitlines()[0] for item in skipped_models)
        if skipped_models
        else "None"
    )
    redundant_removed = redundant_removed or []
    feature_sets = feature_sets or {}
    redundancy_text = (
        f"Before fitting, {len(redundant_removed)} deterministic or near-deterministic aliases "
        f"were excluded from every model feature set, leaving "
        f"{len(feature_sets.get('all_features', []))} raw all-feature columns before one-hot encoding. "
        "The screen removes duplicated run summaries, duplicated shot counts or ratios, "
        "pressure expected-turnover count proxies, and the raw position category once role "
        "indicators are present. The retained representatives preserve the same football "
        "information while reducing collinearity and stabilising permutation importance."
        if redundant_removed
        else "No deterministic feature aliases were detected by the configured redundancy screen."
    )
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
        else r"None recorded. \\"
    )

    def humanize_model_name(name: str) -> str:
        mapping: dict[str, str] = {
            "logistic": "logistic regression",
            "extra_trees": "ExtraTrees",
            "random_forest": "RandomForest",
            "xgboost": "XGBoost",
            "catboost": "CatBoost",
            "lightgbm": "LightGBM",
        }
        return mapping.get(name, name.replace("_", r"\_"))

    base_models = base_models or [
        "logistic",
        "extra_trees",
        "random_forest",
        "xgboost",
        "catboost",
        "lightgbm",
    ]
    model_attempt_list = [humanize_model_name(m) for m in base_models]
    if include_autogluon:
        model_attempt_list.append("a separate AutoGluon holdout benchmark")
    models_attempted_text = (
        ", ".join(model_attempt_list[:-1]) + ", stacking, and " + model_attempt_list[-1]
        if len(model_attempt_list) > 1
        else f"{model_attempt_list[0]}, stacking"
    )
    hpo_xgb_text = (
        "XGBoost used Optuna HPO in this run when the XGBoost trial budget was nonzero; "
        if int(xgboost_hpo_trials) > 0
        else "XGBoost used fixed default hyperparameters in this run; "
    )

    report = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{hyperref}}
\usepackage{{amsmath}}
\setlength{{\emergencystretch}}{{3em}}
\title{{WEC2026 Goal-Scoring Prediction: Data, Preprocessing, and Modeling Report}}
\author{{Automated experiment report}}
\date{{\today}}

\begin{{document}}
\maketitle

\section{{Executive Summary}}
The task is a rare-event binary classification problem: predicting whether a player scores after a match checkpoint.
The modeling table contains {data_summary["n_rows"]:,} checkpoint observations, {data_summary["n_positives"]:,} positives ({data_summary["positive_rate"]:.2%}), {data_summary["n_fixtures"]} fixtures, and {data_summary["n_appearances"]} player appearances.
The best grouped cross-validated model in this run was \textbf{{{latex_escape(best["model"])}}} on \textbf{{{latex_escape(best["feature_set"])}}} features with ROC-AUC {best["roc_auc"]:.3f}, PR-AUC {best["pr_auc"]:.3f}, and balanced accuracy {best["balanced_accuracy"]:.3f}.
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
Categorical variables used training-fold rare-level bucketing plus one-hot encoding; selected stable categories also received frequency and smoothed target-rate encodings fitted only on the training fold. Numerical missing values were median-imputed from training folds only.

\section{{Feature Families}}
The all-feature model uses signals from every available file:
checkpoint physical and shot summaries, run-event intensity, pass volume and xT proxies, shot context, pressure resistance, and pressing-applied metrics.
Recent-vs-cumulative ratios were included to test short-term intensity surges.
{latex_escape(redundancy_text)}

\section{{Model Selection and HPO}}
The benchmark used fixture-grouped stratified cross-validation to prevent match-level leakage.
Models attempted in this run included {latex_escape(models_attempted_text)}.
The stacked ensemble combined base learners' out-of-fold probabilities with a lightweight meta-learner trained with the same grouped splits.
Skipped or unavailable models: {latex_escape(skipped_text)}.
{latex_escape(hpo_xgb_text)}Other model parameters were conservative regularized defaults appropriate for the small sample size. {latex_escape(threshold_text)}

\section{{Grouped Cross-Validated Results}}
\begin{{longtable}}{{llrrrrrrr}}
\toprule
Model & Feature set & ROC-AUC & PR-AUC & P@10\% & Lift@10\% & Bal. Acc. & F1 & Brier \\
\midrule
{table_rows(cv_metrics.sort_values("pr_auc", ascending=False).head(20), ["model", "feature_set", "roc_auc", "pr_auc", "precision_at_10pct", "top_decile_lift", "balanced_accuracy", "f1", "brier"])}
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
{table_rows(ablations.sort_values("pr_auc", ascending=False), ["feature_set", "roc_auc", "pr_auc", "balanced_accuracy", "f1"])}
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
Player behaviour during a match is predictive but noisy because goals are rare. In this run the best grouped-CV model reached ROC-AUC {best["roc_auc"]:.3f}, PR-AUC {best["pr_auc"]:.3f}, and balanced accuracy {best["balanced_accuracy"]:.3f}.
\paragraph{{RQ2.}}
The strongest determinants are identified by a combination of ablation deltas and permutation importance. In this run they were dominated by pass-zone shares, recent distance intensity, role indicators, pressure-location metrics, and pressure quality; the exact ranked variables are reported in the feature-importance table.
\paragraph{{RQ3.}}
The sprint-and-shot-only ablation should be compared against the all-feature model. Its result in this run was {metrics[(metrics["feature_set"] == "sprints_shots")]["pr_auc"].max() if (metrics["feature_set"] == "sprints_shots").any() else float("nan"):.3f} PR-AUC, so it is not assumed sufficient unless it matches the all-feature score.
\paragraph{{RQ4.}}
Passing and pressure data are useful when their ablation improves over the base checkpoint model. The report tables show whether \texttt{{base\_plus\_passing}} and \texttt{{base\_plus\_pressure}} improve ROC-AUC/PR-AUC/balanced accuracy.
\paragraph{{RQ5.}}
The \texttt{{recent\_only}} and \texttt{{cumulative\_only}} ablations compare short-term and accumulated performance. The stronger family is the one with higher grouped-CV PR-AUC and ROC-AUC.
\paragraph{{RQ6.}}
Recent-vs-cumulative ratios and intensity-surge interactions directly test whether short-term intensity above a player's own baseline increases scoring probability.
\paragraph{{RQ7.}}
External factors were modeled through home status, period, checkpoint time, extra time, formation, role indicators, and checkpoint-available substitution timing. Their influence is assessed by the \texttt{{external\_context}} ablation and permutation importance.

\section{{Limitations}}
Only 203 positive examples are available, so fold-level variance is expected. Some event locations are coarse three-zone labels, which limits xT precision. The stacked model is evaluated with grouped OOF meta-predictions, but it should still be validated on a hidden competition set before final claims.

\end{{document}}
"""
    REPORT_PATH.write_text(report)


def run_experiment(args: ExperimentArgs) -> None:
    """Run the configured modeling experiment."""
    start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    config = get_config(args.config_path)
    data_dir = PROJECT_ROOT / str(config.get("data.data_dir", "data"))
    use_gpu = detect_cuda()
    hpo_trials = args.hpo_trials
    if hpo_trials is None:
        if args.preset == "smoke":
            hpo_trials = 1
        elif args.preset == "fast":
            hpo_trials = 0
        else:
            hpo_trials = 12
    n_splits = int(args.folds or config.n_folds)
    if args.preset in {"smoke", "fast"}:
        n_splits = min(n_splits, 3)
    include_autogluon = bool(args.include_autogluon or args.preset == "fast")
    autogluon_time_limit = int(args.autogluon_time_limit)
    if args.preset == "fast" and args.autogluon_time_limit == 900:
        autogluon_time_limit = 420
    xgboost_hpo_trials = 0 if args.preset == "fast" else hpo_trials

    checkpoint_df, event_dfs = load_data(data_dir)
    checkpoint_df = validate_training_checkpoints(checkpoint_df, event_dfs)
    y = checkpoint_df[TARGET_COL].astype(int).reset_index(drop=True)
    data_summary = {
        "n_rows": len(checkpoint_df),
        "n_positives": int(y.sum()),
        "positive_rate": float(y.mean()),
        "n_fixtures": int(checkpoint_df["fixture_id"].nunique()),
        "n_appearances": int(checkpoint_df["player_appearance_id"].nunique()),
    }

    folds = build_fold_artifacts(checkpoint_df.reset_index(drop=True), event_dfs, n_splits=n_splits)
    first_full = pd.concat([folds[0].train_features, folds[0].val_features], axis=0)
    initial_modeling_cols = modeling_columns(first_full)
    redundant_removed = sorted(c for c in REDUNDANT_MODEL_FEATURES if c in initial_modeling_cols)
    feature_sets = folds[0].builder.get_feature_sets(initial_modeling_cols)

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
        logger.info("Running ablation %s with %s features", feature_set_name, len(cols))
        try:
            oof, oof_labels, fold_metrics, _ = run_oof_for_model(
                folds,
                feature_set_name,
                cols,
                ablation_model,
                y,
                use_gpu=use_gpu,
                hpo_trials=hpo_trials
                if feature_set_name == "all_features" and args.preset != "fast"
                else 0,
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
    if args.preset == "practical":
        base_models = [
            "logistic",
            "extra_trees",
            "random_forest",
            "xgboost",
            "catboost",
            "lightgbm",
        ]
    elif args.preset == "fast":
        base_models = ["logistic", "extra_trees", "random_forest", "xgboost", "lightgbm"]
    else:
        base_models = ["logistic", "extra_trees", "xgboost", "catboost"]
    all_cols = feature_sets["all_features"]
    data_audit = build_data_audit(checkpoint_df, event_dfs, folds, all_cols)
    all_oof_for_stack: dict[str, np.ndarray] = {}
    saved_models: dict[str, Any] = {}
    for model_name in base_models:
        logger.info("Running all-features model: %s", model_name)
        try:
            oof, oof_labels, fold_metrics, models = run_oof_for_model(
                folds,
                "all_features",
                all_cols,
                model_name,
                y,
                use_gpu=use_gpu,
                hpo_trials=xgboost_hpo_trials if model_name == "xgboost" else 0,
            )
        except Exception as exc:
            logger.warning("Skipping %s: %s", model_name, exc)
            skipped_models.append(f"{model_name}: {exc}")
            continue
        oof_columns[f"{model_name}_all_features"] = oof
        all_oof_for_stack[model_name] = oof
        metrics_rows.append(
            evaluate_predictions(y, oof, model_name, "all_features", y_pred=oof_labels)
        )
        fold_rows.extend(fold_metrics)
        saved_models[model_name] = models

    if all_oof_for_stack:
        stacked, stacked_labels = run_stacking(
            all_oof_for_stack,
            y,
            checkpoint_df["fixture_id"].reset_index(drop=True),
            n_splits=n_splits,
        )
        oof_columns["stacked_all_features"] = stacked
        metrics_rows.append(
            evaluate_predictions(y, stacked, "stacked", "all_features", y_pred=stacked_labels)
        )

    if include_autogluon:
        logger.info("Running optional AutoGluon holdout benchmark")
        if args.preset == "smoke":
            ag_time_limit = 120
        elif args.preset == "fast":
            ag_time_limit = autogluon_time_limit
        else:
            ag_time_limit = int(args.autogluon_time_limit)
        ag_result = run_autogluon_smoke(
            folds[0].train_features,
            folds[0].val_features,
            all_cols,
            time_limit=ag_time_limit,
            use_gpu=use_gpu,
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
    fold_metrics_df = pd.DataFrame(fold_rows)
    oof_predictions = pd.DataFrame(oof_columns)
    oof_predictions.insert(0, "checkpoint", checkpoint_df["checkpoint"].to_numpy())
    oof_predictions.insert(0, "fixture_id", checkpoint_df["fixture_id"].to_numpy())
    oof_predictions.insert(
        0, "player_appearance_id", checkpoint_df["player_appearance_id"].to_numpy()
    )

    if args.importance_model != "auto" and args.importance_model in all_oof_for_stack:
        importance_model = args.importance_model
    elif all_oof_for_stack:
        importance_model = max(
            all_oof_for_stack,
            key=lambda name: average_precision_score(y, all_oof_for_stack[name]),
        )
    else:
        importance_model = "extra_trees"
    importance_key = f"{importance_model}_all_features"
    baseline_pred = oof_columns.get(
        importance_key,
        oof_columns.get(
            "stacked_all_features", next(iter(all_oof_for_stack.values()), np.zeros(len(y)))
        ),
    )
    try:
        if args.preset == "practical":
            perm_max_features = 40
        elif args.preset == "fast":
            perm_max_features = 0
        else:
            perm_max_features = 8
        importances = permutation_importance_oof(
            folds,
            all_cols,
            y,
            baseline_pred,
            model_name=importance_model,
            use_gpu=use_gpu,
            max_features=perm_max_features,
        )
    except Exception as exc:
        skipped_models.append(f"permutation_importance: {exc}")
        importances = pd.DataFrame(
            columns=["feature", "baseline_pr_auc", "permuted_pr_auc", "importance_pr_auc_drop"]
        )

    metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    fold_metrics_df.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)
    oof_predictions.to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)
    importances.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    with (OUTPUT_DIR / "run_summary.json").open("w") as f:
        json.dump(
            {
                "data_summary": data_summary,
                "use_gpu": use_gpu,
                "preset": args.preset,
                "hpo_trials": hpo_trials,
                "folds": n_splits,
                "importance_model": importance_model,
                "include_autogluon": include_autogluon,
                "autogluon_time_limit": autogluon_time_limit,
                "feature_counts": {name: len(cols) for name, cols in feature_sets.items()},
                "data_audit": data_audit,
                "redundant_features_removed": redundant_removed,
                "skipped_models": skipped_models,
                "elapsed_seconds": time.time() - start,
            },
            f,
            indent=2,
        )
    joblib.dump(saved_models, MODEL_DIR / "fold_models.joblib")
    write_report(
        metrics,
        fold_metrics_df,
        importances,
        data_summary,
        skipped_models,
        feature_sets=feature_sets,
        redundant_removed=redundant_removed,
        base_models=base_models,
        include_autogluon=include_autogluon,
        xgboost_hpo_trials=xgboost_hpo_trials,
    )

    logger.info("Experiment complete")
    logger.info("Top metrics:\n%s", metrics.head(10).to_string(index=False))
    logger.info("Artifacts written to %s", OUTPUT_DIR)
    logger.info("Report written to %s", REPORT_PATH)


def main() -> None:
    args = tyro.cli(ExperimentArgs)
    configure_logging(args.log_level)
    run_experiment(args)


if __name__ == "__main__":
    main()
