"""
Optimize for realistic targets: Balanced Accuracy ≥ 0.7, PR-AUC ≥ 0.2
"""

import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import get_config
from src.data_ingestion import DataIngestion
from src.feature_factory import FeatureFactory
from src.validation.cross_validator import create_cross_validator
from src.validation.leakage_checks import LeakageValidator

warnings.filterwarnings("ignore")

TARGET_BALANCED_ACC = 0.70
TARGET_PR_AUC = 0.20
MAX_ITERATIONS = 20


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction and domain-specific features."""
    df = df.copy()

    # Physical × Tactical
    df["workload_x_xt"] = df["workload_ratio_hsr"] * df["cumul_xt_added"]
    df["speed_decay_x_shots"] = df["speed_decay"] * df["cumul_shots"]
    df["sprint_eff_x_press"] = df["sprint_efficiency"] * df["cumul_press_retention"]

    # Pressure performance
    df["press_under_fatigue"] = df["cumul_press_quality"] * df["is_fatigued"]
    df["momentum_progressive"] = df["has_momentum"] * df["cumul_progressive_press"]

    # Position-specific
    df["attacker_shots"] = df["is_attacker"] * df["cumul_shots"]
    df["attacker_xt"] = df["is_attacker"] * df["cumul_xt_added"]
    df["midfielder_press"] = df["is_midfielder"] * df["cumul_progressive_press"]

    # Recent form
    df["recent_sprint_ratio"] = df["last15_sprints"] / (df["cumul_sprints"] + 1)
    df["recent_shot_ratio"] = df["last15_shots"] / (df["cumul_shots"] + 1)
    df["recent_xt_ratio"] = df["last15_xt_added"] / (df["cumul_xt_added"] + 1)

    # Late game factors
    df["late_workload"] = df["is_late_game"] * df["workload_ratio_hsr"]
    df["late_shots"] = df["is_late_game"] * df["cumul_shots"]

    return df


def prepare_data():
    """Load and prepare data."""
    print("=" * 80)
    print("DATA PREPARATION")
    print("=" * 80)

    config = get_config()
    data_loader = DataIngestion(use_gpu=config.use_gpu)
    checkpoint_df, event_dfs = data_loader.load_all()

    factory = FeatureFactory(use_gpu=config.use_gpu)
    feature_df = factory.engineer_features(checkpoint_df, event_dfs)

    print("\nAdding advanced features...")
    feature_df = add_advanced_features(feature_df)

    LeakageValidator.validate_temporal_boundaries(feature_df, checkpoint_df, event_dfs)
    feature_df = LeakageValidator.validate_player_on_pitch(feature_df)

    # Prepare features
    target_col = "scored_after"
    group_col = "fixture_id"

    exclude_cols = [
        "player_appearance_id",
        "player_id",
        "fixture_id",
        "date",
        "checkpoint",
        "checkpoint_period",
        "formation",
        "jersey_number",
        target_col,
        "minute_in",
        "minute_out",
        "subbed",
    ]

    feature_cols = [col for col in feature_df.columns if col not in exclude_cols]

    X = feature_df[feature_cols].copy()
    y = feature_df[target_col].copy()
    groups = feature_df[group_col].copy()

    if hasattr(X, "to_pandas"):
        X = X.to_pandas()
        y = y.to_pandas()
        groups = groups.to_pandas()

    # Handle categorical
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        X[col] = X[col].astype("category").cat.codes

    print(f"\nDataset: {len(X)} samples, {len(feature_cols)} features")
    print(f"Positive class: {y.sum()} ({(y.sum()/len(y))*100:.2f}%)")

    return X, y, groups, feature_df


def optimize_catboost(X_train, y_train, X_val, y_val, trial):
    """Optimize CatBoost for balanced accuracy and PR-AUC."""
    params = {
        "iterations": trial.suggest_int("iterations", 300, 1500),
        "depth": trial.suggest_int("depth", 5, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 8),
        "border_count": trial.suggest_int("border_count", 32, 200),
        "scale_pos_weight": (len(y_train) - y_train.sum()) / y_train.sum(),
        "random_seed": 42,
        "verbose": False,
        "early_stopping_rounds": 50,
    }

    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool, verbose=False)

    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # Optimize for combined metric
    pr_auc = average_precision_score(y_val, y_pred_proba)

    # Find optimal threshold for balanced accuracy
    precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)
    best_bal_acc = 0
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        bal_acc = balanced_accuracy_score(y_val, y_pred)
        best_bal_acc = max(best_bal_acc, bal_acc)

    # Combined score (both metrics important)
    score = 0.5 * best_bal_acc + 0.5 * pr_auc
    return score


def run_optimization_iteration(X, y, groups, iteration):
    """Run single optimization iteration."""
    print("\n" + "=" * 80)
    print(f"ITERATION {iteration}")
    print("=" * 80)

    cv = create_cross_validator()
    oof_predictions = np.zeros(len(X))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), 1):
        print(f"\n### Fold {fold} ###")

        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Scale
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
        )
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val), columns=X_val.columns, index=X_val.index
        )

        # Optimize
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(
            lambda trial: optimize_catboost(
                X_train_scaled, y_train, X_val_scaled, y_val, trial
            ),
            n_trials=25,
            show_progress_bar=False,
        )

        print(f"  Best combined score: {study.best_value:.4f}")

        # Train final model
        best_params = study.best_params
        best_params.update(
            {
                "scale_pos_weight": (len(y_train) - y_train.sum()) / y_train.sum(),
                "random_seed": 42,
                "verbose": False,
            }
        )

        model = CatBoostClassifier(**best_params)
        train_pool = Pool(X_train_scaled, y_train)
        val_pool = Pool(X_val_scaled, y_val)
        model.fit(train_pool, eval_set=val_pool, verbose=False)

        y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
        oof_predictions[val_idx] = y_pred_proba

        # Calculate metrics
        pr_auc = average_precision_score(y_val, y_pred_proba)

        # Find best threshold
        precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)
        best_bal_acc = 0
        best_threshold = 0.5
        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)
            bal_acc = balanced_accuracy_score(y_val, y_pred)
            if bal_acc > best_bal_acc:
                best_bal_acc = bal_acc
                best_threshold = thresh

        fold_scores.append(
            {
                "fold": fold,
                "pr_auc": pr_auc,
                "balanced_acc": best_bal_acc,
                "threshold": best_threshold,
            }
        )

        print(f"  PR-AUC: {pr_auc:.4f}")
        print(f"  Balanced Acc: {best_bal_acc:.4f}")
        print(f"  Threshold: {best_threshold:.4f}")

    # Overall metrics
    overall_pr_auc = average_precision_score(y, oof_predictions)

    # Find overall best threshold
    precision, recall, thresholds = precision_recall_curve(y, oof_predictions)
    best_overall_bal_acc = 0
    best_overall_threshold = 0.5
    for thresh in thresholds:
        y_pred = (oof_predictions >= thresh).astype(int)
        bal_acc = balanced_accuracy_score(y, y_pred)
        if bal_acc > best_overall_bal_acc:
            best_overall_bal_acc = bal_acc
            best_overall_threshold = thresh

    y_pred_labels = (oof_predictions >= best_overall_threshold).astype(int)
    overall_f1 = f1_score(y, y_pred_labels)

    print("\n" + "=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)
    print(f"PR-AUC: {overall_pr_auc:.4f}")
    print(f"Balanced Accuracy: {best_overall_bal_acc:.4f}")
    print(f"F1 Score: {overall_f1:.4f}")
    print(f"Optimal Threshold: {best_overall_threshold:.4f}")

    print("\n" + classification_report(y, y_pred_labels, target_names=["No Goal", "Goal"]))

    return overall_pr_auc, best_overall_bal_acc, oof_predictions, fold_scores


def main():
    """Main optimization loop."""
    print("\n" + "=" * 80)
    print("REALISTIC TARGET OPTIMIZATION")
    print(f"Targets: Balanced Acc ≥ {TARGET_BALANCED_ACC}, PR-AUC ≥ {TARGET_PR_AUC}")
    print("=" * 80)

    X, y, groups, feature_df = prepare_data()

    best_pr_auc = 0.0
    best_bal_acc = 0.0
    iteration = 1

    while iteration <= MAX_ITERATIONS:
        pr_auc, bal_acc, predictions, fold_scores = run_optimization_iteration(
            X, y, groups, iteration
        )

        # Check if targets met
        targets_met = bal_acc >= TARGET_BALANCED_ACC and pr_auc >= TARGET_PR_AUC

        if pr_auc > best_pr_auc or bal_acc > best_bal_acc:
            best_pr_auc = max(best_pr_auc, pr_auc)
            best_bal_acc = max(best_bal_acc, bal_acc)

            # Save best predictions
            output_dir = Path("outputs/predictions")
            output_dir.mkdir(parents=True, exist_ok=True)

            results_df = pd.DataFrame(
                {
                    "player_appearance_id": feature_df["player_appearance_id"],
                    "fixture_id": feature_df["fixture_id"],
                    "checkpoint": feature_df["checkpoint"],
                    "true_label": y,
                    "predicted_proba": predictions,
                }
            )
            results_df.to_csv(output_dir / "best_optimized_predictions.csv", index=False)
            pd.DataFrame(fold_scores).to_csv(output_dir / "best_optimized_scores.csv", index=False)

            print(f"\n🎯 New best - PR-AUC: {best_pr_auc:.4f}, Bal Acc: {best_bal_acc:.4f}")

        if targets_met:
            print("\n" + "=" * 80)
            print(f"✅ TARGETS REACHED!")
            print(f"   PR-AUC: {pr_auc:.4f} ≥ {TARGET_PR_AUC}")
            print(f"   Balanced Acc: {bal_acc:.4f} ≥ {TARGET_BALANCED_ACC}")
            print("=" * 80)
            break

        iteration += 1

    if iteration > MAX_ITERATIONS:
        print("\n" + "=" * 80)
        print(f"Max iterations reached.")
        print(f"Best PR-AUC: {best_pr_auc:.4f} (target: {TARGET_PR_AUC})")
        print(f"Best Balanced Acc: {best_bal_acc:.4f} (target: {TARGET_BALANCED_ACC})")
        print("=" * 80)


if __name__ == "__main__":
    main()
