"""
Training with advanced feature engineering.
Target: Balanced Acc ≥ 0.7, PR-AUC ≥ 0.2
"""

import warnings
from pathlib import Path

import numpy as np
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

from src.config import get_config
from src.data_ingestion import DataIngestion
from src.feature_factory import FeatureFactory
from src.features.advanced_features import add_all_advanced_features
from src.preprocessing import CategoricalEncoder
from src.validation.cross_validator import create_cross_validator
from src.validation.leakage_checks import LeakageValidator

warnings.filterwarnings("ignore")


def main():
    print("=" * 80)
    print("TRAINING WITH ADVANCED FEATURES")
    print("=" * 80)

    # Load data
    config = get_config()
    data_loader = DataIngestion(use_gpu=config.use_gpu)
    checkpoint_df, event_dfs = data_loader.load_all()

    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING")
    print("=" * 80)

    # Base features
    factory = FeatureFactory(use_gpu=config.use_gpu)
    feature_df = factory.engineer_features(checkpoint_df, event_dfs)

    # Advanced features
    print("\n" + "=" * 80)
    print("ADVANCED FEATURE ENGINEERING")
    print("=" * 80)

    feature_df = add_all_advanced_features(feature_df, event_dfs, use_gpu=config.use_gpu)

    # Validation
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

    print(f"\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    print(f"Samples: {len(X)}")
    print(f"Features: {len(feature_cols)} (was 56, now {len(feature_cols)})")
    print(f"Positive class: {y.sum()} ({(y.sum()/len(y))*100:.2f}%)")
    print(f"Unique matches: {groups.nunique()}")

    # Cross-validation
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION")
    print("=" * 80)

    cv = create_cross_validator()
    oof_predictions = np.zeros(len(X))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), 1):
        print(f"\n### Fold {fold} ###")

        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Encode categorical features consistently
        encoder = CategoricalEncoder()
        X_train = encoder.fit_transform(X_train)
        X_val = encoder.transform(X_val)

        # Scale
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
        )
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val), columns=X_val.columns, index=X_val.index
        )

        # Train CatBoost (best from HPO)
        params = {
            "iterations": 1200,
            "depth": 8,
            "learning_rate": 0.05,
            "l2_leaf_reg": 6.0,
            "border_count": 100,
            "scale_pos_weight": (len(y_train) - y_train.sum()) / y_train.sum(),
            "random_seed": 42,
            "verbose": False,
            "early_stopping_rounds": 100,
        }

        model = CatBoostClassifier(**params)
        train_pool = Pool(X_train_scaled, y_train)
        val_pool = Pool(X_val_scaled, y_val)

        model.fit(train_pool, eval_set=val_pool, verbose=False)

        y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
        oof_predictions[val_idx] = y_pred_proba

        # Metrics
        pr_auc = average_precision_score(y_val, y_pred_proba)

        # Find best threshold for balanced accuracy
        precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)
        best_bal_acc = 0
        best_threshold = 0.5
        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)
            bal_acc = balanced_accuracy_score(y_val, y_pred)
            if bal_acc > best_bal_acc:
                best_bal_acc = bal_acc
                best_threshold = thresh

        y_pred_final = (y_pred_proba >= best_threshold).astype(int)
        fold_f1 = f1_score(y_val, y_pred_final)

        fold_scores.append(
            {
                "fold": fold,
                "pr_auc": pr_auc,
                "balanced_acc": best_bal_acc,
                "f1": fold_f1,
                "threshold": best_threshold,
            }
        )

        print(f"  PR-AUC: {pr_auc:.4f}")
        print(f"  Balanced Acc: {best_bal_acc:.4f}")
        print(f"  F1: {fold_f1:.4f}")
        print(f"  Threshold: {best_threshold:.4f}")

    # Overall results
    print("\n" + "=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)

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

    scores_df = pd.DataFrame(fold_scores)

    print(f"\nCross-Validation Averages:")
    print(f"  PR-AUC: {scores_df['pr_auc'].mean():.4f} ± {scores_df['pr_auc'].std():.4f}")
    print(
        f"  Balanced Acc: {scores_df['balanced_acc'].mean():.4f} ± {scores_df['balanced_acc'].std():.4f}"
    )
    print(f"  F1: {scores_df['f1'].mean():.4f} ± {scores_df['f1'].std():.4f}")

    print(f"\nOut-of-Fold Overall:")
    print(f"  PR-AUC: {overall_pr_auc:.4f}")
    print(f"  Balanced Acc: {best_overall_bal_acc:.4f}")
    print(f"  F1: {overall_f1:.4f}")
    print(f"  Optimal Threshold: {best_overall_threshold:.4f}")

    # Check targets
    print(f"\n" + "=" * 80)
    if best_overall_bal_acc >= 0.70 and overall_pr_auc >= 0.20:
        print("✅ TARGETS REACHED!")
        print(f"   Balanced Acc: {best_overall_bal_acc:.4f} ≥ 0.70")
        print(f"   PR-AUC: {overall_pr_auc:.4f} ≥ 0.20")
    else:
        print("⚠️  Targets not yet reached")
        print(f"   Balanced Acc: {best_overall_bal_acc:.4f} (target: ≥ 0.70)")
        print(f"   PR-AUC: {overall_pr_auc:.4f} (target: ≥ 0.20)")
    print("=" * 80)

    print("\n" + classification_report(y, y_pred_labels, target_names=["No Goal", "Goal"]))

    # Save results
    output_dir = Path("outputs/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(
        {
            "player_appearance_id": feature_df["player_appearance_id"],
            "fixture_id": feature_df["fixture_id"],
            "checkpoint": feature_df["checkpoint"],
            "true_label": y,
            "predicted_proba": oof_predictions,
            "predicted_label": y_pred_labels,
        }
    )

    results_df.to_csv(output_dir / "advanced_features_predictions.csv", index=False)
    scores_df.to_csv(output_dir / "advanced_features_scores.csv", index=False)

    print(f"\n✓ Saved predictions to {output_dir}")


if __name__ == "__main__":
    main()
