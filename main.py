"""
Main execution script for WEC2026 Football Prediction Pipeline

Usage:
    python main.py --optimize  # Run with hyperparameter optimization
    python main.py             # Run with default parameters
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
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

warnings.filterwarnings("ignore", category=FutureWarning)


def main(args):
    """Main pipeline execution"""

    print("=" * 80)
    print("WARSAW ECONOMETRIC CHALLENGE 2026")
    print("Football Goal-Scoring Prediction Pipeline")
    print("=" * 80)

    # 1. Load configuration
    config = get_config(args.config)
    print(f"\nConfiguration loaded:")
    print(f"  GPU available: {config.use_gpu}")
    print(f"  CV folds: {config.n_folds}")
    print(f"  Random seed: {config.random_state}")

    # 2. Load data
    print("\n" + "=" * 80)
    print("STEP 1: Data Loading")
    print("=" * 80)

    data_loader = DataIngestion(use_gpu=config.use_gpu, config_path=args.config)
    checkpoint_df, event_dfs = data_loader.load_all()

    # 3. Engineer features
    print("\n" + "=" * 80)
    print("STEP 2: Feature Engineering")
    print("=" * 80)

    factory = FeatureFactory(config_path=args.config, use_gpu=config.use_gpu)
    feature_df = factory.engineer_features(checkpoint_df, event_dfs)

    # 4. Run leakage checks
    print("\n" + "=" * 80)
    print("STEP 3: Leakage Validation")
    print("=" * 80)

    LeakageValidator.validate_temporal_boundaries(feature_df, checkpoint_df, event_dfs)

    # 5. Prepare data for modeling
    print("\n" + "=" * 80)
    print("STEP 4: Data Preparation")
    print("=" * 80)

    # Filter to valid observations (player on pitch)
    feature_df = LeakageValidator.validate_player_on_pitch(feature_df)

    # Separate features and target
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

    # Convert to pandas if cuDF
    if hasattr(X, "to_pandas"):
        X = X.to_pandas()
        y = y.to_pandas()
        groups = groups.to_pandas()

    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        X[col] = X[col].astype("category").cat.codes

    print(f"\nDataset prepared:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Positive class: {y.sum()} ({(y.sum() / len(y)) * 100:.2f}%)")
    print(f"  Unique matches: {groups.nunique()}")

    # 6. Cross-validation
    print("\n" + "=" * 80)
    print("STEP 5: Cross-Validation")
    print("=" * 80)

    cv = create_cross_validator(args.config)

    # Baseline model - simple XGBoost
    print("\nTraining baseline XGBoost model...")

    oof_predictions = np.zeros(len(X))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), 1):
        print(f"\n--- Fold {fold} ---")

        # Split data
        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Train model
        model = XGBClassifier(
            scale_pos_weight=(1 - y_train.mean()) / y_train.mean(),
            max_depth=6,
            learning_rate=0.05,
            n_estimators=500,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=config.random_state,
            eval_metric="logloss",
            early_stopping_rounds=50,
        )

        model.fit(
            X_train_scaled,
            y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=False,
        )

        # Predict
        y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
        oof_predictions[val_idx] = y_pred_proba

        # Find optimal threshold
        precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
        best_threshold = thresholds[np.argmax(f1_scores)]

        y_pred = (y_pred_proba >= best_threshold).astype(int)

        # Evaluate
        fold_f1 = f1_score(y_val, y_pred)
        fold_prauc = average_precision_score(y_val, y_pred_proba)

        fold_scores.append(
            {
                "fold": fold,
                "f1": fold_f1,
                "pr_auc": fold_prauc,
                "threshold": best_threshold,
            }
        )

        print(f"  F1 Score: {fold_f1:.4f}")
        print(f"  PR-AUC: {fold_prauc:.4f}")
        print(f"  Optimal threshold: {best_threshold:.4f}")

    # Overall performance
    print("\n" + "=" * 80)
    print("OVERALL CROSS-VALIDATION RESULTS")
    print("=" * 80)

    scores_df = pd.DataFrame(fold_scores)
    print(f"\nMean F1 Score: {scores_df['f1'].mean():.4f} (+/- {scores_df['f1'].std():.4f})")
    print(f"Mean PR-AUC: {scores_df['pr_auc'].mean():.4f} (+/- {scores_df['pr_auc'].std():.4f})")

    # Find optimal overall threshold
    precision, recall, thresholds = precision_recall_curve(y, oof_predictions)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    best_overall_threshold = thresholds[np.argmax(f1_scores)]

    oof_pred_labels = (oof_predictions >= best_overall_threshold).astype(int)
    overall_f1 = f1_score(y, oof_pred_labels)
    overall_prauc = average_precision_score(y, oof_predictions)

    print(f"\nOut-of-Fold Performance:")
    print(f"  F1 Score: {overall_f1:.4f}")
    print(f"  PR-AUC: {overall_prauc:.4f}")
    print(f"  Optimal threshold: {best_overall_threshold:.4f}")

    print("\n" + classification_report(y, oof_pred_labels, target_names=["No Goal", "Goal"]))

    # 7. Save results
    print("\n" + "=" * 80)
    print("STEP 6: Saving Results")
    print("=" * 80)

    output_dir = Path("outputs/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save OOF predictions
    results_df = pd.DataFrame(
        {
            "player_appearance_id": feature_df["player_appearance_id"],
            "fixture_id": feature_df["fixture_id"],
            "checkpoint": feature_df["checkpoint"],
            "true_label": y,
            "predicted_proba": oof_predictions,
            "predicted_label": oof_pred_labels,
        }
    )

    results_path = output_dir / "oof_predictions.csv"
    results_df.to_csv(results_path, index=False)
    print(f"  ✓ Saved predictions to {results_path}")

    # Save scores
    scores_path = output_dir / "cv_scores.csv"
    scores_df.to_csv(scores_path, index=False)
    print(f"  ✓ Saved CV scores to {scores_path}")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WEC2026 Football Prediction Pipeline")
    parser.add_argument("--optimize", action="store_true", help="Run hyperparameter optimization")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")

    args = parser.parse_args()
    main(args)
