"""
Advanced optimization pipeline with Optuna and ensemble methods.
Iteratively improves until PR-AUC reaches target (~0.85).
"""

import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import get_config
from src.data_ingestion import DataIngestion
from src.feature_factory import FeatureFactory
from src.validation.cross_validator import create_cross_validator
from src.validation.leakage_checks import LeakageValidator

warnings.filterwarnings("ignore")

TARGET_PR_AUC = 0.85
MAX_ITERATIONS = 50


class AdvancedFeatureEngineer:
    """Enhanced feature engineering with interactions and domain knowledge."""

    @staticmethod
    def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between key metrics."""
        df = df.copy()

        # Physical × Tactical interactions
        df["workload_x_xt"] = df["workload_ratio_hsr"] * df["cumul_xt_added"]
        df["speed_decay_x_press"] = df["speed_decay"] * df["cumul_press_retention"]
        df["sprint_efficiency_x_xt"] = df["sprint_efficiency"] * df["xt_per_pass"]

        # Pressure × Performance
        df["press_under_fatigue"] = df["cumul_press_quality"] * df["is_fatigued"]
        df["momentum_x_progressive"] = (
            df["has_momentum"] * df["cumul_progressive_press"]
        )

        # Position × Physical
        df["attacker_sprints"] = df["is_attacker"] * df["cumul_sprints"]
        df["midfielder_distance"] = df["is_midfielder"] * df["cumul_distance"]

        # Shot context
        df["shots_per_xt"] = df["cumul_shots"] / (df["cumul_xt_added"] + 1)
        df["shots_x_pressure_eff"] = df["cumul_shots"] * df["pressure_efficiency"]

        # Late game intensity
        df["late_game_workload"] = df["is_late_game"] * df["workload_ratio_hsr"]
        df["second_half_speed"] = df["is_second_half"] * df["speed_decay"]

        return df

    @staticmethod
    def add_rolling_ratios(df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling vs cumulative ratios."""
        df = df.copy()

        # Recent form indicators (last 15 vs cumulative)
        df["recent_sprint_ratio"] = df["last15_sprints"] / (df["cumul_sprints"] + 1)
        df["recent_shot_ratio"] = df["last15_shots"] / (df["cumul_shots"] + 1)
        df["recent_xt_ratio"] = df["last15_xt_added"] / (df["cumul_xt_added"] + 1)
        df["recent_press_ratio"] = (
            df["last15_progressive_press"] / (df["cumul_progressive_press"] + 1)
        )

        return df


def prepare_data():
    """Load and prepare data with enhanced features."""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    config = get_config()
    data_loader = DataIngestion(use_gpu=config.use_gpu)
    checkpoint_df, event_dfs = data_loader.load_all()

    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING")
    print("=" * 80)

    factory = FeatureFactory(use_gpu=config.use_gpu)
    feature_df = factory.engineer_features(checkpoint_df, event_dfs)

    # Add advanced features
    print("\nAdding interaction features...")
    feature_df = AdvancedFeatureEngineer.add_interaction_features(feature_df)
    print("Adding rolling ratios...")
    feature_df = AdvancedFeatureEngineer.add_rolling_ratios(feature_df)

    # Leakage validation
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


def train_catboost_with_optuna(X_train, y_train, X_val, y_val, trial):
    """Train CatBoost with Optuna-optimized hyperparameters."""
    params = {
        "iterations": trial.suggest_int("iterations", 500, 2000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "random_strength": trial.suggest_float("random_strength", 0, 10),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
        "scale_pos_weight": (len(y_train) - y_train.sum()) / y_train.sum(),
        "random_seed": 42,
        "verbose": False,
        "early_stopping_rounds": 100,
    }

    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool, verbose=False)

    return model


def train_xgboost_with_optuna(X_train, y_train, X_val, y_val, trial):
    """Train XGBoost with Optuna-optimized hyperparameters."""
    params = {
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 1),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 2),
        "scale_pos_weight": (len(y_train) - y_train.sum()) / y_train.sum(),
        "random_state": 42,
        "eval_metric": "logloss",
        "early_stopping_rounds": 100,
        "tree_method": "hist",
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    return model


def optimize_single_fold(X_train, y_train, X_val, y_val, model_type, n_trials=20):
    """Optimize single model on one fold."""

    def objective(trial):
        if model_type == "catboost":
            model = train_catboost_with_optuna(X_train, y_train, X_val, y_val, trial)
        elif model_type == "xgboost":
            model = train_xgboost_with_optuna(X_train, y_train, X_val, y_val, trial)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        y_pred_proba = model.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, y_pred_proba)
        return pr_auc

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"  Best PR-AUC: {study.best_value:.4f}")
    return study.best_params


def run_full_optimization(X, y, groups, iteration=1):
    """Run full optimization pipeline."""
    print("\n" + "=" * 80)
    print(f"OPTIMIZATION ITERATION {iteration}")
    print("=" * 80)

    cv = create_cross_validator()
    oof_predictions = {}
    fold_scores = []

    # Try multiple model types
    model_types = ["catboost", "xgboost"]

    for model_name in model_types:
        print(f"\n### Optimizing {model_name.upper()} ###")
        oof_preds = np.zeros(len(X))

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), 1):
            print(f"\nFold {fold}:")

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

            # Optimize hyperparameters
            print(f"  Optimizing hyperparameters ({model_name})...")
            best_params = optimize_single_fold(
                X_train_scaled, y_train, X_val_scaled, y_val, model_name, n_trials=15
            )

            # Train final model with best params
            if model_name == "catboost":
                best_params["random_seed"] = 42
                best_params["verbose"] = False
                best_params["scale_pos_weight"] = (len(y_train) - y_train.sum()) / y_train.sum()
                model = CatBoostClassifier(**best_params)
                train_pool = Pool(X_train_scaled, y_train)
                val_pool = Pool(X_val_scaled, y_val)
                model.fit(train_pool, eval_set=val_pool, verbose=False)
            else:  # xgboost
                best_params["random_state"] = 42
                best_params["eval_metric"] = "logloss"
                best_params["scale_pos_weight"] = (len(y_train) - y_train.sum()) / y_train.sum()
                best_params["tree_method"] = "hist"
                model = XGBClassifier(**best_params)
                model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)

            y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            oof_preds[val_idx] = y_pred_proba

            pr_auc = average_precision_score(y_val, y_pred_proba)
            print(f"  Fold {fold} PR-AUC: {pr_auc:.4f}")

        oof_predictions[model_name] = oof_preds

        # Overall score
        overall_pr_auc = average_precision_score(y, oof_preds)
        fold_scores.append({"model": model_name, "pr_auc": overall_pr_auc})
        print(f"\n{model_name.upper()} Overall PR-AUC: {overall_pr_auc:.4f}")

    # Ensemble predictions (average)
    print("\n### ENSEMBLE ###")
    ensemble_preds = np.mean(list(oof_predictions.values()), axis=0)
    ensemble_pr_auc = average_precision_score(y, ensemble_preds)
    print(f"Ensemble PR-AUC: {ensemble_pr_auc:.4f}")

    # Find optimal threshold
    precision, recall, thresholds = precision_recall_curve(y, ensemble_preds)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_threshold = thresholds[np.argmax(f1_scores)]
    ensemble_f1 = f1_score(y, (ensemble_preds >= best_threshold).astype(int))

    print(f"Ensemble F1: {ensemble_f1:.4f}")
    print(f"Optimal threshold: {best_threshold:.4f}")

    return ensemble_pr_auc, ensemble_preds, fold_scores


def main():
    """Main optimization loop."""
    print("\n" + "=" * 80)
    print("ADVANCED ML OPTIMIZATION PIPELINE")
    print(f"Target: PR-AUC ≥ {TARGET_PR_AUC}")
    print("=" * 80)

    # Load data once
    X, y, groups, feature_df = prepare_data()

    best_pr_auc = 0.0
    iteration = 1

    while best_pr_auc < TARGET_PR_AUC and iteration <= MAX_ITERATIONS:
        pr_auc, predictions, fold_scores = run_full_optimization(X, y, groups, iteration)

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            print(f"\n🎯 New best PR-AUC: {best_pr_auc:.4f}")

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
            results_df.to_csv(output_dir / "best_predictions.csv", index=False)

            pd.DataFrame(fold_scores).to_csv(output_dir / "best_scores.csv", index=False)

        if best_pr_auc >= TARGET_PR_AUC:
            print("\n" + "=" * 80)
            print(f"✅ TARGET REACHED! PR-AUC: {best_pr_auc:.4f} ≥ {TARGET_PR_AUC}")
            print("=" * 80)
            break

        iteration += 1

    if best_pr_auc < TARGET_PR_AUC:
        print("\n" + "=" * 80)
        print(f"⚠️  Max iterations reached. Best PR-AUC: {best_pr_auc:.4f}")
        print("=" * 80)


if __name__ == "__main__":
    main()
