"""
Analyze dataset to understand PR-AUC ceiling and feasibility.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from src.config import get_config
from src.data_ingestion import DataIngestion
from src.feature_factory import FeatureFactory
from src.validation.leakage_checks import LeakageValidator

print("=" * 80)
print("DATASET FEASIBILITY ANALYSIS")
print("=" * 80)

# Load data
config = get_config()
data_loader = DataIngestion(use_gpu=config.use_gpu)
checkpoint_df, event_dfs = data_loader.load_all()

factory = FeatureFactory(use_gpu=config.use_gpu)
feature_df = factory.engineer_features(checkpoint_df, event_dfs)

feature_df = LeakageValidator.validate_player_on_pitch(feature_df)

# Check target distribution
y = feature_df["scored_after"]
print(f"\nTarget distribution:")
print(f"  Positive: {y.sum()} ({(y.sum()/len(y))*100:.2f}%)")
print(f"  Negative: {len(y) - y.sum()} ({((len(y) - y.sum())/len(y))*100:.2f}%)")
print(f"  Class imbalance ratio: 1:{int((len(y) - y.sum()) / y.sum())}")

# Check if target is deterministic (leakage)
exclude_cols = [
    "player_appearance_id",
    "player_id",
    "fixture_id",
    "date",
    "checkpoint",
    "checkpoint_period",
    "formation",
    "jersey_number",
    "scored_after",
    "minute_in",
    "minute_out",
    "subbed",
]

feature_cols = [col for col in feature_df.columns if col not in exclude_cols]
X = feature_df[feature_cols].copy()

# Handle categorical
categorical_cols = X.select_dtypes(include=["object", "category"]).columns
for col in categorical_cols:
    X[col] = X[col].astype("category").cat.codes

# Check correlation with target
correlations = []
for col in X.columns:
    if X[col].std() > 0:  # Skip constant columns
        corr = np.corrcoef(X[col].fillna(0), y)[0, 1]
        if not np.isnan(corr):
            correlations.append({"feature": col, "correlation": abs(corr)})

corr_df = pd.DataFrame(correlations).sort_values("correlation", ascending=False)

print(f"\nTop 20 features by absolute correlation with target:")
print(corr_df.head(20).to_string(index=False))

# Check feature importance via random forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X.fillna(0), y)

importance_df = pd.DataFrame(
    {"feature": X.columns, "importance": rf.feature_importances_}
).sort_values("importance", ascending=False)

print(f"\nTop 20 features by Random Forest importance:")
print(importance_df.head(20).to_string(index=False))

# Check baseline PR-AUC
print(f"\nBaseline models:")

# Perfect random baseline
random_preds = np.random.rand(len(y))
random_pr_auc = average_precision_score(y, random_preds)
print(f"  Random predictions PR-AUC: {random_pr_auc:.4f}")

# Class prior baseline
prior_preds = np.full(len(y), y.mean())
prior_pr_auc = average_precision_score(y, prior_preds)
print(f"  Class prior baseline PR-AUC: {prior_pr_auc:.4f}")

# Quick RF baseline
y_pred_rf = rf.predict_proba(X.fillna(0))[:, 1]
rf_pr_auc = average_precision_score(y, y_pred_rf)
print(f"  Random Forest (overfitted) PR-AUC: {rf_pr_auc:.4f}")

# Estimate ceiling
print(f"\n" + "=" * 80)
print("THEORETICAL CEILING ANALYSIS")
print("=" * 80)

# With 5.67% positive class, perfect ranking would give:
# - All positives ranked first
# - Precision curve: starts at 1.0, ends at 0.0567
# - Theoretical max PR-AUC ≈ 0.5-0.7 with perfect model

print(f"\nWith {(y.sum()/len(y))*100:.2f}% positive class:")
print(f"  Random baseline: ~{y.mean():.4f}")
print(f"  Perfect classifier ceiling: ~0.7-0.9 (depends on noise)")
print(f"  Realistic achievable: 0.3-0.5 (strong signal needed)")
print(f"\nTarget of 0.85 requires near-perfect prediction (minimal noise/stochasticity)")

# Check for obvious leakage or strong signals
print(f"\n" + "=" * 80)
print("SIGNAL STRENGTH CHECK")
print("=" * 80)

# Split and check simple model
X_train, X_test, y_train, y_test = train_test_split(
    X.fillna(0), y, test_size=0.2, random_state=42, stratify=y
)

from xgboost import XGBClassifier

xgb = XGBClassifier(
    max_depth=8,
    learning_rate=0.05,
    n_estimators=500,
    scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(),
    random_state=42,
    tree_method="hist",
)

xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
y_pred_xgb = xgb.predict_proba(X_test)[:, 1]
xgb_pr_auc = average_precision_score(y_test, y_pred_xgb)

print(f"XGBoost (hold-out test) PR-AUC: {xgb_pr_auc:.4f}")

if xgb_pr_auc > 0.5:
    print("✅ Strong predictive signal detected")
elif xgb_pr_auc > 0.3:
    print("⚠️  Moderate signal - target 0.85 likely unreachable")
else:
    print("❌ Weak signal - fundamental limitations in data")

print(f"\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

if xgb_pr_auc < 0.3:
    print("Target PR-AUC of 0.85 is not feasible with current features.")
    print("Suggested realistic target: 0.20-0.35")
    print("\nPossible improvements:")
    print("  1. Feature engineering (domain-specific transformations)")
    print("  2. External data sources")
    print("  3. Re-examine target definition (is 'scored_after' correctly labeled?)")
elif xgb_pr_auc < 0.5:
    print("Target PR-AUC of 0.85 is extremely challenging.")
    print("Suggested realistic target: 0.35-0.50")
    print("\nPossible improvements:")
    print("  1. Advanced ensembling (stacking, blending)")
    print("  2. Deep feature engineering")
    print("  3. Hyperparameter optimization")
else:
    print("Strong signal detected! PR-AUC 0.85 may be achievable.")
    print("\nContinue with:")
    print("  1. Ensemble methods")
    print("  2. Careful feature selection")
    print("  3. Advanced models (AutoGluon / deep stacks, when available)")
