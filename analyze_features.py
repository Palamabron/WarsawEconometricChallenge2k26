"""
Analiza feature'ów i identyfikacja możliwości ulepszenia.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

from src.config import get_config
from src.data_ingestion import DataIngestion
from src.feature_factory import FeatureFactory
from src.validation.leakage_checks import LeakageValidator

print("=" * 80)
print("ANALIZA FEATURE'ÓW")
print("=" * 80)

# Load data
config = get_config()
data_loader = DataIngestion(use_gpu=config.use_gpu)
checkpoint_df, event_dfs = data_loader.load_all()

factory = FeatureFactory(use_gpu=config.use_gpu)
feature_df = factory.engineer_features(checkpoint_df, event_dfs)
feature_df = LeakageValidator.validate_player_on_pitch(feature_df)

# Prepare features
target_col = "scored_after"
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

# Handle categorical
categorical_cols = X.select_dtypes(include=["object", "category"]).columns
for col in categorical_cols:
    X[col] = X[col].astype("category").cat.codes

print(f"\nAktualne feature'y: {len(feature_cols)}")
print(f"Positive class: {y.sum()} ({(y.sum()/len(y))*100:.2f}%)")

# 1. Mutual Information
print("\n" + "=" * 80)
print("MUTUAL INFORMATION (top 30)")
print("=" * 80)

mi_scores = mutual_info_classif(X.fillna(0), y, random_state=42)
mi_df = pd.DataFrame({"feature": X.columns, "mi_score": mi_scores}).sort_values(
    "mi_score", ascending=False
)
print(mi_df.head(30).to_string(index=False))

# 2. Random Forest importance
print("\n" + "=" * 80)
print("RANDOM FOREST IMPORTANCE (top 30)")
print("=" * 80)

rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X.fillna(0), y)

importance_df = pd.DataFrame(
    {"feature": X.columns, "importance": rf.feature_importances_}
).sort_values("importance", ascending=False)
print(importance_df.head(30).to_string(index=False))

# 3. Correlation analysis
print("\n" + "=" * 80)
print("KORELACJA Z TARGETEM (top 30)")
print("=" * 80)

correlations = []
for col in X.columns:
    if X[col].std() > 0:
        corr = np.corrcoef(X[col].fillna(0), y)[0, 1]
        if not np.isnan(corr):
            correlations.append({"feature": col, "correlation": abs(corr)})

corr_df = pd.DataFrame(correlations).sort_values("correlation", ascending=False)
print(corr_df.head(30).to_string(index=False))

# 4. Statystyki feature'ów
print("\n" + "=" * 80)
print("STATYSTYKI FEATURE'ÓW")
print("=" * 80)

print(f"\nFeature'y z dużą ilością NaN:")
nan_counts = X.isna().sum().sort_values(ascending=False)
high_nan = nan_counts[nan_counts > len(X) * 0.5]
if len(high_nan) > 0:
    print(high_nan.head(10))
else:
    print("Brak feature'ów z >50% NaN")

print(f"\nFeature'y z niską wariancją:")
low_var = X.std().sort_values()
print(low_var.head(10))

# 5. Analiza event data
print("\n" + "=" * 80)
print("ANALIZA EVENT DATA")
print("=" * 80)

print(f"\nPass events: {len(event_dfs['pass'])} rzędów")
print(f"Run events: {len(event_dfs['run'])} rzędów")
print(f"Shot events: {len(event_dfs['shot'])} rzędów")
print(f"Pressure events: {len(event_dfs['pressure'])} rzędów")

# Shot analysis
shots_df = event_dfs["shot"]
print(f"\nShot analysis:")
print(f"  Body parts: {shots_df['body_part'].value_counts().to_dict()}")
print(f"  Techniques: {shots_df['technique'].value_counts().to_dict()}")
print(f"  Play patterns: {shots_df['play_pattern'].value_counts().to_dict()}")
print(f"  Under pressure: {shots_df['under_pressure'].value_counts().to_dict()}")

# Pressure analysis
pressure_df = event_dfs["pressure"]
print(f"\nPressure analysis:")
print(f"  Outcomes: {pressure_df['press_induced_outcome'].value_counts().to_dict()}")

# Run analysis
run_df = event_dfs["run"]
print(f"\nRun analysis:")
print(f"  Run types: {run_df['run_type'].value_counts().to_dict()}")

print("\n" + "=" * 80)
print("REKOMENDACJE")
print("=" * 80)

print("""
1. AGREGACJE CZASOWE:
   - Dodać rolling windows 5min, 30min (nie tylko 15min)
   - Trend features (różnica last15 vs poprzednie 15min)
   - Acceleration features (przyrost metryki w czasie)

2. KONTEKST MECZU:
   - Score difference (jeśli dostępne)
   - Time since last goal
   - Momentum indicators (sekwencje eventów)

3. SHOTS & xG:
   - Shot quality metrics (dystans, kąt, część ciała)
   - xG model (prosty na bazie play_pattern, body_part, under_pressure)
   - Shot frequency ostatnie N minut

4. PRESSURE:
   - Pressure intensity (liczba pressów na minutę)
   - Successful pressure escapes rate
   - Pressing team patterns

5. PASSES:
   - Pass completion w różnych strefach
   - Progressive passes per minute
   - Key passes (pasy przed strzałem)

6. INTERACTIONS:
   - Position × physical metrics
   - Fatigue × performance drop
   - Home advantage × wszystkie metryki

7. PLAYER FORM:
   - Performance trend w ostatnich checkpointach
   - Consistency metrics (std performance)
   - Peak performance indicators
""")
