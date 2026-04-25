# Graph Report - .  (2026-04-25)

## Corpus Check
- Corpus is ~13,591 words - fits in a single context window. You may not need a graph.

## Summary
- 217 nodes · 305 edges · 20 communities detected
- Extraction: 73% EXTRACTED · 27% INFERRED · 0% AMBIGUOUS · INFERRED: 81 edges (avg confidence: 0.73)
- Token cost: 27,074 input · 5,891 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Configuration & Data Loading|Configuration & Data Loading]]
- [[_COMMUNITY_Expected Threat Analytics|Expected Threat Analytics]]
- [[_COMMUNITY_Core Dataset Schemas|Core Dataset Schemas]]
- [[_COMMUNITY_Config Management|Config Management]]
- [[_COMMUNITY_Data Leakage Prevention|Data Leakage Prevention]]
- [[_COMMUNITY_Model Architecture Rationale|Model Architecture Rationale]]
- [[_COMMUNITY_Focal Loss Implementation|Focal Loss Implementation]]
- [[_COMMUNITY_Physical Fatigue Features|Physical Fatigue Features]]
- [[_COMMUNITY_Cross-Validation Pipeline|Cross-Validation Pipeline]]
- [[_COMMUNITY_Press Resistance Features|Press Resistance Features]]
- [[_COMMUNITY_Module Initializers|Module Initializers]]
- [[_COMMUNITY_Data Ingestion Backend|Data Ingestion Backend]]
- [[_COMMUNITY_CV Fold Parameter|CV Fold Parameter]]
- [[_COMMUNITY_Random Seed Config|Random Seed Config]]
- [[_COMMUNITY_GroupKFold Column|GroupKFold Column]]
- [[_COMMUNITY_Future Data Validation|Future Data Validation]]
- [[_COMMUNITY_CV Split Validation|CV Split Validation]]
- [[_COMMUNITY_Player Filter Logic|Player Filter Logic]]
- [[_COMMUNITY_XGBoost Model|XGBoost Model]]
- [[_COMMUNITY_Architecture Specification|Architecture Specification]]

## God Nodes (most connected - your core abstractions)
1. `ExpectedThreatCalculator` - 20 edges
2. `DataIngestion` - 14 edges
3. `FeatureFactory` - 14 edges
4. `main()` - 12 edges
5. `Config` - 10 edges
6. `LeakageValidator` - 9 edges
7. `FocalLoss` - 8 edges
8. `get_config()` - 7 edges
9. `calculate_physical_features()` - 7 edges
10. `FixtureGroupKFold` - 6 edges

## Surprising Connections (you probably didn't know these)
- `Press Resistance and Cognitive Load Analytics` --semantically_similar_to--> `Press Resistance Feature`  [INFERRED] [semantically similar]
  reports/gemini-deep-researchv1.md → README.md
- `WEC2026 Project Overview` --semantically_similar_to--> `WEC2026 Football Goal-Scoring Prediction Project`  [INFERRED] [semantically similar]
  CLAUDE.md → README.md
- `Three-Zone Markov Expected Threat Model` --semantically_similar_to--> `Expected Threat (xT) Feature`  [INFERRED] [semantically similar]
  reports/gemini-deep-researchv1.md → README.md
- `Relative Intensity and Dynamic Fatigue Modeling` --semantically_similar_to--> `Physical Metrics Feature`  [INFERRED] [semantically similar]
  reports/gemini-deep-researchv1.md → README.md
- `TabPFN Foundational Network Integration` --semantically_similar_to--> `TabPFN Bayesian Transformer Model`  [INFERRED] [semantically similar]
  reports/gemini-deep-researchv1.md → README.md

## Hyperedges (group relationships)
- **End-to-End Data Processing Pipeline** — gemini_data_ingestion_module, gemini_feature_factory_module, gemini_trainer_module, gemini_ensemble_module [EXTRACTED 0.95]
- **Comprehensive Data Leakage Prevention System** — readme_temporal_filtering, readme_grouped_cv, gemini_temporal_masking_rationale, gemini_outcome_sanitization_rationale, gemini_cv_isolation_rationale [EXTRACTED 0.90]
- **Three-Pillar Feature Engineering Strategy** — gemini_expected_threat_markov, gemini_press_resistance_analytics, gemini_fatigue_modeling [EXTRACTED 0.95]

## Communities

### Community 0 - "Configuration & Data Loading"
Cohesion: 0.14
Nodes (15): Get configuration value by key (supports nested keys with dots), Get full path to data file, DataIngestion, Load high-speed run event data, Load shot event data          WARNING: shot outcome columns must be dropped to p, Load behaviour under pressure data, Data loader with automatic GPU/CPU backend detection      Handles loading and va, Load all datasets          Returns:             Tuple of (checkpoint_df, event_d (+7 more)

### Community 1 - "Expected Threat Analytics"
Cohesion: 0.08
Nodes (20): aggregate_xt_features(), ExpectedThreatCalculator, Expected Threat (xT) feature engineering using three-zone Markov model  Calculat, Compute P(shot|zone) from actual shot/pass data and P(goal|shot, zone) from, Iteratively compute Expected Threat values until convergence          xT[zone] =, Calculate Expected Threat values for three-zone pitch model      Zones: bottom (, Fit Expected Threat model          Args:             pass_df: Pass event data, Calculate threat added for each pass          threat_added = xT[destination] - x (+12 more)

### Community 2 - "Core Dataset Schemas"
Cohesion: 0.08
Nodes (28): player_appearance_pass.csv Dataset, player_appearance_run.csv Dataset, player_appearance_shot_limited.csv Dataset, player_appearance_behaviour_under_pressure.csv Dataset, players_quarters_final.csv Dataset, player_appearance_id Primary Key, WEC2026 Project Overview, Three Spatial Zones (top/middle/bottom) (+20 more)

### Community 3 - "Config Management"
Cohesion: 0.11
Nodes (13): Config, get_config(), group_by_column(), n_folds(), random_state(), Configuration loader with environment detection, Configuration manager with GPU/CPU environment detection, Get or create global config instance (+5 more)

### Community 4 - "Data Leakage Prevention"
Cohesion: 0.13
Nodes (19): check_feature_leakage_correlation(), LeakageValidator, Data Leakage Prevention and Validation  Automated checks to ensure temporal inte, Safely merge event data to checkpoint data with temporal filtering      Ensures, Automated data leakage detection and prevention, Check for suspiciously high feature-target correlations      High correlations m, safe_temporal_merge(), validate_cv_splits() (+11 more)

### Community 5 - "Model Architecture Rationale"
Cohesion: 0.12
Nodes (17): CatBoost GPU Focal Loss Growth Policy Constraint, Cross-Validation Pipeline Isolation Rationale, ensemble.py Module, Focal Loss Gradient Boosting Implementation, Grouped Cross-Validation Strategy, Optuna Hyperparameter Optimization, Algorithmic Outcome Sanitization Rationale, TabPFN Foundational Network Integration (+9 more)

### Community 6 - "Focal Loss Implementation"
Cohesion: 0.19
Nodes (10): focal_loss_catboost(), focal_loss_xgboost(), FocalLoss, Focal Loss implementation for imbalanced classification.  FL(p_t) = -α_t * (1 -, CatBoost-compatible focal loss objective.      Args:         y_true: True labels, Focal Loss for binary classification.      Args:         gamma: Focusing paramet, XGBoost-compatible focal loss objective.      XGBoost passes (predt, dtrain) in, Compute mean focal loss.          Args:             y_true: True labels (0 or 1) (+2 more)

### Community 7 - "Physical Fatigue Features"
Cohesion: 0.19
Nodes (13): calculate_fatigue_indicators(), calculate_physical_features(), _calculate_positional_averages(), _calculate_run_diversity(), _calculate_speed_decay(), _calculate_workload_ratio(), Physical Metrics feature engineering  Calculates relative intensity, fatigue ind, Calculate acute:chronic workload ratio      Acute = last 15 minutes     Chronic (+5 more)

### Community 8 - "Cross-Validation Pipeline"
Cohesion: 0.18
Nodes (9): create_cross_validator(), FixtureGroupKFold, get_oof_predictions(), Cross-Validation Strategy with Group-based splitting  Implements fixture-level G, Create cross-validator from config      Args:         config_path: Path to confi, Generate out-of-fold predictions for stacking      Args:         model: Model in, Cross-validation iterator with fixture-level grouping      Ensures entire matche, Initialize fixture-grouped cross-validator          Args:             n_splits: (+1 more)

### Community 9 - "Press Resistance Features"
Cohesion: 0.24
Nodes (9): calculate_baseline_pass_angles(), _calculate_baseline_turnover_rates(), calculate_press_resistance_features(), _calculate_single_checkpoint_features(), Press Resistance feature engineering  Analyzes player behavior under defensive p, Calculate press resistance features for a single checkpoint      Args:         p, Calculate baseline turnover rates by zone across all players      Args:, Calculate baseline pass angle std per player from regular passes      This provi (+1 more)

### Community 10 - "Module Initializers"
Cohesion: 0.4
Nodes (1): Validation and cross-validation utilities

### Community 11 - "Data Ingestion Backend"
Cohesion: 0.5
Nodes (3): Data ingestion module with GPU/CPU backend support, Apply strict temporal filtering to prevent data leakage      Only includes event, safe_temporal_filter()

### Community 12 - "CV Fold Parameter"
Cohesion: 1.0
Nodes (1): Number of cross-validation folds

### Community 13 - "Random Seed Config"
Cohesion: 1.0
Nodes (1): Random seed for reproducibility

### Community 14 - "GroupKFold Column"
Cohesion: 1.0
Nodes (1): Column name for GroupKFold (fixture_id)

### Community 15 - "Future Data Validation"
Cohesion: 1.0
Nodes (1): Validate no future data leakage in features          Checks:         1. All even

### Community 16 - "CV Split Validation"
Cohesion: 1.0
Nodes (1): Validate cross-validation split integrity          Ensures:         1. No match

### Community 17 - "Player Filter Logic"
Cohesion: 1.0
Nodes (1): Filter out observations where player is not on pitch          Args:

### Community 18 - "XGBoost Model"
Cohesion: 1.0
Nodes (1): XGBoost Baseline Model

### Community 19 - "Architecture Specification"
Cohesion: 1.0
Nodes (1): End-to-End Predictive Architecture Technical Specification

## Knowledge Gaps
- **78 isolated node(s):** `Configuration loader with environment detection`, `Configuration manager with GPU/CPU environment detection`, `Load configuration from YAML file`, `Detect CUDA-capable GPU availability          Returns:             bool: True if`, `Get configuration value by key (supports nested keys with dots)` (+73 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Module Initializers`** (5 nodes): `Validation and cross-validation utilities`, `__init__.py`, `__init__.py`, `__init__.py`, `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `CV Fold Parameter`** (1 nodes): `Number of cross-validation folds`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Random Seed Config`** (1 nodes): `Random seed for reproducibility`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `GroupKFold Column`** (1 nodes): `Column name for GroupKFold (fixture_id)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Future Data Validation`** (1 nodes): `Validate no future data leakage in features          Checks:         1. All even`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `CV Split Validation`** (1 nodes): `Validate cross-validation split integrity          Ensures:         1. No match`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Player Filter Logic`** (1 nodes): `Filter out observations where player is not on pitch          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `XGBoost Model`** (1 nodes): `XGBoost Baseline Model`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Architecture Specification`** (1 nodes): `End-to-End Predictive Architecture Technical Specification`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `main()` connect `Configuration & Data Loading` to `Cross-Validation Pipeline`, `Expected Threat Analytics`, `Config Management`, `Data Leakage Prevention`?**
  _High betweenness centrality (0.102) - this node is a cross-community bridge._
- **Why does `FeatureFactory` connect `Configuration & Data Loading` to `Expected Threat Analytics`, `Config Management`?**
  _High betweenness centrality (0.095) - this node is a cross-community bridge._
- **Why does `ExpectedThreatCalculator` connect `Expected Threat Analytics` to `Configuration & Data Loading`, `Config Management`?**
  _High betweenness centrality (0.079) - this node is a cross-community bridge._
- **Are the 12 inferred relationships involving `ExpectedThreatCalculator` (e.g. with `FeatureFactory` and `Feature Factory - Orchestrates all feature engineering  Coordinates Expected Thr`) actually correct?**
  _`ExpectedThreatCalculator` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `DataIngestion` (e.g. with `Main execution script for WEC2026 Football Prediction Pipeline  Usage:     pytho` and `Main pipeline execution`) actually correct?**
  _`DataIngestion` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `FeatureFactory` (e.g. with `Main execution script for WEC2026 Football Prediction Pipeline  Usage:     pytho` and `Main pipeline execution`) actually correct?**
  _`FeatureFactory` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `main()` (e.g. with `get_config()` and `DataIngestion`) actually correct?**
  _`main()` has 10 INFERRED edges - model-reasoned connections that need verification._