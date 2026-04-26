# Warsaw Econometric Challenge 2026 - Agent Guide

## Project Purpose

This repository builds a football goal-scoring prediction pipeline for WEC2026. The task is rare-event binary classification: predict whether a player scores after a 15-minute checkpoint. The main table has 3,486 checkpoint observations across 31 fixtures, with roughly 5.8% positives.

The project is not just raw data anymore. It contains fold-safe temporal feature engineering, leakage validation, grouped cross-validation, modeling experiments, and a LaTeX/PDF report with generated figures.

## Repository Map

```text
WarsawEconometricChallenge2k26/
├── data/                         # Raw CSV competition data
├── src/
│   ├── config.py                 # YAML config loader and GPU detection
│   ├── data_ingestion.py         # pandas/cuDF data loading and schema handling
│   ├── temporal_features.py      # Current fold-safe temporal feature builder
│   ├── feature_factory.py        # Older feature engineering orchestrator
│   ├── features/                 # xT, pressure, and physical feature helpers
│   ├── models/focal_loss.py      # Custom focal-loss utilities
│   └── validation/               # Leakage checks and fixture-grouped CV
├── tests/                        # pytest coverage for feature, leakage, and CV behavior
├── scripts/create_eda_plots.py   # Regenerates report figures and summary CSV/JSON files
├── run_experiment.py             # Main report-grade experiment runner
├── main.py                       # Simpler baseline pipeline entry point
├── config.yaml                   # Data paths, CV, model, and feature settings
├── reports/                      # Problem PDFs, LaTeX report, generated figures
├── pyproject.toml                # Python 3.12 project metadata and tool config
└── Makefile                      # Setup, lint, type-check, test, cleanup commands
```

## Setup And Commands

Use Python 3.12 and `uv`.

```bash
uv venv
source .venv/bin/activate
make install-dev
```

Common commands:

```bash
make format      # ruff format + ruff check --fix on src/ and tests/
make lint        # ruff check src/ tests/
make type-check  # mypy src/
make test        # pytest tests/ -v
make test-cov    # pytest with coverage
make all         # format, lint, type-check, test
```

Pipeline and analysis commands:

```bash
python main.py
python main.py --optimize
python run_experiment.py --preset smoke
python run_experiment.py --preset practical --hpo-trials 20 --include-autogluon
python scripts/create_eda_plots.py
```

`outputs/` is the default runtime artifact directory for predictions, models, and experiment outputs. The report artifacts currently live under `reports/` and `reports/figures/`.

## Data Files

The primary join key across event files is `player_appearance_id`. Each appearance is a player in one fixture; `fixture_id` identifies the match and must be used for grouped validation.


| File                                             | Role                                                       | Approx. rows |
| ------------------------------------------------ | ---------------------------------------------------------- | ------------ |
| `players_quarters_final.csv`                     | Checkpoint-level modeling table with target `scored_after` | 3,486        |
| `player_appearance_pass.csv`                     | Pass events, accuracy, addressee, stage, period/minute     | 29,795       |
| `player_appearance_run.csv`                      | High-speed run/sprint events, speed, distance, run type    | 35,133       |
| `player_appearance_shot_limited.csv`             | Shot context without full outcome data                     | 780          |
| `player_appearance_behaviour_under_pressure.csv` | Pressed-player decisions and pressing-player IDs           | 12,185       |


CSV conventions: comma-separated, booleans are usually uppercase `TRUE`/`FALSE`, missing values may be literal `NULL`, dates are `YYYY-MM-DD`, and IDs are integers.

Important checkpoint columns:

- `checkpoint`, `checkpoint_period`, `checkpoint_min`: period-relative checkpoint time.
- `minute_in`, `minute_out`, `subbed`: substitution context. Treat `minute_out` and `subbed` as future information for modeling unless a specific analysis intentionally audits substitutions.
- `last15_*` and `cumul_*`: supplied rolling and cumulative physical/shot statistics.
- `scored_after`: target.

## Modeling Workflow

Prefer `src/temporal_features.py` and `run_experiment.py` for current report-grade modeling. `FeatureFactory` and `main.py` are useful historical/simple baselines, but they are less careful than the fold-safe workflow.

Current fold-safe flow:

1. Load `players_quarters_final.csv` plus pass/run/shot/pressure event logs.
2. Split with fixture-grouped, stratified CV (`StratifiedGroupKFold` or `FixtureGroupKFold`) so one fixture never appears in both train and validation.
3. Fit `TemporalFeatureBuilder` on the training checkpoints only.
4. Transform train and validation checkpoints separately using the same fitted builder.
5. Encode categoricals and impute numerics using training-fold data only.
6. Evaluate with ROC-AUC, PR-AUC, balanced accuracy, F1, Brier score, and log loss.

`run_experiment.py` builds ablations for:

- `base_checkpoint`
- `sprints_shots`
- `base_plus_passing`
- `base_plus_pressure`
- `recent_only`
- `cumulative_only`
- `external_context`
- `all_features`

It then benchmarks logistic regression, ExtraTrees, RandomForest, XGBoost, CatBoost, LightGBM, TabPFN, stacking, and optional AutoGluon when installed.

## Leakage And Temporal Integrity Rules

Leakage prevention is the most important project constraint.

- Always compare events to checkpoints using absolute match time, not only `minute`, because minutes restart by period. Period offsets are `half_1=0`, `half_2=45`, `extra_time_1=90`, `extra_time_2=105`.
- Event-derived features must satisfy `minute_in <= event_abs_min <= checkpoint_abs_min`.
- Do not use future substitution fields as features. `src/temporal_features.py` excludes `minute_out` and `subbed` in `modeling_columns()`.
- Do not use shot outcome/result/goal columns as predictors. `src/data_ingestion.py` drops obvious shot leakage columns, and `LeakageValidator` rejects suspicious outcome-like feature names.
- Fit learned reference quantities inside each training fold only: xT zone values, pressure turnover baselines, position baselines, imputers, encoders, scalers, thresholds, and meta-learners.
- Group all CV by `fixture_id`; never random-split checkpoint rows from the same match across folds.
- Run relevant tests after touching temporal logic, especially `tests/test_temporal_features.py`, `tests/test_temporal_integrity.py`, and `tests/test_cross_validator.py`.

## Feature Engineering Notes

`TemporalFeatureBuilder` creates the current main modeling table. It normalizes checkpoints, derives absolute time/context columns, aggregates event blocks over `last5`, `last15`, and cumulative windows, and provides research-question-oriented feature sets.

Main feature families:

- Checkpoint context: absolute checkpoint minute, half/extra-time flags, late-game flag, home/away, position, formation, minutes played.
- Supplied physical and shooting state: `last15_*`, `cumul_*`, per-90 rates, recent-vs-cumulative ratios, intensity surge indicators.
- Passing: counts, accuracy, receiver diversity, field-stage shares, xT-like stage values.
- Running: event counts, distance, speed, HSR/sprint counts, stage shares.
- Shots: shot context counts, pressure/share features, stage/body/technique/play-pattern signals.
- Pressure: retention, turnovers, forward escape, pass angle summaries, stage shares.
- Pressing applied: features from appearances listed as `pressing_player_appearance_id`.

## Current Results And Report Artifacts

Generated reporting artifacts are in `reports/`:

- `reports/wec2026_modeling_report.tex`
- `reports/wec2026_modeling_report.pdf`
- `reports/figures/model_metrics.csv`
- `reports/figures/model_feature_importance.csv`
- `reports/figures/feature_correlations.csv`
- `reports/figures/eda_plot_summary.json`
- PDF figures for target rates, event-stage quality, model comparison, feature correlations, and feature importance.

Recent figure-summary results show the task is noisy and imbalanced. The best recorded PR-AUC in `reports/figures/model_metrics.csv` is XGBoost at about 0.124, while RandomForest has the strongest ROC-AUC at about 0.683. Useful signals include attacker/defender position, pass stage shares, pressure quality/retention, recent HSR/distance ratios, and checkpoint timing.

## Development Conventions

- Keep changes scoped and preserve the leakage guardrails above.
- Prefer pandas-compatible implementations unless GPU acceleration is isolated and optional.
- cuDF/RAPIDS support is optional; code should fall back cleanly to pandas/CPU.
- Configuration belongs in `config.yaml` when it changes data paths, CV, feature settings, model settings, or performance targets.
- Tests use plain `pytest`; `pyproject.toml` configures coverage by default for `pytest`, while `make test` runs `pytest tests/ -v`.
- Ruff line length is 100 and target Python is 3.12.
- Use deterministic `random_state=42` unless there is a clear reason to expose a seed.
- Avoid committing generated caches, virtual environments, model binaries, or large transient outputs unless explicitly requested.

## When Adding Or Changing Code

Before modeling changes:

1. Identify whether the code path is the current fold-safe path (`TemporalFeatureBuilder`/`run_experiment.py`) or the older baseline path (`FeatureFactory`/`main.py`).
2. Confirm the feature is knowable at checkpoint time.
3. Add or update a focused test if the change affects time conversion, event filtering, feature inclusion/exclusion, or grouped CV.
4. Run the smallest relevant test first, then broaden to `make test` when practical.

For report updates, regenerate the source CSV/JSON/PDF artifacts with `scripts/create_eda_plots.py` or `run_experiment.py` rather than hand-editing derived tables.