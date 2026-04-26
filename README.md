# Warsaw Econometric Challenge 2026

Football goal-scoring prediction pipeline for WEC2026. The task is rare-event binary classification: predict whether a player scores after a 15-minute checkpoint.

The current modeling path is fold-safe: event features are built inside fixture-grouped cross-validation folds, temporal boundaries use absolute match time, and known leakage fields are excluded before training.

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
uv venv
source .venv/bin/activate
make install-dev
```

For optional RAPIDS/cuDF support:

```bash
make install-gpu
```

## Common Commands

```bash
make format      # ruff check --fix + ruff format
make lint        # ruff lint + format check
make type-check  # mypy
make test        # pytest
make all         # lint, type-check, test
```

Run the main report-grade experiment:

```bash
python run_experiment.py --preset smoke
python run_experiment.py --preset practical --hpo-trials 20
```

The older baseline entry point is still available:

```bash
python main.py
python main.py --optimize
```

## Repository Map

```text
data/                    Competition CSV files
src/config.py            Pydantic-validated YAML config loader
src/data_ingestion.py    pandas/cuDF data loading helpers
src/temporal_features.py Fold-safe temporal feature builder
src/preprocessing.py     Train-only categorical encoding utilities
src/features/            xT, pressure, and physical feature helpers
src/validation/          Leakage checks and grouped CV
run_experiment.py        Main experiment and report runner
main.py                  Older baseline pipeline
reports/                 Report sources, PDFs, and generated figures
tests/                   Pytest coverage for temporal and modeling helpers
```

## Data

| File | Role |
| --- | --- |
| `players_quarters_final.csv` | Checkpoint modeling table with target `scored_after` |
| `player_appearance_pass.csv` | Pass events |
| `player_appearance_run.csv` | High-speed run and sprint events |
| `player_appearance_shot_limited.csv` | Shot context without full outcome data |
| `player_appearance_behaviour_under_pressure.csv` | Pressed-player decisions and pressing-player IDs |

## Leakage Rules

- Compare event times to checkpoints using absolute match time, because minutes restart by period.
- Fit encoders, imputers, feature builders, thresholds, and meta-learners on training folds only.
- Group cross-validation by `fixture_id`; one fixture must not appear in both train and validation.
- Do not use future substitution fields, shot outcomes, or goal/result-like columns as predictors.

## Configuration

Edit `config.yaml` for data paths, CV settings, feature settings, and model options. The loader validates the main sections with pydantic while keeping dotted-key access for existing code:

```python
from src.config import get_config

config = get_config()
n_folds = config.get("cross_validation.n_folds")
```

## Outputs

Runtime outputs go under `outputs/`. Report artifacts live under `reports/` and `reports/figures/`.
