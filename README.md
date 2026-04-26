# Warsaw Econometric Challenge 2026 - Football Goal-Scoring Prediction

Binary classification: predict whether a player scores a goal after a 15-minute match checkpoint. Dataset: 3,486 observations, 5.82% positive class, 31 matches.

## Setup

Requires [uv](https://github.com/astral-sh/uv).

```bash
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate.bat
make sync-core             # core deps
make sync-dev              # add ruff, mypy, pytest
make install-gpu           # add RAPIDS cuDF (CUDA 11.8+ required)
```

## Usage

```bash
python main.py             # run pipeline
python main.py --optimize  # with Optuna hyperparameter search
```

## Development

```bash
make fmt        # ruff auto-fix + format
make lint       # ruff lint + format check
make types      # mypy on src and tests
make test       # pytest
make test-cov   # pytest with coverage
make all        # lint, types, and tests
```

Notebook checks are optional and skip cleanly when `notebooks/` is absent:

```bash
make nb-format  # nbQA ruff auto-fix + format via uv
make nb-lint    # nbQA ruff lint + format check via uv
make nb-types   # nbQA mypy via uv
```

## Project Structure

```
src/
  config.py                    # configuration loader
  data_ingestion.py            # data loading (CPU/GPU auto-detect)
  feature_factory.py           # feature engineering orchestrator
  features/
    expected_threat.py         # three-zone Markov xT model
    press_resistance.py        # behavior under pressure metrics
    physical_metrics.py        # workload ratios, fatigue, speed decay
  models/
    focal_loss.py              # custom focal loss (gamma=2.0, alpha=0.94)
  validation/
    cross_validator.py         # fixture-based GroupKFold
    leakage_checks.py          # temporal integrity validation
data/                          # CSV datasets
outputs/                       # predictions, models, reports
config.yaml                    # model and pipeline configuration
main.py                        # entry point
```

## Data

| File | Description | Rows |
|------|-------------|------|
| `players_quarters_final.csv` | Checkpoint observations (target: `scored_after`) | 3,486 |
| `player_appearance_pass.csv` | Pass events | 29,795 |
| `player_appearance_run.csv` | Physical tracking | 35,133 |
| `player_appearance_shot_limited.csv` | Shot attempts | 780 |
| `player_appearance_behaviour_under_pressure.csv` | Pressure situations | 12,185 |

## Features Engineered

**Expected Threat (xT):** Three-zone Markov model (bottom/middle/top) computing threat added per pass, aggregated as rolling 15-min and cumulative totals.

**Press Resistance:** Retention rate, progressive evasion count, pass angle distortion, and expected vs actual turnovers under defensive pressure.

**Physical Metrics:** Acute:chronic workload ratio, positional momentum deviation, speed decay, run type diversity, fatigue/momentum indicators.

## Model Architecture

Baseline: XGBoost with `scale_pos_weight`.

Planned: TabPFN (zero-shot Bayesian transformer) + CatBoost with focal loss, stacked via logistic regression meta-learner.

## Data Leakage Safeguards

- All event features filtered to `event_minute <= checkpoint_minute`
- Shot outcome columns excluded from features
- Cross-validation grouped by `fixture_id` (no match overlap between folds)
- Scaler fit exclusively on training data within each CV fold

## Configuration

Key settings in `config.yaml`:

```yaml
cross_validation:
  n_folds: 5
  group_by: "fixture_id"

models:
  catboost:
    grow_policy: "Lossguide"  # required for custom loss on GPU
  focal_loss:
    gamma: 2.0
    alpha: 0.94

hyperparameter_optimization:
  enabled: false
  n_trials: 100
```

## GPU Support

GPU (RTX 4090) accelerates feature engineering ~20x via RAPIDS cuDF. Auto-detected at runtime; falls back to pandas on CPU.

```bash
make install-gpu
python -c "import cudf; print('GPU ready')"
```

## Performance Targets

| Metric | Target | Baseline (dummy) |
|--------|--------|-----------------|
| F1 Score | > 0.30 | 0.11 |
| PR-AUC | > 0.40 | - |
| Brier Score | < 0.06 | - |

Expected runtime: ~45 min (ARM CPU) / ~15 min (RTX 4090).

## References

- Expected Threat: Karun Singh, https://karun.in/blog/expected-threat.html
- Focal Loss: Lin et al., 2017, https://arxiv.org/abs/1708.02002
- TabPFN: Hollmann et al., 2023, https://arxiv.org/abs/2207.01848
