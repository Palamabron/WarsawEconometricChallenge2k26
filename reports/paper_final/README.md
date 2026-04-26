# `paper_final` package

This folder contains a self-contained final paper bundle based on current experiment artifacts.

## Contents

- `paper_final.tex` - updated manuscript with preprocessing, feature engineering, model setup, and results aligned to current CSV exports.
- `paper_final.pdf` - compiled PDF from `paper_final.tex`.
- `figures/` - all report-relevant plots used in the manuscript:
  - `target_rates.pdf`
  - `event_stage_quality.pdf`
  - `feature_correlations.pdf`
  - `model_comparison.pdf`
  - `model_feature_importance.pdf`
  - `oof_precision_recall.pdf`
  - `stacked_meta_weights.pdf`
- `data/` - source artifacts used for the manuscript tables/text:
  - `metrics.csv`
  - `run_summary.json`
  - `oof_predictions.csv`
  - `ensemble_oof_performance.csv`
  - `stacked_meta_weights.csv`
  - `model_feature_importance.csv`
  - `model_metrics.csv`
  - `feature_correlations.csv`
  - `eda_plot_summary.json`

## Compile

From this directory:

```bash
pdflatex -interaction=nonstopmode -halt-on-error paper_final.tex
pdflatex -interaction=nonstopmode -halt-on-error paper_final.tex
```

The second pass resolves references/hyperlinks.

## Notes

- Main model results are OOF grouped-CV rows from `data/metrics.csv` (n=3486), while `autogluon_holdout` is a separate holdout benchmark (n=1096).
- Explainability artifacts come from scan mode (fast diagnostics), so SHAP/permutation heavy outputs are intentionally omitted here.
