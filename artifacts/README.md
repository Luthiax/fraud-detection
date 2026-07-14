# Generated evaluation artifacts

`python run_analysis.py` writes:

- `validation_results.csv` — candidate performance used for model selection;
- `test_metrics.json` — the one-time final-test evaluation and separate cost baselines;
- `champion_model.joblib` — calibrated model, feature order, and selected threshold.

The binary model is excluded from Git. The CSV and JSON should be committed only after a clean, reproducible run has been reviewed.
