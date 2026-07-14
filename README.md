# Credit Card Fraud Alert Ranking — Cost-Aware Offline Evaluation

This project demonstrates how to evaluate a rare-event classifier when fraud represents only 0.17% of transactions. It focuses on alert ranking, review capacity, threshold selection, and explicit cost assumptions—not headline accuracy.

> **Scope:** this is an offline portfolio case study using a two-day, anonymized historical dataset. It is not a real-time production system, a bank deployment recommendation, or proof of future savings.

## Decision question

Given a limited manual-review queue, which transactions should be prioritized, and how should an alert threshold be selected when reviews and missed fraud have different costs?

The repaired workflow makes three separate decisions:

1. fit candidate models on the earliest 60% of transactions;
2. select the model and cost threshold on the next 20%;
3. evaluate that frozen choice once on the final 20%.

This prevents the final test period from being reused for model selection or threshold tuning.

## Clean-run results

Reproduced locally with Python 3.12.10 using the declared dependencies.

### Decisions made on validation

| Metric | Result |
|---|---:|
| Champion model | XGBoost with sigmoid calibration |
| Validation PR-AUC | 0.7830 |
| Validation Recall@100 | 78.95% |
| Cost-selected threshold | 0.569 |

### One-time final test

The final period contains 56,962 transactions and 75 fraud cases.

| Metric | Result |
|---|---:|
| Test PR-AUC | **0.8013** |
| Test Recall@100 | **77.33%** |
| Alerts at frozen threshold | 61 (0.1071%) |
| True positives / false positives / false negatives | 57 / 4 / 18 |
| Precision / recall at threshold | 93.44% / 76.00% |
| Review cost | EUR 305.00 |
| Missed-fraud amount | EUR 2,638.27 |
| Total modeled cost | **EUR 2,943.27** |
| Review-nothing baseline | EUR 7,729.26 |
| Savings vs review nothing | **EUR 4,785.99** on the test period |
| Normalized savings vs review nothing | **EUR 8,402.08 per 100k transactions** |
| Review-everything baseline | EUR 284,810.00 |

The per-100k value is a normalization of this historical test sample under the stated cost assumptions—not a forecast or bank ROI claim.

The previous public metrics—PR-AUC 0.8346, threshold 0.003, and EUR 17,172.91 per 100k—are withdrawn because they were selected and reported on the same holdout and used an inconsistent cost formula.

### Generated evidence

![Final test precision-recall curve](screenshots/pr_curve.png)

![Validation cost curve](screenshots/cost_curve.png)

## Correct cost definition

For a threshold `t`:

```text
alerts = all transactions with calibrated fraud score >= t
review cost = number of alerts × assumed review cost
missed fraud loss = sum(Amount for missed frauds) × loss multiplier
total cost = review cost + missed fraud loss
```

Two baselines answer different questions:

```text
review nothing = sum(Amount for every fraud)
review everything = number of transactions × review cost
```

The old formula charged review cost only to false positives and replaced each missed fraud with a full-dataset average. The repaired formula charges **every** alert and uses the actual transaction value of each missed fraud within the evaluation period.

Even the repaired formula is illustrative. It assumes transaction `Amount` equals avoidable loss, a detected fraud prevents the full amount, and each review costs EUR 5. A real institution would replace those with net loss, recovery, investigation time, customer-friction, and chargeback estimates.

## Dataset

Source: [Kaggle — Credit Card Fraud Detection, ULB Machine Learning Group / Worldline](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

The local file contains 284,807 transactions, including 492 labeled fraud cases (0.1727%), across roughly 48 hours. `V1`–`V28` are anonymized PCA features; `Time` and `Amount` are the only named predictors.

Important limits:

- only about two days of activity, so weekly, monthly, and seasonal behavior is absent;
- PCA anonymity prevents operational root-cause explanations and most subgroup fairness audits;
- label timing and feature availability cannot be fully audited from anonymized fields;
- a chronological split is more deployment-like than a random split, but 48 hours is still too short for robust drift testing;
- the 2013 sample may not represent current fraud patterns, controls, or payment rails.

See [`data/README.md`](data/README.md) for the local integrity snapshot and evaluation caveats.

## Methodology

- Sort transactions by `Time` and create 60% train / 20% validation / 20% final-test periods.
- Compare calibrated class-weighted Logistic Regression and calibrated XGBoost.
- Select the champion on validation PR-AUC, not accuracy.
- Select the operating threshold on validation cost using explicit assumptions.
- Evaluate the frozen model and threshold once on the later test period.
- Report PR-AUC, Recall@100, alert rate, precision, recall, confusion counts, cost, and both baselines.
- Save the fitted preprocessing/model/threshold bundle together.

Probability calibration is learned inside the training period. The threshold is therefore presented as a calibrated risk threshold, but calibration on a short historical sample still requires monitoring in any real deployment.

## Run locally

Requires Python 3.11 or 3.12.

```bash
git clone https://github.com/Luthiax/fraud-detection.git
cd fraud-detection
python -m venv .venv

# Windows
.venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_analysis.py
python -m unittest discover -s tests
```

If `data/creditcard.csv` is absent, the analysis downloads the public Kaggle dataset with `kagglehub`.

The script regenerates:

- `artifacts/validation_results.csv`;
- `artifacts/test_metrics.json`;
- `artifacts/champion_model.joblib` (excluded from Git);
- `screenshots/pr_curve.png`;
- `screenshots/cost_curve.png`.

## Repository map

```text
Fraud-Detection/
├── README.md
├── requirements.txt
├── data/README.md
├── src/evaluation.py
├── tests/test_evaluation.py
├── run_analysis.py
├── Fraud_Detection_Analysis.ipynb
├── artifacts/
└── screenshots/
```

## What this project demonstrates

- why 99.83% accuracy can be useless in rare-event detection;
- ranking metrics and fixed-capacity evaluation;
- time-ordered train/validation/test design;
- threshold selection before final testing;
- calibrated scores and explicit operating assumptions;
- cost accounting that a reviewer can inspect and challenge.

It does not demonstrate streaming infrastructure, real-time latency, rule-engine integration, customer authentication, auto-decline safety, compliance approval, or production monitoring.

## Interview explanation

> “My first version overstated the result because I chose the model and cost threshold on the same holdout used for reporting. I repaired it with chronological train, validation, and final-test periods. Model and threshold decisions happen before the final test. I also corrected the cost function so every alert incurs review cost and missed loss uses each missed transaction's actual amount. The remaining economics are assumptions, not a bank ROI claim. I would deploy only in shadow mode first; I would not recommend auto-decline from this dataset.”

## About

I'm **Leonardo Flores**, a bilingual English/Spanish operations and business-analytics professional based in Lima, Peru. This project is part of a portfolio focused on translating analytical work into clear decisions while stating evidence limits honestly.

[LinkedIn](https://www.linkedin.com/in/leonardo-floresg/) · [GitHub](https://github.com/Luthiax)
