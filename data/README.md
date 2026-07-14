# Data card — Credit Card Fraud Detection

## Source

| Field | Detail |
|---|---|
| Dataset | Credit Card Fraud Detection |
| Provider | Machine Learning Group at ULB, from a research collaboration with Worldline |
| Source page | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Local shape | 284,807 rows × 31 columns |
| Local file | `data/creditcard.csv` (excluded from Git) |

Reuse is subject to the terms shown on the source page. This repository does not make a separate claim about ownership or relicensing of the data.

## Local integrity snapshot

- 284,807 transactions;
- 492 fraud labels and 284,315 non-fraud labels;
- 0.1727% fraud rate;
- no missing values in the supplied local copy;
- `Time` ranges from 0 to 172,792 seconds;
- 1,825 zero-amount transactions, including 27 labeled frauds;
- mean amount among fraud rows: approximately EUR 122.21.

## Columns

| Column | Description |
|---|---|
| `Time` | Seconds elapsed from the first transaction in the file. It provides ordering, not a verified calendar timestamp. |
| `V1`–`V28` | PCA-transformed, anonymized predictors. Their original business meaning is unavailable. |
| `Amount` | Transaction amount. The portfolio cost model treats it as an illustrative avoided-loss proxy. |
| `Class` | Target label: 1 for fraud, 0 otherwise. |

## Leakage and availability caveats

There is no obvious target-derived predictor in the published schema, but the PCA anonymity means feature definitions and exact availability times cannot be audited. The project therefore cannot prove that every `V` feature would be available at authorization time in a real payment system.

Labels may also arrive after chargebacks or investigations. The dataset does not expose label delay, recovery, or investigation status, all of which matter in production evaluation.

## Split decision

The analysis sorts rows by `Time` and uses:

- earliest 60% for training;
- next 20% for model and threshold selection;
- latest 20% for one final evaluation.

This avoids mixing later transactions into earlier training data. It is more deployment-like than a random split, but the full dataset covers only about 48 hours, so it is not a strong test of long-term temporal drift.

## Interpretation limits

- PCA feature importance cannot be translated into an operational action such as blocking a merchant category or country.
- Missing demographic and account attributes prevent a meaningful subgroup fairness audit.
- `Amount` is not necessarily equal to final net fraud loss after recovery or chargebacks.
- A EUR 5 review cost is an explicit sensitivity assumption, not an observed value.
- Performance on 2013 activity does not establish performance against current fraud tactics.

To reproduce the analysis, place `creditcard.csv` in this folder or let `run_analysis.py` retrieve it through `kagglehub`.
