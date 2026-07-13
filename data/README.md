# Dataset — Credit Card Fraud Detection

## Source

| Field | Detail |
|---|---|
| Name | Credit Card Fraud Detection |
| Author | Machine Learning Group - ULB (Université Libre de Bruxelles) |
| Platform | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Rows × Columns | 284,807 × 31 (28 PCA features, Time, Amount, Class) |
| License | Open for research |
| File | `../creditcard.csv` |

> **Note:** The CSV is excluded from this repository via `.gitignore` (~150 MB).  
> To reproduce the analysis, the notebook automatically downloads the dataset using `kagglehub`.

---

## Raw Column Dictionary

| Column | Type | Description |
|---|---|---|
| `Time` | float | Seconds elapsed between this transaction and the first transaction in the dataset (0–172,792, approx 48 hours). |
| `V1`–`V28` | float | Principal Component Analysis (PCA) transformations of original features. Anonymized to protect user identities and merchant data. |
| `Amount` | float | Transaction amount (EUR). Mean ≈ €88, Max ≈ €25,691. |
| `Class` | int | **Target variable.** 1 in case of fraud and 0 otherwise. Only 492 frauds (0.1727% positive rate). |

---

## Leakage & Anonymization Discussion

**PCA Anonymization Trade-off:**
The 28 `V` features are the result of PCA. While this preserves privacy (we don't know the merchant category, cardholder country, or device type), it eliminates operational root-cause analysis. We can say "V14 is highly suspicious," but we cannot translate that back into "Block IP ranges from Country X." 

**Leakage:**
All features are collected at the moment of the transaction attempt. There is no target leakage in this dataset. The critical challenge is the extreme class imbalance.
