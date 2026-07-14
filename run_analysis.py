"""Run leakage-aware model selection and cost evaluation for fraud alerts."""

from __future__ import annotations

import json
import sys
from importlib.metadata import version
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.evaluation import (
    baseline_costs,
    chronological_split,
    find_cost_optimal_threshold,
    operating_cost,
    recall_at_k,
)


RANDOM_STATE = 42
REVIEW_COST_EUR = 5.0
FRAUD_LOSS_MULTIPLIER = 1.0


def load_data() -> pd.DataFrame:
    """Load the local Kaggle CSV, downloading it only when necessary."""
    local_path = Path("data/creditcard.csv")
    if not local_path.exists():
        import kagglehub

        download_path = Path(kagglehub.dataset_download("mlg-ulb/creditcardfraud"))
        local_path = download_path / "creditcard.csv"

    df = pd.read_csv(local_path)
    expected = {"Time", "Amount", "Class", *{f"V{i}" for i in range(1, 29)}}
    missing = sorted(expected.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if df[list(expected)].isna().any().any():
        raise ValueError("Required columns contain missing values.")
    if not set(df["Class"].unique()).issubset({0, 1}):
        raise ValueError("Class must contain only 0 and 1.")
    return df


def _build_candidates(y_train: pd.Series):
    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())
    imbalance_weight = negatives / positives

    logistic = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    xgboost = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=imbalance_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    calibration_cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    return {
        "Logistic Regression": CalibratedClassifierCV(
            logistic,
            method="sigmoid",
            cv=calibration_cv,
        ),
        "XGBoost": CalibratedClassifierCV(
            xgboost,
            method="sigmoid",
            cv=calibration_cv,
        ),
    }


def _save_charts(
    y_test,
    test_scores,
    champion_name: str,
    validation_cost_curve: pd.DataFrame,
    selected_threshold: float,
):
    output_dir = Path("screenshots")
    output_dir.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    precision, recall, _ = precision_recall_curve(y_test, test_scores)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(recall, precision, color="#6F4E7C", linewidth=2)
    ax.set_title(f"Final Test Precision–Recall Curve — {champion_name}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curve.png", dpi=180)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        validation_cost_curve["threshold"],
        validation_cost_curve["total_cost_eur"],
        color="#C44E52",
        linewidth=2,
    )
    ax.axvline(
        selected_threshold,
        color="black",
        linestyle="--",
        label=f"Selected on validation: {selected_threshold:.3f}",
    )
    selected_row = validation_cost_curve.iloc[
        (validation_cost_curve["threshold"] - selected_threshold).abs().argmin()
    ]
    ax.scatter(
        [selected_threshold],
        [selected_row["total_cost_eur"]],
        color="black",
        s=45,
        zorder=3,
    )
    ax.set_yscale("log")
    ax.set_title("Validation Cost Sensitivity — Threshold Selected Before Test")
    ax.set_xlabel("Calibrated fraud-risk threshold")
    ax.set_ylabel("Review cost + missed transaction value (EUR, log scale)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "cost_curve.png", dpi=180)
    plt.close()


def run_analysis() -> dict:
    df = load_data()
    train, validation, test = chronological_split(df)
    feature_columns = [column for column in df.columns if column != "Class"]

    X_train, y_train = train[feature_columns], train["Class"]
    X_validation, y_validation = validation[feature_columns], validation["Class"]
    X_test, y_test = test[feature_columns], test["Class"]

    validation_rows = []
    fitted = {}
    for name, model in _build_candidates(y_train).items():
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_validation)[:, 1]
        validation_rows.append(
            {
                "model": name,
                "validation_pr_auc": average_precision_score(y_validation, scores),
                "validation_recall_at_100": recall_at_k(y_validation, scores, k=100),
            }
        )
        fitted[name] = (model, scores)

    validation_results = pd.DataFrame(validation_rows).sort_values(
        "validation_pr_auc",
        ascending=False,
    )
    champion_name = validation_results.iloc[0]["model"]
    champion, validation_scores = fitted[champion_name]

    threshold_result, cost_curve = find_cost_optimal_threshold(
        y_true=y_validation,
        scores=validation_scores,
        amounts=validation["Amount"],
        review_cost_per_alert=REVIEW_COST_EUR,
        fraud_loss_multiplier=FRAUD_LOSS_MULTIPLIER,
    )
    selected_threshold = float(threshold_result["threshold"])

    # The final test period is evaluated once, after model and threshold choice.
    test_scores = champion.predict_proba(X_test)[:, 1]
    test_operating = operating_cost(
        y_true=y_test,
        scores=test_scores,
        amounts=test["Amount"],
        threshold=selected_threshold,
        review_cost_per_alert=REVIEW_COST_EUR,
        fraud_loss_multiplier=FRAUD_LOSS_MULTIPLIER,
    )
    test_baselines = baseline_costs(
        y_true=y_test,
        amounts=test["Amount"],
        review_cost_per_alert=REVIEW_COST_EUR,
        fraud_loss_multiplier=FRAUD_LOSS_MULTIPLIER,
    )

    test_summary = {
        "champion_model": champion_name,
        "selected_validation_threshold": selected_threshold,
        "test_pr_auc": float(average_precision_score(y_test, test_scores)),
        "test_recall_at_100": recall_at_k(y_test, test_scores, k=100),
        **test_operating,
        **test_baselines,
    }
    test_summary["savings_vs_review_nothing_eur"] = (
        test_summary["review_nothing_eur"] - test_summary["total_cost_eur"]
    )
    test_summary["savings_vs_review_everything_eur"] = (
        test_summary["review_everything_eur"] - test_summary["total_cost_eur"]
    )
    test_summary["savings_vs_review_nothing_per_100k_eur"] = (
        test_summary["savings_vs_review_nothing_eur"] * 100_000 / len(test)
    )

    _save_charts(
        y_test=y_test,
        test_scores=test_scores,
        champion_name=champion_name,
        validation_cost_curve=cost_curve,
        selected_threshold=selected_threshold,
    )

    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    validation_results.to_csv(artifacts / "validation_results.csv", index=False)
    artifact_payload = {
        "environment": {
            "python": sys.version.split()[0],
            "pandas": version("pandas"),
            "numpy": version("numpy"),
            "scikit-learn": version("scikit-learn"),
            "xgboost": version("xgboost"),
            "matplotlib": version("matplotlib"),
            "seaborn": version("seaborn"),
        },
        "metrics": test_summary,
    }
    with (artifacts / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(artifact_payload, handle, indent=2)
    joblib.dump(
        {
            "model": champion,
            "features": feature_columns,
            "threshold": selected_threshold,
        },
        artifacts / "champion_model.joblib",
    )

    return {
        "data_quality": {
            "rows": len(df),
            "frauds": int(df["Class"].sum()),
            "fraud_rate": float(df["Class"].mean()),
            "missing_cells": int(df.isna().sum().sum()),
        },
        "split_summary": pd.DataFrame(
            [
                {
                    "partition": name,
                    "rows": len(partition),
                    "frauds": int(partition["Class"].sum()),
                    "fraud_rate": partition["Class"].mean(),
                    "start_time": partition["Time"].min(),
                    "end_time": partition["Time"].max(),
                }
                for name, partition in [
                    ("train", train),
                    ("validation", validation),
                    ("test", test),
                ]
            ]
        ),
        "validation_results": validation_results,
        "test_summary": test_summary,
    }


def main():
    result = run_analysis()
    print("\nDATA QUALITY")
    print(pd.Series(result["data_quality"]).to_string())
    print("\nCHRONOLOGICAL SPLITS")
    print(result["split_summary"].to_string(index=False))
    print("\nVALIDATION MODEL SELECTION")
    print(result["validation_results"].to_string(index=False))
    print("\nFINAL TEST — EVALUATED AFTER MODEL AND THRESHOLD SELECTION")
    print(pd.Series(result["test_summary"]).to_string())


if __name__ == "__main__":
    main()
