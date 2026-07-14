"""Evaluation and cost functions for fraud-alert ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd


def chronological_split(
    df: pd.DataFrame,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return earlier train, middle validation, and later test partitions."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    if not 0 < validation_fraction < 1 - train_fraction:
        raise ValueError("validation_fraction leaves no room for a test set.")
    if "Time" not in df.columns:
        raise ValueError("A Time column is required for chronological splitting.")

    ordered = df.sort_values("Time", kind="mergesort").reset_index(drop=True)
    train_target = int(len(ordered) * train_fraction)
    validation_target = int(len(ordered) * (train_fraction + validation_fraction))
    train_cutoff = ordered.iloc[train_target]["Time"]
    test_cutoff = ordered.iloc[validation_target]["Time"]

    # Keep equal timestamps together so no instant appears in two partitions.
    train = ordered.loc[ordered["Time"] < train_cutoff].copy()
    validation = ordered.loc[
        (ordered["Time"] >= train_cutoff) & (ordered["Time"] < test_cutoff)
    ].copy()
    test = ordered.loc[ordered["Time"] >= test_cutoff].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Chronological cutoffs produced an empty partition.")
    return train, validation, test


def recall_at_k(y_true, scores, k: int = 100) -> float:
    """Fraction of all fraud cases captured in the k highest scores."""
    labels = np.asarray(y_true, dtype=int)
    risk_scores = np.asarray(scores, dtype=float)
    if labels.shape != risk_scores.shape:
        raise ValueError("y_true and scores must have the same shape.")
    if k < 1:
        raise ValueError("k must be positive.")
    positives = labels.sum()
    if positives == 0:
        return float("nan")

    selected = np.argsort(risk_scores)[::-1][: min(k, len(labels))]
    return float(labels[selected].sum() / positives)


def operating_cost(
    y_true,
    scores,
    amounts,
    threshold: float,
    review_cost_per_alert: float = 5.0,
    fraud_loss_multiplier: float = 1.0,
) -> dict[str, float | int]:
    """Calculate alert-review cost plus missed-fraud transaction value.

    Every alert incurs review cost, whether it is a true or false positive.
    Missed loss is the observed Amount of false negatives multiplied by an
    explicit loss assumption.
    """
    labels = np.asarray(y_true, dtype=int)
    risk_scores = np.asarray(scores, dtype=float)
    transaction_amounts = np.asarray(amounts, dtype=float)

    if not (labels.shape == risk_scores.shape == transaction_amounts.shape):
        raise ValueError("Labels, scores, and amounts must have the same shape.")
    if review_cost_per_alert < 0 or fraud_loss_multiplier < 0:
        raise ValueError("Cost assumptions must be non-negative.")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    alerts = risk_scores >= threshold
    fraud = labels == 1
    true_positives = int(np.sum(alerts & fraud))
    false_positives = int(np.sum(alerts & ~fraud))
    false_negatives = int(np.sum(~alerts & fraud))
    alert_count = int(alerts.sum())

    review_cost = float(alert_count * review_cost_per_alert)
    missed_fraud_loss = float(
        transaction_amounts[~alerts & fraud].sum() * fraud_loss_multiplier
    )
    total_cost = review_cost + missed_fraud_loss

    precision = true_positives / alert_count if alert_count else 0.0
    fraud_count = int(fraud.sum())
    recall = true_positives / fraud_count if fraud_count else float("nan")

    return {
        "threshold": float(threshold),
        "alerts": alert_count,
        "alert_rate": alert_count / len(labels),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": float(precision),
        "recall": float(recall),
        "review_cost_eur": review_cost,
        "missed_fraud_loss_eur": missed_fraud_loss,
        "total_cost_eur": total_cost,
    }


def find_cost_optimal_threshold(
    y_true,
    scores,
    amounts,
    review_cost_per_alert: float = 5.0,
    fraud_loss_multiplier: float = 1.0,
    thresholds=None,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Choose a threshold on validation data and return the full cost curve."""
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 1001)

    rows = [
        operating_cost(
            y_true=y_true,
            scores=scores,
            amounts=amounts,
            threshold=float(threshold),
            review_cost_per_alert=review_cost_per_alert,
            fraud_loss_multiplier=fraud_loss_multiplier,
        )
        for threshold in thresholds
    ]
    curve = pd.DataFrame(rows)
    best = curve.loc[curve["total_cost_eur"].idxmin()].to_dict()
    return best, curve


def baseline_costs(
    y_true,
    amounts,
    review_cost_per_alert: float = 5.0,
    fraud_loss_multiplier: float = 1.0,
) -> dict[str, float]:
    """Return comparable review-everything and review-nothing costs."""
    labels = np.asarray(y_true, dtype=int)
    transaction_amounts = np.asarray(amounts, dtype=float)
    if labels.shape != transaction_amounts.shape:
        raise ValueError("Labels and amounts must have the same shape.")

    return {
        "review_nothing_eur": float(
            transaction_amounts[labels == 1].sum() * fraud_loss_multiplier
        ),
        "review_everything_eur": float(len(labels) * review_cost_per_alert),
    }
