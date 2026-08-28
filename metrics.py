"""
Metrics utilities for the Credit Scoring Prediction System.
"""

import numpy as np


def calculate_metrics(y_true, y_pred, y_proba=None):
    """Calculate common classification metrics."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix
    )

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_proba is not None and len(set(y_true)) > 1:
        try:
            metrics["roc_auc_macro"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            )
        except ValueError:
            metrics["roc_auc_macro"] = None
    else:
        metrics["roc_auc_macro"] = None

    return metrics


def risk_level_to_score(risk):
    """Convert risk level to numeric score for charts."""
    mapping = {"Low": 1, "Medium": 2, "High": 3}
    return mapping.get(risk, 0)
