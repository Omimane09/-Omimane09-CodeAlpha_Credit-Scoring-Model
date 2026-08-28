"""
Evaluation module for the Credit Scoring Prediction System.

Computes metrics, confusion matrix, and classification report.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


class ModelEvaluator:
    """Compute and store evaluation metrics for trained models."""

    def __init__(self, model, X_test, y_test, label_encoder=None):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.label_encoder = label_encoder
        self.y_pred = None
        self.y_proba = None
        self.metrics = {}
        self._evaluate()

    def _evaluate(self):
        """Perform the evaluation and store results."""
        self.y_pred = self.model.predict(self.X_test)

        # Try to get probabilities
        try:
            self.y_proba = self.model.predict_proba(self.X_test)
        except AttributeError:
            self.y_proba = None

        self.metrics["accuracy"] = accuracy_score(self.y_test, self.y_pred)
        average = "macro"

        # For multiclass AUC
        if self.y_proba is not None and len(set(self.y_test)) > 1:
            try:
                self.metrics["roc_auc"] = roc_auc_score(
                    self.y_test, self.y_proba, multi_class="ovr", average=average
                )
            except ValueError:
                self.metrics["roc_auc"] = None
        else:
            self.metrics["roc_auc"] = None

        self.metrics["precision"] = precision_score(
            self.y_test, self.y_pred, average=average, zero_division=0
        )
        self.metrics["recall"] = recall_score(
            self.y_test, self.y_pred, average=average, zero_division=0
        )
        self.metrics["f1"] = f1_score(
            self.y_test, self.y_pred, average=average, zero_division=0
        )
        self.metrics["confusion_matrix"] = confusion_matrix(self.y_test, self.y_pred).tolist()

        # Classification report as dict
        report = classification_report(
            self.y_test, self.y_pred, output_dict=True, zero_division=0
        )
        # Convert numpy types
        self.metrics["report"] = json_serialize(report)

    def get_summary(self):
        """Return a clean summary of metrics."""
        return {
            "accuracy": round(self.metrics["accuracy"] * 100, 2),
            "precision": round(self.metrics["precision"] * 100, 2),
            "recall": round(self.metrics["recall"] * 100, 2),
            "f1": round(self.metrics["f1"] * 100, 2),
            "roc_auc": round(self.metrics["roc_auc"] * 100, 2) if self.metrics["roc_auc"] else "N/A",
        }


def json_serialize(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {k: json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [json_serialize(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
