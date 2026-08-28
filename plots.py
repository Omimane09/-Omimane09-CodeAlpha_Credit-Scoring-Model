"""
Plotting utilities for the Credit Scoring Prediction System.
Generates matplotlib figures saved to static/images.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    """Plot and save a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100)
    plt.close()


def plot_feature_importance(model, feature_names, save_path, top_n=15):
    """Plot and save feature importance."""
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], color="teal")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.gca().invert_yaxis()
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100)
    plt.close()


def plot_roc_curve(y_true, y_proba, save_path, n_classes=3, labels=None):
    """Plot and save ROC curves for multiclass."""
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        y_true_bin = (y_true == i).astype(int)
        if y_proba.shape[1] > i:
            fpr, tpr, _ = roc_curve(y_true_bin, y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            label = labels[i] if labels and i < len(labels) else f"Class {i}"
            plt.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100)
    plt.close()
