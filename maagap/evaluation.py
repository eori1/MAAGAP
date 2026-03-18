"""Evaluation metrics, visualisations, and reporting for all models.

Covers the metrics specified in the thesis Objective 2:
  Accuracy, Precision, Recall, F1-Score, AUC-ROC, MAE
"""

import numpy as np
import pandas as pd
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_absolute_error,
    classification_report, confusion_matrix, roc_curve, auc,
)

from .config import OUTPUTS_DIR, RISK_LABELS


def _safe_auc(y_true, y_proba):
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return roc_auc_score(y_true, y_proba)
    except ValueError:
        return float("nan")


def find_optimal_threshold(y_true, y_proba, metric="f1"):
    """Search for the probability threshold that maximises F1 on a validation set."""
    best_t, best_score = 0.5, 0.0
    for t in np.arange(0.20, 0.80, 0.01):
        preds = (y_proba >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_score:
            best_t, best_score = t, score
    return round(best_t, 2)


def binary_metrics(y_true, y_pred, y_proba=None, label="Model"):
    """Compute all thesis-specified binary classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc_val = _safe_auc(y_true, y_proba) if y_proba is not None else float("nan")
    return {
        "Model": label,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "AUC-ROC": round(auc_val, 4),
    }


def regression_metrics(y_true, y_pred, label="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    return {"Model": label, "MAE": round(mae, 4)}


def multiclass_metrics(y_true, y_pred, label="Model"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {
        "Model": label,
        "Accuracy": round(acc, 4),
        "Precision (macro)": round(prec, 4),
        "Recall (macro)": round(rec, 4),
        "F1-Score (macro)": round(f1, 4),
    }


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_DIR, filename), dpi=150)
    plt.close(fig)


def plot_roc_curves(curves_data, filename="roc_curves.png"):
    """curves_data: list of (fpr, tpr, auc_val, label)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for fpr, tpr, auc_val, label in curves_data:
        ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Delay Prediction")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_DIR, filename), dpi=150)
    plt.close(fig)


def plot_feature_importance(importances, feature_names, title, filename, top_n=20):
    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.35)))
    ax.barh(range(len(idx)), importances[idx], color="steelblue")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_DIR, filename), dpi=150)
    plt.close(fig)


def plot_training_history(history, filename="lstm_training.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="Train")
    axes[0].plot(history.history["val_loss"], label="Validation")
    axes[0].set_title("LSTM Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history.history["accuracy"], label="Train")
    axes[1].plot(history.history["val_accuracy"], label="Validation")
    axes[1].set_title("LSTM Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_DIR, filename), dpi=150)
    plt.close(fig)


def plot_risk_distribution(y_true, y_pred, filename="risk_distribution.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, title in zip(axes, [y_true, y_pred], ["Actual", "Predicted"]):
        labels, counts = np.unique(data, return_counts=True)
        label_names = [RISK_LABELS[i] if i < len(RISK_LABELS) else str(i) for i in labels]
        colors = ["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"]
        ax.bar(label_names, counts, color=[colors[i] for i in labels])
        ax.set_title(f"{title} Risk Distribution")
        ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_DIR, filename), dpi=150)
    plt.close(fig)


def generate_full_report(all_metrics, filename="evaluation_report.csv"):
    df = pd.DataFrame(all_metrics)
    df.to_csv(os.path.join(OUTPUTS_DIR, filename), index=False)
    return df
