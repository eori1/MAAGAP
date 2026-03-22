"""Evaluation metrics, visualisations (Plotly), and reporting for all models.

Covers the metrics specified in the thesis Objective 2:
  Accuracy, Precision, Recall, F1-Score, AUC-ROC, MAE
"""

import numpy as np
import pandas as pd
import os

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_absolute_error,
    confusion_matrix, roc_curve,
)

from .config import OUTPUTS_DIR, RISK_LABELS

MODEL_COLORS = {
    "Random Forest": "#2ecc71",
    "XGBoost": "#3498db",
    "LSTM": "#e74c3c",
    "Meta-Ensemble": "#9b59b6",
    "RF Risk": "#2ecc71",
    "XGB Risk": "#3498db",
}

_PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Segoe UI, sans-serif", size=13),
    margin=dict(l=60, r=30, t=60, b=50),
)


def _save_fig(fig, filename):
    """Save as interactive HTML and static PNG."""
    base, _ = os.path.splitext(filename)
    fig.write_html(os.path.join(OUTPUTS_DIR, base + ".html"), include_plotlyjs="cdn")
    try:
        fig.write_image(os.path.join(OUTPUTS_DIR, filename), scale=2, width=900, height=600)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _safe_auc(y_true, y_proba):
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return roc_auc_score(y_true, y_proba)
    except ValueError:
        return float("nan")


def find_optimal_threshold(y_true, y_proba, metric="f1"):
    best_t, best_score = 0.5, 0.0
    for t in np.arange(0.20, 0.80, 0.01):
        preds = (y_proba >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_score:
            best_t, best_score = t, score
    return round(best_t, 2)


def binary_metrics(y_true, y_pred, y_proba=None, label="Model"):
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
# Plotly visualisations
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    text = [[str(v) for v in row] for row in cm]

    fig = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels,
        text=text, texttemplate="%{text}", textfont=dict(size=16),
        colorscale="Blues", showscale=True,
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Predicted",
        yaxis_title="Actual",
        yaxis=dict(autorange="reversed"),
        **_PLOTLY_LAYOUT,
    )
    _save_fig(fig, filename)
    return fig


def plot_roc_curves(curves_data, filename="roc_curves_delay.png"):
    """curves_data: list of (fpr, tpr, auc_val, label)."""
    fig = go.Figure()

    for fpr, tpr, auc_val, label in curves_data:
        color = MODEL_COLORS.get(label, "#7f8c8d")
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{label} (AUC = {auc_val:.3f})",
            line=dict(color=color, width=2.5),
            hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random baseline",
        line=dict(color="grey", width=1, dash="dash"), showlegend=True,
    ))
    fig.update_layout(
        title=dict(text="ROC Curves — Delay Prediction", x=0.5),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.55, y=0.05, bgcolor="rgba(255,255,255,0.8)"),
        **_PLOTLY_LAYOUT,
    )
    _save_fig(fig, filename)
    return fig


def plot_feature_importance(importances, feature_names, title, filename, top_n=20):
    idx = np.argsort(importances)[-top_n:]
    names = [feature_names[i] for i in idx]
    vals = importances[idx]

    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color="#3498db",
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Importance",
        yaxis=dict(automargin=True),
        height=max(500, top_n * 28),
        **_PLOTLY_LAYOUT,
    )
    _save_fig(fig, filename)
    return fig


def plot_training_history(history, filename="lstm_training_history.png"):
    epochs = list(range(1, len(history.history["loss"]) + 1))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("LSTM Loss", "LSTM Accuracy"),
        horizontal_spacing=0.12,
    )

    fig.add_trace(go.Scatter(
        x=epochs, y=history.history["loss"], mode="lines",
        name="Train Loss", line=dict(color="#e74c3c", width=2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=epochs, y=history.history["val_loss"], mode="lines",
        name="Val Loss", line=dict(color="#3498db", width=2, dash="dash"),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=epochs, y=history.history["accuracy"], mode="lines",
        name="Train Acc", line=dict(color="#e74c3c", width=2),
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=epochs, y=history.history["val_accuracy"], mode="lines",
        name="Val Acc", line=dict(color="#3498db", width=2, dash="dash"),
    ), row=1, col=2)

    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)
    fig.update_layout(
        width=1000, height=450,
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        **_PLOTLY_LAYOUT,
    )
    _save_fig(fig, filename)
    return fig


def plot_risk_distribution(y_true, y_pred, filename="risk_distribution.png"):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Actual Risk Distribution", "Predicted Risk Distribution"),
        horizontal_spacing=0.12,
    )

    colors = ["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"]

    for col_idx, (data, subtitle) in enumerate([(y_true, "Actual"), (y_pred, "Predicted")], 1):
        labels_idx, counts = np.unique(data, return_counts=True)
        names = [RISK_LABELS[i] if i < len(RISK_LABELS) else str(i) for i in labels_idx]
        bar_colors = [colors[i] if i < len(colors) else "#95a5a6" for i in labels_idx]
        fig.add_trace(go.Bar(
            x=names, y=counts, marker_color=bar_colors,
            text=counts, textposition="auto",
            hovertemplate="%{x}: %{y}<extra></extra>",
            showlegend=False,
        ), row=1, col=col_idx)

    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_layout(width=900, height=450, **_PLOTLY_LAYOUT)
    _save_fig(fig, filename)
    return fig


def plot_model_comparison(metrics_list, filename="model_comparison.png"):
    """Grouped bar chart comparing all models across key metrics."""
    models = [m["Model"] for m in metrics_list]
    metric_keys = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
    metric_keys = [k for k in metric_keys if k in metrics_list[0]]

    fig = go.Figure()
    bar_colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6", "#f39c12"]

    for i, key in enumerate(metric_keys):
        vals = [m.get(key, 0) for m in metrics_list]
        fig.add_trace(go.Bar(
            name=key, x=models, y=vals,
            marker_color=bar_colors[i % len(bar_colors)],
            text=[f"{v:.3f}" for v in vals], textposition="auto",
        ))

    fig.update_layout(
        title=dict(text="Model Performance Comparison — Delay Prediction", x=0.5),
        barmode="group",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.05]),
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        width=950, height=500,
        **_PLOTLY_LAYOUT,
    )
    _save_fig(fig, filename)
    return fig


def generate_full_report(all_metrics, filename="evaluation_report.csv"):
    df = pd.DataFrame(all_metrics)
    df.to_csv(os.path.join(OUTPUTS_DIR, filename), index=False)
    return df
