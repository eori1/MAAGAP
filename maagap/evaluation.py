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


def _risk_counts_bar(data, colors):
    """Return bar trace for a single risk label vector (0/1/2/3)."""
    labels_idx, counts = np.unique(data, return_counts=True)
    names = [RISK_LABELS[i] if i < len(RISK_LABELS) else str(i) for i in labels_idx]
    bar_colors = [colors[i] if i < len(colors) else "#95a5a6" for i in labels_idx]
    return go.Bar(
        x=names, y=counts, marker_color=bar_colors,
        text=counts, textposition="auto",
        hovertemplate="%{x}: %{y}<extra></extra>",
        showlegend=False,
    )


def plot_risk_distribution(y_true, y_pred, filename="risk_distribution.png", title=None):
    """Side-by-side bar chart: actual vs predicted risk distribution."""
    pred_title = "Predicted Risk Distribution"
    if title:
        pred_title = title
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Actual Risk Distribution", pred_title),
        horizontal_spacing=0.12,
    )

    colors = ["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"]

    fig.add_trace(_risk_counts_bar(y_true, colors), row=1, col=1)
    fig.add_trace(_risk_counts_bar(y_pred, colors), row=1, col=2)

    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_layout(width=900, height=450, **_PLOTLY_LAYOUT)
    _save_fig(fig, filename)
    return fig


def plot_risk_distribution_rf_xgb(y_true, y_rf_pred, y_xgb_pred, filename="risk_distribution_rf_xgb.png"):
    """One figure: Actual | RF predicted | XGBoost predicted (4-class risk)."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            "Actual Risk Distribution",
            "Predicted — Random Forest (Risk)",
            "Predicted — XGBoost (Risk)",
        ),
        horizontal_spacing=0.08,
    )
    colors = ["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"]
    for col, data in enumerate([y_true, y_rf_pred, y_xgb_pred], 1):
        fig.add_trace(_risk_counts_bar(data, colors), row=1, col=col)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_layout(width=1100, height=450, **_PLOTLY_LAYOUT)
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


# ---------------------------------------------------------------------------
# Objective 3 — Risk Scoring Engine visualisations
# ---------------------------------------------------------------------------

_TIER_COLORS = {
    "Low":      "#2ecc71",
    "Medium":   "#f39c12",
    "High":     "#e74c3c",
    "Critical": "#8e44ad",
}


def plot_risk_score_distribution(risk_scores, risk_tiers_arr, filename="risk_score_distribution.png"):
    """Histogram of continuous risk scores, colour-coded by assigned tier."""
    from plotly.subplots import make_subplots as _msp
    scores = np.asarray(risk_scores, dtype=float)
    tiers  = np.asarray(risk_tiers_arr, dtype=object)

    fig = go.Figure()
    for tier in RISK_LABELS:
        mask = (tiers == tier)
        if mask.sum() == 0:
            continue
        fig.add_trace(go.Histogram(
            x=scores[mask],
            name=tier,
            marker_color=_TIER_COLORS.get(tier, "#95a5a6"),
            opacity=0.75,
            nbinsx=30,
            hovertemplate=f"{tier}: %{{y}} projects<extra></extra>",
        ))

    # Add vertical threshold lines
    for label, threshold in [
        ("Low/Medium",   0.30),
        ("Medium/High",  0.70),
        ("High/Critical", 0.90),
    ]:
        fig.add_vline(
            x=threshold, line_dash="dash", line_color="#7f8c8d",
            annotation_text=f"{label}<br>{threshold:.2f}",
            annotation_position="top right",
            annotation_font_size=10,
        )

    fig.update_layout(
        title=dict(text="Risk Score Distribution by Tier — Test Set", x=0.5),
        xaxis_title="Risk Score",
        yaxis_title="Number of Projects",
        barmode="overlay",
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        **_PLOTLY_LAYOUT,
    )
    _save_fig(fig, filename)
    return fig


def plot_risk_tier_comparison(actual_tiers, predicted_tiers, filename="risk_tier_comparison.png"):
    """Side-by-side bar chart: actual vs predicted tier distribution."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Actual Risk Tiers (Ground Truth)", "Predicted Risk Tiers (Risk Engine)"),
        horizontal_spacing=0.12,
    )
    colors = [_TIER_COLORS.get(t, "#95a5a6") for t in RISK_LABELS]
    for col, tiers_arr in enumerate([actual_tiers, predicted_tiers], 1):
        counts = [np.sum(np.asarray(tiers_arr) == t) for t in RISK_LABELS]
        fig.add_trace(go.Bar(
            x=RISK_LABELS, y=counts,
            marker_color=colors,
            text=counts, textposition="auto",
            hovertemplate="%{x}: %{y} projects<extra></extra>",
            showlegend=False,
        ), row=1, col=col)

    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_layout(width=900, height=450, **_PLOTLY_LAYOUT)
    _save_fig(fig, filename)
    return fig


def plot_logic_consistency(consistency_result, n_projects, filename="logic_consistency.png"):
    """Gauge-style indicator showing zero (or near-zero) logic violations."""
    violations = consistency_result.get("violations", 0)
    pct_ok     = (1 - violations / max(n_projects, 1)) * 100

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct_ok,
        delta={"reference": 100, "valueformat": ".2f"},
        title={"text": "Risk Tier Logic Consistency (%)"},
        gauge={
            "axis":  {"range": [0, 100]},
            "bar":   {"color": "#2ecc71"},
            "steps": [
                {"range": [0,  80], "color": "#e74c3c"},
                {"range": [80, 95], "color": "#f39c12"},
                {"range": [95, 100], "color": "#2ecc71"},
            ],
            "threshold": {
                "line": {"color": "#2c3e50", "width": 4},
                "thickness": 0.75,
                "value": 100,
            },
        },
        number={"suffix": "%", "valueformat": ".2f"},
    ))
    fig.add_annotation(
        text=f"Violations: {violations} / {n_projects}",
        x=0.5, y=0.15, xref="paper", yref="paper",
        showarrow=False, font=dict(size=14),
    )
    fig.update_layout(height=400, width=500, **_PLOTLY_LAYOUT)
    _save_fig(fig, filename)
    return fig


# ---------------------------------------------------------------------------
# Objective 4 — Optimization visualisations
# ---------------------------------------------------------------------------

def plot_optimization_comparison(
    base_eff, lp_eff, improvement_pct, mc_results,
    filename="optimization_comparison.png",
):
    """Bar chart comparing Baseline vs LP allocation efficiency with MC ribbon."""
    labels = ["Baseline (Manual/Random)", "LP Optimized"]
    values = [base_eff, lp_eff]
    colors = ["#e74c3c", "#2ecc71"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Allocation Efficiency: Baseline vs LP",
            "Monte Carlo Improvement Distribution",
        ),
        horizontal_spacing=0.15,
    )

    # Panel 1 — Bar comparison
    fig.add_trace(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{v:.4f}" for v in values], textposition="auto",
        hovertemplate="%{x}<br>Efficiency: %{y:.4f}<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    # Improvement annotation
    fig.add_annotation(
        text=f"+{improvement_pct:.1f}% improvement",
        x=1, y=lp_eff * 1.02,
        xref="x", yref="y",
        showarrow=True, arrowhead=2,
        ax=-60, ay=-30,
        font=dict(color="#2ecc71", size=13, family="Segoe UI"),
        row=1, col=1,
    )

    # Panel 2 — MC histogram of improvements
    if mc_results and mc_results.get("n_successful", 0) > 0:
        mean_i  = mc_results["mean_improvement_pct"]
        std_i   = mc_results["std_improvement_pct"]
        # Approximate normal distribution for display
        x_range = np.linspace(
            max(0, mean_i - 4 * std_i),
            mean_i + 4 * std_i, 200
        )
        from scipy.stats import norm as _norm
        y_vals  = _norm.pdf(x_range, mean_i, max(std_i, 1e-6))
        fig.add_trace(go.Scatter(
            x=x_range, y=y_vals, mode="lines", fill="tozeroy",
            line=dict(color="#3498db", width=2),
            fillcolor="rgba(52,152,219,0.25)",
            name="MC distribution",
            hovertemplate="Improvement: %{x:.1f}%<extra></extra>",
        ), row=1, col=2)
        fig.add_vline(
            x=15.0, line_dash="dash", line_color="#e74c3c",
            annotation_text="15% target",
            annotation_position="top left",
            annotation_font=dict(color="#e74c3c"),
            row=1, col=2,
        )
        fig.add_vline(
            x=mean_i, line_dash="dot", line_color="#2ecc71",
            annotation_text=f"Mean {mean_i:.1f}%",
            annotation_position="top right",
            annotation_font=dict(color="#2ecc71"),
            row=1, col=2,
        )
        fig.update_xaxes(title_text="Efficiency Improvement (%)", row=1, col=2)
        fig.update_yaxes(title_text="Probability Density", row=1, col=2)

    fig.update_yaxes(title_text="Avg Captured Risk Score", row=1, col=1)
    fig.update_layout(width=1100, height=480, **_PLOTLY_LAYOUT)
    _save_fig(fig, filename)
    return fig


def plot_lp_selection_profile(
    risk_scores, risk_tiers_arr, lp_selected_idx, base_selected_idx,
    filename="lp_selection_profile.png",
):
    """Scatter: risk score vs project index, highlighting LP vs baseline selections."""
    scores = np.asarray(risk_scores, dtype=float)
    tiers  = np.asarray(risk_tiers_arr, dtype=object)
    n      = len(scores)
    proj_x = np.arange(n)

    lp_set   = set(lp_selected_idx.tolist())
    base_set = set(base_selected_idx.tolist())

    def _mask(condition):
        return np.array([condition(i) for i in range(n)])

    lp_only   = _mask(lambda i: i in lp_set and i not in base_set)
    base_only = _mask(lambda i: i in base_set and i not in lp_set)
    both      = _mask(lambda i: i in lp_set and i in base_set)
    neither   = _mask(lambda i: i not in lp_set and i not in base_set)

    fig = go.Figure()
    for mask, label, color, sym in [
        (neither,  "Not Selected",       "#bdc3c7", "circle"),
        (both,     "Selected by Both",    "#f39c12", "diamond"),
        (base_only,"Baseline Only",       "#e74c3c", "square"),
        (lp_only,  "LP Only",             "#2ecc71", "star"),
    ]:
        if mask.sum() == 0:
            continue
        fig.add_trace(go.Scatter(
            x=proj_x[mask], y=scores[mask], mode="markers",
            name=label,
            marker=dict(color=color, size=6, symbol=sym, opacity=0.7),
            customdata=tiers[mask],
            hovertemplate="Project %{x}<br>Risk: %{y:.4f}<br>Tier: %{customdata}<extra></extra>",
        ))

    # Threshold lines
    for thr, lbl in [(0.30, "Low/Med"), (0.70, "Med/High"), (0.90, "High/Crit")]:
        fig.add_hline(y=thr, line_dash="dash", line_color="#95a5a6",
                      annotation_text=lbl, annotation_position="right")

    fig.update_layout(
        title=dict(text="LP vs Baseline Selection Profile — Test Projects", x=0.5),
        xaxis_title="Project Index",
        yaxis_title="Risk Score",
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
        width=1100, height=500,
        **_PLOTLY_LAYOUT,
    )
    _save_fig(fig, filename)
    return fig
