"""Evaluation metrics, visualisations (Plotly), and reporting for all models.

Covers the metrics specified in the thesis Objective 2:
  Accuracy, Precision, Recall, F1-Score, AUC-ROC, MAE
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Any, Optional, Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_absolute_error,
    confusion_matrix, roc_curve,
)

from .config import OUTPUTS_DIR, RISK_LABELS
from .logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    """Computes evaluation metrics across the MAAGAP predictive framework."""

    @staticmethod
    def _safe_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
        try:
            if len(np.unique(y_true)) < 2:
                return float("nan")
            return roc_auc_score(y_true, y_proba)
        except ValueError:
            return float("nan")

    @staticmethod
    def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray, metric: str = "f1") -> float:
        best_t, best_score = 0.5, 0.0
        for t in np.arange(0.20, 0.80, 0.01):
            preds = (y_proba >= t).astype(int)
            score = f1_score(y_true, preds, zero_division=0)
            if score > best_score:
                best_t, best_score = t, score
        return round(best_t, 2)

    @classmethod
    def binary_metrics(cls, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None, label: str = "Model") -> Dict[str, Any]:
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc_val = cls._safe_auc(y_true, y_proba) if y_proba is not None else float("nan")
        return {
            "Model": label,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "AUC-ROC": round(auc_val, 4),
        }

    @staticmethod
    def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "Model") -> Dict[str, Any]:
        mae = mean_absolute_error(y_true, y_pred)
        return {"Model": label, "MAE": round(mae, 4)}

    @staticmethod
    def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "Model") -> Dict[str, Any]:
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

    @staticmethod
    def generate_full_report(all_metrics: List[Dict[str, Any]], filename: str = "evaluation_report.csv") -> pd.DataFrame:
        df = pd.DataFrame(all_metrics)
        df.to_csv(os.path.join(OUTPUTS_DIR, filename), index=False)
        return df

    @staticmethod
    def compare_tuning_impact(baseline_metrics: List[Dict[str, Any]], tuned_metrics: List[Dict[str, Any]]) -> pd.DataFrame:
        """Combine baseline (untuned) and tuned metrics into side-by-side comparison DataFrame with Δ improvements."""
        df_base = pd.DataFrame(baseline_metrics).copy()
        df_tuned = pd.DataFrame(tuned_metrics).copy()
        
        df_base.rename(columns={col: f"{col} (Untuned)" for col in df_base.columns if col != "Model"}, inplace=True)
        df_tuned.rename(columns={col: f"{col} (Tuned)" for col in df_tuned.columns if col != "Model"}, inplace=True)

        df_merged = pd.merge(df_base, df_tuned, on="Model", how="outer")
        
        for col in ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]:
            u_col = f"{col} (Untuned)"
            t_col = f"{col} (Tuned)"
            if u_col in df_merged.columns and t_col in df_merged.columns:
                df_merged[f"{col} Δ"] = round(df_merged[t_col] - df_merged[u_col], 4)
                
        out_path = os.path.join(OUTPUTS_DIR, "tuning_comparison_report.csv")
        df_merged.to_csv(out_path, index=False)
        logger.info(f"Saved hyperparameter tuning comparison report to {out_path}")
        return df_merged



class Visualizer:
    """Generates and saves Plotly visualizations."""
    
    MODEL_COLORS = {
        "Random Forest": "#2ecc71",
        "XGBoost": "#3498db",
        "LSTM": "#e74c3c",
        "Meta-Ensemble": "#9b59b6",
        "RF Risk": "#2ecc71",
        "XGB Risk": "#3498db",
    }
    
    TIER_COLORS = {
        "Low":      "#2ecc71",
        "Medium":   "#f39c12",
        "High":     "#e74c3c",
        "Critical": "#8e44ad",
    }
    
    _PLOTLY_LAYOUT = dict(
        template="plotly_white",
        font=dict(family="Segoe UI, sans-serif", size=13),
        margin=dict(l=60, r=30, t=60, b=50),
    )

    @staticmethod
    def _save_fig(fig: go.Figure, filename: str) -> None:
        """Save as interactive HTML and static PNG."""
        base, _ = os.path.splitext(filename)
        out_html = os.path.join(OUTPUTS_DIR, base + ".html")
        out_png = os.path.join(OUTPUTS_DIR, filename)
        
        fig.write_html(out_html, include_plotlyjs="cdn")
        try:
            fig.write_image(out_png, scale=2, width=900, height=600)
        except Exception as e:
            logger.debug(f"Failed to write image {filename}: {e}")
            pass

    @classmethod
    def plot_confusion_matrix(cls, y_true: np.ndarray, y_pred: np.ndarray, labels: List[str], title: str, filename: str) -> go.Figure:
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
            **cls._PLOTLY_LAYOUT,
        )
        cls._save_fig(fig, filename)
        return fig

    @classmethod
    def plot_roc_curves(cls, curves_data: List[Tuple[np.ndarray, np.ndarray, float, str]], filename: str = "roc_curves_delay.png") -> go.Figure:
        """curves_data: list of (fpr, tpr, auc_val, label)."""
        fig = go.Figure()

        for fpr, tpr, auc_val, label in curves_data:
            color = cls.MODEL_COLORS.get(label, "#7f8c8d")
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
            **cls._PLOTLY_LAYOUT,
        )
        cls._save_fig(fig, filename)
        return fig

    @classmethod
    def plot_feature_importance(cls, importances: np.ndarray, feature_names: List[str], title: str, filename: str, top_n: int = 20) -> go.Figure:
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
            **cls._PLOTLY_LAYOUT,
        )
        cls._save_fig(fig, filename)
        return fig

    @classmethod
    def plot_training_history(cls, history: Any, filename: str = "lstm_training_history.png") -> go.Figure:
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
            **cls._PLOTLY_LAYOUT,
        )
        cls._save_fig(fig, filename)
        return fig

    @classmethod
    def plot_all_training_histories(cls, lstm_history: Any, xgb_evals: Dict, rf_lc: Tuple, meta_lc: Tuple, filename: str = "all_training_histories.png") -> go.Figure:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "LSTM Loss (Epochs)",
                "XGBoost Log-Loss (Boosting Rounds)",
                "Random Forest Learning Curve",
                "Meta-Ensemble Learning Curve"
            ),
            horizontal_spacing=0.12,
            vertical_spacing=0.25
        )

        # 1. LSTM
        epochs = list(range(1, len(lstm_history.history["loss"]) + 1))
        fig.add_trace(go.Scatter(x=epochs, y=lstm_history.history["loss"], mode="lines", name="LSTM Train", line=dict(color="#e74c3c")), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=lstm_history.history["val_loss"], mode="lines", name="LSTM Val", line=dict(color="#3498db", dash="dash")), row=1, col=1)

        # 2. XGBoost
        if xgb_evals and "validation_0" in xgb_evals and "validation_1" in xgb_evals:
            xgb_train = xgb_evals["validation_0"]["logloss"]
            xgb_val = xgb_evals["validation_1"]["logloss"]
            rounds = list(range(1, len(xgb_train) + 1))
            fig.add_trace(go.Scatter(x=rounds, y=xgb_train, mode="lines", name="XGB Train", line=dict(color="#e74c3c")), row=1, col=2)
            fig.add_trace(go.Scatter(x=rounds, y=xgb_val, mode="lines", name="XGB Val", line=dict(color="#3498db", dash="dash")), row=1, col=2)

        # 3. RF Learning Curve
        train_sizes_rf, train_scores_rf, val_scores_rf = rf_lc
        fig.add_trace(go.Scatter(x=train_sizes_rf, y=np.mean(train_scores_rf, axis=1), mode="lines+markers", name="RF Train Score", line=dict(color="#2ecc71")), row=2, col=1)
        fig.add_trace(go.Scatter(x=train_sizes_rf, y=np.mean(val_scores_rf, axis=1), mode="lines+markers", name="RF Val Score", line=dict(color="#9b59b6", dash="dash")), row=2, col=1)

        # 4. Meta-Ensemble Learning Curve
        train_sizes_meta, train_scores_meta, val_scores_meta = meta_lc
        fig.add_trace(go.Scatter(x=train_sizes_meta, y=np.mean(train_scores_meta, axis=1), mode="lines+markers", name="Meta Train Score", line=dict(color="#2ecc71")), row=2, col=2)
        fig.add_trace(go.Scatter(x=train_sizes_meta, y=np.mean(val_scores_meta, axis=1), mode="lines+markers", name="Meta Val Score", line=dict(color="#9b59b6", dash="dash")), row=2, col=2)

        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_yaxes(title_text="Log Loss", row=1, col=1)
        fig.update_xaxes(title_text="Boosting Round", row=1, col=2)
        fig.update_yaxes(title_text="Log Loss", row=1, col=2)
        fig.update_xaxes(title_text="Training Examples", row=2, col=1)
        fig.update_yaxes(title_text="F1 Score", row=2, col=1)
        fig.update_xaxes(title_text="Training Examples", row=2, col=2)
        fig.update_yaxes(title_text="F1 Score", row=2, col=2)

        fig.update_layout(
            height=700, width=1000,
            title=dict(text="Model Training Histories & Learning Curves", x=0.5),
            **cls._PLOTLY_LAYOUT
        )
        cls._save_fig(fig, filename)
        return fig

    @staticmethod
    def _risk_counts_bar(data: np.ndarray, colors: List[str]) -> go.Bar:
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

    @classmethod
    def plot_risk_distribution(cls, y_true: np.ndarray, y_pred: np.ndarray, filename: str = "risk_distribution.png", title: Optional[str] = None) -> go.Figure:
        pred_title = "Predicted Risk Distribution"
        if title:
            pred_title = title
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Actual Risk Distribution", pred_title),
            horizontal_spacing=0.12,
        )

        colors = ["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"]

        fig.add_trace(cls._risk_counts_bar(y_true, colors), row=1, col=1)
        fig.add_trace(cls._risk_counts_bar(y_pred, colors), row=1, col=2)

        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_layout(width=900, height=450, **cls._PLOTLY_LAYOUT)
        cls._save_fig(fig, filename)
        return fig

    @classmethod
    def plot_risk_distribution_rf_xgb(cls, y_true: np.ndarray, y_rf_pred: np.ndarray, y_xgb_pred: np.ndarray, filename: str = "risk_distribution_rf_xgb.png") -> go.Figure:
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
            fig.add_trace(cls._risk_counts_bar(data, colors), row=1, col=col)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_layout(width=1100, height=450, **cls._PLOTLY_LAYOUT)
        cls._save_fig(fig, filename)
        return fig

    @classmethod
    def plot_model_comparison(cls, metrics_list: List[Dict[str, Any]], filename: str = "model_comparison.png") -> go.Figure:
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
            **cls._PLOTLY_LAYOUT,
        )
        cls._save_fig(fig, filename)
        return fig

    @classmethod
    def plot_risk_score_distribution(cls, risk_scores: np.ndarray, risk_tiers_arr: np.ndarray, filename: str = "risk_score_distribution.png") -> go.Figure:
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
                marker_color=cls.TIER_COLORS.get(tier, "#95a5a6"),
                opacity=0.75,
                nbinsx=30,
                hovertemplate=f"{tier}: %{{y}} projects<extra></extra>",
            ))

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
            **cls._PLOTLY_LAYOUT,
        )
        cls._save_fig(fig, filename)
        return fig

    @classmethod
    def plot_risk_tier_comparison(cls, actual_tiers: np.ndarray, predicted_tiers: np.ndarray, filename: str = "risk_tier_comparison.png") -> go.Figure:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Actual Risk Tiers (Ground Truth)", "Predicted Risk Tiers (Risk Engine)"),
            horizontal_spacing=0.12,
        )
        colors = [cls.TIER_COLORS.get(t, "#95a5a6") for t in RISK_LABELS]
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
        fig.update_layout(width=900, height=450, **cls._PLOTLY_LAYOUT)
        cls._save_fig(fig, filename)
        return fig

    @classmethod
    def plot_logic_consistency(cls, consistency_result: Dict[str, int], n_projects: int, filename: str = "logic_consistency.png") -> go.Figure:
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
        fig.update_layout(height=400, width=500, **cls._PLOTLY_LAYOUT)
        cls._save_fig(fig, filename)
        return fig

    @classmethod
    def plot_optimization_comparison(cls, base_eff: float, lp_eff: float, improvement_pct: float, mc_results: Dict[str, Any], filename: str = "optimization_comparison.png") -> go.Figure:
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

        fig.add_trace(go.Bar(
            x=labels, y=values,
            marker_color=colors,
            text=[f"{v:.4f}" for v in values], textposition="auto",
            hovertemplate="%{x}<br>Efficiency: %{y:.4f}<extra></extra>",
            showlegend=False,
        ), row=1, col=1)

        fig.add_annotation(
            text=f"+{improvement_pct:.1f}% improvement",
            x=1, y=lp_eff * 1.02,
            xref="x", yref="y",
            showarrow=True, arrowhead=2,
            ax=-60, ay=-30,
            font=dict(color="#2ecc71", size=13, family="Segoe UI"),
            row=1, col=1,
        )

        if mc_results and mc_results.get("n_successful", 0) > 0:
            mean_i  = mc_results["mean_improvement_pct"]
            std_i   = mc_results["std_improvement_pct"]
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
        fig.update_layout(width=1100, height=480, **cls._PLOTLY_LAYOUT)
        cls._save_fig(fig, filename)
        return fig

    @classmethod
    def plot_lp_selection_profile(cls, risk_scores: np.ndarray, risk_tiers_arr: np.ndarray, lp_selected_idx: np.ndarray, base_selected_idx: np.ndarray, filename: str = "lp_selection_profile.png") -> go.Figure:
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

        for thr, lbl in [(0.30, "Low/Med"), (0.70, "Med/High"), (0.90, "High/Crit")]:
            fig.add_hline(y=thr, line_dash="dash", line_color="#95a5a6",
                          annotation_text=lbl, annotation_position="right")

        fig.update_layout(
            title=dict(text="LP vs Baseline Selection Profile — Test Projects", x=0.5),
            xaxis_title="Project Index",
            yaxis_title="Risk Score",
            legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
            width=1100, height=500,
            **cls._PLOTLY_LAYOUT,
        )
        cls._save_fig(fig, filename)
        return fig

    @classmethod
    def plot_tuning_comparison(cls, comparison_df: pd.DataFrame, metric: str = "AUC-ROC", filename: str = "tuning_comparison_chart.png") -> go.Figure:
        """Grouped bar chart comparing untuned vs tuned model performance."""
        fig = go.Figure()
        
        models = comparison_df["Model"].values
        u_col = f"{metric} (Untuned)"
        t_col = f"{metric} (Tuned)"
        
        if u_col in comparison_df.columns:
            fig.add_trace(go.Bar(
                x=models, y=comparison_df[u_col].values, name="Untuned (Default Params)",
                marker_color="#95a5a6", text=comparison_df[u_col].values, textposition="auto"
            ))
        if t_col in comparison_df.columns:
            fig.add_trace(go.Bar(
                x=models, y=comparison_df[t_col].values, name="Tuned (RandomizedSearchCV)",
                marker_color="#2ecc71", text=comparison_df[t_col].values, textposition="auto"
            ))

        fig.update_layout(
            title=dict(text=f"Untuned vs. Tuned Model Comparison ({metric})", x=0.5),
            xaxis_title="Model Architecture",
            yaxis_title=metric,
            barmode="group",
            legend=dict(x=0.01, y=0.99),
            **cls._PLOTLY_LAYOUT,
        )
        cls._save_fig(fig, filename)
        return fig

    @classmethod
    def plot_training_curves_overlay(cls, history_untuned: dict, history_tuned: dict, model_name: str = "LSTM", filename: str = "training_history_comparison.png") -> go.Figure:
        """Overlay loss and accuracy curves for untuned vs tuned model training."""
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss Comparison", "Accuracy Comparison"))

        if history_untuned:
            fig.add_trace(go.Scatter(y=history_untuned.get("loss", []), name="Untuned Train Loss", line=dict(color="#95a5a6", dash="dash")), row=1, col=1)
            fig.add_trace(go.Scatter(y=history_untuned.get("val_loss", []), name="Untuned Val Loss", line=dict(color="#7f8c8d", dash="dash")), row=1, col=1)
            fig.add_trace(go.Scatter(y=history_untuned.get("accuracy", []), name="Untuned Train Acc", line=dict(color="#95a5a6", dash="dash")), row=1, col=2)
            fig.add_trace(go.Scatter(y=history_untuned.get("val_accuracy", []), name="Untuned Val Acc", line=dict(color="#7f8c8d", dash="dash")), row=1, col=2)

        if history_tuned:
            fig.add_trace(go.Scatter(y=history_tuned.get("loss", []), name="Tuned Train Loss", line=dict(color="#e74c3c")), row=1, col=1)
            fig.add_trace(go.Scatter(y=history_tuned.get("val_loss", []), name="Tuned Val Loss", line=dict(color="#c0392b")), row=1, col=1)
            fig.add_trace(go.Scatter(y=history_tuned.get("accuracy", []), name="Tuned Train Acc", line=dict(color="#2ecc71")), row=1, col=2)
            fig.add_trace(go.Scatter(y=history_tuned.get("val_accuracy", []), name="Tuned Val Acc", line=dict(color="#27ae60")), row=1, col=2)

        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=2)
        fig.update_layout(title=dict(text=f"{model_name} Training History: Untuned vs. Tuned", x=0.5), width=1100, height=480, **cls._PLOTLY_LAYOUT)
        cls._save_fig(fig, filename)
        return fig

# Backward compatibility functions
def find_optimal_threshold(y_true, y_proba, metric="f1"):
    return Evaluator.find_optimal_threshold(y_true, y_proba, metric)

def binary_metrics(y_true, y_pred, y_proba=None, label="Model"):
    return Evaluator.binary_metrics(y_true, y_pred, y_proba, label)

def regression_metrics(y_true, y_pred, label="Model"):
    return Evaluator.regression_metrics(y_true, y_pred, label)

def multiclass_metrics(y_true, y_pred, label="Model"):
    return Evaluator.multiclass_metrics(y_true, y_pred, label)

def generate_full_report(all_metrics, filename="evaluation_report.csv"):
    return Evaluator.generate_full_report(all_metrics, filename)

def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    return Visualizer.plot_confusion_matrix(y_true, y_pred, labels, title, filename)

def plot_roc_curves(curves_data, filename="roc_curves_delay.png"):
    return Visualizer.plot_roc_curves(curves_data, filename)

def plot_feature_importance(importances, feature_names, title, filename, top_n=20):
    return Visualizer.plot_feature_importance(importances, feature_names, title, filename, top_n)

def plot_training_history(history, filename="lstm_training_history.png"):
    return Visualizer.plot_training_history(history, filename)

def plot_all_training_histories(lstm_history, xgb_evals, rf_lc, meta_lc, filename="all_training_histories.png"):
    return Visualizer.plot_all_training_histories(lstm_history, xgb_evals, rf_lc, meta_lc, filename)

def plot_risk_distribution(y_true, y_pred, filename="risk_distribution.png", title=None):
    return Visualizer.plot_risk_distribution(y_true, y_pred, filename, title)

def plot_risk_distribution_rf_xgb(y_true, y_rf_pred, y_xgb_pred, filename="risk_distribution_rf_xgb.png"):
    return Visualizer.plot_risk_distribution_rf_xgb(y_true, y_rf_pred, y_xgb_pred, filename)

def plot_model_comparison(metrics_list, filename="model_comparison.png"):
    return Visualizer.plot_model_comparison(metrics_list, filename)

def plot_risk_score_distribution(risk_scores, risk_tiers_arr, filename="risk_score_distribution.png"):
    return Visualizer.plot_risk_score_distribution(risk_scores, risk_tiers_arr, filename)

def plot_risk_tier_comparison(actual_tiers, predicted_tiers, filename="risk_tier_comparison.png"):
    return Visualizer.plot_risk_tier_comparison(actual_tiers, predicted_tiers, filename)

def plot_logic_consistency(consistency_result, n_projects, filename="logic_consistency.png"):
    return Visualizer.plot_logic_consistency(consistency_result, n_projects, filename)

def plot_optimization_comparison(base_eff, lp_eff, improvement_pct, mc_results, filename="optimization_comparison.png"):
    return Visualizer.plot_optimization_comparison(base_eff, lp_eff, improvement_pct, mc_results, filename)

def plot_lp_selection_profile(risk_scores, risk_tiers_arr, lp_selected_idx, base_selected_idx, filename="lp_selection_profile.png"):
    return Visualizer.plot_lp_selection_profile(risk_scores, risk_tiers_arr, lp_selected_idx, base_selected_idx, filename)

# Expose _save_fig
_save_fig = Visualizer._save_fig
