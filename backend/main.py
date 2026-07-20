"""Orchestrates the MAAGAP framework execution.

Objectives covered:
1. Data pipeline (Real dataset mapping -> Synthetic generation -> Feature Engineering & Preprocessing Pipeline)
2. Predictive Framework (RF, XGBoost, LSTM, Meta-Ensemble) + Tuning Comparison
3. Dynamic Risk Scoring Engine + Threshold Boundary Verification
4. Resource Allocation Optimization (PuLP LP Solver + Baseline Comparison)
5. Explainable AI (SHAP attributions & visualizations)
6. ERD-aligned Relational Table Export
"""

import os
import sys
import argparse
import traceback
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.model_selection import learning_curve

from maagap.config import SEED, OUTPUTS_DIR, DATA_PROCESSED_DIR
from maagap.logger import get_logger, banner
from maagap.data_preprocessing import DataPreprocessor
from maagap.synthetic_generator import SyntheticDataGenerator
from maagap.feature_engineering import FeatureEngineer, split_data
from maagap.preprocessing_pipeline import MAAGAPPreprocessor
from maagap.models import TreeModelTrainer, LSTMTrainer, MetaEnsembleTrainer, RegressionModelTrainer
from maagap.risk_scoring import compute_all_risk_scores, get_risk_tier, check_logic_consistency
from maagap.optimization import InspectorAssignmentOptimizer
from maagap.explainability import ExplanationService
from maagap.evaluation import Evaluator, Visualizer

logger = get_logger("MAAGAP.Main")


class MainPipeline:
    """Encapsulates the end-to-end execution of the MAAGAP framework."""

    def __init__(self, tune_models: bool = False, compare_tuning: bool = False):
        self.tune_models = tune_models
        self.compare_tuning = compare_tuning
        self.evaluator = Evaluator()
        self.visualizer = Visualizer()
        self.feature_engineer = FeatureEngineer()
        self.preprocessor_pipeline = MAAGAPPreprocessor()
        self.optimizer = InspectorAssignmentOptimizer()
        self.metrics_list = []

    def _train_and_evaluate_pass(self, X_static, X_temporal, y_delay, df_projects, train_idx, val_idx, test_idx, tune: bool, label_prefix: str = "") -> dict:
        """Run training pass for RF, XGB, LSTM, Meta and return predictions & metrics."""
        static_cols = self.preprocessor_pipeline.engineer.static_feature_names
        
        # 1. Random Forest
        rf_model = TreeModelTrainer.train_random_forest(
            X_static[train_idx], y_delay[train_idx], task="delay", tune=tune
        )
        rf_prob_test = rf_model.predict_proba(X_static[test_idx])[:, 1]
        rf_pred_test = rf_model.predict(X_static[test_idx])

        # 2. XGBoost
        xgb_model, xgb_evals = TreeModelTrainer.train_xgboost(
            X_static[train_idx], y_delay[train_idx], 
            X_static[val_idx], y_delay[val_idx],
            task="delay", tune=tune
        )
        xgb_prob_test = xgb_model.predict_proba(X_static[test_idx])[:, 1]
        xgb_pred_test = xgb_model.predict(X_static[test_idx])

        # 3. LSTM
        lstm_model, lstm_history, _ = LSTMTrainer.train_lstm(
            X_temporal[train_idx], y_delay[train_idx],
            X_temporal[val_idx], y_delay[val_idx],
            task="delay", tune=tune
        )
        lstm_prob_test = lstm_model.predict(X_temporal[test_idx], verbose=0).flatten()
        lstm_pred_test = (lstm_prob_test >= 0.5).astype(int)

        # 4. Meta Ensemble Stacking
        rf_prob_train = rf_model.predict_proba(X_static[train_idx])[:, 1]
        xgb_prob_train = xgb_model.predict_proba(X_static[train_idx])[:, 1]
        lstm_prob_train = lstm_model.predict(X_temporal[train_idx], verbose=0).flatten()

        meta_model = MetaEnsembleTrainer.train_meta_ensemble(
            rf_prob_train, xgb_prob_train, lstm_prob_train, y_delay[train_idx],
            artifact_name="meta_ensemble_delay.pkl"
        )
        meta_pred_test, meta_prob_full = MetaEnsembleTrainer.predict_meta(
            meta_model, rf_prob_test, xgb_prob_test, lstm_prob_test
        )
        meta_prob_test = meta_prob_full[:, 1]

        # Compute test metrics
        models_data = [
            ("Random Forest" + label_prefix, rf_pred_test, rf_prob_test),
            ("XGBoost" + label_prefix, xgb_pred_test, xgb_prob_test),
            ("LSTM" + label_prefix, lstm_pred_test, lstm_prob_test),
            ("Meta-Ensemble" + label_prefix, meta_pred_test, meta_prob_test),
        ]
        
        metrics = []
        for name, preds, probs in models_data:
            m_bin = self.evaluator.binary_metrics(y_delay[test_idx], preds, probs, label=name)
            metrics.append(m_bin)

        return {
            "rf_model": rf_model, "xgb_model": xgb_model, "lstm_model": lstm_model, "meta_model": meta_model,
            "rf_prob_test": rf_prob_test, "xgb_prob_test": xgb_prob_test, "lstm_prob_test": lstm_prob_test, "meta_prob_test": meta_prob_test,
            "rf_pred_test": rf_pred_test, "xgb_pred_test": xgb_pred_test, "lstm_pred_test": lstm_pred_test, "meta_pred_test": meta_pred_test,
            "rf_prob_train": rf_prob_train, "xgb_prob_train": xgb_prob_train, "lstm_prob_train": lstm_prob_train,
            "lstm_history": lstm_history, "xgb_evals": xgb_evals,
            "metrics": metrics,
        }

    def _export_frontend_assignments(self, df_assignments: pd.DataFrame, inspectors_df: pd.DataFrame, n_projects: int) -> None:
        """Write the LP deployment schedule as JSON for the Next.js dashboard.

        Skipped silently when the frontend directory is not present (e.g.
        backend-only checkouts)."""
        frontend_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "public", "data")
        if not os.path.isdir(frontend_data_dir):
            logger.info("Frontend data directory not found; skipping assignments JSON export.")
            return

        capacities = InspectorAssignmentOptimizer.compute_capacities(inspectors_df)
        inspectors_payload = []
        for j, (_, row) in enumerate(inspectors_df.iterrows()):
            insp_rows = df_assignments[df_assignments["inspector_id"] == row["inspector_id"]]
            projects = [
                {
                    "projectId": r["project_id"],
                    "name": r["project_id"],
                    "location": r["location"],
                    "type": r["project_type"],
                    "riskScore": float(r["risk_score"]),
                    "riskTier": r["risk_tier"],
                    "priority": r["priority"],
                    "urgency": r["urgency"],
                }
                for _, r in insp_rows.iterrows()
            ]
            inspectors_payload.append({
                "id": row["inspector_id"],
                "name": row["inspector_name"],
                "availability": row["availability_status"],
                "vehicleAccess": bool(row["vehicle_access"]),
                "capacity": int(capacities[j]),
                "currentWorkload": int(row["current_workload"]),
                "totalProjects": len(projects),
                "projects": projects,
            })

        payload = {
            "generatedAt": pd.Timestamp.now().isoformat(),
            "solver": "PuLP CBC (Integer LP)",
            "totalProjects": n_projects,
            "assignedProjects": int(len(df_assignments)),
            "unassignedProjects": int(n_projects - len(df_assignments)),
            "criticalAssignments": int((df_assignments["risk_tier"] == "Critical").sum()),
            "inspectors": inspectors_payload,
        }
        out_path = os.path.join(frontend_data_dir, "assignments.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Exported frontend assignments JSON to {out_path}")

    @staticmethod
    def _frontend_data_dir() -> str:
        return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "public", "data")

    def _export_frontend_extras(
        self,
        df_test_projects: pd.DataFrame,
        df_quarterly: pd.DataFrame,
        inspectors_df: pd.DataFrame,
        df_assignments: pd.DataFrame,
        risk_scores_test: np.ndarray,
        risk_tiers_test: np.ndarray,
        predicted_delay_days: np.ndarray,
        df_prev_predictions: "pd.DataFrame | None",
    ) -> None:
        """Export timeline, inspection-report, roster, and alert data for the
        Next.js dashboard. Skipped when the frontend directory is absent."""
        data_dir = self._frontend_data_dir()
        if not os.path.isdir(data_dir):
            logger.info("Frontend data directory not found; skipping extras export.")
            return

        test_ids = df_test_projects["project_id"].values
        risk_by_id = dict(zip(test_ids, risk_scores_test))
        tier_by_id = dict(zip(test_ids, risk_tiers_test))
        delay_days_by_id = dict(zip(test_ids, predicted_delay_days))

        # --- timeline.json: actual-vs-scheduled deviation per test project ---
        timeline = []
        for _, p in df_test_projects.iterrows():
            status = "Delayed" if p["is_delayed"] == 1 else ("Ongoing" if p["year"] == max(df_test_projects["year"]) else "Completed")
            timeline.append({
                "id": p["project_id"],
                "name": p["project_id"],
                "location": p["location"],
                "type": p["project_type"],
                "year": int(p["year"]),
                "startDate": p["start_date"],
                "plannedEndDate": p["planned_end_date"],
                "plannedMonths": int(p["planned_duration_months"]),
                "actualDelayDays": int(p["delay_days"]),
                "predictedDelayDays": round(float(delay_days_by_id[p["project_id"]]), 1),
                "riskTier": tier_by_id[p["project_id"]],
                "status": status,
            })
        with open(os.path.join(data_dir, "timeline.json"), "w", encoding="utf-8") as f:
            json.dump(timeline, f, indent=2)

        # --- reports.json: latest quarterly inspection log per test project ---
        insp_name_by_id = dict(zip(inspectors_df["inspector_id"], inspectors_df["inspector_name"]))
        df_q_test = df_quarterly[df_quarterly["project_id"].isin(test_ids)]
        latest_q = df_q_test.sort_values("quarter").groupby("project_id").tail(1)
        start_by_id = dict(zip(df_test_projects["project_id"], pd.to_datetime(df_test_projects["start_date"])))
        reports = []
        for _, q in latest_q.iterrows():
            pid = q["project_id"]
            slippage = float(q["slippage_pct"])
            if slippage > 20:
                r_status = "Flagged"
            elif slippage > 5:
                r_status = "Pending Review"
            else:
                r_status = "Validated"
            # deterministic inspector attribution mirrors tbl_inspection_log round-robin
            insp_idx = int(q.name) % len(inspectors_df)
            insp_id = inspectors_df["inspector_id"].values[insp_idx]
            report_date = start_by_id[pid] + pd.DateOffset(months=3 * int(q["quarter"]))
            reports.append({
                "projectId": pid,
                "quarter": int(q["quarter"]),
                "totalQuarters": int(q["total_quarters"]),
                "plannedProgress": float(q["planned_progress_pct"]),
                "actualProgress": float(q["actual_progress_pct"]),
                "slippage": slippage,
                "issues": int(q["issues_count"]),
                "date": report_date.strftime("%Y-%m-%d"),
                "status": r_status,
                "inspectorId": insp_id,
                "inspectorName": insp_name_by_id[insp_id],
                "riskTier": tier_by_id.get(pid, "Low"),
            })
        reports.sort(key=lambda r: r["date"], reverse=True)
        with open(os.path.join(data_dir, "reports.json"), "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2)

        # --- inspectors.json: roster + LP-assigned workload (Users page) ---
        capacities = InspectorAssignmentOptimizer.compute_capacities(inspectors_df)
        assigned_counts = df_assignments["inspector_id"].value_counts().to_dict()
        roster = []
        for j, (_, r) in enumerate(inspectors_df.iterrows()):
            handle = r["inspector_name"].split(".")[-1].strip().lower().replace(" ", "")
            roster.append({
                "id": r["inspector_id"],
                "name": r["inspector_name"],
                "email": f"{handle}@iloilo.gov.ph",
                "position": "Project Inspector",
                "role": "Inspector",
                "status": "Active" if str(r["availability_status"]).lower() == "available" else "On Duty",
                "vehicleAccess": bool(r["vehicle_access"]),
                "capacity": int(capacities[j]),
                "assigned": int(assigned_counts.get(r["inspector_id"], 0)),
            })
        with open(os.path.join(data_dir, "inspectors.json"), "w", encoding="utf-8") as f:
            json.dump(roster, f, indent=2)

        # --- alerts.json: tier transitions vs previous run + critical alerts ---
        tier_rank = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
        alerts = []
        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        if df_prev_predictions is not None and "risk_tier" in df_prev_predictions.columns:
            prev_tier = dict(zip(df_prev_predictions["project_id"], df_prev_predictions["risk_tier"]))
            for pid in test_ids:
                old, new = prev_tier.get(pid), tier_by_id[pid]
                if old is not None and tier_rank.get(new, 0) > tier_rank.get(old, 0):
                    alerts.append({
                        "type": "TIER_ESCALATION",
                        "projectId": pid,
                        "fromTier": old,
                        "toTier": new,
                        "riskScore": round(float(risk_by_id[pid]), 4),
                        "message": f"{pid} escalated from {old} to {new} risk",
                        "date": today,
                    })
        for pid in test_ids:
            if tier_by_id[pid] == "Critical":
                alerts.append({
                    "type": "CRITICAL_RISK",
                    "projectId": pid,
                    "fromTier": None,
                    "toTier": "Critical",
                    "riskScore": round(float(risk_by_id[pid]), 4),
                    "message": f"{pid} is at Critical risk ({risk_by_id[pid]:.0%}) — immediate inspection required",
                    "date": today,
                })
        alerts.sort(key=lambda a: (a["type"] != "TIER_ESCALATION", -a["riskScore"]))
        for k, a in enumerate(alerts):
            a["id"] = f"ALERT-{k+1:04d}"
        with open(os.path.join(data_dir, "alerts.json"), "w", encoding="utf-8") as f:
            json.dump(alerts, f, indent=2)

        logger.info(f"Exported timeline/reports/inspectors/alerts JSON to {data_dir}")

    def run(self) -> None:
        """Execute the full pipeline."""
        try:
            banner(logger, "INITIALISING MAAGAP FRAMEWORK")
            logger.info(f"Seed set to {SEED}")
            logger.info(f"Outputs will be saved to: {OUTPUTS_DIR}")

            # ---------------------------------------------------------
            # DATA PIPELINE & PREPROCESSING ARTIFACT
            # ---------------------------------------------------------
            banner(logger, "DATA PIPELINE: REAL DATA & SYNTHETIC GENERATION")
            
            # Snapshot the previous run's predictions (if any) before they are
            # overwritten, so tier-transition alerts can be computed by diff.
            prev_pred_path = os.path.join(DATA_PROCESSED_DIR, "tbl_predictions.csv")
            df_prev_predictions = pd.read_csv(prev_pred_path) if os.path.exists(prev_pred_path) else None

            preprocessor = DataPreprocessor()
            df_real = preprocessor.load_and_clean_ppdo()
            dist = preprocessor.extract_distributions(df_real)
            
            generator = SyntheticDataGenerator()
            df_projects, df_quarterly = generator.generate_synthetic_dataset(dist)

            X_static, X_temporal, static_cols, temporal_cols = self.preprocessor_pipeline.fit_transform(df_projects, df_quarterly)
            pipeline_path = self.preprocessor_pipeline.save()
            logger.info(f"Saved reusable preprocessor artifact to {pipeline_path}")

            y_delay, y_overrun, y_risk_true, y_delay_days, y_overrun_pct = self.feature_engineer.build_targets(df_projects)

            n_samples = len(X_static)
            train_idx, val_idx, test_idx = split_data(n_samples)
            
            logger.info(f"Train samples: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

            # ---------------------------------------------------------
            # UNTUNED VS TUNED COMPARISON WORKFLOW (IF REQUESTED)
            # ---------------------------------------------------------
            untuned_pass_res = None
            if self.compare_tuning:
                banner(logger, "HYPERPARAMETER TUNING COMPARISON: UNTUNED BASELINE PASS")
                untuned_pass_res = self._train_and_evaluate_pass(
                    X_static, X_temporal, y_delay, df_projects, train_idx, val_idx, test_idx, tune=False
                )

            # Primary Run
            banner(logger, "OBJECTIVE 1: MULTI-STAGE PREDICTIVE FRAMEWORK")
            pass_res = self._train_and_evaluate_pass(
                X_static, X_temporal, y_delay, df_projects, train_idx, val_idx, test_idx, tune=self.tune_models
            )

            rf_model = pass_res["rf_model"]
            xgb_model = pass_res["xgb_model"]
            lstm_model = pass_res["lstm_model"]
            meta_model = pass_res["meta_model"]
            
            rf_prob_test = pass_res["rf_prob_test"]
            xgb_prob_test = pass_res["xgb_prob_test"]
            lstm_prob_test = pass_res["lstm_prob_test"]
            meta_prob_test = pass_res["meta_prob_test"]
            
            rf_pred_test = pass_res["rf_pred_test"]
            xgb_pred_test = pass_res["xgb_pred_test"]
            lstm_pred_test = pass_res["lstm_pred_test"]
            meta_pred_test = pass_res["meta_pred_test"]
            
            lstm_history = pass_res["lstm_history"]
            xgb_evals = pass_res["xgb_evals"]

            # ---------------------------------------------------------
            # OBJECTIVE 2: METRICS & VISUALISATIONS
            # ---------------------------------------------------------
            banner(logger, "OBJECTIVE 2: METRICS & VISUALISATIONS")

            logger.info("Generating learning curves for RF and Meta-Ensemble...")
            train_sizes_rf, train_scores_rf, val_scores_rf = learning_curve(
                rf_model, X_static[train_idx], y_delay[train_idx], cv=3, n_jobs=-1, scoring="f1",
                train_sizes=np.linspace(0.2, 1.0, 5)
            )
            rf_lc = (train_sizes_rf, train_scores_rf, val_scores_rf)

            meta_X_train = np.column_stack([pass_res["rf_prob_train"], pass_res["xgb_prob_train"], pass_res["lstm_prob_train"]])
            train_sizes_meta, train_scores_meta, val_scores_meta = learning_curve(
                meta_model, meta_X_train, y_delay[train_idx], cv=3, n_jobs=-1, scoring="f1",
                train_sizes=np.linspace(0.2, 1.0, 5)
            )
            meta_lc = (train_sizes_meta, train_scores_meta, val_scores_meta)

            self.visualizer.plot_all_training_histories(lstm_history, xgb_evals, rf_lc, meta_lc, "all_training_histories.png")

            if self.compare_tuning and untuned_pass_res is not None:
                logger.info("Generating Untuned vs. Tuned Comparison Deliverables...")
                comp_df = Evaluator.compare_tuning_impact(untuned_pass_res["metrics"], pass_res["metrics"])
                self.visualizer.plot_tuning_comparison(comp_df, metric="AUC-ROC", filename="tuning_comparison_auc.png")
                self.visualizer.plot_training_curves_overlay(
                    untuned_pass_res["lstm_history"].history if untuned_pass_res["lstm_history"] else {},
                    lstm_history.history if lstm_history else {},
                    model_name="LSTM", filename="lstm_training_comparison.png"
                )

            logger.info("Generating feature importance plots...")
            self.visualizer.plot_feature_importance(rf_model.feature_importances_, static_cols, "Random Forest Feature Importance", "fi_rf_delay.png")
            if hasattr(xgb_model, "feature_importances_"):
                self.visualizer.plot_feature_importance(xgb_model.feature_importances_, static_cols, "XGBoost Feature Importance", "fi_xgb_delay.png")

            contribs = MetaEnsembleTrainer.meta_ensemble_percent_contributions(
                meta_model, rf_prob_test, xgb_prob_test, lstm_prob_test
            )
            logger.info("Meta-Ensemble Sub-Model Contributions:")
            for m, c in contribs.items():
                logger.info(f"  - {m}: {c:.1f}%")

            models_data = [
                ("Random Forest", rf_pred_test, rf_prob_test),
                ("XGBoost", xgb_pred_test, xgb_prob_test),
                ("LSTM", lstm_pred_test, lstm_prob_test),
                ("Meta-Ensemble", meta_pred_test, meta_prob_test),
            ]

            roc_curves = []
            for name, preds, probs in models_data:
                m_bin = self.evaluator.binary_metrics(y_delay[test_idx], preds, probs, label=name)
                self.metrics_list.append(m_bin)
                fpr, tpr, _ = roc_curve(y_delay[test_idx], probs)
                roc_curves.append((fpr, tpr, m_bin["AUC-ROC"], name))

            self.visualizer.plot_roc_curves(roc_curves, "roc_curves_delay.png")
            self.visualizer.plot_model_comparison(self.metrics_list, "model_comparison_delay.png")

            # ---------------------------------------------------------
            # EXPLAINABLE AI (SHAP)
            # ---------------------------------------------------------
            banner(logger, "EXPLAINABLE AI: SHAP FEATURE ATTRIBUTION")
            rf_shap, rf_base = ExplanationService.explain_random_forest(rf_model, X_static[test_idx], static_cols)
            xgb_shap, xgb_base = ExplanationService.explain_xgboost(xgb_model, X_static[test_idx], static_cols)
            
            ExplanationService.generate_summary_plot(rf_shap, X_static[test_idx], static_cols, "Random Forest SHAP Summary", "shap_summary_rf.png")
            ExplanationService.generate_summary_plot(xgb_shap, X_static[test_idx], static_cols, "XGBoost SHAP Summary", "shap_summary_xgb.png")
            
            # Formulate sample JSON explanation for prediction
            sample_shap_json = ExplanationService.format_shap_json(rf_shap, static_cols, rf_base, sample_idx=0)
            logger.info(f"Sample prediction SHAP explanation: {json.dumps(sample_shap_json, indent=2)}")

            # ---------------------------------------------------------
            # COST OVERRUN MODEL & MAE REGRESSION (Objective 1 & 2)
            # ---------------------------------------------------------
            banner(logger, "COST OVERRUN CLASSIFIER & MAE REGRESSION MODELS")

            xgb_overrun, _ = TreeModelTrainer.train_xgboost(
                X_static[train_idx], y_overrun[train_idx],
                X_static[val_idx], y_overrun[val_idx],
                task="overrun", tune=self.tune_models
            )
            overrun_prob_test = xgb_overrun.predict_proba(X_static[test_idx])[:, 1]
            overrun_pred_test = xgb_overrun.predict(X_static[test_idx])
            overrun_metrics = self.evaluator.binary_metrics(
                y_overrun[test_idx], overrun_pred_test, overrun_prob_test, label="XGBoost (Cost Overrun)"
            )

            from sklearn.metrics import mean_absolute_error
            reg_delay_days = RegressionModelTrainer.train_xgboost_regressor(
                X_static[train_idx], y_delay_days[train_idx], task="delay_days"
            )
            mae_delay_days = mean_absolute_error(y_delay_days[test_idx], reg_delay_days.predict(X_static[test_idx]))

            reg_overrun_pct = RegressionModelTrainer.train_xgboost_regressor(
                X_static[train_idx], y_overrun_pct[train_idx], task="overrun_pct"
            )
            mae_overrun_pct = mean_absolute_error(y_overrun_pct[test_idx], reg_overrun_pct.predict(X_static[test_idx]))

            logger.info(f"MAE (delay duration): {mae_delay_days:.2f} days")
            logger.info(f"MAE (cost overrun): {mae_overrun_pct * 100:.2f} percentage points of budget")

            pd.DataFrame([
                {"target": "delay_days", "model": "XGBoost Regressor", "MAE": round(float(mae_delay_days), 2), "unit": "days"},
                {"target": "cost_overrun_pct", "model": "XGBoost Regressor", "MAE": round(float(mae_overrun_pct), 4), "unit": "fraction of budget"},
                {"target": "is_cost_overrun", "model": "XGBoost Classifier", "MAE": None, "unit": f"F1={overrun_metrics['F1-Score']:.4f}, AUC={overrun_metrics['AUC-ROC']:.4f}"},
            ]).to_csv(os.path.join(OUTPUTS_DIR, "regression_mae_report.csv"), index=False)

            # ---------------------------------------------------------
            # OBJECTIVE 3: RISK ENGINE & BOUNDARY VERIFICATION
            # ---------------------------------------------------------
            banner(logger, "OBJECTIVE 3: DYNAMIC RISK SCORING ENGINE")

            risk_scores_test = compute_all_risk_scores(
                rf_prob_test, xgb_prob_test, lstm_prob_test, 
                weights=(0.35, 0.35, 0.30)
            )
            
            v_get_tier = np.vectorize(get_risk_tier)
            risk_tiers_test = v_get_tier(risk_scores_test)

            self.visualizer.plot_risk_score_distribution(risk_scores_test, risk_tiers_test, "risk_score_distribution.png")
            
            actual_tiers_test = df_projects.iloc[test_idx]["risk_category"].values
            self.visualizer.plot_risk_tier_comparison(actual_tiers_test, risk_tiers_test, "risk_tier_comparison.png")

            consistency = check_logic_consistency(risk_scores_test, risk_tiers_test)
            self.visualizer.plot_logic_consistency(consistency, len(risk_scores_test), "logic_consistency.png")

            # ---------------------------------------------------------
            # OBJECTIVE 4: OPTIMIZATION
            # ---------------------------------------------------------
            banner(logger, "OBJECTIVE 4: INSPECTOR DEPLOYMENT OPTIMIZATION")

            inspectors_df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "tbl_inspector.csv"))
            logger.info(f"Loaded PPDO inspector roster: {len(inspectors_df)} inspectors")

            assignment, lp_status = self.optimizer.optimize_assignment(
                risk_scores_test, inspectors_df,
                min_critical_coverage=1.0, critical_threshold=0.90,
            )
            lp_util = self.optimizer.captured_utility(risk_scores_test, assignment)

            manual_assignment = self.optimizer.run_manual_baseline(risk_scores_test, inspectors_df)
            manual_util = self.optimizer.captured_utility(risk_scores_test, manual_assignment)
            improvement = self.optimizer.analyze_improvement(manual_util, lp_util)

            mc_results = self.optimizer.run_random_baseline_mc(
                risk_scores_test, inspectors_df, lp_utility=lp_util, n_iterations=100
            )

            self.visualizer.plot_optimization_comparison(
                manual_util, lp_util, improvement["improvement_pct"], mc_results, "optimization_comparison.png"
            )

            self.visualizer.plot_lp_selection_profile(
                risk_scores_test, risk_tiers_test,
                np.where(assignment >= 0)[0], np.where(manual_assignment >= 0)[0],
                "lp_selection_profile.png"
            )

            # ---------------------------------------------------------
            # EXPORT RELATIONAL TABLES: PREDICTIONS + ASSIGNMENTS
            # ---------------------------------------------------------
            df_test_projects = df_projects.iloc[test_idx].copy()
            inspector_ids = inspectors_df["inspector_id"].values
            assignment_refs = [
                inspector_ids[assignment[idx]] if assignment[idx] >= 0 else "UNASSIGNED"
                for idx in range(len(test_idx))
            ]

            df_predictions_tbl = pd.DataFrame({
                "prediction_id": [f"PRED-{idx+1:05d}" for idx in range(len(test_idx))],
                "project_id": df_test_projects["project_id"].values,
                "prediction_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
                "delay_probability": np.round(meta_prob_test, 4),
                "cost_overrun_probability": np.round(overrun_prob_test, 4),
                "predicted_delay_days": np.round(reg_delay_days.predict(X_static[test_idx]), 1),
                "risk_score": np.round(risk_scores_test, 4),
                "risk_tier": risk_tiers_test,
                "shap_explanation": [json.dumps(ExplanationService.format_shap_json(rf_shap, static_cols, rf_base, idx)) for idx in range(len(test_idx))],
                "optimized_assignment_ref": assignment_refs,
            })
            pred_tbl_path = os.path.join(DATA_PROCESSED_DIR, "tbl_predictions.csv")
            df_predictions_tbl.to_csv(pred_tbl_path, index=False)
            logger.info(f"Exported relational predictions table to {pred_tbl_path}")

            # Deployment schedule table (DFD data store D4)
            tier_priority = {"Critical": "HIGH", "High": "HIGH", "Medium": "MEDIUM", "Low": "LOW"}
            tier_urgency = {"Critical": "VISIT ASAP", "High": "VISIT SOON", "Medium": "CHECK IN A WEEK", "Low": "ROUTINE"}
            assigned_mask = assignment >= 0
            assigned_indices = np.where(assigned_mask)[0]
            df_assignments = pd.DataFrame({
                "assignment_id": [f"ASSIGN-{k+1:05d}" for k in range(len(assigned_indices))],
                "project_id": df_test_projects["project_id"].values[assigned_indices],
                "project_type": df_test_projects["project_type"].values[assigned_indices],
                "location": df_test_projects["location"].values[assigned_indices],
                "inspector_id": [inspector_ids[assignment[i]] for i in assigned_indices],
                "inspector_name": [inspectors_df["inspector_name"].values[assignment[i]] for i in assigned_indices],
                "risk_score": np.round(risk_scores_test[assigned_indices], 4),
                "risk_tier": risk_tiers_test[assigned_indices],
                "priority": [tier_priority[t] for t in risk_tiers_test[assigned_indices]],
                "urgency": [tier_urgency[t] for t in risk_tiers_test[assigned_indices]],
            }).sort_values(["inspector_id", "risk_score"], ascending=[True, False])
            assign_tbl_path = os.path.join(DATA_PROCESSED_DIR, "tbl_assignments.csv")
            df_assignments.to_csv(assign_tbl_path, index=False)
            logger.info(f"Exported deployment schedule table to {assign_tbl_path}")

            # Frontend export: per-inspector grouped assignments JSON
            self._export_frontend_assignments(df_assignments, inspectors_df, len(test_idx))

            # Frontend export: timeline, inspection reports, roster, and tier-transition alerts
            self._export_frontend_extras(
                df_test_projects, df_quarterly, inspectors_df, df_assignments,
                risk_scores_test, risk_tiers_test,
                reg_delay_days.predict(X_static[test_idx]),
                df_prev_predictions,
            )

            banner(logger, "MAAGAP EXECUTION COMPLETE!")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            logger.error(traceback.format_exc())
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="MAAGAP Pipeline Execution")
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning for models.")
    parser.add_argument("--compare-tuning", action="store_true", help="Run side-by-side untuned vs. tuned comparison.")
    args = parser.parse_args()

    pipeline = MainPipeline(tune_models=args.tune, compare_tuning=args.compare_tuning)
    pipeline.run()


if __name__ == "__main__":
    main()
