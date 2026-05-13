"""MAAGAP — Objectives 1–4 Pipeline
===========================================
Objective 1: Multi-stage predictive framework
  Stage 1: Random Forest + XGBoost (static features)
  Stage 2: LSTM (temporal quarterly sequences)
  Meta:    Stacking ensemble (fuses Stage 1 + Stage 2)

Objective 2: Model evaluation
  Metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC, MAE

Objective 3: Dynamic Risk Scoring Engine
  Translates probability outputs into actionable tiers
  (Low / Medium / High / Critical) with logic consistency testing

Objective 4: LP Resource Allocation Optimization
  Linear programming to maximise high-risk project inspection coverage
  Evaluated against baseline (manual/random) with >=15% improvement target
  Validated via 200-iteration Monte Carlo simulation

Data: Synthetic data grounded in real PPDO 2026 and Fund Transfer Con
(21k+ records, 2013-2026) distributions.
"""

import os, sys, warnings, time
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from sklearn.metrics import roc_curve

from maagap.config import SEED, OUTPUTS_DIR, RISK_LABELS, RISK_THRESHOLDS
from maagap.data_preprocessing import (
    load_and_clean_ppdo, extract_distributions,
    load_fund_transfer_con, extract_fund_transfer_distributions,
)
from maagap.synthetic_generator import generate_synthetic_dataset
from maagap.feature_engineering import (
    build_static_features, build_temporal_sequences,
    build_targets, split_data,
)
from maagap.models import (
    train_random_forest, train_xgboost,
    train_lstm, train_meta_ensemble, predict_meta, meta_ensemble_percent_contributions,
)
from maagap.evaluation import (
    binary_metrics, regression_metrics, multiclass_metrics,
    plot_confusion_matrix, plot_roc_curves, plot_feature_importance,
    plot_training_history, plot_risk_distribution, plot_risk_distribution_rf_xgb,
    plot_model_comparison,
    generate_full_report, find_optimal_threshold,
    # Objective 3
    plot_risk_score_distribution, plot_risk_tier_comparison, plot_logic_consistency,
    # Objective 4
    plot_optimization_comparison, plot_lp_selection_profile,
)
from maagap.risk_scoring import (
    compute_risk_score, risk_tiers, logic_consistency_check, RiskScoringConfig,
)

np.random.seed(SEED)
DIVIDER = "=" * 70


def banner(text):
    print(f"\n{DIVIDER}\n  {text}\n{DIVIDER}")


def print_hardware_info():
    """Print GPU/CPU info (for Chapter 4 reproducibility)."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print(f"  TensorFlow GPUs detected: {[g.name for g in gpus]}")
        else:
            print("  TensorFlow GPUs detected: none (CPU fallback)")
    except Exception as e:
        print(f"  TensorFlow hardware check failed: {e}")


def main():
    t0 = time.time()
    banner("MAAGAP — Objective 1: Multi-Stage Predictive Framework")
    banner("Hardware Acceleration Check (CUDA/GPU)")
    print_hardware_info()

    # ==================================================================
    # STEP 1 — Preprocess real PPDO data + Fund Transfer Con sheet
    # ==================================================================
    banner("Step 1/9: Loading & Cleaning Real PPDO 2026 Data")
    ppdo_df = load_and_clean_ppdo()
    distributions = extract_distributions(ppdo_df)
    print(f"  MONITORING REPORT Con — Cleaned records : {len(ppdo_df)}")
    print(f"  Budget log-mean : {distributions['budget_log_mean']:.2f}")
    print(f"  Budget log-std  : {distributions['budget_log_std']:.2f}")
    print(f"  Type distribution:")
    for k, v in distributions["type_probs"].items():
        print(f"    {k}: {v:.1%}")

    # --- Fund Transfer Con supplementary data ---
    print()
    df_ft = load_fund_transfer_con()
    if df_ft is not None:
        ft_dist = extract_fund_transfer_distributions(df_ft)
        print(f"  Fund Transfer Con — Records loaded    : {len(df_ft):,}")
        print(f"  Fund Transfer Con — Year range        : "
              f"{int(df_ft['year'].min())} – {int(df_ft['year'].max())}")
        if "ft_budget_log_mean" in ft_dist:
            print(f"  Fund Transfer Con — Budget log-mean  : {ft_dist['ft_budget_log_mean']:.2f}")
            print(f"  Fund Transfer Con — Budget median    : "
                  f"PHP {ft_dist.get('ft_budget_median', 0):,.0f}")
        if "liquidation_rate" in ft_dist:
            print(f"  Fund Transfer Con — Liquidation rate : {ft_dist['liquidation_rate']:.1%}")
        if "unliquidated_rate" in ft_dist:
            print(f"  Fund Transfer Con — Unliquidated rate: {ft_dist['unliquidated_rate']:.1%}")
        if "ft_type_probs" in ft_dist:
            print(f"  Fund Transfer Con — Project types:")
            for k, v in ft_dist["ft_type_probs"].items():
                print(f"    {k}: {v:.1%}")
        if "municipality_probs" in ft_dist:
            top_munis = sorted(
                ft_dist["municipality_probs"].items(), key=lambda x: -x[1]
            )[:5]
            print(f"  Fund Transfer Con — Top municipalities:")
            for m, p in top_munis:
                print(f"    {m}: {p:.1%}")
        # Merge Fund Transfer budget distribution into main distributions
        # (larger sample = more representative log-normal parameters)
        distributions["ft_distributions"] = ft_dist
    else:
        print("  Fund Transfer Con — Sheet not available (skipped)")
        ft_dist = {}

    # ==================================================================
    # STEP 2 — Generate synthetic multi-year dataset
    # ==================================================================
    banner("Step 2/9: Generating Synthetic Dataset (2016-2025)")
    df_proj, df_qtr = generate_synthetic_dataset(distributions)
    print(f"  Projects generated  : {len(df_proj)}")
    print(f"  Quarterly records   : {len(df_qtr)}")
    print(f"  Delayed projects    : {df_proj['is_delayed'].sum()} ({df_proj['is_delayed'].mean():.1%})")
    print(f"  Cost-overrun projects: {df_proj['is_cost_overrun'].sum()} ({df_proj['is_cost_overrun'].mean():.1%})")
    print(f"  Risk distribution:")
    for cat in RISK_LABELS:
        n = (df_proj["risk_category"] == cat).sum()
        print(f"    {cat}: {n} ({n/len(df_proj):.1%})")

    # ==================================================================
    # STEP 3 — Feature engineering
    # ==================================================================
    banner("Step 3/9: Feature Engineering")
    X_static, feat_names, static_scaler, _ = build_static_features(df_proj)
    X_temporal, temp_feat_names, temp_scaler = build_temporal_sequences(df_proj, df_qtr)
    y_delay, y_overrun, y_risk, y_delay_days, y_overrun_pct = build_targets(df_proj)

    print(f"  Static features  : {X_static.shape[1]} columns")
    print(f"  Temporal tensor  : {X_temporal.shape} (projects x timesteps x features)")
    print(f"  Delay label dist : 0={np.sum(y_delay==0)}, 1={np.sum(y_delay==1)}")
    print(f"  Risk label dist  : " + ", ".join(
        f"{RISK_LABELS[i]}={np.sum(y_risk==i)}" for i in range(len(RISK_LABELS))
    ))

    # ==================================================================
    # STEP 4 — Train / Val / Test split (70 / 15 / 15)
    # ==================================================================
    banner("Step 4/9: Splitting Data (70/15/15)")
    idx_tr, idx_va, idx_te = split_data(len(df_proj))
    print(f"  Train : {len(idx_tr)}  |  Val : {len(idx_va)}  |  Test : {len(idx_te)}")

    Xs_tr, Xs_va, Xs_te = X_static[idx_tr], X_static[idx_va], X_static[idx_te]
    Xt_tr, Xt_va, Xt_te = X_temporal[idx_tr], X_temporal[idx_va], X_temporal[idx_te]
    yd_tr, yd_va, yd_te = y_delay[idx_tr], y_delay[idx_va], y_delay[idx_te]
    yr_tr, yr_va, yr_te = y_risk[idx_tr], y_risk[idx_va], y_risk[idx_te]
    ydd_te = y_delay_days[idx_te]

    # ==================================================================
    # STEP 5 — Train models: baseline (for meta comparison) then tuned
    # ==================================================================
    banner("Step 5/9: Training Models")

    print("\n  [Baseline] Delay models (no hyperparameter tuning) — for meta-ensemble comparison ...")
    rf_b = train_random_forest(Xs_tr, yd_tr, task="delay", tune=False)
    rf_prob_va_b = rf_b.predict_proba(Xs_va)[:, 1]
    rf_prob_te_b = rf_b.predict_proba(Xs_te)[:, 1]
    xgb_b = train_xgboost(Xs_tr, yd_tr, task="delay", tune=False)
    xgb_prob_va_b = xgb_b.predict_proba(Xs_va)[:, 1]
    xgb_prob_te_b = xgb_b.predict_proba(Xs_te)[:, 1]
    lstm_b, _, _ = train_lstm(Xt_tr, yd_tr, Xt_va, yd_va, task="delay", tune=False)
    lstm_prob_va_b = lstm_b.predict(Xt_va, verbose=0).flatten()
    lstm_prob_te_b = lstm_b.predict(Xt_te, verbose=0).flatten()
    meta_b = train_meta_ensemble(
        rf_prob_va_b, xgb_prob_va_b, lstm_prob_va_b, yd_va,
        artifact_name="meta_ensemble_baseline.pkl",
    )
    meta_b_pct = meta_ensemble_percent_contributions(meta_b, rf_prob_va_b, xgb_prob_va_b, lstm_prob_va_b)
    print(f"    Meta contribution (%): {meta_b_pct}")
    meta_b_pred_te, meta_b_prob_te = predict_meta(
        meta_b, rf_prob_te_b, xgb_prob_te_b, lstm_prob_te_b,
    )
    meta_b_prob_pos = meta_b_prob_te[:, 1] if meta_b_prob_te.ndim > 1 else meta_b_prob_te
    print("    -> meta-learner trained on BASELINE base models (saved: meta_ensemble_baseline.pkl)")

    print("\n  [Stage 1a] Random Forest — delay prediction (tuning) ...")
    rf = train_random_forest(Xs_tr, yd_tr, task="delay", tune=True)
    rf_pred_te = rf.predict(Xs_te)
    rf_prob_te = rf.predict_proba(Xs_te)[:, 1]
    rf_prob_va = rf.predict_proba(Xs_va)[:, 1]
    print("    -> trained")

    print("  [Stage 1b] XGBoost — delay prediction (tuning) ...")
    xgb = train_xgboost(Xs_tr, yd_tr, task="delay", tune=True)
    xgb_pred_te = xgb.predict(Xs_te)
    xgb_prob_te = xgb.predict_proba(Xs_te)[:, 1]
    xgb_prob_va = xgb.predict_proba(Xs_va)[:, 1]
    print("    -> trained")

    print("  [Stage 1c] Random Forest — risk categorisation (tuning) ...")
    rf_risk = train_random_forest(Xs_tr, yr_tr, task="risk", tune=True)
    rf_risk_pred_te = rf_risk.predict(Xs_te)
    print("    -> trained")

    print("  [Stage 1d] XGBoost — risk categorisation (tuning) ...")
    xgb_risk = train_xgboost(Xs_tr, yr_tr, task="risk", tune=True)
    xgb_risk_pred_te = xgb_risk.predict(Xs_te)
    print("    -> trained")

    print("  [Stage 2]  LSTM — temporal delay prediction (tuning) ...")
    lstm, history, lstm_best_params = train_lstm(Xt_tr, yd_tr, Xt_va, yd_va, task="delay", tune=True)
    lstm_prob_te = lstm.predict(Xt_te, verbose=0).flatten()
    lstm_pred_te = (lstm_prob_te >= 0.5).astype(int)
    lstm_prob_va = lstm.predict(Xt_va, verbose=0).flatten()
    print(f"    -> trained ({len(history.history['loss'])} epochs)")

    print("  [Meta]     Stacking ensemble on TUNED bases (RF + XGB + LSTM) ...")
    meta = train_meta_ensemble(
        rf_prob_va, xgb_prob_va, lstm_prob_va, yd_va,
        artifact_name="meta_ensemble.pkl",
    )
    meta_pct = meta_ensemble_percent_contributions(meta, rf_prob_va, xgb_prob_va, lstm_prob_va)
    print(f"    Meta contribution (%): {meta_pct}")
    meta_pred_te, meta_prob_te = predict_meta(meta, rf_prob_te, xgb_prob_te, lstm_prob_te)
    meta_prob_pos = meta_prob_te[:, 1] if meta_prob_te.ndim > 1 else meta_prob_te
    print("    -> trained (saved: meta_ensemble.pkl)")

    # ==================================================================
    # STEP 6 — Evaluation (Objective 2)
    # ==================================================================
    banner("Step 6/9: Evaluation — Binary Delay Prediction")

    all_metrics = []
    roc_data = []
    binary_delay_metrics = []

    for name, y_p, y_pr in [
        ("Random Forest", rf_pred_te, rf_prob_te),
        ("XGBoost", xgb_pred_te, xgb_prob_te),
        ("LSTM", lstm_pred_te, lstm_prob_te),
        ("Meta-Ensemble (baseline bases)", meta_b_pred_te, meta_b_prob_pos),
        ("Meta-Ensemble (tuned bases)", meta_pred_te, meta_prob_pos),
    ]:
        m = binary_metrics(yd_te, y_p, y_pr, label=name)
        all_metrics.append(m)
        binary_delay_metrics.append(m)
        print(f"\n  {name}:")
        for k, v in m.items():
            if k != "Model":
                print(f"    {k:12s}: {v}")

        fpr, tpr, _ = roc_curve(yd_te, y_pr)
        roc_data.append((fpr, tpr, m["AUC-ROC"], name))

    banner("Meta-ensemble: baseline vs tuned base models (test set)")
    mb = binary_metrics(yd_te, meta_b_pred_te, meta_b_prob_pos, label="Meta baseline")
    mt = binary_metrics(yd_te, meta_pred_te, meta_prob_pos, label="Meta tuned")
    for k in ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]:
        delta = mt[k] - mb[k]
        print(f"  {k:12s}:  baseline {mb[k]:.4f}  |  tuned {mt[k]:.4f}  |  delta {delta:+.4f}")

    banner("Evaluation — Risk Categorisation (Multiclass)")
    for name, y_p in [
        ("RF Risk",  rf_risk_pred_te),
        ("XGB Risk", xgb_risk_pred_te),
    ]:
        m = multiclass_metrics(yr_te, y_p, label=name)
        all_metrics.append(m)
        print(f"\n  {name}:")
        for k, v in m.items():
            if k != "Model":
                print(f"    {k:18s}: {v}")

    banner("Evaluation — Regression (Delay Days MAE)")
    max_delay = df_proj["delay_days"].max()
    for name, proba in [
        ("Random Forest", rf_prob_te),
        ("XGBoost", xgb_prob_te),
        ("LSTM", lstm_prob_te),
        ("Meta-Ensemble (baseline bases)", meta_b_prob_pos),
        ("Meta-Ensemble (tuned bases)", meta_prob_pos),
    ]:
        # Fix MAE Denominator Trap: Only evaluate on projects predicted or known to be delayed
        mask = (yd_te == 1) | (proba >= 0.5)
        if mask.sum() > 0:
            pred_days = proba[mask] * max_delay
            m = regression_metrics(ydd_te[mask], pred_days, label=name)
            all_metrics.append(m)
            print(f"  {name:30s} MAE: {m['MAE']:.2f} days (evaluated on {mask.sum()} delayed/flagged projects)")
        else:
            pred_days = proba * max_delay
            m = regression_metrics(ydd_te, pred_days, label=name)
            all_metrics.append(m)
            print(f"  {name:30s} MAE: {m['MAE']:.2f} days")

    # ==================================================================
    # STEP 7 — Objective 3: Dynamic Risk Scoring Engine
    # ==================================================================
    banner("Step 7/9: Objective 3 — Dynamic Risk Scoring Engine")

    # Use tuned meta-ensemble delay probability + synthetic overrun probability
    overrun_proba_te = df_proj["overrun_probability"].values[idx_te]
    risk_cfg         = RiskScoringConfig(w_delay=0.55, w_overrun=0.45)
    risk_scores_te   = compute_risk_score(meta_prob_pos, overrun_proba_te, cfg=risk_cfg)
    risk_tiers_te    = risk_tiers(risk_scores_te)
    actual_tiers_te  = np.array([RISK_LABELS[i] for i in yr_te])

    # Logic consistency check
    consistency = logic_consistency_check(risk_scores_te, risk_tiers_te)
    print(f"  Logic consistency violations : {consistency['violations']} / {len(risk_scores_te)}")
    print(f"  Unknown tier labels          : {consistency['unknown_tier_labels']}")

    # Tier distribution
    print("\n  Risk Tier Distribution (Test Set):")
    for tier in RISK_LABELS:
        mask = (risk_tiers_te == tier)
        cnt  = mask.sum()
        lo, hi = RISK_THRESHOLDS[tier]
        scores_in = risk_scores_te[mask]
        mean_s = scores_in.mean() if cnt > 0 else 0.0
        print(f"    {tier:8s}: {cnt:4d} projects ({cnt/len(risk_tiers_te):.1%})  "
              f"threshold [{lo:.2f}–{hi:.2f}]  mean_score={mean_s:.4f}")

    # Agreement with ground-truth risk categories from synthetic generation
    tier_match = (risk_tiers_te == actual_tiers_te)
    print(f"\n  Tier agreement with ground truth  : "
          f"{tier_match.sum()} / {len(tier_match)} ({tier_match.mean():.1%})")

    # Per-tier agreement breakdown
    print("\n  Per-Tier Agreement:")
    print(f"    {'Tier':8s} {'N_pred':>8s} {'N_actual':>9s} {'Match':>8s}")
    print("    " + "-" * 40)
    for tier in RISK_LABELS:
        pred_mask   = (risk_tiers_te   == tier)
        actual_mask = (actual_tiers_te == tier)
        match       = np.sum(pred_mask & actual_mask)
        print(f"    {tier:8s} {pred_mask.sum():>8d} {actual_mask.sum():>9d} {match:>8d}")

    # ==================================================================
    # STEP 8 — Objective 4: LP Resource Allocation Optimization
    # ==================================================================
    banner("Step 8/9: Objective 4 — LP Resource Allocation Optimization")

    OPT_AVAILABLE = False
    result_lp = result_base = None
    mc_results = None
    improvement_pct = 0.0
    eff_lp_val = eff_base_val = 0.0

    try:
        from maagap.optimization import (
            OptimizationConfig,
            optimize_inspection_allocation,
            baseline_manual_allocation,
            allocation_efficiency,
            compute_efficiency_improvement,
            monte_carlo_robustness,
        )

        # Realistic capacity: 6 inspectors × 5 inspections/period = 30 slots
        opt_cfg = OptimizationConfig(
            inspectors_available=6,
            inspections_per_inspector=5,
            seed=SEED,
        )
        capacity = opt_cfg.inspectors_available * opt_cfg.inspections_per_inspector
        n_test   = len(risk_scores_te)

        print(f"  Test-set projects   : {n_test}")
        print(f"  Inspector capacity  : {opt_cfg.inspectors_available} inspectors "
              f"× {opt_cfg.inspections_per_inspector} = {capacity} inspection slots")

        # Baseline (manual/random) allocation
        result_base  = baseline_manual_allocation(risk_scores_te, cfg=opt_cfg)
        eff_base_res = allocation_efficiency(
            result_base["selected_idx"], risk_scores_te, risk_tiers_te
        )
        eff_base_val = eff_base_res["avg_captured_risk"]

        # LP optimized allocation
        result_lp   = optimize_inspection_allocation(risk_scores_te, risk_tiers_te, cfg=opt_cfg)
        eff_lp_res  = allocation_efficiency(
            result_lp["selected_idx"], risk_scores_te, risk_tiers_te
        )
        eff_lp_val  = eff_lp_res["avg_captured_risk"]

        # Efficiency improvement
        eff_summary = compute_efficiency_improvement(
            result_lp["selected_idx"], result_base["selected_idx"],
            risk_scores_te, risk_tiers_te,
        )
        improvement_pct = eff_summary["improvement_pct"]

        print(f"\n  Allocation Results:")
        print(f"    {'Method':32s} {'Selected':>10s} {'Avg Risk Score':>16s}")
        print("    " + "-" * 62)
        print(f"    {'Baseline (Manual/Random)':32s} "
              f"{result_base['selected_count']:>10d} {eff_base_val:>16.4f}")
        print(f"    {'LP Optimized':32s} "
              f"{result_lp['selected_count']:>10d} {eff_lp_val:>16.4f}")
        target_label = "PASS (>= 15%)" if eff_summary["target_met"] else "FAIL (< 15%)"
        print(f"    {'Efficiency Improvement':32s} {'':>10s} "
              f"{improvement_pct:>15.2f}%  [{target_label}]")
        print(f"  LP Solver Status    : {result_lp['status']}")

        # Coverage by tier
        print("\n  Coverage by Risk Tier — Inspected Projects:")
        print(f"    {'Tier':10s} {'Total':>8s} {'Baseline':>10s} {'LP Opt':>10s} "
              f"{'LP Recall':>12s}")
        print("    " + "-" * 55)
        base_sel_set = set(result_base["selected_idx"].tolist())
        lp_sel_set   = set(result_lp["selected_idx"].tolist())
        for tier in RISK_LABELS:
            tier_idx  = np.where(risk_tiers_te == tier)[0]
            n_total   = len(tier_idx)
            n_base    = sum(1 for i in tier_idx if i in base_sel_set)
            n_lp      = sum(1 for i in tier_idx if i in lp_sel_set)
            lp_recall = n_lp / max(n_total, 1)
            print(f"    {tier:10s} {n_total:>8d} {n_base:>10d} {n_lp:>10d} "
                  f"{lp_recall:>11.1%}")

        # Monte Carlo robustness
        print("\n  Running Monte Carlo Robustness (200 iterations, noise_std=0.05)...")
        mc_results = monte_carlo_robustness(
            risk_scores_te, risk_tiers_te,
            n_simulations=200, noise_std=0.05, cfg=opt_cfg,
        )
        print(f"    Successful runs     : {mc_results['n_successful']} / {mc_results['n_simulations']}")
        print(f"    Mean improvement    : {mc_results['mean_improvement_pct']:.2f}%")
        print(f"    Std dev             : {mc_results['std_improvement_pct']:.2f}%")
        print(f"    Range               : [{mc_results['min_improvement_pct']:.2f}%, "
              f"{mc_results['max_improvement_pct']:.2f}%]")
        print(f"    Simulations >= 15%  : {mc_results['pct_above_15']:.1f}%")

        OPT_AVAILABLE = True

    except ImportError as e:
        print(f"  [SKIP] PuLP not installed ({e}). Install with: pip install pulp")
    except Exception as e:
        print(f"  [ERROR] Optimization step failed: {e}")
        import traceback; traceback.print_exc()

    # ==================================================================
    # STEP 9 — Generate visualisations & report
    # ==================================================================
    banner("Step 9/9: Generating Outputs (Plotly)")

    plot_roc_curves(roc_data, "roc_curves_delay.png")
    print("  Saved: roc_curves_delay.png / .html")

    plot_model_comparison(binary_delay_metrics, "model_comparison.png")
    print("  Saved: model_comparison.png / .html")

    for name, y_p, label_set, fname in [
        ("Meta-Ensemble (tuned bases)", meta_pred_te, ["Not Delayed", "Delayed"], "cm_meta_delay.png"),
        ("Meta-Ensemble (baseline bases)", meta_b_pred_te, ["Not Delayed", "Delayed"], "cm_meta_baseline_delay.png"),
        ("Random Forest", rf_pred_te, ["Not Delayed", "Delayed"], "cm_rf_delay.png"),
        ("XGBoost", xgb_pred_te, ["Not Delayed", "Delayed"], "cm_xgb_delay.png"),
        ("LSTM", lstm_pred_te, ["Not Delayed", "Delayed"], "cm_lstm_delay.png"),
    ]:
        plot_confusion_matrix(yd_te, y_p, label_set, f"{name} — Delay Prediction", fname)
        print(f"  Saved: {fname}")

    for name, y_p, fname in [
        ("Random Forest", rf_risk_pred_te, "cm_rf_risk.png"),
        ("XGBoost",       xgb_risk_pred_te, "cm_xgb_risk.png"),
    ]:
        plot_confusion_matrix(yr_te, y_p, RISK_LABELS, f"{name} — Risk Categories", fname)
        print(f"  Saved: {fname}")

    plot_feature_importance(
        rf.feature_importances_, feat_names,
        "Random Forest — Top 20 Features (Delay)", "fi_rf_delay.png",
    )
    print("  Saved: fi_rf_delay.png")

    plot_feature_importance(
        xgb.feature_importances_, feat_names,
        "XGBoost — Top 20 Features (Delay)", "fi_xgb_delay.png",
    )
    print("  Saved: fi_xgb_delay.png")

    plot_training_history(history, "lstm_training_history.png")
    print("  Saved: lstm_training_history.png")

    plot_risk_distribution(
        yr_te, rf_risk_pred_te, "risk_distribution_rf.png",
        title="Predicted — Random Forest (Risk)",
    )
    print("  Saved: risk_distribution_rf.png / .html")
    plot_risk_distribution(
        yr_te, xgb_risk_pred_te, "risk_distribution_xgb.png",
        title="Predicted — XGBoost (Risk)",
    )
    print("  Saved: risk_distribution_xgb.png / .html")
    plot_risk_distribution_rf_xgb(
        yr_te, rf_risk_pred_te, xgb_risk_pred_te, "risk_distribution.png",
    )
    print("  Saved: risk_distribution.png / .html (Actual | RF | XGB — combined)")

    # --- Objective 3 plots ---
    plot_risk_score_distribution(risk_scores_te, risk_tiers_te, "risk_score_distribution.png")
    print("  Saved: risk_score_distribution.png / .html")

    plot_risk_tier_comparison(actual_tiers_te, risk_tiers_te, "risk_tier_comparison.png")
    print("  Saved: risk_tier_comparison.png / .html")

    plot_logic_consistency(consistency, len(risk_scores_te), "logic_consistency.png")
    print("  Saved: logic_consistency.png / .html")

    # --- Objective 4 plots ---
    if OPT_AVAILABLE and result_lp is not None and result_base is not None:
        plot_optimization_comparison(
            eff_base_val, eff_lp_val, improvement_pct, mc_results,
            "optimization_comparison.png",
        )
        print("  Saved: optimization_comparison.png / .html")

        plot_lp_selection_profile(
            risk_scores_te, risk_tiers_te,
            result_lp["selected_idx"], result_base["selected_idx"],
            "lp_selection_profile.png",
        )
        print("  Saved: lp_selection_profile.png / .html")

    report_df = generate_full_report(all_metrics)
    print("  Saved: evaluation_report.csv")

    # ------------------------------------------------------------------
    # Hyperparameter tuning comparison (Default vs Tuned for all models)
    # ------------------------------------------------------------------
    banner("Hyperparameter Tuning Comparison")

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    def _calc(y_true, y_pred, y_prob, label):
        return {
            "Model": label,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, zero_division=0),
            "AUC-ROC": roc_auc_score(y_true, y_prob),
        }

    print("  Reusing baseline delay models from Step 5 (rf_b, xgb_b, lstm_b)...")
    rf_bl_p, rf_bl_pr = rf_b.predict(Xs_te), rf_b.predict_proba(Xs_te)[:, 1]
    xgb_bl_p, xgb_bl_pr = xgb_b.predict(Xs_te), xgb_b.predict_proba(Xs_te)[:, 1]
    lstm_bl_pr = lstm_b.predict(Xt_te, verbose=0).flatten()
    lstm_bl_p = (lstm_bl_pr >= 0.5).astype(int)

    pairs = [
        (_calc(yd_te, rf_bl_p, rf_bl_pr, "RF (Default)"),
         _calc(yd_te, rf_pred_te, rf_prob_te, "RF (Tuned)"), "Random Forest"),
        (_calc(yd_te, xgb_bl_p, xgb_bl_pr, "XGB (Default)"),
         _calc(yd_te, xgb_pred_te, xgb_prob_te, "XGB (Tuned)"), "XGBoost"),
        (_calc(yd_te, lstm_bl_p, lstm_bl_pr, "LSTM (Default)"),
         _calc(yd_te, lstm_pred_te, lstm_prob_te, "LSTM (Tuned)"), "LSTM"),
    ]

    keys = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
    print(f"\n  {'Model':20s} {'Acc':>8s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s} {'AUC':>8s}")
    print("  " + "-" * 54)
    for bl, tu, _ in pairs:
        for m in (bl, tu):
            print(f"  {m['Model']:20s} " + " ".join(f"{m[k]:8.4f}" for k in keys))

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("Random Forest", "XGBoost", "LSTM"),
                        horizontal_spacing=0.08)
    for col, (bl, tu, name) in enumerate(pairs, 1):
        bl_vals = [bl[k] for k in keys]
        tu_vals = [tu[k] for k in keys]
        fig.add_trace(go.Bar(x=keys, y=bl_vals, name="Default" if col == 1 else "",
                             marker_color="#e74c3c", opacity=0.7,
                             text=[f"{v:.3f}" for v in bl_vals], textposition="auto",
                             showlegend=(col == 1), legendgroup="default"), row=1, col=col)
        fig.add_trace(go.Bar(x=keys, y=tu_vals, name="Tuned" if col == 1 else "",
                             marker_color="#2ecc71", opacity=0.9,
                             text=[f"{v:.3f}" for v in tu_vals], textposition="auto",
                             showlegend=(col == 1), legendgroup="tuned"), row=1, col=col)

    fig.update_layout(
        title=dict(text="Impact of Hyperparameter Tuning on All Models", x=0.5),
        barmode="group", template="plotly_white",
        height=500, width=1200,
        yaxis=dict(range=[0, 1.05]), yaxis2=dict(range=[0, 1.05]), yaxis3=dict(range=[0, 1.05]),
        legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"),
        font=dict(family="Segoe UI, sans-serif", size=12),
    )
    from maagap.evaluation import _save_fig
    _save_fig(fig, "hyperparameter_tuning_comparison.png")
    print("  Saved: hyperparameter_tuning_comparison.png / .html")

    # ==================================================================
    # Final summary
    # ==================================================================
    elapsed = time.time() - t0
    banner(f"COMPLETE  —  Total time: {elapsed:.1f}s")
    print("\n  Summary of Binary Delay Prediction (Test Set):\n")
    summary = [m for m in all_metrics if "AUC-ROC" in m]
    header = f"  {'Model':18s} {'Accuracy':>9s} {'Precision':>10s} {'Recall':>8s} {'F1':>8s} {'AUC-ROC':>9s}"
    print(header)
    print("  " + "-" * len(header.strip()))
    for m in summary:
        print(f"  {m['Model']:18s} {m['Accuracy']:9.4f} {m['Precision']:10.4f}"
              f" {m['Recall']:8.4f} {m['F1-Score']:8.4f} {m['AUC-ROC']:9.4f}")

    print(f"\n  All outputs saved to: {os.path.abspath(OUTPUTS_DIR)}")
    print(f"  Trained models in  : {os.path.abspath(os.path.join(os.path.dirname(__file__), 'models'))}")
    print()


if __name__ == "__main__":
    main()
