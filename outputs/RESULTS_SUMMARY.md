# MAAGAP Objective 1 — Results Summary

**Multi-Stage Predictive Framework for Government Project Delay and Cost Overrun Prediction**

Generated from the executed pipeline using 3,000 synthetic projects grounded in real PPDO Iloilo 2026 data.

---

## 1. Binary Delay Prediction (Test Set — 450 samples)

**Stacking meta-learner (two variants):** Both use logistic regression on stacked validation probabilities from RF, XGBoost, and LSTM.

- **Meta (baseline bases):** Base models trained with **default** hyperparameters (`tune=False`).
- **Meta (tuned bases):** Base models **tuned** via `RandomizedSearchCV` (RF/XGB) and LSTM search — primary artifact `models/meta_ensemble.pkl`.

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | 0.7533 | 0.6099 | 0.7351 | 0.6667 | 0.8429 |
| XGBoost | 0.7467 | 0.5979 | 0.7483 | 0.6647 | 0.8435 |
| LSTM | 0.8778 | 0.8478 | 0.7748 | 0.8097 | 0.9273 |
| Meta-Ensemble (baseline bases) | *(run `python main.py`)* | *(printed)* | *(printed)* | *(printed)* | *(printed)* |
| **Meta-Ensemble (tuned bases)** | **0.8733** | **0.8730** | **0.7285** | **0.7942** | **0.9340** |

*Re-run `python main.py` to print both meta rows and the **Δ (tuned − baseline)** banner on the test set.*

---

## 2. Risk Categorisation — 4-Class (Low / Medium / High / Critical)

Risk thresholds per manuscript: Low (0.0–0.3), Medium (0.3–0.7), High (0.7–0.9), Critical (0.9–1.0)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1-Score (macro) |
|-------|----------|-------------------|----------------|------------------|
| RF Risk | 0.8022 | 0.7660 | 0.7770 | 0.7686 |
| XGB Risk | 0.8067 | 0.7762 | 0.7765 | 0.7745 |

---

## 3. Regression — Delay Days (Mean Absolute Error)

| Model | MAE (days) |
|-------|------------|
| Random Forest | 81.33 |
| XGBoost | 85.76 |
| LSTM | 57.10 |
| Meta-Ensemble (baseline bases) | *(run `python main.py`)* |
| **Meta-Ensemble (tuned bases)** | **46.20** |

---

## 4. Output Visualisations (PNG files in `outputs/` folder)

### ROC Curves
- **`roc_curves_delay.png`** — ROC curves for RF, XGBoost, LSTM, and both Meta-Ensemble variants (baseline vs tuned bases) with AUC values

### Model Performance Comparison
- **`model_comparison.png`** — Grouped bar chart comparing Accuracy, Precision, Recall, F1-Score, and AUC-ROC (includes both meta variants)

### Confusion Matrices — Binary Delay Prediction
- **`cm_rf_delay.png`** — Random Forest: Not Delayed vs Delayed
- **`cm_xgb_delay.png`** — XGBoost: Not Delayed vs Delayed
- **`cm_lstm_delay.png`** — LSTM: Not Delayed vs Delayed
- **`cm_meta_delay.png`** — Meta-Ensemble (tuned bases): Not Delayed vs Delayed
- **`cm_meta_baseline_delay.png`** — Meta-Ensemble (baseline bases): Not Delayed vs Delayed

### Confusion Matrices — Risk Categorisation (4-class)
- **`cm_rf_risk.png`** — Random Forest: Low / Medium / High / Critical
- **`cm_xgb_risk.png`** — XGBoost: Low / Medium / High / Critical

### Feature Importance
- **`fi_rf_delay.png`** — Random Forest top 20 most important features for delay prediction
- **`fi_xgb_delay.png`** — XGBoost top 20 most important features for delay prediction

### LSTM Training History
- **`lstm_training_history.png`** — Training vs validation loss and accuracy curves over epochs

### Risk Distribution (4-class: Low / Medium / High / Critical)

These charts compare **class frequencies**, not per-project correctness (see confusion matrices for that).

- **`risk_distribution.png`** — **Combined:** Actual | Random Forest (Risk) | XGBoost (Risk) — three panels in one figure
- **`risk_distribution_rf.png`** — Actual vs **RF** predicted only (two panels)
- **`risk_distribution_xgb.png`** — Actual vs **XGBoost** predicted only (two panels)

**Why not the meta-ensemble?** The meta-learner fuses **RF + XGB + LSTM** for **binary delay** only. The LSTM is not trained for **4-class risk**, so there is no LSTM risk probability to stack. Risk tiers use **RF and XGB** on static features, per the manuscript.

### Hyperparameter Tuning Comparison
- **`hyperparameter_tuning_comparison.png`** — Side-by-side bar chart: Default vs Tuned for RF and XGBoost

---

## 5. Hyperparameter Tuning Comparison

Baseline models use **default hyperparameters** (sklearn/xgboost defaults for trees; single default LSTM architecture). Tuned models use `RandomizedSearchCV` for trees (40 iter, 5-fold CV, F1 scoring) and manual config search for LSTM (8 configurations, best by validation loss).

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| RF (Default) | 0.7711 | 0.6690 | 0.6291 | 0.6485 | 0.8285 |
| **RF (Tuned)** | **0.7533** | **0.6099** | **0.7351** | **0.6667** | **0.8429** |
| XGB (Default) | 0.7489 | 0.6131 | 0.6821 | 0.6458 | 0.8101 |
| **XGB (Tuned)** | **0.7467** | **0.5979** | **0.7483** | **0.6647** | **0.8435** |
| LSTM (Default) | 0.8578 | 0.8991 | 0.6490 | 0.7538 | 0.9251 |
| **LSTM (Tuned)** | **0.8733** | **0.8456** | **0.7616** | **0.8014** | **0.9266** |

**Key tuned hyperparameters:**
- RF: n_estimators=300, max_depth=10, min_samples_split=5, min_samples_leaf=4, max_features=log2
- XGB: n_estimators=300, max_depth=8, lr=0.01, subsample=0.7, colsample_bytree=0.7
- LSTM: Best configuration selected from 8 candidates by validation loss

**Tuning improved F1-Score, AUC-ROC, and Recall across all models.** LSTM tuning showed the clearest gains: F1 improved from 0.754 to 0.801 (+6.3%), and Recall from 0.649 to 0.762 (+17.4%). Higher recall means the tuned models catch more genuinely at-risk projects, which is essential for government accountability.

---

## 6. Key Findings

1. **Random Forest and XGBoost achieve ~75% accuracy and ~0.84 AUC-ROC** on static project features alone (contractor reliability, typhoon exposure, budget, agency capacity), demonstrating meaningful predictive signal for delay forecasting from project-level characteristics.

2. **LSTM achieves ~88% accuracy and ~0.93 AUC-ROC** by leveraging temporal quarterly monitoring patterns (progress slippage, expenditure ratios), confirming that sequential inspection data significantly improves prediction.

3. **The Meta-Ensemble achieves the best AUC-ROC (0.934)** by fusing all three models through logistic regression stacking, with the lowest MAE at 46.20 days.

4. **Risk categorisation models achieve ~80% accuracy** on the 4-class problem (Low/Medium/High/Critical), using manuscript-defined thresholds.

5. **Feature importance analysis** identifies `contractor_reliability`, `typhoon_exposure`, `is_infrastructure`, and `approved_budget` as the most influential static risk factors.

---

## 7. Feature Definitions (39 total: 30 static + 9 temporal)

### Static Features (30) — used by Random Forest & XGBoost

| # | Feature | Description |
|---|---------|-------------|
| 1 | `approved_budget` | Total approved budget allocation for the project in pesos |
| 2 | `planned_duration_months` | Number of months the project is officially scheduled to take |
| 3 | `start_month` | Calendar month (1–12) when the project began, capturing seasonality |
| 4 | `has_contractor` | Whether the project has an assigned contractor (1) or is agency-managed (0) |
| 5 | `contractor_reliability` | Historical performance score (0–1) of the assigned contractor based on past delivery |
| 6 | `agency_capacity` | Capability score (0–1) of the implementing government agency |
| 7 | `typhoon_exposure` | Number of typhoon-affected days in the project's province during its year (PAGASA) |
| 8 | `cpi_at_start` | Consumer Price Index at the project's start year (PSA) |
| 9 | `cmrpi_at_start` | Construction Materials Retail Price Index at the project's start year (PSA) |
| 10 | `cpi_change` | Year-over-year percentage change in CPI, indicating inflation pressure |
| 11 | `cmrpi_change` | Year-over-year percentage change in construction material prices |
| 12 | `budget_log` | Log-transformed budget to reduce skewness across orders of magnitude |
| 13 | `is_infrastructure` | Binary: 1 if infrastructure (roads, bridges, buildings), 0 if non-infrastructure |
| 14 | `is_typhoon_start` | Binary: 1 if project starts during typhoon season (June–November) |
| 15 | `infra_x_typhoon` | Interaction: infrastructure projects disproportionately affected by typhoon exposure |
| 16 | `infra_x_budget` | Interaction: large-budget infrastructure projects carry compounded risk |
| 17 | `contractor_x_typhoon` | Interaction: unreliable contractors combined with high typhoon exposure |
| 18 | `budget_x_cpi_change` | Interaction: large budgets under high inflation face greater cost overrun pressure |
| 19 | `low_contractor_flag` | Binary: 1 if contractor reliability is below 0.5 |
| 20 | `high_budget_flag` | Binary: 1 if project budget exceeds the dataset median |
| 21 | `agency_risk` | Inverse of agency capacity (1 − agency_capacity), representing institutional weakness |
| 22 | `contractor_x_agency` | Interaction: weak contractor combined with weak agency ("double risk") |
| 23 | `infra_x_low_contractor` | Interaction: infrastructure projects assigned to low-reliability contractors |
| 24 | `typhoon_x_budget` | Interaction: weather exposure scaled by project size |
| 25 | `econ_pressure` | Combined magnitude of CPI and CMRPI changes, representing overall economic stress |
| 26 | `composite_risk_features` | Weighted composite score of infrastructure type, budget, contractor, weather, agency, and economic factors |
| 27 | `project_type_enc` | Label-encoded project type (Infrastructure vs Non-Infrastructure) |
| 28 | `implementing_agency_enc` | Label-encoded implementing agency (e.g., Provincial Engineering Office, PHO) |
| 29 | `procurement_mode_enc` | Label-encoded mode of procurement (e.g., public bidding, negotiated) |
| 30 | `funding_source_enc` | Label-encoded funding source (e.g., General Fund, Special Education Fund) |

### Temporal Features (9) — used by LSTM (per quarterly monitoring period)

| # | Feature | Description |
|---|---------|-------------|
| 1 | `planned_progress_pct` | Expected cumulative percentage of work completed by this quarter |
| 2 | `actual_progress_pct` | Actual cumulative percentage of work completed as recorded during inspection |
| 3 | `slippage_pct` | Gap between planned and actual progress (planned − actual), indicating schedule deviation |
| 4 | `expenditure_ratio` | Ratio of actual spending to planned budget at this quarter |
| 5 | `issues_count` | Number of reported issues (material shortages, permit delays, workforce problems) |
| 6 | `rainfall_mm` | Average monthly rainfall in millimeters for the province this quarter (PAGASA) |
| 7 | `typhoon_days` | Number of typhoon-affected days in the province this quarter (PAGASA) |
| 8 | `cpi_quarterly` | Consumer Price Index for this quarter (PSA) |
| 9 | `cmrpi_quarterly` | Construction Materials Retail Price Index for this quarter (PSA) |

---

## 8. Data Split

- **Train:** 2,100 samples (70%)
- **Validation:** 450 samples (15%)
- **Test:** 450 samples (15%)

Per manuscript specification: 70/30 train-test split, with the 30% subdivided into 15% validation + 15% test to support LSTM early stopping and meta-ensemble training.

---

## 9. Interactive Charts

All visualisations are also saved as interactive HTML files (`.html`) in the `outputs/` folder. Open them in any browser for hover tooltips and zoom.
