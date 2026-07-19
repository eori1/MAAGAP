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
| LSTM | 0.8689 | 0.8286 | 0.7682 | 0.7973 | 0.9254 |
| Meta-Ensemble (baseline bases) | 0.8644 | 0.8571 | 0.7152 | 0.7798 | 0.9315 |
| **Meta-Ensemble (tuned bases)** | **0.8689** | **0.8594** | **0.7285** | **0.7885** | **0.9355** |

**Δ (tuned − baseline) on test:** Accuracy +0.0045, Precision +0.0023, Recall +0.0133, F1 +0.0087, AUC-ROC +0.0040.

*Values from the last executed `MAAGAP_Objective1.ipynb` / `evaluation_report.csv` (minor differences vs `python main.py` are possible due to deep-learning non-determinism).*

---

## 2. Risk Categorisation — 4-Class (Low / Medium / High / Critical)

Risk thresholds per manuscript: Low (0.0–0.3), Medium (0.3–0.7), High (0.7–0.9), Critical (0.9–1.0)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1-Score (macro) |
|-------|----------|-------------------|----------------|------------------|
| RF Risk | 0.8000 | 0.7692 | 0.7632 | 0.7634 |
| XGB Risk | 0.8178 | 0.7989 | 0.7893 | 0.7909 |

---

## 3. Regression — Delay Days (Mean Absolute Error)

| Model | MAE (days) |
|-------|------------|
| Random Forest | 81.33 |
| XGBoost | 85.76 |
| LSTM | 57.07 |
| Meta-Ensemble (baseline bases) | 47.12 |
| **Meta-Ensemble (tuned bases)** | **44.78** |

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
| RF (Default) | 0.7711 | 0.6644 | 0.6424 | 0.6532 | 0.8330 |
| **RF (Tuned)** | **0.7533** | **0.6099** | **0.7351** | **0.6667** | **0.8429** |
| XGB (Default) | 0.7578 | 0.6329 | 0.6623 | 0.6472 | 0.8063 |
| **XGB (Tuned)** | **0.7467** | **0.5979** | **0.7483** | **0.6647** | **0.8435** |
| LSTM (Default) | 0.8622 | 0.8112 | 0.7682 | 0.7891 | 0.9252 |
| **LSTM (Tuned)** | **0.8689** | **0.8594** | **0.7285** | **0.7885** | **0.9248** |

**Key tuned hyperparameters:**
- RF: n_estimators=300, max_depth=10, min_samples_split=5, min_samples_leaf=4, max_features=log2
- XGB: n_estimators=300, max_depth=8, lr=0.01, subsample=0.7, colsample_bytree=0.7
- LSTM: Best configuration selected from 8 candidates by validation loss (e.g. units 128/64, dropout 0.4, lr 0.002, batch 64 in the latest run)

**Interpretation:** Metrics above are **test-set** comparisons of default vs tuned base models. Tree tuning improves **Recall** and **AUC-ROC** vs defaults; LSTM default vs tuned trade off precision/recall (defaults had higher recall on this run). The **stacking meta-learner on tuned bases** still edges the baseline-meta on Accuracy, Precision, F1, and AUC (see §1).

---

## 6. Key Findings

1. **Random Forest and XGBoost achieve ~75% accuracy and ~0.84 AUC-ROC** on static project features alone (contractor reliability, typhoon exposure, budget, agency capacity), demonstrating meaningful predictive signal for delay forecasting from project-level characteristics.

2. **LSTM achieves ~87% accuracy and ~0.92 AUC-ROC** by leveraging temporal quarterly monitoring patterns (progress slippage, expenditure ratios), confirming that sequential inspection data significantly improves prediction.

3. **The Meta-Ensemble (tuned bases) achieves the best AUC-ROC (~0.936)** by fusing all three models through logistic regression stacking, with the lowest MAE at **~44.8** days (vs **~47.1** for meta on baseline bases in the latest notebook run).

4. **Risk categorisation models achieve ~80–82% accuracy** on the 4-class problem (Low/Medium/High/Critical), using manuscript-defined thresholds.

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
