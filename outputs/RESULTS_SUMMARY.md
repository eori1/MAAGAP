# MAAGAP Objective 1 — Results Summary

**Multi-Stage Predictive Framework for Government Project Delay and Cost Overrun Prediction**

Generated from the executed pipeline using 3,000 synthetic projects grounded in real PPDO Iloilo 2026 data.

---

## 1. Binary Delay Prediction (Test Set — 450 samples)

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | 0.7533 | 0.6099 | 0.7351 | 0.6667 | 0.8429 |
| XGBoost | 0.7467 | 0.5979 | 0.7483 | 0.6647 | 0.8435 |
| LSTM | 0.8778 | 0.8478 | 0.7748 | 0.8097 | 0.9273 |
| **Meta-Ensemble** | **0.8733** | **0.8730** | **0.7285** | **0.7942** | **0.9340** |

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
| **Meta-Ensemble** | **46.20** |

---

## 4. Output Visualisations (PNG files in `outputs/` folder)

### ROC Curves
- **`roc_curves_delay.png`** — ROC curves for all four models (RF, XGBoost, LSTM, Meta-Ensemble) showing AUC values

### Model Performance Comparison
- **`model_comparison.png`** — Grouped bar chart comparing Accuracy, Precision, Recall, F1-Score, and AUC-ROC across all four models

### Confusion Matrices — Binary Delay Prediction
- **`cm_rf_delay.png`** — Random Forest: Not Delayed vs Delayed
- **`cm_xgb_delay.png`** — XGBoost: Not Delayed vs Delayed
- **`cm_lstm_delay.png`** — LSTM: Not Delayed vs Delayed
- **`cm_meta_delay.png`** — Meta-Ensemble: Not Delayed vs Delayed

### Confusion Matrices — Risk Categorisation (4-class)
- **`cm_rf_risk.png`** — Random Forest: Low / Medium / High / Critical
- **`cm_xgb_risk.png`** — XGBoost: Low / Medium / High / Critical

### Feature Importance
- **`fi_rf_delay.png`** — Random Forest top 20 most important features for delay prediction
- **`fi_xgb_delay.png`** — XGBoost top 20 most important features for delay prediction

### LSTM Training History
- **`lstm_training_history.png`** — Training vs validation loss and accuracy curves over epochs

### Risk Distribution
- **`risk_distribution.png`** — Actual vs Predicted risk category distribution (side-by-side)

---

## 5. Key Findings

1. **Random Forest and XGBoost achieve ~75% accuracy and ~0.84 AUC-ROC** on static project features alone (contractor reliability, typhoon exposure, budget, agency capacity), demonstrating meaningful predictive signal for delay forecasting from project-level characteristics.

2. **LSTM achieves ~88% accuracy and ~0.93 AUC-ROC** by leveraging temporal quarterly monitoring patterns (progress slippage, expenditure ratios), confirming that sequential inspection data significantly improves prediction.

3. **The Meta-Ensemble achieves the best AUC-ROC (0.934)** by fusing all three models through logistic regression stacking, with the lowest MAE at 46.20 days.

4. **Risk categorisation models achieve ~80% accuracy** on the 4-class problem (Low/Medium/High/Critical), using manuscript-defined thresholds.

5. **Feature importance analysis** identifies `contractor_reliability`, `typhoon_exposure`, `is_infrastructure`, and `approved_budget` as the most influential static risk factors.

---

## 6. Data Split

- **Train:** 2,100 samples (70%)
- **Validation:** 450 samples (15%)
- **Test:** 450 samples (15%)

Per manuscript specification: 70/30 train-test split, with the 30% subdivided into 15% validation + 15% test to support LSTM early stopping and meta-ensemble training.

---

## 7. Interactive Charts

All visualisations are also saved as interactive HTML files (`.html`) in the `outputs/` folder. Open them in any browser for hover tooltips and zoom.
