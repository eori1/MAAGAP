# MAAGAP — Objective 1: Multi-Stage Predictive Framework

**Machine Analytics for Allocation, Governance and Assessment of Projects**

A multi-stage machine learning framework for predicting government project delays and cost overruns, developed as part of an undergraduate thesis at the College of Information and Communications Technology.

---

## Overview

MAAGAP Objective 1 implements a **two-stage predictive framework** that combines traditional ensemble classifiers with deep learning to forecast whether government infrastructure and non-infrastructure projects will experience delays or cost overruns.

| Stage | Model | Input | Purpose |
|-------|-------|-------|---------|
| **Stage 1** | Random Forest | Static project features (30 features) | Feature-based risk classification |
| **Stage 1** | XGBoost (Gradient Boosting) | Static project features (30 features) | Feature-based risk classification |
| **Stage 2** | LSTM Neural Network | Temporal quarterly sequences (4 timesteps x 9 features) | Capture sequential monitoring patterns |
| **Meta** | Logistic Regression (Stacking) | Probabilities from all three models | Fuse Stage 1 + Stage 2 predictions |

---

## Results Summary

### Binary Delay Prediction (Test Set)

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | 0.7533 | 0.6099 | 0.7351 | 0.6667 | 0.8429 |
| XGBoost | 0.7467 | 0.5979 | 0.7483 | 0.6647 | 0.8435 |
| LSTM | 0.8778 | 0.8478 | 0.7748 | 0.8097 | 0.9273 |
| **Meta-Ensemble** | **0.8733** | **0.8730** | **0.7285** | **0.7942** | **0.9340** |

### Risk Categorisation (4-class: Low / Medium / High / Critical)

Thresholds per manuscript: Low (0.0–0.3), Medium (0.3–0.7), High (0.7–0.9), Critical (0.9–1.0)

| Model | Accuracy | Macro F1-Score |
|-------|----------|----------------|
| RF Risk | 0.8022 | 0.7686 |
| XGB Risk | 0.8067 | 0.7745 |

### Regression — Delay Days (MAE)

| Model | MAE (days) |
|-------|------------|
| Random Forest | 81.33 |
| XGBoost | 85.76 |
| LSTM | 57.10 |
| **Meta-Ensemble** | **46.20** |

### Hyperparameter Tuning Impact

All three model types were tuned: RF/XGBoost via `RandomizedSearchCV` (40 iter, 5-fold CV, F1 scoring); LSTM via manual search over 8 configurations (varying units, dropout, learning rate, batch size), selected by validation loss.

| Model | F1-Score | AUC-ROC | Recall |
|-------|----------|---------|--------|
| RF (Default) | 0.6485 | 0.8285 | 0.6291 |
| **RF (Tuned)** | **0.6667** | **0.8429** | **0.7351** |
| XGB (Default) | 0.6458 | 0.8101 | 0.6821 |
| **XGB (Tuned)** | **0.6647** | **0.8435** | **0.7483** |
| LSTM (Default) | 0.7538 | 0.9251 | 0.6490 |
| **LSTM (Tuned)** | **0.8014** | **0.9266** | **0.7616** |

Tuning improved F1, AUC-ROC, and Recall across all models — the metrics most critical for catching at-risk projects. LSTM showed the largest gain (+6.3% F1, +17.4% Recall).

### Key Findings

1. **LSTM and Meta-Ensemble significantly outperform static-feature models** (~88% accuracy vs ~75%), confirming that temporal monitoring data is the strongest predictor of project delays.
2. **AUC-ROC of 0.93** for the Meta-Ensemble indicates excellent discriminatory ability between delayed and on-time projects, well above the 0.75 threshold for good performance cited in the manuscript.
3. **Hyperparameter tuning improved Recall by +10–15%** for tree-based models, ensuring more at-risk projects are correctly identified.
4. **Random Forest and XGBoost achieve ~80% accuracy on 4-class risk categorisation**, with balanced precision and recall across risk tiers.
5. **Feature importance analysis** reveals `contractor_reliability`, `typhoon_exposure`, `approved_budget`, and `is_infrastructure` as the top static risk drivers.
6. **The Meta-Ensemble achieves the lowest MAE** (~46 days), demonstrating the value of fusing multiple model perspectives.

---

## Project Structure

```
MAAGAP/
├── maagap/                        # Core Python package
│   ├── __init__.py                # Package metadata
│   ├── config.py                  # Constants, hyperparameters, PAGASA/PSA data
│   ├── data_preprocessing.py      # Clean real PPDO Excel data
│   ├── synthetic_generator.py     # Generate 3,000 synthetic projects (2016-2025)
│   ├── feature_engineering.py     # Build static features + temporal tensors
│   ├── models.py                  # RF, XGBoost, LSTM, Meta-ensemble training
│   └── evaluation.py              # Metrics, Plotly visualisations, reports
│
├── main.py                        # Full pipeline orchestration script
├── MAAGAP_Objective1.ipynb        # Jupyter Notebook (presentation version)
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
├── LIST-OF-ALL-ONGOING-PPAS-2026.xlsx   # Real PPDO dataset (773 records)
│
├── data/
│   └── processed/                 # Generated datasets
│       ├── ppdo_2026_cleaned.csv
│       ├── synthetic_projects.csv
│       └── synthetic_quarterly.csv
│
├── outputs/                       # Evaluation charts and reports
│   ├── RESULTS_SUMMARY.md              # Complete results summary document
│   ├── evaluation_report.csv            # Full metrics table
│   ├── model_comparison.png / .html     # Grouped bar chart — all models
│   ├── roc_curves_delay.png / .html     # ROC curves for all models
│   ├── cm_rf_delay.png / .html          # Confusion matrix — RF (delay)
│   ├── cm_xgb_delay.png / .html         # Confusion matrix — XGBoost (delay)
│   ├── cm_lstm_delay.png / .html        # Confusion matrix — LSTM (delay)
│   ├── cm_meta_delay.png / .html        # Confusion matrix — Meta-Ensemble (delay)
│   ├── cm_rf_risk.png / .html           # Confusion matrix — RF Risk (4-class)
│   ├── cm_xgb_risk.png / .html          # Confusion matrix — XGB Risk (4-class)
│   ├── fi_rf_delay.png / .html          # Feature importance — RF (top 20)
│   ├── fi_xgb_delay.png / .html         # Feature importance — XGBoost (top 20)
│   ├── lstm_training_history.png / .html # LSTM loss/accuracy curves
│   ├── risk_distribution.png / .html    # Actual | RF | XGB risk (3 panels)
│   ├── risk_distribution_rf.png / .html   # Actual vs RF risk only
│   ├── risk_distribution_xgb.png / .html  # Actual vs XGB risk only
│   └── hyperparameter_tuning_comparison.png / .html  # Default vs Tuned bar chart
│
└── models/                        # Trained model artifacts (git-ignored, ~50MB)
    ├── rf_delay.pkl
    ├── rf_risk.pkl
    ├── xgb_delay.pkl
    ├── xgb_risk.pkl
    ├── lstm_delay.keras
    └── meta_ensemble.pkl
```

> **Note:** Trained models in `models/` are git-ignored due to file size (~50MB total). Run the pipeline to regenerate them.

---

## Data Sources

| Source | Description | Usage |
|--------|-------------|-------|
| **PPDO Iloilo** | LIST-OF-ALL-ONGOING-PPAS-2026.xlsx — 773 real project records | Extract statistical distributions for synthetic data generation |
| **PAGASA** | Historical monthly rainfall (mm) and typhoon exposure days for Iloilo Province | External contextual variable — weather risk |
| **PSA** | Consumer Price Index (CPI) and Construction Materials Retail Price Index (CMRPI), 2016–2025 | External contextual variable — economic conditions |
| **Synthetic** | 3,000 projects across 2016–2025 with quarterly monitoring data, generated from real distributions | Training, validation, and testing dataset |

---

## How to Run

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/eori1/MAAGAP.git
cd MAAGAP
pip install -r requirements.txt
```

### Run the Full Pipeline (CLI)

```bash
python main.py
```

This will:
1. Load and clean the real PPDO 2026 dataset
2. Generate 3,000 synthetic projects with quarterly monitoring data
3. Engineer static features (30 columns) and temporal tensors (4 x 9)
4. Split data into 70% train / 15% validation / 15% test
5. Train Random Forest, XGBoost, LSTM, and Meta-Ensemble (with hyperparameter tuning)
6. Evaluate all models and save outputs to `outputs/`

**Expected runtime:** ~2–3 minutes (depending on hardware).

### Run the Jupyter Notebook

```bash
jupyter notebook MAAGAP_Objective1.ipynb
```

The notebook contains the same pipeline with markdown explanations and inline interactive Plotly visualisations.

---

## Module Descriptions

### `maagap/config.py`
Centralises all constants and hyperparameters: PAGASA weather data (monthly rainfall, typhoon days), PSA economic indicators (CPI, CMRPI), Iloilo-specific agencies, contractors with seeded reliability scores, risk thresholds (Low 0-0.3, Medium 0.3-0.7, High 0.7-0.9, Critical 0.9-1.0), and model hyperparameters.

### `maagap/data_preprocessing.py`
Reads the raw PPDO Excel file, cleans column names, extracts project status from unstructured "remarks" text, classifies projects as Infrastructure or Non-Infrastructure, and computes statistical distributions (budget log-normal parameters, agency/type probabilities).

### `maagap/synthetic_generator.py`
Generates 3,000 synthetic projects (2016–2025) grounded in real PPDO distributions. Simulates contractor assignments, budget allocation, PAGASA weather exposure, PSA economic conditions, and quarterly monitoring progress. Uses sharpened logistic probability functions for delay/overrun outcomes with temporal noise to prevent data leakage.

### `maagap/feature_engineering.py`
Transforms raw data into ML-ready formats:
- **Static features** (30 columns): numeric fields + label-encoded categoricals + engineered interaction terms
- **Temporal tensor** (3000 x 4 x 9): MinMax-scaled quarterly sequences for LSTM input

<details>
<summary><b>Full Feature Definitions (30 static + 9 temporal = 39 total)</b></summary>

#### Static Features (30) — used by Random Forest & XGBoost

| # | Feature | Description |
|---|---------|-------------|
| 1 | `approved_budget` | Total approved budget allocation for the project in pesos |
| 2 | `planned_duration_months` | Number of months the project is officially scheduled to take |
| 3 | `start_month` | Calendar month (1–12) when the project began, capturing seasonality |
| 4 | `has_contractor` | Whether the project has an assigned contractor (1) or is agency-managed (0) |
| 5 | `contractor_reliability` | Historical performance score (0–1) of the assigned contractor |
| 6 | `agency_capacity` | Capability score (0–1) of the implementing government agency |
| 7 | `typhoon_exposure` | Typhoon-affected days in the province during the project year (PAGASA) |
| 8 | `cpi_at_start` | Consumer Price Index at the project's start year (PSA) |
| 9 | `cmrpi_at_start` | Construction Materials Retail Price Index at the project's start year (PSA) |
| 10 | `cpi_change` | Year-over-year % change in CPI |
| 11 | `cmrpi_change` | Year-over-year % change in construction material prices |
| 12 | `budget_log` | Log-transformed budget to reduce skewness |
| 13 | `is_infrastructure` | Binary: 1 if infrastructure, 0 if non-infrastructure |
| 14 | `is_typhoon_start` | Binary: 1 if project starts during typhoon season (Jun–Nov) |
| 15 | `infra_x_typhoon` | Interaction: infrastructure × typhoon exposure |
| 16 | `infra_x_budget` | Interaction: infrastructure × budget scale |
| 17 | `contractor_x_typhoon` | Interaction: contractor unreliability × typhoon exposure |
| 18 | `budget_x_cpi_change` | Interaction: budget × inflation pressure |
| 19 | `low_contractor_flag` | Binary: 1 if contractor reliability < 0.5 |
| 20 | `high_budget_flag` | Binary: 1 if budget > dataset median |
| 21 | `agency_risk` | 1 − agency_capacity (institutional weakness) |
| 22 | `contractor_x_agency` | Interaction: weak contractor × weak agency |
| 23 | `infra_x_low_contractor` | Interaction: infrastructure × low-reliability contractor |
| 24 | `typhoon_x_budget` | Interaction: weather exposure × project size |
| 25 | `econ_pressure` | Combined CPI + CMRPI change magnitude |
| 26 | `composite_risk_features` | Weighted composite of all key risk indicators |
| 27 | `project_type_enc` | Label-encoded project type |
| 28 | `implementing_agency_enc` | Label-encoded implementing agency |
| 29 | `procurement_mode_enc` | Label-encoded mode of procurement |
| 30 | `funding_source_enc` | Label-encoded funding source |

#### Temporal Features (9) — used by LSTM (per quarterly monitoring period)

| # | Feature | Description |
|---|---------|-------------|
| 1 | `planned_progress_pct` | Expected cumulative % of work completed by this quarter |
| 2 | `actual_progress_pct` | Actual cumulative % of work completed (from inspection) |
| 3 | `slippage_pct` | Planned − actual progress, indicating schedule deviation |
| 4 | `expenditure_ratio` | Actual spending / planned budget at this quarter |
| 5 | `issues_count` | Number of reported issues (shortages, permit delays, etc.) |
| 6 | `rainfall_mm` | Average monthly rainfall this quarter (PAGASA) |
| 7 | `typhoon_days` | Typhoon-affected days this quarter (PAGASA) |
| 8 | `cpi_quarterly` | Consumer Price Index this quarter (PSA) |
| 9 | `cmrpi_quarterly` | Construction Materials RPI this quarter (PSA) |

</details>

### `maagap/models.py`
Implements training with hyperparameter tuning for all model types:
- **Random Forest** — `RandomizedSearchCV` (40 iter, 5-fold CV, F1 scoring) over depth, estimators, split criteria
- **XGBoost** — `RandomizedSearchCV` (40 iter, 5-fold CV, F1 scoring) over depth, learning rate, regularisation, subsampling
- **LSTM** — Manual hyperparameter search over 8 configurations (units, dropout, learning rate, batch size); Keras Sequential (2-layer LSTM + Dropout + BatchNorm + Dense), trained with early stopping
- **Meta-Ensemble** — Logistic Regression stacking classifier fusing RF, XGBoost, and LSTM probability outputs

### `maagap/evaluation.py`
Computes all thesis-specified metrics (Accuracy, Precision, Recall, F1-Score, AUC-ROC, MAE) and generates interactive Plotly visualisations (ROC curves, confusion matrices, feature importance, LSTM training history, risk distribution, model comparison charts). Saves both `.html` (interactive) and `.png` (static) formats.

---

## Technologies Used

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10+ |
| **ML / Deep Learning** | Scikit-learn, XGBoost, TensorFlow / Keras |
| **Data Processing** | Pandas, NumPy, OpenPyXL |
| **Visualisation** | Plotly (interactive + static export via Kaleido) |
| **Model Persistence** | Joblib |
| **Presentation** | Jupyter Notebook |

---

## Evaluation Metrics (Thesis Objective 2)

All metrics follow the thesis specification using a 70/15/15 train/validation/test split:

- **Accuracy** — Overall correctness of predictions
- **Precision** — Of predicted positives, how many are actually positive
- **Recall** — Of actual positives, how many are correctly identified
- **F1-Score** — Harmonic mean of Precision and Recall
- **AUC-ROC** — Area Under the ROC curve; values above 0.75 indicate good discriminative ability (per manuscript)
- **MAE** — Mean Absolute Error for delay-day regression (in days)

---

## Authors

MAAGAP Research Team — College of Information and Communications Technology

---

## License

This project is developed for academic purposes as part of an undergraduate thesis.
