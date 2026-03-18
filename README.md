# MAAGAP — Objective 1: Multi-Stage Predictive Framework

**Machine Analytics for Allocation, Governance and Assessment of Projects**

A multi-stage machine learning framework for predicting government project delays and cost overruns, developed as part of an undergraduate thesis at the College of Information and Communications Technology.

---

## Overview

MAAGAP Objective 1 implements a **two-stage predictive framework** that combines traditional ensemble classifiers with deep learning to forecast whether government infrastructure and non-infrastructure projects will experience delays or cost overruns.

| Stage | Model | Input | Purpose |
|-------|-------|-------|---------|
| **Stage 1** | Random Forest | Static project features (26 features) | Feature-based risk classification |
| **Stage 1** | XGBoost (Gradient Boosting) | Static project features (26 features) | Feature-based risk classification |
| **Stage 2** | LSTM Neural Network | Temporal quarterly sequences (4 timesteps × 9 features) | Capture sequential monitoring patterns |
| **Meta** | Logistic Regression (Stacking) | Probabilities from all three models | Fuse Stage 1 + Stage 2 predictions |

---

## Results Summary

### Binary Delay Prediction (Test Set)

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | 0.6778 | 0.5500 | 0.2876 | 0.3777 | 0.6620 |
| XGBoost | 0.6622 | 0.5039 | 0.4248 | 0.4610 | 0.6426 |
| **LSTM** | **0.9556** | **0.9463** | **0.9216** | **0.9338** | **0.9829** |
| **Meta-Ensemble** | **0.9556** | **0.9463** | **0.9216** | **0.9338** | **0.9799** |

### Risk Categorisation (4-class: Low / Medium / High / Critical)

| Model | Accuracy | Macro F1-Score |
|-------|----------|----------------|
| RF Risk | 0.7444 | 0.6768 |
| XGB Risk | 0.7244 | 0.6590 |

### Regression — Delay Days (MAE)

| Model | MAE (days) |
|-------|------------|
| Random Forest | 76.89 |
| XGBoost | 81.53 |
| LSTM | 59.25 |
| Meta-Ensemble | 59.30 |

### Key Findings

1. **LSTM and the Meta-Ensemble dramatically outperform static-feature models** (95.6% accuracy vs ~67%), confirming that temporal monitoring data is the strongest predictor of project delays.
2. **AUC-ROC of 0.98** for LSTM/Meta indicates near-perfect discriminatory ability between delayed and on-time projects.
3. **Random Forest and XGBoost achieve ~74% accuracy on 4-class risk categorisation**, with strongest performance on "Medium" risk projects.
4. **Feature importance analysis** reveals `contractor_reliability`, `typhoon_exposure`, `approved_budget`, and `is_infrastructure` as the top static risk drivers.
5. **The Meta-Ensemble achieves the lowest MAE** (~59 days), demonstrating the value of fusing multiple model perspectives.

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
│   └── evaluation.py              # Metrics, plots, report generation
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
│       ├── ppdo_2026_cleaned.csv        # Cleaned real data
│       ├── synthetic_projects.csv       # 3,000 synthetic projects
│       └── synthetic_quarterly.csv      # Quarterly monitoring records
│
├── outputs/                       # Evaluation charts and reports
│   ├── evaluation_report.csv            # Full metrics table
│   ├── roc_curves_delay.png             # ROC curves for all models
│   ├── cm_rf_delay.png                  # Confusion matrix — RF
│   ├── cm_xgb_delay.png                 # Confusion matrix — XGBoost
│   ├── cm_meta_delay.png               # Confusion matrix — Meta-Ensemble
│   ├── cm_rf_risk.png                   # Confusion matrix — RF Risk (4-class)
│   ├── cm_xgb_risk.png                  # Confusion matrix — XGB Risk (4-class)
│   ├── fi_rf_delay.png                  # Feature importance — RF
│   ├── fi_xgb_delay.png                 # Feature importance — XGBoost
│   ├── lstm_training_history.png        # LSTM loss/accuracy curves
│   └── risk_distribution.png            # Actual vs Predicted risk bars
│
└── models/                        # Trained model artifacts (git-ignored, ~50MB)
    ├── rf_delay.pkl                     # Random Forest — delay
    ├── rf_risk.pkl                      # Random Forest — risk
    ├── xgb_delay.pkl                    # XGBoost — delay
    ├── xgb_risk.pkl                     # XGBoost — risk
    ├── lstm_delay.keras                 # LSTM — delay
    └── meta_ensemble.pkl                # Stacking meta-learner
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
git clone https://github.com/<your-username>/MAAGAP.git
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
3. Engineer static features (26 columns) and temporal tensors (4 × 9)
4. Split data into 70% train / 15% validation / 15% test
5. Train Random Forest, XGBoost, LSTM, and Meta-Ensemble models
6. Evaluate all models and save outputs to `outputs/`

**Expected runtime:** ~2–4 minutes (depending on hardware).

### Run the Jupyter Notebook

```bash
jupyter notebook MAAGAP_Objective1.ipynb
```

The notebook contains the same pipeline with markdown explanations and inline visualisations — designed for thesis presentation.

---

## Module Descriptions

### `maagap/config.py`
Centralises all constants and hyperparameters: PAGASA weather data (monthly rainfall, typhoon days), PSA economic indicators (CPI, CMRPI), Iloilo-specific agencies, contractors with seeded reliability scores, model hyperparameters (RF: 300 trees, XGB: 300 estimators, LSTM: 64 units, 60 epochs max).

### `maagap/data_preprocessing.py`
Reads the raw PPDO Excel file, cleans column names, extracts project status from unstructured "remarks" text, classifies projects as Infrastructure or Non-Infrastructure using keyword matching, and computes statistical distributions (budget log-normal parameters, agency/type probabilities).

### `maagap/synthetic_generator.py`
Generates 3,000 synthetic projects (2016–2025) grounded in real PPDO distributions. Simulates contractor assignments, budget allocation, PAGASA weather exposure, PSA economic conditions, and quarterly monitoring progress. Uses logistic probability functions to determine delay and cost overrun outcomes, with deliberate temporal noise and "misleading early progress" patterns to prevent data leakage.

### `maagap/feature_engineering.py`
Transforms raw data into ML-ready formats:
- **Static features** (26 columns): numeric fields + label-encoded categoricals + engineered interaction terms (`infra_x_typhoon`, `contractor_x_typhoon`, `composite_risk_features`, etc.)
- **Temporal tensor** (3000 × 4 × 9): MinMax-scaled quarterly sequences for LSTM input

### `maagap/models.py`
Implements training for all four model types:
- **Random Forest** — `sklearn.ensemble.RandomForestClassifier` with balanced class weights
- **XGBoost** — `xgboost.XGBClassifier` with scale_pos_weight for imbalance handling
- **LSTM** — Keras Sequential model (2-layer LSTM + Dropout + BatchNorm + Dense), trained with early stopping
- **Meta-Ensemble** — Logistic Regression stacking classifier that fuses probability outputs from RF, XGBoost, and LSTM

### `maagap/evaluation.py`
Computes all thesis-specified metrics (Accuracy, Precision, Recall, F1-Score, AUC-ROC, MAE) and generates visualisations (ROC curves, confusion matrices, feature importance bar charts, LSTM training history, risk distribution plots).

---

## Technologies Used

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10+ |
| **ML / Deep Learning** | Scikit-learn, XGBoost, TensorFlow / Keras |
| **Data Processing** | Pandas, NumPy, OpenPyXL |
| **Visualisation** | Matplotlib, Seaborn |
| **Model Persistence** | Joblib |
| **Presentation** | Jupyter Notebook |

---

## Evaluation Metrics (Thesis Objective 2)

All metrics follow the thesis specification using a 70/15/15 train/validation/test split:

- **Accuracy** — Overall correctness of predictions
- **Precision** — Of predicted positives, how many are actually positive
- **Recall** — Of actual positives, how many are correctly identified
- **F1-Score** — Harmonic mean of Precision and Recall
- **AUC-ROC** — Area Under the Receiver Operating Characteristic curve (threshold-independent)
- **MAE** — Mean Absolute Error for delay-day regression

---

## Output Visualisations

All visualisations are saved to the `outputs/` directory:

| File | Description |
|------|-------------|
| `roc_curves_delay.png` | ROC curves comparing all four models |
| `cm_rf_delay.png` | Confusion matrix — Random Forest (delay) |
| `cm_xgb_delay.png` | Confusion matrix — XGBoost (delay) |
| `cm_meta_delay.png` | Confusion matrix — Meta-Ensemble (delay) |
| `cm_rf_risk.png` | Confusion matrix — RF Risk (4-class) |
| `cm_xgb_risk.png` | Confusion matrix — XGB Risk (4-class) |
| `fi_rf_delay.png` | Feature importance — Random Forest (top 20) |
| `fi_xgb_delay.png` | Feature importance — XGBoost (top 20) |
| `lstm_training_history.png` | LSTM training loss and accuracy curves |
| `risk_distribution.png` | Actual vs Predicted risk category distribution |
| `evaluation_report.csv` | Complete metrics table for all models |

---

## Authors

MAAGAP Research Team — College of Information and Communications Technology

---

## License

This project is developed for academic purposes as part of an undergraduate thesis.
