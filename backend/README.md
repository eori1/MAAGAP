# MAAGAP — Predictive Risk Assessment and Resource Optimization System

**Machine Analytics for Allocation, Governance and Assessment of Projects**

A multi-stage machine learning and operations research framework for predicting government project delays, scoring risk, and optimizing resource allocation. Developed as an undergraduate thesis at the College of Information and Communications Technology.

---

## Overview

MAAGAP implements a comprehensive pipeline addressing the first four objectives of the thesis:

1. **Objective 1 & 2 (Predictive Framework & Evaluation):** A stacking meta-ensemble that fuses Random Forest and XGBoost (static features) with a Long Short-Term Memory (LSTM) neural network (temporal quarterly sequences).
2. **Objective 3 (Dynamic Risk Scoring Engine):** Translates predicted probabilities into actionable management tiers (Low, Medium, High, Critical) with strict threshold logic consistency.
3. **Objective 4 (Resource Allocation Optimization):** A Linear Programming (LP) model that optimizes the deployment of limited inspectors to high-utility projects, benchmarked against manual/random allocation.

---

## Key Results Summary

### 1. Meta-Ensemble Dominance
The three-model stacking architecture consistently outperforms any individual base model. The Logistic Regression meta-learner assigned the following contributions (based on learned coefficients and output variance):
- **LSTM (72.51%):** The primary driver of predictive accuracy, proving that temporal sequential monitoring data (quarter-by-quarter slippage) is highly predictive.
- **Random Forest (14.74%) & XGBoost (12.75%):** Provide complementary stability and feature-interaction signals from static project attributes (budget, contractor reliability, typhoon exposure).

**Binary Delay Prediction (Test Set):**
- **Accuracy:** ~89.11%
- **F1-Score:** ~0.6260
- **AUC-ROC:** ~0.9023

### 2. Risk Categorisation & Logic Consistency
The pipeline successfully classifies projects into 4 tiers: Low, Medium, High, and Critical. 
- A programmatic **Logic Consistency Check** confirms 0 violations across the test set, proving the threshold boundaries act exactly as defined in the manuscript.

### 3. Allocation Optimization (LP vs Baseline)
- **Efficiency Improvement:** The LP optimization achieved a **267% improvement** in captured risk utility compared to the baseline manual/random allocation strategy, far exceeding the ≥15% target.
- **Monte Carlo Robustness:** Across a 200-iteration simulation with Gaussian noise (σ=0.05), 100% of the runs exceeded the 15% improvement target, proving the LP approach is highly robust against model uncertainty.

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
│   ├── risk_scoring.py            # Objective 3: Dynamic Risk Scoring Engine
│   ├── optimization.py            # Objective 4: LP Resource Allocation
│   └── evaluation.py              # Metrics, Plotly visualisations, reports
│
├── scripts/
│   ├── get_ensemble_pct.py        # Computes actual Meta-Ensemble percentages
│   └── ...                        # Notebook utilities
│
├── main.py                        # Full pipeline orchestration script (MainPipeline)
├── MAAGAP_Objective1_Clean.ipynb  # Jupyter Notebook interactive version
├── requirements.txt               # Python dependencies
├── walkthrough.md                 # Complete defense guide and documentation
├── chapter4_draft.md              # Manuscript Chapter 4 implementation draft
└── README.md                      # This file
```

> **Note:** Trained models in `models/` are git-ignored due to file size (~50MB total). Run the pipeline to regenerate them.

---

## Data Sources

| Source | Description | Usage |
|--------|-------------|-------|
| **PPDO Iloilo** | Excel workbooks (e.g., `MONITORING REPORT Con` and `Fund Transfer Con`) | Extract statistical distributions (budgets, municipalities, funding) |
| **PAGASA** | Historical monthly rainfall (mm) and typhoon exposure days | External contextual variable — weather risk |
| **PSA** | Consumer Price Index (CPI) and Construction Materials RPI | External contextual variable — economic conditions |
| **Synthetic** | 3,000 projects with quarterly monitoring data | Training, validation, and testing dataset generated from real PPDO distributions |

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

This will execute Steps 1 through 9:
1. Load real PPDO 2026 dataset and extract distributions
2. Generate 3,000 synthetic projects with quarterly records
3. Engineer features (static and temporal)
4. Split data (70/15/15)
5. Train models (RF, XGB, LSTM, Meta-Ensemble)
6. Evaluate predictive performance
7. Run Risk Scoring Engine (Objective 3)
8. Run LP Optimization and Monte Carlo (Objective 4)
9. Save all plots and reports to `outputs/`

### Defense Walkthrough
For a complete guide to presenting this codebase to panelists, refer to the `walkthrough.md` file located in the root directory.

---

## Authors

**MAAGAP Research Team** — College of Information and Communications Technology

## License

This project is developed for academic purposes as part of an undergraduate thesis.
