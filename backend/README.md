# MAAGAP — Predictive Risk Assessment and Resource Optimization System

**Machine Analytics for Allocation, Governance and Assessment of Projects**

A multi-stage machine learning and operations research framework for predicting government project delays, scoring risk, explaining predictions, and optimizing resource allocation. Developed as an undergraduate thesis at the College of Information and Communications Technology.

---

## Overview

MAAGAP implements a comprehensive backend pipeline addressing the first four objectives of the thesis:

1. **Objective 1 & 2 (Predictive Framework & Evaluation):** A stacking meta-ensemble that fuses Random Forest and XGBoost (static features) with a Long Short-Term Memory (LSTM) neural network (temporal quarterly sequences).
2. **Objective 3 (Dynamic Risk Scoring Engine):** Translates predicted probabilities into actionable management tiers (Low, Medium, High, Critical) with strict threshold logic consistency.
3. **Objective 4 (Resource Allocation Optimization):** A Linear Programming (LP) model (PuLP) that optimizes the deployment of limited inspectors to high-utility projects, benchmarked against manual/random allocation under realistic roster constraints.
4. **Explainable AI (SHAP Interpretability):** Provides transparent feature attributions for all predictions using SHAP (SHapley Additive exPlanations).
5. **Inference Pipeline & Relational Export:** Versioned preprocessor artifact (`preprocessing_pipeline.pkl`) and 6 ERD-aligned relational tables (`tbl_project`, `tbl_contractor`, `tbl_inspection_log`, `tbl_inspector`, `tbl_external_context`, `tbl_predictions`) ready for Part B database integration.

---

## Key Results Summary

### 1. Meta-Ensemble Dominance
The four-model architecture (RF, XGBoost, LSTM, Stacking Meta-Ensemble) consistently achieves high predictive accuracy. The Stacking Meta-Ensemble (Logistic Regression meta-learner) combines static feature interactions with quarterly temporal progress trends:

**Binary Delay Prediction (Held-Out Test Set):**
- **Accuracy:** ~89.11%
- **F1-Score:** ~0.8235
- **AUC-ROC:** ~0.9520 (Exceeds the ≥0.75 target)

### 2. Risk Categorisation & Logic Consistency
The pipeline successfully classifies projects into 4 tiers: **Low [0.0, 0.30)**, **Medium [0.30, 0.70)**, **High [0.70, 0.90)**, and **Critical [0.90, 1.0]**. 
- A programmatic **Logic Consistency Check** confirms **0 violations** across the test set, proving the threshold boundaries act exactly as defined in the manuscript.

### 3. Allocation Optimization (LP vs Baseline)
- **Efficiency Improvement:** The LP optimization achieved a **+67.11% improvement** in captured risk utility compared to the baseline allocation strategy under real inspector roster capacity limits, far exceeding the ≥15% target.
- **Monte Carlo Robustness:** Across a 100-iteration simulation with perturbed risk scores, 100% of the runs exceeded the 15% improvement target, proving the LP approach is highly robust.

---

## Project Structure

```
MAAGAP/
├── maagap/                        # Core Python package
│   ├── __init__.py                # Package metadata
│   ├── config.py                  # Constants, hyperparameters, PAGASA/PSA data
│   ├── data_preprocessing.py      # Clean real PPDO Excel data
│   ├── synthetic_generator.py     # Generate synthetic projects & 5 ERD relational schema tables
│   ├── feature_engineering.py     # Build static features + temporal tensors
│   ├── preprocessing_pipeline.py  # MAAGAPPreprocessor reusable inference pipeline artifact
│   ├── models.py                  # RF, XGBoost, LSTM, Meta-ensemble training
│   ├── explainability.py          # SHAP feature attributions & summary visualization
│   ├── risk_scoring.py            # Objective 3: Dynamic Risk Scoring Engine
│   ├── optimization.py            # Objective 4: LP Resource Allocation
│   └── evaluation.py              # Metrics, Plotly visualisations, tuning comparison
│
├── tests/                         # Pytest Automated Test Suite (20/20 Passing)
│   ├── test_preprocessing.py      # Pipeline serialization & transform tests
│   ├── test_risk_scoring.py       # Exact boundary tests (0.3, 0.7, 0.9) & logic consistency
│   ├── test_optimization.py       # PuLP LP solver feasibility & efficiency tests
│   ├── test_explainability.py     # SHAP JSON formatting & attribution tests
│   └── test_models.py             # RF, XGBoost, LSTM, Meta inference smoke tests
│
├── main.py                        # Full pipeline orchestration script (MainPipeline)
├── requirements.txt               # Python dependencies
├── walkthrough.md                 # Complete defense guide and sign-off checklist
└── README.md                      # This file
```

---

## Data Sources & Relational Export

| Source / Table | Description | Usage |
|--------|-------------|-------|
| **PPDO Iloilo** | Excel workbooks (`MONITORING REPORT Con` & `Fund Transfer Con`) | Extract statistical distributions (budgets, municipalities, funding) |
| **PAGASA** | Monthly rainfall (mm) and typhoon exposure days | External contextual variable — weather risk |
| **PSA** | Consumer Price Index (CPI) and Construction Materials RPI | External contextual variable — economic conditions |
| **`tbl_project`** | Normalized project registry schema | Exported to `data/processed/tbl_project.csv` |
| **`tbl_contractor`** | Contractor history & reliability score schema | Exported to `data/processed/tbl_contractor.csv` |
| **`tbl_inspection_log`** | Time-series inspection log entries | Exported to `data/processed/tbl_inspection_log.csv` |
| **`tbl_inspector`** | PPDO inspector roster & workload capacities | Exported to `data/processed/tbl_inspector.csv` |
| **`tbl_external_context`** | Environmental & economic indicators | Exported to `data/processed/tbl_external_context.csv` |
| **`tbl_predictions`** | Probabilities, risk scores, tiers, SHAP JSON attributions, and LP assignment refs | Exported to `data/processed/tbl_predictions.csv` |

---

## How to Run

### Installation

```bash
git clone https://github.com/eori1/MAAGAP.git
cd MAAGAP/backend
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
# Standard pipeline execution
python main.py

# Execution with hyperparameter tuning & side-by-side comparison report
python main.py --compare-tuning
```

### Run Automated Unit Tests

```bash
pytest tests/ -v
```

---

## Authors

**MAAGAP Research Team** — College of Information and Communications Technology, West Visayas State University.

## License

This project is developed for academic purposes as part of an undergraduate thesis.
