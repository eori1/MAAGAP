import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Cell 1: Markdown
text_1 = """# MAAGAP — Objectives 1–4: Multi-Stage Predictive Framework

**Machine Analytics for Allocation, Governance and Assessment of Projects**

This notebook implements and evaluates all four study objectives using the newly refactored Object-Oriented Pipeline:
1. **Objective 1:** Multi-stage predictive framework (RF + XGBoost + LSTM + Meta-ensemble)
2. **Objective 2:** Model evaluation — Accuracy, Precision, Recall, F1-Score, AUC-ROC, MAE
3. **Objective 3:** Dynamic Risk Scoring Engine
4. **Objective 4:** LP Resource Allocation Optimization
"""
nb.cells.append(nbf.v4.new_markdown_cell(text_1))

# Cell 2: Imports
code_2 = """import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from maagap.config import SEED
from maagap.logger import get_logger, banner
from maagap.data_preprocessing import DataPreprocessor
from maagap.synthetic_generator import SyntheticDataGenerator
from maagap.feature_engineering import FeatureEngineer, split_data
from maagap.models import TreeModelTrainer, LSTMTrainer, MetaEnsembleTrainer
from maagap.risk_scoring import compute_all_risk_scores, get_risk_tier, check_logic_consistency
from maagap.optimization import ResourceOptimizer
from maagap.evaluation import Evaluator, Visualizer

np.random.seed(SEED)
print("All modules imported successfully.")"""
nb.cells.append(nbf.v4.new_code_cell(code_2))

# Cell 3: Markdown
text_3 = """## Step 1 — Data Pipeline
Load real PPDO records, extract distributions, and generate the synthetic project & quarterly data. We then build static and temporal feature matrices."""
nb.cells.append(nbf.v4.new_markdown_cell(text_3))

# Cell 4: Data Pipeline
code_4 = """preprocessor = DataPreprocessor()
df_real = preprocessor.load_and_clean_ppdo()
dist = preprocessor.extract_distributions(df_real)

generator = SyntheticDataGenerator()
df_projects, df_quarterly = generator.generate_synthetic_dataset(dist, n_projects=3000)

fe = FeatureEngineer()
X_static, static_cols, _, _ = fe.build_static_features(df_projects)
X_temporal, temporal_cols, _ = fe.build_temporal_sequences(df_projects, df_quarterly)
y_delay, y_overrun, y_risk_true, _, _ = fe.build_targets(df_projects)

train_idx, val_idx, test_idx = split_data(len(X_static))
print(f"Train samples: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")"""
nb.cells.append(nbf.v4.new_code_cell(code_4))

# Cell 5: Markdown
text_5 = """## Step 2 — Objective 1: Predictive Framework
Train Stage 1 (RF, XGBoost) and Stage 2 (LSTM) models, then fuse them via a Meta-Ensemble."""
nb.cells.append(nbf.v4.new_markdown_cell(text_5))

# Cell 6: Models
code_6 = """# Stage 1: Random Forest
rf_model = TreeModelTrainer.train_random_forest(X_static[train_idx], y_delay[train_idx], task="delay_nb", tune=False)
rf_prob_test = rf_model.predict_proba(X_static[test_idx])[:, 1]
rf_pred_test = rf_model.predict(X_static[test_idx])

# Stage 1: XGBoost
xgb_model = TreeModelTrainer.train_xgboost(X_static[train_idx], y_delay[train_idx], task="delay_nb", tune=False)
xgb_prob_test = xgb_model.predict_proba(X_static[test_idx])[:, 1]
xgb_pred_test = xgb_model.predict(X_static[test_idx])

# Stage 2: LSTM
lstm_model, lstm_history, _ = LSTMTrainer.train_lstm(
    X_temporal[train_idx], y_delay[train_idx],
    X_temporal[val_idx], y_delay[val_idx],
    task="delay_nb", tune=False
)
lstm_prob_test = lstm_model.predict(X_temporal[test_idx]).flatten()
lstm_pred_test = (lstm_prob_test >= 0.5).astype(int)

# Meta-Ensemble
rf_prob_train = rf_model.predict_proba(X_static[train_idx])[:, 1]
xgb_prob_train = xgb_model.predict_proba(X_static[train_idx])[:, 1]
lstm_prob_train = lstm_model.predict(X_temporal[train_idx]).flatten()

meta_model = MetaEnsembleTrainer.train_meta_ensemble(
    rf_prob_train, xgb_prob_train, lstm_prob_train, y_delay[train_idx],
    artifact_name="meta_ensemble_nb.pkl"
)
meta_pred_test, meta_prob_full = MetaEnsembleTrainer.predict_meta(
    meta_model, rf_prob_test, xgb_prob_test, lstm_prob_test
)
meta_prob_test = meta_prob_full[:, 1]

print("Model training complete.")"""
nb.cells.append(nbf.v4.new_code_cell(code_6))

# Cell 7: Markdown
text_7 = """## Step 3 — Objective 2: Evaluation Metrics
We compute standard classification metrics and visualize ROC curves."""
nb.cells.append(nbf.v4.new_markdown_cell(text_7))

# Cell 8: Eval
code_8 = """evaluator = Evaluator()
visualizer = Visualizer()

models_data = [
    ("Random Forest", rf_pred_test, rf_prob_test),
    ("XGBoost", xgb_pred_test, xgb_prob_test),
    ("LSTM", lstm_pred_test, lstm_prob_test),
    ("Meta-Ensemble", meta_pred_test, meta_prob_test),
]

metrics_list = []
for name, preds, probs in models_data:
    m = evaluator.binary_metrics(y_delay[test_idx], preds, probs, label=name)
    metrics_list.append(m)

df_metrics = pd.DataFrame(metrics_list)
display(df_metrics)"""
nb.cells.append(nbf.v4.new_code_cell(code_8))

# Cell 9: Markdown
text_9 = """## Step 4 — Objective 3: Dynamic Risk Engine & Objective 4: Optimization
Convert probabilities to continuous risk scores (0 to 1), assign Risk Tiers, and allocate resources via LP Optimization."""
nb.cells.append(nbf.v4.new_markdown_cell(text_9))

# Cell 10: Risk & Opt
code_10 = """# Risk Engine
risk_scores_test = compute_all_risk_scores(rf_prob_test, xgb_prob_test, lstm_prob_test)
v_get_tier = np.vectorize(get_risk_tier)
risk_tiers_test = v_get_tier(risk_scores_test)

consistency = check_logic_consistency(risk_scores_test, risk_tiers_test)

# Optimization
budgets_test = df_projects.iloc[test_idx]["approved_budget"].values
agency_names_test = df_projects.iloc[test_idx]["implementing_agency"].values
total_budget = np.sum(budgets_test) * 0.40

opt = ResourceOptimizer()
base_eff, base_sel = opt.run_monte_carlo_baseline(budgets_test, risk_scores_test, total_budget)

lp_sel, lp_status = opt.optimize_allocation(
    budgets_test, risk_scores_test, total_budget,
    agency_ids=agency_names_test, max_projects_per_agency=15
)

lp_eff = opt.evaluate_allocation_efficiency(risk_scores_test, lp_sel)
improvement = opt.analyze_improvement(base_eff, lp_eff)

print(f"Baseline Efficiency: {base_eff:.4f}")
print(f"LP Optimization Efficiency: {lp_eff:.4f}")
print(f"Improvement: +{improvement['improvement_pct']:.2f}%")"""
nb.cells.append(nbf.v4.new_code_cell(code_10))

# Save notebook
with open('c:\\Users\\ASUS\\Desktop\\Tisis\\MAAGAP_Objective1_Clean.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook created successfully.")
