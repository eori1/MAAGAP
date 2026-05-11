"""MAAGAP: Machine Analytics for Allocation, Governance and Assessment of Projects.

Multi-stage predictive framework for government project risk assessment.

Objectives:
  1. Multi-stage predictive framework (RF + XGBoost + LSTM + Meta-ensemble)
  2. Model evaluation (Accuracy, Precision, Recall, F1-Score, AUC-ROC, MAE)
  3. Dynamic Risk Scoring Engine (Low / Medium / High / Critical tiers)
  4. LP Resource Allocation Optimization (>=15% improvement over baseline)
"""

from . import (
    config,
    data_preprocessing,
    synthetic_generator,
    feature_engineering,
    models,
    evaluation,
    risk_scoring,
    optimization,
)
