"""Unit tests for SHAP Explainability Service."""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from maagap.explainability import ExplanationService


def test_shap_tree_explainer_formatting():
    # Fit simple dummy RF
    X = np.random.randn(50, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)
    
    feature_names = ["f1", "f2", "f3", "f4", "f5"]
    shap_vals, base_val = ExplanationService.explain_random_forest(rf, X, feature_names)
    
    assert shap_vals.shape == (50, 5)
    
    json_summary = ExplanationService.format_shap_json(shap_vals, feature_names, base_val, sample_idx=0)
    
    assert "base_value" in json_summary
    assert "top_contributing_factors" in json_summary
    assert len(json_summary["top_contributing_factors"]) == 5
