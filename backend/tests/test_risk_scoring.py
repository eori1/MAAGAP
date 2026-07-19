"""Unit tests for Risk Scoring Engine threshold boundaries and logic consistency."""

import pytest
import numpy as np
from maagap.risk_scoring import get_risk_tier, compute_all_risk_scores, check_logic_consistency


@pytest.mark.parametrize("score,expected_tier", [
    (0.00, "Low"),
    (0.15, "Low"),
    (0.29, "Low"),
    (0.30, "Medium"),
    (0.50, "Medium"),
    (0.69, "Medium"),
    (0.70, "High"),
    (0.85, "High"),
    (0.89, "High"),
    (0.90, "Critical"),
    (0.95, "Critical"),
    (1.00, "Critical"),
])
def test_risk_threshold_boundaries(score, expected_tier):
    tier = get_risk_tier(score)
    assert tier == expected_tier, f"Score {score} expected {expected_tier}, got {tier}"


def test_logic_consistency_zero_violations():
    # Generate test array of probabilities
    np.random.seed(42)
    p_delay = np.random.uniform(0, 1, 100)
    p_overrun = np.random.uniform(0, 1, 100)
    
    scores = compute_all_risk_scores(p_delay, p_overrun)
    tiers = np.vectorize(get_risk_tier)(scores)
    consistency = check_logic_consistency(scores, tiers)
    
    assert consistency["is_consistent"], f"Logic consistency check failed with violations: {consistency['violations']}"
    assert consistency["violations"] == 0


