"""Objective 3 -- Dynamic Risk Scoring Engine.

Computes a continuous risk score (0 to 1) based on weighted ensemble outputs.
Also implements logic checking to ensure generated tiers are consistent.
"""

import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional

from .config import RISK_THRESHOLDS
from .logger import get_logger

logger = get_logger(__name__)


def compute_risk_score(rf_prob: float, xgb_prob: float, lstm_prob: float, rf_weight: float = 0.35, xgb_weight: float = 0.35, lstm_weight: float = 0.30) -> float:
    """Compute weighted composite risk score from individual model delay probabilities."""
    return float(np.clip(
        rf_prob * rf_weight + xgb_prob * xgb_weight + lstm_prob * lstm_weight,
        0.0, 1.0
    ))


def get_risk_tier(score: float) -> str:
    """Map a continuous risk score to a categorical tier using defined thresholds."""
    if score >= RISK_THRESHOLDS["Critical"][0]:
        return "Critical"
    elif score >= RISK_THRESHOLDS["High"][0]:
        return "High"
    elif score >= RISK_THRESHOLDS["Medium"][0]:
        return "Medium"
    return "Low"


def compute_all_risk_scores(rf_probs: np.ndarray, xgb_probs: np.ndarray, lstm_probs: np.ndarray, weights: Optional[Tuple[float, float, float]] = None) -> np.ndarray:
    """Vectorised computation of risk scores for an entire dataset."""
    w_rf, w_xgb, w_lstm = weights if weights else (0.35, 0.35, 0.30)
    scores = (
        np.asarray(rf_probs) * w_rf
        + np.asarray(xgb_probs) * w_xgb
        + np.asarray(lstm_probs) * w_lstm
    )
    return np.clip(scores, 0.0, 1.0)


def check_logic_consistency(risk_scores: np.ndarray, assigned_tiers: np.ndarray) -> Dict[str, Any]:
    """Verify that assigned tiers perfectly match the score thresholds."""
    violations = 0
    details = []

    for idx, (score, tier) in enumerate(zip(risk_scores, assigned_tiers)):
        expected = get_risk_tier(float(score))
        if tier != expected:
            violations += 1
            details.append({
                "index": idx,
                "score": float(score),
                "assigned": tier,
                "expected": expected
            })

    is_consistent = (violations == 0)
    if is_consistent:
        logger.info("Logic Check PASSED: 100% of risk tiers match threshold rules.")
    else:
        logger.warning(f"Logic Check FAILED: {violations} logic violations detected!")

    return {
        "is_consistent": is_consistent,
        "violations": violations,
        "details": details,
    }
