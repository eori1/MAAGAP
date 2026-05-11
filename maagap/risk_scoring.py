"""Objective 3 — Dynamic Risk Scoring Engine.

Transforms continuous probability outputs (delay + cost overrun) into:
  - a continuous risk score in [0, 1]
  - an actionable tier: Low / Medium / High / Critical

Includes basic logic consistency tests to ensure tier assignment respects
the defined threshold boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

from .config import RISK_LABELS, RISK_THRESHOLDS


@dataclass(frozen=True)
class RiskScoringConfig:
    w_delay: float = 0.55
    w_overrun: float = 0.45
    w_importance: float = 0.00  # optional strategic importance weight


def compute_risk_score(
    delay_proba: np.ndarray,
    overrun_proba: np.ndarray,
    strategic_importance: np.ndarray | None = None,
    cfg: RiskScoringConfig = RiskScoringConfig(),
) -> np.ndarray:
    """Compute a continuous risk score in [0, 1]."""
    d = np.asarray(delay_proba, dtype=float)
    o = np.asarray(overrun_proba, dtype=float)
    if d.shape != o.shape:
        raise ValueError("delay_proba and overrun_proba must have the same shape")

    score = cfg.w_delay * d + cfg.w_overrun * o

    if strategic_importance is not None and cfg.w_importance != 0:
        s = np.asarray(strategic_importance, dtype=float)
        if s.shape != d.shape:
            raise ValueError("strategic_importance must have the same shape as probabilities")
        # importance is assumed already in [0, 1]
        score = (1 - cfg.w_importance) * score + cfg.w_importance * s

    return np.clip(score, 0.0, 1.0)


def risk_tier_from_score(score: float) -> str:
    """Convert a scalar score into a tier label using manuscript thresholds."""
    for label, (lo, hi) in RISK_THRESHOLDS.items():
        if lo <= score < hi:
            return label
    return "Critical"


def risk_tiers(scores: np.ndarray) -> np.ndarray:
    """Vectorized tier mapping."""
    s = np.asarray(scores, dtype=float).reshape(-1)
    out = np.empty_like(s, dtype=object)
    for i, v in enumerate(s):
        out[i] = risk_tier_from_score(float(v))
    return out


def logic_consistency_check(scores: np.ndarray, tiers: Iterable[str]) -> Dict[str, int]:
    """Return counts of any tier assignment violations."""
    s = np.asarray(scores, dtype=float).reshape(-1)
    t = np.asarray(list(tiers), dtype=object).reshape(-1)
    if len(s) != len(t):
        raise ValueError("scores and tiers length mismatch")

    violations = 0
    unknown = 0
    for v, tier in zip(s, t):
        if tier not in RISK_LABELS and tier != "Critical":
            unknown += 1
            continue
        lo, hi = RISK_THRESHOLDS.get(tier, (0.0, 1.0))
        ok = (lo <= v < hi) if tier != "Critical" else (v >= RISK_THRESHOLDS["Critical"][0])
        if not ok:
            violations += 1

    return {"violations": int(violations), "unknown_tier_labels": int(unknown)}

