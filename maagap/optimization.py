"""Objective 4 — Linear Programming Optimization for Resource Allocation.

Implements a simple LP that allocates limited inspection capacity to projects
to maximize expected portfolio benefit.

This is designed for thesis demonstration using simulated scenarios:
  - baseline: manual/random allocation
  - optimized: LP allocation (PuLP)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class OptimizationConfig:
    inspectors_available: int = 10
    inspections_per_inspector: int = 6  # per planning period
    min_assignments: int = 0  # optional minimum number of projects to cover
    seed: int = 42


def _tier_weight(tier: str) -> float:
    # Prioritize higher-risk tiers more heavily (manuscript-aligned)
    return {
        "Low": 1.0,
        "Medium": 2.0,
        "High": 3.0,
        "Critical": 4.0,
    }.get(str(tier), 1.0)


def optimize_inspection_allocation(
    risk_score: np.ndarray,
    risk_tier: np.ndarray,
    strategic_importance: np.ndarray | None = None,
    cfg: OptimizationConfig = OptimizationConfig(),
) -> Dict[str, object]:
    """Select projects to inspect under capacity constraints via LP.

    Decision variable x_j in {0,1} indicates whether project j is selected.
    Objective: maximize sum_j x_j * utility_j
    """
    try:
        import pulp  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PuLP is required for optimization. Install via `pip install pulp`.") from e

    rs = np.asarray(risk_score, dtype=float).reshape(-1)
    tiers = np.asarray(risk_tier, dtype=object).reshape(-1)
    n = len(rs)
    if len(tiers) != n:
        raise ValueError("risk_score and risk_tier length mismatch")

    if strategic_importance is None:
        imp = np.zeros(n, dtype=float)
    else:
        imp = np.asarray(strategic_importance, dtype=float).reshape(-1)
        if len(imp) != n:
            raise ValueError("strategic_importance length mismatch")
        imp = np.clip(imp, 0.0, 1.0)

    # Utility favors high risk score + tier + importance.
    tier_w = np.array([_tier_weight(t) for t in tiers], dtype=float)
    utility = (0.65 * rs + 0.25 * (tier_w / 4.0) + 0.10 * imp)

    capacity = int(cfg.inspectors_available) * int(cfg.inspections_per_inspector)

    prob = pulp.LpProblem("maagap_inspection_allocation", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", range(n), lowBound=0, upBound=1, cat=pulp.LpBinary)

    prob += pulp.lpSum([x[j] * float(utility[j]) for j in range(n)])
    prob += pulp.lpSum([x[j] for j in range(n)]) <= capacity
    if cfg.min_assignments and cfg.min_assignments > 0:
        prob += pulp.lpSum([x[j] for j in range(n)]) >= int(cfg.min_assignments)

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    selected = np.array([int(pulp.value(x[j]) or 0) for j in range(n)], dtype=int)
    selected_idx = np.where(selected == 1)[0]

    return {
        "capacity": capacity,
        "selected_count": int(selected.sum()),
        "selected_idx": selected_idx,
        "utility": utility,
        "objective_value": float(pulp.value(prob.objective) or 0.0),
        "status": pulp.LpStatus.get(prob.status, str(prob.status)),
    }


def baseline_manual_allocation(
    risk_score: np.ndarray,
    cfg: OptimizationConfig = OptimizationConfig(),
) -> Dict[str, object]:
    """Baseline 'manual' allocation = random selection under same capacity."""
    rs = np.asarray(risk_score, dtype=float).reshape(-1)
    n = len(rs)
    capacity = int(cfg.inspectors_available) * int(cfg.inspections_per_inspector)
    rng = np.random.RandomState(cfg.seed)
    idx = rng.permutation(n)[: min(capacity, n)]
    selected = np.zeros(n, dtype=int)
    selected[idx] = 1
    return {"capacity": capacity, "selected_count": int(selected.sum()), "selected_idx": idx}


def allocation_efficiency(
    selected_idx: np.ndarray,
    risk_score: np.ndarray,
    risk_tier: np.ndarray,
    strategic_importance: np.ndarray | None = None,
) -> Dict[str, float]:
    """Compute weighted risk captured per selected project (thesis efficiency metric)."""
    rs    = np.asarray(risk_score, dtype=float).reshape(-1)
    tiers = np.asarray(risk_tier,  dtype=object).reshape(-1)
    sel   = np.asarray(selected_idx, dtype=int)

    if strategic_importance is None:
        imp = np.zeros_like(rs)
    else:
        imp = np.clip(np.asarray(strategic_importance, dtype=float).reshape(-1), 0, 1)

    tier_w   = np.array([_tier_weight(t) for t in tiers], dtype=float)
    captured = 0.70 * rs + 0.20 * (tier_w / 4.0) + 0.10 * imp
    if len(sel) == 0:
        return {"avg_captured_risk": 0.0}
    return {"avg_captured_risk": float(np.mean(captured[sel]))}


def compute_efficiency_improvement(
    selected_lp: np.ndarray,
    selected_base: np.ndarray,
    risk_score: np.ndarray,
    risk_tier: np.ndarray,
    strategic_importance: np.ndarray | None = None,
) -> Dict[str, float]:
    """Return efficiency metrics for both allocations and the % improvement."""
    eff_lp   = allocation_efficiency(selected_lp,   risk_score, risk_tier, strategic_importance)
    eff_base = allocation_efficiency(selected_base, risk_score, risk_tier, strategic_importance)
    base_val = eff_base["avg_captured_risk"]
    lp_val   = eff_lp["avg_captured_risk"]
    improvement = (lp_val - base_val) / max(base_val, 1e-9) * 100.0
    return {
        "baseline_efficiency":  round(base_val,   4),
        "lp_efficiency":        round(lp_val,     4),
        "improvement_pct":      round(improvement, 4),
        "target_met":           improvement >= 15.0,
    }


def monte_carlo_robustness(
    risk_score: np.ndarray,
    risk_tier: np.ndarray,
    n_simulations: int = 200,
    noise_std: float = 0.05,
    cfg: OptimizationConfig = OptimizationConfig(),
    strategic_importance: np.ndarray | None = None,
) -> Dict[str, object]:
    """Monte Carlo simulation: perturb risk scores and re-run LP vs baseline.

    Tests whether the >=15% efficiency improvement is robust to uncertainty
    in the risk score estimates (e.g. model confidence intervals).
    """
    rng = np.random.RandomState(cfg.seed)
    rs  = np.asarray(risk_score, dtype=float).reshape(-1)

    lp_efficiencies   = []
    base_efficiencies = []
    improvements      = []

    for _ in range(n_simulations):
        noise     = rng.normal(0.0, noise_std, size=len(rs))
        perturbed = np.clip(rs + noise, 0.0, 1.0)

        try:
            res_lp   = optimize_inspection_allocation(
                perturbed, risk_tier, strategic_importance, cfg
            )
            res_base = baseline_manual_allocation(perturbed, cfg)

            eff_lp   = allocation_efficiency(
                res_lp["selected_idx"],   perturbed, risk_tier, strategic_importance
            )["avg_captured_risk"]
            eff_base = allocation_efficiency(
                res_base["selected_idx"], perturbed, risk_tier, strategic_importance
            )["avg_captured_risk"]

            lp_efficiencies.append(eff_lp)
            base_efficiencies.append(eff_base)
            if eff_base > 1e-9:
                improvements.append((eff_lp - eff_base) / eff_base * 100.0)
        except Exception:
            pass

    impr = np.array(improvements, dtype=float)
    return {
        "n_simulations":        n_simulations,
        "n_successful":         len(impr),
        "mean_improvement_pct": float(np.mean(impr))  if len(impr) else 0.0,
        "std_improvement_pct":  float(np.std(impr))   if len(impr) else 0.0,
        "min_improvement_pct":  float(np.min(impr))   if len(impr) else 0.0,
        "max_improvement_pct":  float(np.max(impr))   if len(impr) else 0.0,
        "pct_above_15":         float((impr >= 15.0).mean() * 100) if len(impr) else 0.0,
        "lp_mean_efficiency":   float(np.mean(lp_efficiencies))   if lp_efficiencies   else 0.0,
        "base_mean_efficiency": float(np.mean(base_efficiencies)) if base_efficiencies else 0.0,
    }


