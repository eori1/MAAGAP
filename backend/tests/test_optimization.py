"""Unit tests for Linear Programming Resource Allocation Optimization engine."""

import pytest
import numpy as np
import pandas as pd

from maagap.synthetic_generator import SyntheticDataGenerator
from maagap.risk_scoring import compute_all_risk_scores, get_risk_tier
from maagap.optimization import ResourceOptimizer


@pytest.fixture(scope="module")
def sample_projects(tmp_path_factory):
    gen = SyntheticDataGenerator(seed=42)
    out_dir = tmp_path_factory.mktemp("test_optimization_data")
    df_proj, _ = gen.generate_synthetic_dataset(n_projects=60, output_dir=str(out_dir))
    p_delay = df_proj["delay_probability"].values
    p_overrun = df_proj["overrun_probability"].values
    scores = compute_all_risk_scores(p_delay, p_overrun)
    tiers = np.vectorize(get_risk_tier)(scores)
    df_proj["risk_score"] = scores
    df_proj["risk_tier"] = tiers
    return df_proj


def test_lp_optimization_feasibility(sample_projects):
    optimizer = ResourceOptimizer()
    budgets = sample_projects["approved_budget"].values
    risk_scores = sample_projects["risk_score"].values
    total_budget = np.sum(budgets) * 0.40

    selection_mask, status = optimizer.optimize_allocation(budgets, risk_scores, total_budget)
    
    assert status == 1, f"Expected LP status 1 (Optimal), got {status}"
    assert selection_mask.shape[0] == len(budgets)
    assert np.sum(selection_mask) > 0



def test_monte_carlo_baseline_and_efficiency(sample_projects):
    optimizer = ResourceOptimizer()
    budgets = sample_projects["approved_budget"].values
    risk_scores = sample_projects["risk_score"].values
    total_budget = np.sum(budgets) * 0.40

    base_eff, base_sel = optimizer.run_monte_carlo_baseline(budgets, risk_scores, total_budget, n_iterations=20)
    lp_sel, status = optimizer.optimize_allocation(budgets, risk_scores, total_budget)
    lp_eff = optimizer.evaluate_allocation_efficiency(risk_scores, lp_sel)
    
    improvement = optimizer.analyze_improvement(base_eff, lp_eff)

    assert base_eff > 0.0
    assert lp_eff >= base_eff
    assert improvement["improvement_pct"] >= 0.0
