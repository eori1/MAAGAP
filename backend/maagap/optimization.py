"""Objective 4 -- LP Optimization for Resource Allocation.

Formulates and solves the Knapsack-like LP problem to maximise risk mitigation 
subject to budget constraints.
"""

import numpy as np
import pandas as pd
import pulp
import logging
from typing import Dict, Any, List, Optional, Tuple

from .logger import get_logger

logger = get_logger(__name__)


class ResourceOptimizer:
    """Class to formulate and solve the Resource Allocation LP problem."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def optimize_allocation(
        self,
        budgets: np.ndarray,
        risk_scores: np.ndarray,
        total_budget_limit: float,
        agency_ids: Optional[np.ndarray] = None,
        max_projects_per_agency: Optional[int] = None,
        min_critical_coverage: float = 0.50,
        critical_threshold: float = 0.90,
    ) -> Tuple[np.ndarray, pulp.LpStatus]:
        """
        Solve integer LP for optimal resource allocation.
        """
        logger.info("Formulating Resource Allocation LP problem...")
        n = len(budgets)
        
        prob = pulp.LpProblem("Resource_Allocation_Optimization", pulp.LpMaximize)
        
        # Decision variables: x[i] in {0, 1}
        x = pulp.LpVariable.dicts("project", range(n), cat="Binary")

        # Objective: Maximise total captured risk score
        prob += pulp.lpSum([risk_scores[i] * x[i] for i in range(n)]), "Total_Mitigated_Risk"

        # Constraint 1: Global Budget
        prob += pulp.lpSum([budgets[i] * x[i] for i in range(n)]) <= total_budget_limit, "Global_Budget_Constraint"

        # Constraint 2: Agency Capacity Limits (Optional)
        if agency_ids is not None and max_projects_per_agency is not None:
            unique_agencies = np.unique(agency_ids)
            for a in unique_agencies:
                idx_a = np.where(agency_ids == a)[0]
                prob += pulp.lpSum([x[i] for i in idx_a]) <= max_projects_per_agency, f"Capacity_{a}"

        # Constraint 3: Minimum Critical Project Coverage
        critical_idx = np.where(risk_scores >= critical_threshold)[0]
        if len(critical_idx) > 0:
            min_critical_count = int(np.ceil(len(critical_idx) * min_critical_coverage))
            prob += pulp.lpSum([x[i] for i in critical_idx]) >= min_critical_count, "Min_Critical_Coverage"

        # Solve using default CBC solver
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=60)
        prob.solve(solver)
        
        status = pulp.LpStatus[prob.status]
        logger.info(f"LP Solver Status: {status}")

        if prob.status != pulp.LpStatusOptimal:
            logger.warning("Optimization did not find an optimal solution!")
            return np.zeros(n, dtype=int), prob.status

        selected = np.array([int(x[i].varValue) for i in range(n)])
        return selected, prob.status

    def evaluate_allocation_efficiency(self, risk_scores: np.ndarray, selected_idx: np.ndarray) -> float:
        """Calculate average risk score of selected projects (Efficiency)."""
        if sum(selected_idx) == 0:
            return 0.0
        return float(np.mean(risk_scores[selected_idx == 1]))

    def run_monte_carlo_baseline(self, budgets: np.ndarray, risk_scores: np.ndarray, total_budget_limit: float, n_iterations: int = 100) -> Tuple[float, np.ndarray]:
        """Run random allocations to establish a baseline efficiency."""
        logger.info(f"Running {n_iterations} Monte Carlo baseline allocations...")
        n = len(budgets)
        baseline_efficiencies = []
        best_efficiency = 0.0
        best_selection = np.zeros(n, dtype=int)

        for _ in range(n_iterations):
            idx = self.rng.permutation(n)
            cum_budget = 0.0
            sel = np.zeros(n, dtype=int)
            
            for i in idx:
                if cum_budget + budgets[i] <= total_budget_limit:
                    sel[i] = 1
                    cum_budget += budgets[i]
            
            eff = self.evaluate_allocation_efficiency(risk_scores, sel)
            baseline_efficiencies.append(eff)
            
            if eff > best_efficiency:
                best_efficiency = eff
                best_selection = sel.copy()

        avg_baseline = float(np.mean(baseline_efficiencies))
        logger.info(f"Average baseline efficiency: {avg_baseline:.4f}")
        return avg_baseline, best_selection

    def analyze_improvement(self, base_eff: float, lp_eff: float) -> Dict[str, float]:
        """Compute relative percentage improvement of LP over baseline."""
        if base_eff <= 0:
            return {"improvement_pct": 0.0}
        
        imp_pct = ((lp_eff - base_eff) / base_eff) * 100.0
        logger.info(f"LP Optimization achieved +{imp_pct:.2f}% efficiency improvement over baseline.")
        return {"improvement_pct": imp_pct}


# Backward compatibility wrappers
def optimize_allocation(
    budgets, risk_scores, total_budget_limit,
    agency_ids=None, max_projects_per_agency=None,
    min_critical_coverage=0.50, critical_threshold=0.90
):
    return ResourceOptimizer().optimize_allocation(
        budgets, risk_scores, total_budget_limit,
        agency_ids, max_projects_per_agency,
        min_critical_coverage, critical_threshold
    )

def evaluate_allocation_efficiency(risk_scores, selected_idx):
    return ResourceOptimizer().evaluate_allocation_efficiency(risk_scores, selected_idx)

def run_monte_carlo_baseline(budgets, risk_scores, total_budget_limit, n_iterations=100, seed=42):
    return ResourceOptimizer(seed).run_monte_carlo_baseline(budgets, risk_scores, total_budget_limit, n_iterations)
