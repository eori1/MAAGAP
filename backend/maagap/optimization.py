"""Objective 4 -- LP Optimization for Resource Allocation.

Two formulations:

1. ``InspectorAssignmentOptimizer`` -- the primary engine matching the manuscript
   (DFD Level 2, Fig. 13): assigns the limited PPDO inspector roster to
   elevated-risk projects, constrained by per-inspector workload capacity,
   availability status, and vehicle access. Benchmarked against the manual
   round-robin practice and a Monte Carlo random baseline.

2. ``ResourceOptimizer`` -- the earlier Knapsack-like budget-constrained project
   selection LP, retained for comparison experiments.
"""

import numpy as np
import pandas as pd
import pulp
import logging
from typing import Dict, Any, List, Optional, Tuple

from .logger import get_logger

logger = get_logger(__name__)

# Inspection-cycle capacity assumptions (visits per quarterly cycle).
# An inspector with a service vehicle can reach more distributed sites than one
# relying on shared transport -- reflecting the logistics delimitation in the
# manuscript (vehicle availability limits inspection frequency).
BASE_VISITS_WITH_VEHICLE = 12
BASE_VISITS_WITHOUT_VEHICLE = 7
BUSY_STATUS_PENALTY = 4  # visits removed when roster status is not "Available"


class InspectorAssignmentOptimizer:
    """Assign PPDO inspectors to projects via integer LP (PuLP/CBC).

    Decision variable x[i][j] = 1 when inspector j is scheduled to visit
    project i during the inspection cycle. The objective maximises the total
    risk score captured by scheduled visits, so high-risk projects are
    prioritised under the roster's manpower limits.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    @staticmethod
    def compute_capacities(inspectors_df: pd.DataFrame) -> np.ndarray:
        """Derive per-inspector visit capacity from the roster attributes."""
        caps = []
        for _, row in inspectors_df.iterrows():
            base = BASE_VISITS_WITH_VEHICLE if row["vehicle_access"] else BASE_VISITS_WITHOUT_VEHICLE
            if str(row["availability_status"]).strip().lower() != "available":
                base -= BUSY_STATUS_PENALTY
            cap = max(0, base - int(row["current_workload"]))
            caps.append(cap)
        return np.array(caps, dtype=int)

    def optimize_assignment(
        self,
        risk_scores: np.ndarray,
        inspectors_df: pd.DataFrame,
        min_critical_coverage: float = 1.0,
        critical_threshold: float = 0.90,
    ) -> Tuple[np.ndarray, str]:
        """Solve the inspector-to-project assignment LP.

        Returns
        -------
        assignment : np.ndarray of shape (n_projects,)
            Index of the assigned inspector per project, or -1 if unvisited.
        status : str
            PuLP solver status name.
        """
        n = len(risk_scores)
        capacities = self.compute_capacities(inspectors_df)
        m = len(capacities)
        logger.info(
            f"Formulating inspector assignment LP: {n} projects, {m} inspectors, "
            f"total capacity {capacities.sum()} visits/cycle"
        )

        prob = pulp.LpProblem("Inspector_Deployment_Optimization", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("visit", (range(n), range(m)), cat="Binary")

        # Objective: maximise total risk utility captured by scheduled visits
        prob += pulp.lpSum(risk_scores[i] * x[i][j] for i in range(n) for j in range(m)), "Captured_Risk_Utility"

        # Each project receives at most one inspector this cycle
        for i in range(n):
            prob += pulp.lpSum(x[i][j] for j in range(m)) <= 1, f"Single_Visit_{i}"

        # Each inspector is bounded by their derived workload capacity
        for j in range(m):
            prob += pulp.lpSum(x[i][j] for i in range(n)) <= int(capacities[j]), f"Capacity_{j}"

        # Critical projects must be covered (up to total capacity)
        critical_idx = np.where(np.asarray(risk_scores) >= critical_threshold)[0]
        if len(critical_idx) > 0:
            required = int(np.ceil(len(critical_idx) * min_critical_coverage))
            required = min(required, int(capacities.sum()))
            prob += pulp.lpSum(x[i][j] for i in critical_idx for j in range(m)) >= required, "Critical_Coverage"

        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=120)
        prob.solve(solver)
        status = pulp.LpStatus[prob.status]
        logger.info(f"Assignment LP solver status: {status}")

        assignment = np.full(n, -1, dtype=int)
        if prob.status == pulp.LpStatusOptimal:
            for i in range(n):
                for j in range(m):
                    if x[i][j].varValue is not None and round(x[i][j].varValue) == 1:
                        assignment[i] = j
                        break
        else:
            logger.warning("Assignment LP did not reach an optimal solution.")
        return assignment, status

    @staticmethod
    def captured_utility(risk_scores: np.ndarray, assignment: np.ndarray) -> float:
        """Total risk score captured by the scheduled visits."""
        mask = assignment >= 0
        return float(np.sum(np.asarray(risk_scores)[mask]))

    def run_manual_baseline(self, risk_scores: np.ndarray, inspectors_df: pd.DataFrame) -> np.ndarray:
        """Mimic current PPDO manual practice: visit projects in list order
        (as they appear on the monitoring report), rotating inspectors
        round-robin until each reaches capacity -- no risk prioritisation."""
        n = len(risk_scores)
        capacities = self.compute_capacities(inspectors_df).astype(int)
        remaining = capacities.copy()
        assignment = np.full(n, -1, dtype=int)
        j = 0
        m = len(capacities)
        for i in range(n):
            if remaining.sum() == 0:
                break
            # advance to next inspector with remaining capacity
            tries = 0
            while remaining[j % m] == 0 and tries < m:
                j += 1
                tries += 1
            if remaining[j % m] > 0:
                assignment[i] = j % m
                remaining[j % m] -= 1
                j += 1
        return assignment

    def run_random_baseline_mc(
        self,
        risk_scores: np.ndarray,
        inspectors_df: pd.DataFrame,
        lp_utility: float,
        n_iterations: int = 100,
        target_pct: float = 15.0,
    ) -> Dict[str, Any]:
        """Monte Carlo random assignment respecting capacities.

        Computes the actual distribution of LP improvement over random
        baselines -- no summary statistic is assumed or hardcoded.
        """
        n = len(risk_scores)
        capacities = self.compute_capacities(inspectors_df).astype(int)
        m = len(capacities)
        utilities, improvements = [], []

        for _ in range(n_iterations):
            remaining = capacities.copy()
            assignment = np.full(n, -1, dtype=int)
            order = self.rng.permutation(n)
            for i in order:
                if remaining.sum() == 0:
                    break
                avail = np.where(remaining > 0)[0]
                j = avail[self.rng.randint(len(avail))]
                assignment[i] = j
                remaining[j] -= 1
            util = self.captured_utility(risk_scores, assignment)
            utilities.append(util)
            improvements.append(((lp_utility - util) / util) * 100.0 if util > 0 else 0.0)

        improvements = np.asarray(improvements)
        results = {
            "n_iterations": n_iterations,
            "n_successful": int(np.sum(improvements >= target_pct)),
            "mean_baseline_utility": float(np.mean(utilities)),
            "mean_improvement_pct": float(np.mean(improvements)),
            "std_improvement_pct": float(np.std(improvements)),
            "min_improvement_pct": float(np.min(improvements)),
            "max_improvement_pct": float(np.max(improvements)),
        }
        logger.info(
            f"Monte Carlo random baseline: mean improvement {results['mean_improvement_pct']:.2f}% "
            f"(std {results['std_improvement_pct']:.2f}), "
            f"{results['n_successful']}/{n_iterations} runs >= {target_pct}% target"
        )
        return results

    @staticmethod
    def analyze_improvement(base_utility: float, lp_utility: float) -> Dict[str, float]:
        """Relative percentage improvement of LP over a baseline utility."""
        if base_utility <= 0:
            return {"improvement_pct": 0.0}
        imp = ((lp_utility - base_utility) / base_utility) * 100.0
        logger.info(f"LP assignment achieved +{imp:.2f}% captured-risk improvement over baseline.")
        return {"improvement_pct": imp}


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
