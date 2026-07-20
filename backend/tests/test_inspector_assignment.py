"""Unit tests for the inspector-to-project assignment LP (Objective 4)."""

import numpy as np
import pandas as pd
import pytest

from maagap.optimization import (
    InspectorAssignmentOptimizer,
    BASE_VISITS_WITH_VEHICLE,
    BASE_VISITS_WITHOUT_VEHICLE,
    BUSY_STATUS_PENALTY,
)


@pytest.fixture(scope="module")
def roster():
    return pd.DataFrame([
        {"inspector_id": "INSP-001", "inspector_name": "A", "availability_status": "Available", "current_workload": 2, "vehicle_access": True},
        {"inspector_id": "INSP-002", "inspector_name": "B", "availability_status": "Available", "current_workload": 3, "vehicle_access": True},
        {"inspector_id": "INSP-003", "inspector_name": "C", "availability_status": "Available", "current_workload": 1, "vehicle_access": False},
        {"inspector_id": "INSP-004", "inspector_name": "D", "availability_status": "Busy", "current_workload": 5, "vehicle_access": True},
    ])


@pytest.fixture(scope="module")
def risk_scores():
    rng = np.random.RandomState(7)
    scores = rng.uniform(0.05, 0.85, 80)
    scores[:5] = rng.uniform(0.91, 0.99, 5)  # ensure critical projects exist
    return scores


def test_capacity_computation(roster):
    caps = InspectorAssignmentOptimizer.compute_capacities(roster)
    assert caps[0] == BASE_VISITS_WITH_VEHICLE - 2
    assert caps[1] == BASE_VISITS_WITH_VEHICLE - 3
    assert caps[2] == BASE_VISITS_WITHOUT_VEHICLE - 1
    assert caps[3] == max(0, BASE_VISITS_WITH_VEHICLE - BUSY_STATUS_PENALTY - 5)
    assert (caps >= 0).all()


def test_assignment_respects_constraints(roster, risk_scores):
    opt = InspectorAssignmentOptimizer(seed=42)
    assignment, status = opt.optimize_assignment(risk_scores, roster)

    assert status == "Optimal"
    caps = InspectorAssignmentOptimizer.compute_capacities(roster)
    # each project has at most one inspector by construction (single index)
    assert assignment.shape[0] == len(risk_scores)
    # per-inspector load within capacity
    for j in range(len(roster)):
        assert np.sum(assignment == j) <= caps[j]
    # total visits equal total capacity (more projects than capacity here)
    assert np.sum(assignment >= 0) == min(caps.sum(), len(risk_scores))


def test_critical_projects_are_covered(roster, risk_scores):
    opt = InspectorAssignmentOptimizer(seed=42)
    assignment, _ = opt.optimize_assignment(risk_scores, roster, min_critical_coverage=1.0, critical_threshold=0.90)
    critical_idx = np.where(risk_scores >= 0.90)[0]
    assert (assignment[critical_idx] >= 0).all(), "All critical projects must be scheduled"


def test_lp_beats_manual_and_random_baselines(roster, risk_scores):
    opt = InspectorAssignmentOptimizer(seed=42)
    assignment, _ = opt.optimize_assignment(risk_scores, roster)
    lp_util = opt.captured_utility(risk_scores, assignment)

    manual = opt.run_manual_baseline(risk_scores, roster)
    manual_util = opt.captured_utility(risk_scores, manual)
    assert lp_util >= manual_util

    mc = opt.run_random_baseline_mc(risk_scores, roster, lp_utility=lp_util, n_iterations=20)
    assert mc["n_iterations"] == 20
    assert 0 <= mc["n_successful"] <= 20
    assert mc["std_improvement_pct"] >= 0.0
    assert lp_util >= mc["mean_baseline_utility"]


def test_manual_baseline_ignores_risk(roster):
    """Manual round-robin assigns in list order regardless of risk score."""
    opt = InspectorAssignmentOptimizer(seed=42)
    scores = np.concatenate([np.full(40, 0.05), np.full(40, 0.95)])
    manual = opt.run_manual_baseline(scores, roster)
    caps = InspectorAssignmentOptimizer.compute_capacities(roster)
    # manual fills the low-risk head of the list first
    assert np.sum(manual[:40] >= 0) == min(40, caps.sum())
