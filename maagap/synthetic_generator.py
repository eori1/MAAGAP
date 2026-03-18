"""Generate realistic synthetic multi-year project data for model training.

Uses distributions extracted from the real PPDO 2026 dataset and
incorporates simulated PAGASA weather and PSA economic context.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

from .config import (
    SEED, SYNTHETIC_NUM_PROJECTS, SYNTHETIC_YEARS,
    IMPLEMENTING_AGENCIES, AGENCY_INFRA_RATIO, AGENCY_CAPACITY_SCORE,
    CONTRACTORS, CONTRACTOR_RELIABILITY, ILOILO_MUNICIPALITIES,
    FUNDING_SOURCES, PROCUREMENT_MODES,
    INFRA_DURATION_MONTHS, NON_INFRA_DURATION_MONTHS,
    ILOILO_MONTHLY_RAINFALL_MM, ILOILO_MONTHLY_TYPHOON_DAYS,
    PSA_CPI_ANNUAL, PSA_CMRPI_ANNUAL,
    RISK_THRESHOLDS, LSTM_MAX_TIMESTEPS, DATA_PROCESSED_DIR,
)

RNG = np.random.RandomState(SEED)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _quarter_months(start_month, quarter_idx):
    """Return list of calendar months for a given quarter offset from start."""
    return [(start_month + quarter_idx * 3 + m - 1) % 12 + 1 for m in range(3)]


def _quarter_weather(year, months):
    """Simulated PAGASA quarterly weather for Iloilo."""
    yr_noise = RNG.normal(1.0, 0.15)
    rainfall = sum(ILOILO_MONTHLY_RAINFALL_MM[m] for m in months) * max(yr_noise, 0.3)
    typhoon = sum(ILOILO_MONTHLY_TYPHOON_DAYS[m] for m in months) * max(RNG.normal(1.0, 0.25), 0.0)
    return round(rainfall, 1), round(typhoon, 1)


def _quarter_economics(year, quarter_idx):
    """Simulated PSA quarterly economic indicators."""
    cpi_base = PSA_CPI_ANNUAL.get(year, 132.5)
    cmrpi_base = PSA_CMRPI_ANNUAL.get(year, 139.2)
    q_drift = (quarter_idx - 1.5) * 0.5
    cpi = cpi_base + q_drift + RNG.normal(0, 0.8)
    cmrpi = cmrpi_base + q_drift + RNG.normal(0, 1.2)
    return round(cpi, 2), round(cmrpi, 2)


def _compute_delay_risk(row, contractor_rel, agency_cap, typhoon_exposure, cpi_change):
    """Logistic model for delay probability based on project features.

    Coefficients are tuned so tree-based models can learn meaningful
    decision boundaries while retaining stochastic variance.
    """
    z = (
        -0.80
        + 1.20 * (1 if row["project_type"] == "Infrastructure" else 0)
        + 0.90 * np.clip((np.log(row["approved_budget"]) - 13.5) / 3.0, -1, 1)
        + 0.85 * np.clip(typhoon_exposure / 10.0, 0, 1)
        - 1.10 * contractor_rel
        + 0.60 * np.clip(cpi_change / 10.0, -0.5, 0.5)
        - 0.70 * agency_cap
        + RNG.normal(0, 0.08)
    )
    return _sigmoid(z)


def _compute_overrun_risk(is_delayed, delay_severity, is_infra, cmrpi_change):
    z = (
        -0.90
        + 1.20 * is_delayed
        + 0.60 * delay_severity
        + 0.50 * (1 if is_infra else 0)
        + 0.40 * np.clip(cmrpi_change / 10.0, -0.5, 0.5)
        + RNG.normal(0, 0.12)
    )
    return _sigmoid(z)


def _risk_category(delay_prob, overrun_prob):
    # Blended score that also accounts for individual worst-case risk
    combined = max(
        0.55 * delay_prob + 0.45 * overrun_prob,
        0.70 * max(delay_prob, overrun_prob),
    )
    for label, (lo, hi) in RISK_THRESHOLDS.items():
        if lo <= combined < hi:
            return label
    return "Critical"


def generate_synthetic_dataset(distributions=None, n_projects=None):
    """Generate the full synthetic dataset: projects + quarterly monitoring."""
    n = n_projects or SYNTHETIC_NUM_PROJECTS
    projects = []
    quarterly_records = []

    budget_log_mean = distributions["budget_log_mean"] if distributions else 14.2
    budget_log_std = distributions["budget_log_std"] if distributions else 1.5

    for pid in range(1, n + 1):
        year = RNG.choice(SYNTHETIC_YEARS)
        agency = RNG.choice(IMPLEMENTING_AGENCIES)
        infra_ratio = AGENCY_INFRA_RATIO.get(agency, 0.5)
        is_infra = RNG.random() < infra_ratio
        proj_type = "Infrastructure" if is_infra else "Non-Infrastructure"

        budget = float(np.clip(
            np.exp(RNG.normal(budget_log_mean, budget_log_std)),
            10_000, 80_000_000,
        ))
        budget = round(budget, 2)

        duration_months = INFRA_DURATION_MONTHS if is_infra else NON_INFRA_DURATION_MONTHS
        total_quarters = duration_months // 3
        start_month = RNG.randint(1, 13)
        start_date = datetime(year, start_month, RNG.randint(1, 15))
        planned_end = start_date + timedelta(days=duration_months * 30)

        location = RNG.choice(ILOILO_MUNICIPALITIES)
        funding = RNG.choice(FUNDING_SOURCES)
        procurement = RNG.choice(PROCUREMENT_MODES)
        has_contractor = is_infra or (RNG.random() < 0.25)
        contractor = RNG.choice(CONTRACTORS) if has_contractor else "N/A"
        contractor_rel = CONTRACTOR_RELIABILITY.get(contractor, 0.5)
        agency_cap = AGENCY_CAPACITY_SCORE.get(agency, 0.65)

        # Weather exposure: count typhoon-season months in project span
        span_months = [((start_month + m - 1) % 12) + 1 for m in range(duration_months)]
        typhoon_exposure = sum(ILOILO_MONTHLY_TYPHOON_DAYS[m] for m in span_months)

        cpi_start = PSA_CPI_ANNUAL.get(year, 132.5)
        cpi_end = PSA_CPI_ANNUAL.get(min(year + 1, 2025), cpi_start + 3)
        cpi_change = cpi_end - cpi_start
        cmrpi_start = PSA_CMRPI_ANNUAL.get(year, 139.2)
        cmrpi_end = PSA_CMRPI_ANNUAL.get(min(year + 1, 2025), cmrpi_start + 3)
        cmrpi_change = cmrpi_end - cmrpi_start

        delay_prob = _compute_delay_risk(
            {"project_type": proj_type, "approved_budget": budget},
            contractor_rel, agency_cap, typhoon_exposure, cpi_change,
        )
        is_delayed = int(RNG.random() < delay_prob)
        delay_severity = RNG.uniform(0.15, 0.80) if is_delayed else 0.0
        delay_days = int(delay_severity * duration_months * 30) if is_delayed else 0

        overrun_prob = _compute_overrun_risk(is_delayed, delay_severity, is_infra, cmrpi_change)
        is_overrun = int(RNG.random() < overrun_prob)
        overrun_pct = round(RNG.uniform(0.05, 0.35), 3) if is_overrun else 0.0

        risk_cat = _risk_category(delay_prob, overrun_prob)

        projects.append({
            "project_id": f"PROJ-{year}-{pid:04d}",
            "year": year,
            "project_type": proj_type,
            "implementing_agency": agency,
            "location": location,
            "contractor": contractor,
            "has_contractor": int(has_contractor),
            "contractor_reliability": contractor_rel,
            "procurement_mode": procurement,
            "funding_source": funding,
            "approved_budget": budget,
            "planned_duration_months": duration_months,
            "start_month": start_month,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "planned_end_date": planned_end.strftime("%Y-%m-%d"),
            "agency_capacity": agency_cap,
            "typhoon_exposure": round(typhoon_exposure, 2),
            "cpi_at_start": round(cpi_start, 2),
            "cmrpi_at_start": round(cmrpi_start, 2),
            "cpi_change": round(cpi_change, 2),
            "cmrpi_change": round(cmrpi_change, 2),
            "is_delayed": is_delayed,
            "delay_days": delay_days,
            "delay_probability": round(delay_prob, 4),
            "is_cost_overrun": is_overrun,
            "cost_overrun_pct": overrun_pct,
            "overrun_probability": round(overrun_prob, 4),
            "risk_category": risk_cat,
        })

        # --- Quarterly monitoring records ---
        cum_actual_progress = 0.0
        cum_actual_expenditure = 0.0

        # Temporal noise: sometimes delayed projects look fine early,
        # sometimes on-time projects have temporary dips — prevents leakage
        misleading_early = RNG.random() < 0.30
        temporal_noise_scale = RNG.uniform(5.0, 15.0)

        for q in range(total_quarters):
            planned_progress = (q + 1) / total_quarters * 100.0
            qtr_months = _quarter_months(start_month, q)
            rainfall, typhoon_days = _quarter_weather(year, qtr_months)
            cpi_q, cmrpi_q = _quarter_economics(year, q)

            weather_drag = np.clip(typhoon_days / 8.0, 0, 0.3)
            quarter_ratio = (q + 1) / total_quarters

            if is_delayed:
                if misleading_early and q < total_quarters // 2:
                    actual_progress = max(0, planned_progress + RNG.normal(0, 4))
                else:
                    progress_lag = delay_severity * quarter_ratio * 100.0
                    actual_progress = max(0, planned_progress - progress_lag * (0.4 + 0.6 * RNG.random())
                                          - weather_drag * 8 + RNG.normal(0, temporal_noise_scale))
            else:
                temp_dip = temporal_noise_scale * 0.5 if (misleading_early and q == 0) else 0
                actual_progress = max(0, planned_progress - weather_drag * 4 - temp_dip
                                      + RNG.normal(0, temporal_noise_scale * 0.6))

            actual_progress = np.clip(actual_progress, 0, 100.0)
            cum_actual_progress = max(cum_actual_progress, actual_progress)
            actual_progress = cum_actual_progress

            slippage = round(planned_progress - actual_progress, 2)

            planned_exp = budget * quarter_ratio
            if is_overrun:
                overrun_factor = 1.0 + overrun_pct * quarter_ratio
            else:
                overrun_factor = 1.0 + RNG.normal(0, 0.04)
            actual_exp = planned_exp * overrun_factor + RNG.normal(0, budget * 0.02)
            actual_exp = max(0, actual_exp)
            cum_actual_expenditure = max(cum_actual_expenditure, actual_exp)

            issues = max(0, int(
                RNG.poisson(2.0)
                + (2 if typhoon_days > 4 else 0)
                + (1 if is_delayed and q >= total_quarters // 2 else 0)
                - (1 if contractor_rel > 0.75 else 0)
            ))

            quarterly_records.append({
                "project_id": f"PROJ-{year}-{pid:04d}",
                "quarter": q + 1,
                "total_quarters": total_quarters,
                "planned_progress_pct": round(planned_progress, 2),
                "actual_progress_pct": round(actual_progress, 2),
                "slippage_pct": slippage,
                "planned_expenditure": round(planned_exp, 2),
                "actual_expenditure": round(cum_actual_expenditure, 2),
                "expenditure_ratio": round(cum_actual_expenditure / max(budget, 1), 4),
                "issues_count": issues,
                "rainfall_mm": rainfall,
                "typhoon_days": typhoon_days,
                "cpi_quarterly": cpi_q,
                "cmrpi_quarterly": cmrpi_q,
            })

    df_projects = pd.DataFrame(projects)
    df_quarterly = pd.DataFrame(quarterly_records)

    df_projects.to_csv(os.path.join(DATA_PROCESSED_DIR, "synthetic_projects.csv"), index=False)
    df_quarterly.to_csv(os.path.join(DATA_PROCESSED_DIR, "synthetic_quarterly.csv"), index=False)

    return df_projects, df_quarterly
