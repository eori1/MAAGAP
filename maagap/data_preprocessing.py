"""Clean and preprocess the real PPDO 2026 dataset.

Extracts statistical distributions used to parameterise synthetic data
generation so that the synthetic records mirror real-world patterns.
"""

import pandas as pd
import numpy as np
import os
from .config import REAL_DATA_FILE, DATA_PROCESSED_DIR


def _extract_status(remark):
    if pd.isna(remark):
        return "unknown"
    r = str(remark).lower().strip()
    if "completed" in r:
        return "completed"
    if any(kw in r for kw in ["ongoing", "on going", "on-going"]):
        return "ongoing"
    if "to be implemented" in r:
        return "to_be_implemented"
    if "pr on process" in r:
        return "procurement"
    if any(kw in r for kw in ["cancelled", "reversion"]):
        return "cancelled"
    return "other"


def _classify_type(row):
    combined = f"{row.get('project_name', '')} {row.get('brief_description', '')}".lower()
    infra_kw = [
        "construction", "rehabilitation", "improvement", "repair",
        "maintenance", "building", "road", "bridge", "water system",
        "facility", "infrastructure", "hospital", "school",
    ]
    return "Infrastructure" if any(k in combined for k in infra_kw) else "Non-Infrastructure"


def load_and_clean_ppdo(filepath=None):
    """Return a cleaned DataFrame from the raw PPDO Excel file."""
    filepath = filepath or REAL_DATA_FILE
    raw = pd.read_excel(filepath, sheet_name="Sheet1", header=None)

    cols = [
        "project_name", "implementing_agency", "brief_description",
        "expected_output", "location", "contractor", "procurement_mode",
        "funding_source", "approved_budget", "start_date", "end_date",
        "physical_accomplishment", "financial_accomplishment", "remarks",
    ]

    data = raw.iloc[8:].copy()
    data.columns = cols
    data.reset_index(drop=True, inplace=True)

    for c in ["implementing_agency", "contractor", "procurement_mode",
              "funding_source", "remarks", "project_name", "brief_description"]:
        data[c] = data[c].astype(str).str.strip().replace(["nan", "None", ""], np.nan)

    data["approved_budget"] = pd.to_numeric(data["approved_budget"], errors="coerce")
    data["status"] = data["remarks"].apply(_extract_status)
    data["project_type"] = data.apply(_classify_type, axis=1)

    # Drop pure section-header rows (no agency AND no budget)
    data = data.dropna(subset=["implementing_agency", "approved_budget"], how="all")

    out_path = os.path.join(DATA_PROCESSED_DIR, "ppdo_2026_cleaned.csv")
    data.to_csv(out_path, index=False)
    return data


def extract_distributions(df):
    """Derive statistical distributions from cleaned PPDO data."""
    dist = {}

    budgets = df["approved_budget"].dropna()
    budgets = budgets[budgets > 0]
    log_b = np.log(budgets)
    dist["budget_log_mean"] = float(log_b.mean())
    dist["budget_log_std"] = float(log_b.std())
    dist["budget_min"] = float(budgets.min())
    dist["budget_max"] = float(budgets.max())

    agency_vc = df["implementing_agency"].value_counts(normalize=True)
    dist["agency_probs"] = agency_vc.to_dict()

    type_vc = df["project_type"].value_counts(normalize=True)
    dist["type_probs"] = type_vc.to_dict()

    status_vc = df["status"].value_counts(normalize=True)
    dist["status_probs"] = status_vc.to_dict()

    return dist
