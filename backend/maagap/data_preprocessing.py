"""Clean and preprocess the real PPDO 2026 dataset.

Extracts statistical distributions used to parameterise synthetic data
generation so that the synthetic records mirror real-world patterns.
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, Any, Tuple, Optional

from .config import REAL_DATA_FILE, DATA_PROCESSED_DIR
from .logger import get_logger

logger = get_logger(__name__)

class DataPreprocessor:
    """Class to encapsulate logic for reading and processing PPDO datasets."""

    def __init__(self, filepath: Optional[str] = None):
        self.filepath = filepath or REAL_DATA_FILE

    @staticmethod
    def _extract_status(remark: Any) -> str:
        """Parse status from a raw remark string."""
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

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> Dict[str, str]:
        """Convert columns to lower snake case."""
        return {c: str(c).strip().lower().replace("\n", " ").replace("  ", " ") for c in df.columns}

    def _load_ppdo_like_table(self) -> Tuple[pd.DataFrame, str]:
        """Load a PPDO-like project monitoring table from a workbook."""
        try:
            df = pd.read_excel(self.filepath, sheet_name="MONITORING REPORT Con")
            colmap = self._normalise_columns(df)
            df = df.rename(columns=colmap)
            return df, "MONITORING REPORT Con"
        except Exception as e:
            logger.debug(f"Failed to load 'MONITORING REPORT Con': {e}")
            pass

        # Fallback: previous fixed-layout sheet
        logger.info("Falling back to 'Sheet1' parsing...")
        raw = pd.read_excel(self.filepath, sheet_name="Sheet1", header=None)
        cols = [
            "project_name", "implementing_agency", "brief_description",
            "expected_output", "location", "contractor", "procurement_mode",
            "funding_source", "approved_budget", "start_date", "end_date",
            "physical_accomplishment", "financial_accomplishment", "remarks",
        ]
        data = raw.iloc[8:].copy()
        data.columns = cols
        data.reset_index(drop=True, inplace=True)
        return data, "Sheet1"

    @staticmethod
    def _classify_type(row: pd.Series) -> str:
        """Classify project as Infrastructure or Non-Infrastructure."""
        combined = f"{row.get('project_name', '')} {row.get('brief_description', '')}".lower()
        infra_kw = [
            "construction", "rehabilitation", "improvement", "repair",
            "maintenance", "building", "road", "bridge", "water system",
            "facility", "infrastructure", "hospital", "school",
        ]
        return "Infrastructure" if any(k in combined for k in infra_kw) else "Non-Infrastructure"

    def load_and_clean_ppdo(self) -> pd.DataFrame:
        """Return a cleaned DataFrame from the raw PPDO Excel file."""
        logger.info(f"Loading raw PPDO dataset from {self.filepath}")
        data, sheet_used = self._load_ppdo_like_table()

        if sheet_used == "MONITORING REPORT Con":
            junk_cols = [c for c in data.columns if c.strip() == "" or str(c).startswith("unnamed")]
            if junk_cols:
                data = data.drop(columns=junk_cols, errors="ignore")

            rename_map = {
                "name of project": "project_name",
                "location": "location",
                "amount (php)": "approved_budget",
                "funds released to:": "implementing_agency",
                "remarks": "remarks",
                "status": "status_raw",
                "file name": "funding_source",
            }
            data = data.rename(columns=rename_map)

            for c in ["brief_description", "expected_output", "contractor",
                      "procurement_mode", "start_date", "end_date",
                      "physical_accomplishment", "financial_accomplishment"]:
                if c not in data.columns:
                    data[c] = np.nan

            status_text = data.get("status_raw")
            remarks_text = data.get("remarks")
            data["remarks"] = (
                status_text.astype(str).replace({"nan": ""}).fillna("")
                + " | "
                + remarks_text.astype(str).replace({"nan": ""}).fillna("")
            ).str.strip(" |")
            data = data.drop(columns=["status_raw"], errors="ignore")

            if "funding_source" in data.columns:
                data["brief_description"] = data["funding_source"]

        for c in ["implementing_agency", "contractor", "procurement_mode",
                  "funding_source", "remarks", "project_name", "brief_description"]:
            if c in data.columns:
                data[c] = data[c].astype(str).str.strip().replace(["nan", "None", ""], np.nan)

        data["approved_budget"] = pd.to_numeric(data.get("approved_budget"), errors="coerce")
        data["status"] = data["remarks"].apply(self._extract_status)
        data["project_type"] = data.apply(self._classify_type, axis=1)

        data = data.dropna(subset=["implementing_agency", "approved_budget"], how="all")

        out_path = os.path.join(DATA_PROCESSED_DIR, "ppdo_2026_cleaned.csv")
        data.to_csv(out_path, index=False)
        logger.info(f"Saved cleaned PPDO data to {out_path} ({len(data)} rows)")
        return data

    def extract_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Derive statistical distributions from cleaned PPDO data."""
        dist = {}

        budgets = df["approved_budget"].dropna()
        budgets = budgets[budgets > 0]
        log_b = np.log(budgets)
        
        dist["budget_log_mean"] = float(log_b.mean())
        dist["budget_log_std"] = float(log_b.std())
        dist["budget_min"] = float(budgets.min())
        dist["budget_max"] = float(budgets.max())

        dist["agency_probs"] = df["implementing_agency"].value_counts(normalize=True).to_dict()
        dist["type_probs"] = df["project_type"].value_counts(normalize=True).to_dict()
        dist["status_probs"] = df["status"].value_counts(normalize=True).to_dict()

        return dist

    # --- Fund Transfer Con Processing ---

    _FUND_SOURCE_MAP = {
        "IRA":  "General Fund",
        "MOOE": "MOOE",
        "SEF":  "SEF",
        "NTA":  "20% NTA",
        "GAD":  "GAD Fund",
        "LDRRMF": "LDRRMF",
        "HFEP": "HFEP",
        "TRUST": "Trust Fund",
        "SUPPLEMENTAL": "Supplemental Budget",
        "SPECIAL": "Special Budget",
    }

    @classmethod
    def _normalise_fund_source(cls, raw: str) -> str:
        """Map raw Fund Transfer source labels canonical FUNDING_SOURCES entries."""
        upper = str(raw).upper().strip()
        for key, canon in cls._FUND_SOURCE_MAP.items():
            if key in upper:
                return canon
        return "General Fund"

    def load_fund_transfer_con(self) -> Optional[pd.DataFrame]:
        """Load and clean the 'Fund Transfer Con' sheet."""
        try:
            df = pd.read_excel(self.filepath, sheet_name="Fund Transfer Con", header=0)
        except Exception as e:
            logger.warning(f"Could not load 'Fund Transfer Con' sheet: {e}")
            return None

        df.columns = [str(c).strip() for c in df.columns]
        df = df[[c for c in df.columns if not c.startswith("Unnamed")]].copy()

        # Deduplicate column names
        seen = {}
        new_cols = []
        for col in df.columns:
            if col in seen:
                seen[col] += 1
                new_cols.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                new_cols.append(col)
        df.columns = new_cols

        rename = {
            "Year":              "year",
            "Municipality":      "municipality",
            "Barangay":          "barangay",
            "Name of Project":   "project_name",
            "Amount":            "approved_budget",
            "Source of Fund":    "source_of_fund",
            "Amount Liquidated": "amount_liquidated",
            "Balance":           "balance",
            "Amount Refunded":   "amount_refunded",
            "Status":            "status",
            "Remarks":           "remarks",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        for col in ["project_name", "municipality", "barangay", "source_of_fund", "status", "remarks"]:
            if col in df.columns:
                cleaned = df[col].astype(str).str.strip()
                cleaned = cleaned.replace({"nan": np.nan, "None": np.nan, "": np.nan})
                df[col] = cleaned

        df["approved_budget"] = pd.to_numeric(df.get("approved_budget"), errors="coerce")
        df["amount_liquidated"] = pd.to_numeric(df.get("amount_liquidated"), errors="coerce")
        df["balance"] = pd.to_numeric(df.get("balance"), errors="coerce")
        df["year"] = pd.to_numeric(df.get("year"), errors="coerce")

        df = df[df["approved_budget"].notna() & (df["approved_budget"] > 0)].copy()

        df["is_liquidated"] = df["status"].astype(str).str.lower().str.contains("liquidated", na=False)
        df["is_monitored"] = df["status"].astype(str).str.lower().str.contains("monitored", na=False)
        df["project_type"] = df.apply(self._classify_type, axis=1)
        df["source_of_fund_canon"] = df["source_of_fund"].astype(str).apply(self._normalise_fund_source)

        return df

    def extract_fund_transfer_distributions(self, df_ft: pd.DataFrame) -> Dict[str, Any]:
        """Derive usable distributions from the Fund Transfer Con sheet."""
        if df_ft is None or len(df_ft) == 0:
            return {}

        dist = {}
        budgets = df_ft["approved_budget"].dropna()
        budgets = budgets[budgets > 0]
        
        if len(budgets) > 0:
            log_b = np.log(budgets)
            dist["ft_budget_log_mean"] = float(log_b.mean())
            dist["ft_budget_log_std"]  = float(log_b.std())
            dist["ft_budget_min"]      = float(budgets.min())
            dist["ft_budget_max"]      = float(budgets.max())
            dist["ft_budget_median"]   = float(budgets.median())

        if "municipality" in df_ft.columns:
            dist["municipality_probs"] = df_ft["municipality"].dropna().value_counts(normalize=True).to_dict()

        if "source_of_fund_canon" in df_ft.columns:
            dist["funding_source_probs"] = df_ft["source_of_fund_canon"].value_counts(normalize=True).to_dict()

        if "project_type" in df_ft.columns:
            dist["ft_type_probs"] = df_ft["project_type"].value_counts(normalize=True).to_dict()

        if "is_liquidated" in df_ft.columns:
            dist["liquidation_rate"] = float(df_ft["is_liquidated"].mean())

        if "balance" in df_ft.columns and "amount_liquidated" in df_ft.columns:
            has_bal = (df_ft["balance"].fillna(0) > 0)
            dist["unliquidated_rate"] = float(has_bal.mean())

        if "year" in df_ft.columns:
            dist["fund_transfer_year_probs"] = df_ft["year"].dropna().value_counts(normalize=True).to_dict()

        return dist

# Provide simple wrappers for backward-compatibility or module usage
def load_and_clean_ppdo(filepath: Optional[str] = None) -> pd.DataFrame:
    return DataPreprocessor(filepath).load_and_clean_ppdo()

def extract_distributions(df: pd.DataFrame) -> Dict[str, Any]:
    return DataPreprocessor().extract_distributions(df)

def load_fund_transfer_con(filepath: Optional[str] = None) -> Optional[pd.DataFrame]:
    return DataPreprocessor(filepath).load_fund_transfer_con()

def extract_fund_transfer_distributions(df_ft: pd.DataFrame) -> Dict[str, Any]:
    return DataPreprocessor().extract_fund_transfer_distributions(df_ft)
