"""Serializable preprocessing pipeline for MAAGAP model inference.

Encapsulates feature transformations, scalers, and encoders into a versioned artifact
(models/preprocessing_pipeline.pkl) to ensure inference in Part B runs identically to training.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional

from .feature_engineering import FeatureEngineer
from .config import MODELS_DIR, DATA_PROCESSED_DIR
from .logger import get_logger

logger = get_logger(__name__)


class MAAGAPPreprocessor:
    """Versioned, serializable preprocessor artifact for static & temporal inputs."""

    def __init__(self):
        self.engineer = FeatureEngineer()
        self.is_fitted = False

    def fit_transform(self, df_projects: pd.DataFrame, df_quarterly: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Fit feature transformations on dataset and return X_static, X_temporal."""
        logger.info("Fitting MAAGAP preprocessor pipeline...")
        X_static, static_cols, _, _ = self.engineer.build_static_features(df_projects)
        X_temporal, temporal_cols, _ = self.engineer.build_temporal_sequences(df_projects, df_quarterly)
        
        self.is_fitted = True
        return X_static, X_temporal, static_cols, temporal_cols

    def transform_static(self, df_projects: pd.DataFrame) -> np.ndarray:
        """Transform static project features for inference using fitted encoders/scalers."""
        if not self.is_fitted:
            raise RuntimeError("MAAGAPPreprocessor must be fitted before calling transform_static()")
        
        df = self.engineer._add_engineered_features(df_projects)
        engineered_cols = [
            "budget_log", "is_infrastructure", "is_typhoon_start",
            "infra_x_typhoon", "infra_x_budget", "contractor_x_typhoon",
            "budget_x_cpi_change", "low_contractor_flag", "high_budget_flag",
            "agency_risk", "contractor_x_agency", "infra_x_low_contractor",
            "typhoon_x_budget", "econ_pressure", "composite_risk_features",
        ]

        cat_encoded = pd.DataFrame(index=df.index)
        for col in self.engineer.STATIC_CATEGORICAL:
            le = self.engineer.label_encoders.get(col)
            val_col = df[col].fillna("Unknown").astype(str)
            if le is not None:
                known = set(le.classes_)
                val_col = val_col.apply(lambda x: x if x in known else le.classes_[0])
                cat_encoded[col + "_enc"] = le.transform(val_col)
            else:
                cat_encoded[col + "_enc"] = 0

        X_df = pd.concat([
            df[self.engineer.STATIC_NUMERIC].fillna(0).astype(float),
            df[engineered_cols].fillna(0).astype(float),
            cat_encoded.astype(float),
        ], axis=1)

        return X_df.values.astype(np.float32)

    def transform_temporal(self, df_projects: pd.DataFrame, df_quarterly: pd.DataFrame) -> np.ndarray:
        """Transform quarterly time-series features for inference using fitted temporal scaler."""
        if not self.is_fitted or self.engineer.temporal_scaler is None:
            raise RuntimeError("MAAGAPPreprocessor must be fitted before calling transform_temporal()")
        
        from .config import LSTM_MAX_TIMESTEPS
        n_features = len(self.engineer.TEMPORAL_FEATURES)
        project_ids = df_projects["project_id"].values
        n = len(project_ids)

        X = np.zeros((n, LSTM_MAX_TIMESTEPS, n_features), dtype=np.float32)
        q_grouped = df_quarterly.groupby("project_id")

        for i, pid in enumerate(project_ids):
            if pid not in q_grouped.groups:
                continue
            grp = q_grouped.get_group(pid).sort_values("quarter")
            vals = grp[self.engineer.TEMPORAL_FEATURES].values.astype(np.float32)
            vals_scaled = self.engineer.temporal_scaler.transform(vals)
            t = min(len(vals_scaled), LSTM_MAX_TIMESTEPS)
            X[i, :t, :] = vals_scaled[:t]

        return X

    def save(self, filepath: Optional[str] = None) -> str:
        """Serialize preprocessor pipeline to disk."""
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, "preprocessing_pipeline.pkl")
        joblib.dump(self, filepath)
        logger.info(f"Saved preprocessing pipeline to {filepath}")
        
        data_dict = {
            "static_numeric_features": self.engineer.STATIC_NUMERIC,
            "static_categorical_features": self.engineer.STATIC_CATEGORICAL,
            "all_static_features": self.engineer.static_feature_names,
            "temporal_features": self.engineer.temporal_feature_names,
        }
        dict_path = os.path.join(DATA_PROCESSED_DIR, "data_dictionary.json")
        with open(dict_path, "w") as f:
            json.dump(data_dict, f, indent=2)
        logger.info(f"Saved data dictionary to {dict_path}")
        return filepath

    @classmethod
    def load(cls, filepath: Optional[str] = None) -> "MAAGAPPreprocessor":
        """Load serialized preprocessor pipeline from disk."""
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, "preprocessing_pipeline.pkl")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessing pipeline file not found at {filepath}")
        instance = joblib.load(filepath)
        logger.info(f"Loaded preprocessing pipeline from {filepath}")
        return instance
