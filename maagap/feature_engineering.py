"""Transform raw project and quarterly data into ML-ready feature matrices.

Produces:
  - Static feature matrix (X_static) for RF / XGBoost
  - Temporal 3-D tensor (X_temporal) for LSTM
  - Target vectors (y_delay, y_risk, y_delay_days)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from .config import LSTM_MAX_TIMESTEPS, RISK_LABELS


# Columns used as static features for tree-based models
_STATIC_NUMERIC = [
    "approved_budget", "planned_duration_months", "start_month",
    "has_contractor", "contractor_reliability", "agency_capacity",
    "typhoon_exposure", "cpi_at_start", "cmrpi_at_start",
    "cpi_change", "cmrpi_change",
]

_STATIC_CATEGORICAL = [
    "project_type", "implementing_agency", "procurement_mode", "funding_source",
]


def _add_engineered_features(df):
    """Create interaction and derived features that amplify the risk signal."""
    df = df.copy()
    df["budget_log"] = np.log1p(df["approved_budget"])
    df["is_infrastructure"] = (df["project_type"] == "Infrastructure").astype(float)
    df["is_typhoon_start"] = df["start_month"].isin([6, 7, 8, 9, 10, 11]).astype(float)

    # Key interaction: infrastructure projects in typhoon season with low contractor quality
    df["infra_x_typhoon"] = df["is_infrastructure"] * df["typhoon_exposure"]
    df["infra_x_budget"] = df["is_infrastructure"] * df["budget_log"]
    df["contractor_x_typhoon"] = (1 - df["contractor_reliability"]) * df["typhoon_exposure"]
    df["budget_x_cpi_change"] = df["budget_log"] * np.abs(df["cpi_change"])
    df["low_contractor_flag"] = (df["contractor_reliability"] < 0.5).astype(float)
    df["high_budget_flag"] = (df["approved_budget"] > df["approved_budget"].median()).astype(float)
    df["agency_risk"] = 1.0 - df["agency_capacity"]
    df["composite_risk_features"] = (
        df["is_infrastructure"] * 0.3
        + df["high_budget_flag"] * 0.2
        + df["low_contractor_flag"] * 0.2
        + df["is_typhoon_start"] * 0.15
        + df["agency_risk"] * 0.15
    )
    return df

# Per-timestep features for LSTM
_TEMPORAL_FEATURES = [
    "planned_progress_pct", "actual_progress_pct", "slippage_pct",
    "expenditure_ratio", "issues_count",
    "rainfall_mm", "typhoon_days", "cpi_quarterly", "cmrpi_quarterly",
]


def build_static_features(df_projects):
    """Label-encode categoricals + engineered interactions → feature matrix.

    Tree-based models handle raw numeric ranges natively and don't benefit
    from MinMax scaling; label encoding is preferred over one-hot to avoid
    sparse high-dimensional splits.
    """
    df = _add_engineered_features(df_projects)

    engineered_cols = [
        "budget_log", "is_infrastructure", "is_typhoon_start",
        "infra_x_typhoon", "infra_x_budget", "contractor_x_typhoon",
        "budget_x_cpi_change", "low_contractor_flag", "high_budget_flag",
        "agency_risk", "composite_risk_features",
    ]

    label_encoders = {}
    cat_encoded = pd.DataFrame(index=df.index)
    for col in _STATIC_CATEGORICAL:
        le = LabelEncoder()
        df[col] = df[col].fillna("Unknown")
        cat_encoded[col + "_enc"] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = pd.concat([
        df[_STATIC_NUMERIC].fillna(0).astype(float),
        df[engineered_cols].fillna(0).astype(float),
        cat_encoded.astype(float),
    ], axis=1)
    feature_names = list(X.columns)

    # No scaling — trees split on raw values; LSTM uses its own scaler
    scaler = None
    return X.values.astype(np.float32), feature_names, scaler, label_encoders


def build_temporal_sequences(df_projects, df_quarterly):
    """Pad/truncate quarterly records into fixed-length 3-D tensor for LSTM.

    Returns shape (n_projects, LSTM_MAX_TIMESTEPS, n_features).
    """
    n_features = len(_TEMPORAL_FEATURES)
    project_ids = df_projects["project_id"].values
    n = len(project_ids)

    X = np.zeros((n, LSTM_MAX_TIMESTEPS, n_features), dtype=np.float32)
    q_grouped = df_quarterly.groupby("project_id")

    scaler = MinMaxScaler()
    all_temporal = df_quarterly[_TEMPORAL_FEATURES].values.astype(np.float32)
    scaler.fit(all_temporal)

    for i, pid in enumerate(project_ids):
        if pid not in q_grouped.groups:
            continue
        grp = q_grouped.get_group(pid).sort_values("quarter")
        vals = grp[_TEMPORAL_FEATURES].values.astype(np.float32)
        vals_scaled = scaler.transform(vals)
        t = min(len(vals_scaled), LSTM_MAX_TIMESTEPS)
        X[i, :t, :] = vals_scaled[:t]

    return X, _TEMPORAL_FEATURES, scaler


def build_targets(df_projects):
    """Extract target vectors aligned with project rows."""
    y_delay = df_projects["is_delayed"].values.astype(int)
    y_overrun = df_projects["is_cost_overrun"].values.astype(int)
    y_delay_days = df_projects["delay_days"].values.astype(float)
    y_overrun_pct = df_projects["cost_overrun_pct"].values.astype(float)

    risk_map = {label: idx for idx, label in enumerate(RISK_LABELS)}
    y_risk = df_projects["risk_category"].map(risk_map).values.astype(int)

    return y_delay, y_overrun, y_risk, y_delay_days, y_overrun_pct


def split_data(n, train_ratio=0.70, val_ratio=0.15, seed=42):
    """Return shuffled train / val / test index arrays."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_train = int(n * train_ratio)
    n_val = int(n * (train_ratio + val_ratio))
    return idx[:n_train], idx[n_train:n_val], idx[n_val:]
