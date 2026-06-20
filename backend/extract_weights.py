import os
import joblib
import pandas as pd
import numpy as np
from maagap.config import MODELS_DIR

# Load models
rf = joblib.load(os.path.join(MODELS_DIR, "rf_delay.pkl"))
xgb = joblib.load(os.path.join(MODELS_DIR, "xgb_delay.pkl"))

# We need the feature names. Since we generated them dynamically, let's load the data.
from maagap.data_preprocessing import DataPreprocessor
from maagap.synthetic_generator import SyntheticDataGenerator
from maagap.feature_engineering import FeatureEngineer

preprocessor = DataPreprocessor()
df_real = preprocessor.load_and_clean_ppdo()
dist = preprocessor.extract_distributions(df_real)

generator = SyntheticDataGenerator()
df_projects, df_quarterly = generator.generate_synthetic_dataset(dist, n_projects=50) # just need columns

fe = FeatureEngineer()
X_static, static_cols, _, _ = fe.build_static_features(df_projects)

rf_importances = rf.feature_importances_
xgb_importances = xgb.feature_importances_ if hasattr(xgb, "feature_importances_") else np.zeros_like(rf_importances)

# Normalize to 100%
rf_w = (rf_importances / rf_importances.sum()) * 100
xgb_w = (xgb_importances / xgb_importances.sum()) * 100

df_weights = pd.DataFrame({
    "Feature": static_cols,
    "Random_Forest_Weight_Pct": rf_w,
    "XGBoost_Weight_Pct": xgb_w
})

# Sort by XGBoost weight
df_weights = df_weights.sort_values(by="XGBoost_Weight_Pct", ascending=False).round(2)

# Format as table manually
print(f"{'Feature':<40} | {'RF Weight (%)':<15} | {'XGBoost Weight (%)':<15}")
print("-" * 75)
for _, row in df_weights.iterrows():
    print(f"{row['Feature']:<40} | {row['Random_Forest_Weight_Pct']:<15.2f} | {row['XGBoost_Weight_Pct']:<15.2f}")
