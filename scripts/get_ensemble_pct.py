import sys, os, warnings
sys.path.insert(0, r'c:\Users\ASUS\Desktop\Tisis')
os.chdir(r'c:\Users\ASUS\Desktop\Tisis')
warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import joblib
import tensorflow as tf

from maagap.config import SEED
from maagap.data_preprocessing import load_and_clean_ppdo, extract_distributions, load_fund_transfer_con, extract_fund_transfer_distributions
from maagap.synthetic_generator import generate_synthetic_dataset
from maagap.feature_engineering import build_static_features, build_temporal_sequences, build_targets, split_data
from maagap.models import meta_ensemble_percent_contributions

print("Step 1: Loading real PPDO data...")
real_df = load_and_clean_ppdo()
distributions = extract_distributions(real_df)

df_ft = load_fund_transfer_con()
if df_ft is not None:
    ft_dist = extract_fund_transfer_distributions(df_ft)
    distributions["ft_distributions"] = ft_dist

print("Step 2: Generating synthetic dataset...")
df_proj, df_qtr = generate_synthetic_dataset(distributions)
print("  Projects:", len(df_proj), "| Quarterly records:", len(df_qtr))

print("Step 3: Engineering features...")
X_static, feat_names, static_scaler, _ = build_static_features(df_proj)
X_temporal, temp_feat_names, temp_scaler = build_temporal_sequences(df_proj, df_qtr)
y_delay, y_overrun, y_risk, y_delay_days, y_overrun_pct = build_targets(df_proj)

print("Step 4: Reproducing 70/15/15 split...")
idx_tr, idx_va, idx_te = split_data(len(df_proj))
Xs_va = X_static[idx_va]
Xt_va = X_temporal[idx_va]
yd_va = y_delay[idx_va]
Xs_te = X_static[idx_te]
Xt_te = X_temporal[idx_te]
yd_te = y_delay[idx_te]
print("  Train:{} Val:{} Test:{}".format(len(idx_tr), len(idx_va), len(idx_te)))

print("Step 5: Loading saved tuned models...")
rf   = joblib.load('models/rf_delay.pkl')
xgb  = joblib.load('models/xgb_delay.pkl')
meta = joblib.load('models/meta_ensemble.pkl')
lstm = tf.keras.models.load_model('models/lstm_delay.keras')

print("Step 6: Getting VALIDATION SET probabilities (same as training)...")
rf_prob_va   = rf.predict_proba(Xs_va)[:, 1]
xgb_prob_va  = xgb.predict_proba(Xs_va)[:, 1]
lstm_prob_va = lstm.predict(Xt_va, verbose=0).flatten()

print("Step 7: Computing contribution percentages...")
pcts = meta_ensemble_percent_contributions(meta, rf_prob_va, xgb_prob_va, lstm_prob_va)

# Also compute manually to show the math
S = np.column_stack([rf_prob_va, xgb_prob_va, lstm_prob_va])
coefs = np.abs(meta.coef_[0])
stds  = S.std(axis=0)
raw   = coefs * stds
manual_pcts = (raw / raw.sum()) * 100.0

names = ['Random Forest', 'XGBoost', 'LSTM']

print()
print("=" * 60)
print("  META-ENSEMBLE CONTRIBUTION PERCENTAGES")
print("  (Computed on Validation Set, n={})".format(len(yd_va)))
print("=" * 60)
print("  {:<22} {:>10}  {:>9}  {:>9}".format("Model", "Contrib %", "|coef|", "Std(prob)"))
print("  " + "-" * 56)
for name, pct, c, s in zip(names, manual_pcts, coefs, stds):
    print("  {:<22} {:>9.2f}%  {:>9.4f}  {:>9.4f}".format(name, pct, c, s))
print("  " + "-" * 56)
print("  {:<22} {:>9.2f}%".format("TOTAL", manual_pcts.sum()))
print()
print("  Raw meta coef_  : {}".format(np.round(meta.coef_[0], 6)))
print("  Intercept       : {:.4f}".format(meta.intercept_[0]))
print("  Function output : {}".format(pcts))
print("=" * 60)
