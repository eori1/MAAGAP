"""Smoke tests for Random Forest, XGBoost, LSTM, and Meta-Ensemble models."""

import pytest
import numpy as np
import pandas as pd

from maagap.synthetic_generator import SyntheticDataGenerator
from maagap.preprocessing_pipeline import MAAGAPPreprocessor
from maagap.models import TreeModelTrainer, LSTMTrainer, MetaEnsembleTrainer
from maagap.feature_engineering import split_data


@pytest.fixture(scope="module")
def prepared_features():
    gen = SyntheticDataGenerator(seed=42)
    df_proj, df_qtr = gen.generate_synthetic_dataset(n_projects=80)
    
    preprocessor = MAAGAPPreprocessor()
    X_static, X_temporal, static_cols, temporal_cols = preprocessor.fit_transform(df_proj, df_qtr)
    y_delay = df_proj["is_delayed"].values
    
    train_idx, val_idx, test_idx = split_data(len(df_proj))
    
    return X_static, X_temporal, y_delay, train_idx, val_idx, test_idx


def test_tree_models_training_and_prediction(prepared_features):
    X_static, _, y_delay, train_idx, val_idx, test_idx = prepared_features
    
    rf = TreeModelTrainer.train_random_forest(X_static[train_idx], y_delay[train_idx], tune=False)
    rf_probs = rf.predict_proba(X_static[test_idx])[:, 1]
    
    assert rf_probs.shape[0] == len(test_idx)
    assert np.all((rf_probs >= 0.0) & (rf_probs <= 1.0))

    xgb, _ = TreeModelTrainer.train_xgboost(X_static[train_idx], y_delay[train_idx], X_static[val_idx], y_delay[val_idx], tune=False)
    xgb_probs = xgb.predict_proba(X_static[test_idx])[:, 1]
    
    assert xgb_probs.shape[0] == len(test_idx)
    assert np.all((xgb_probs >= 0.0) & (xgb_probs <= 1.0))


def test_meta_ensemble_stacking(prepared_features):
    X_static, X_temporal, y_delay, train_idx, val_idx, test_idx = prepared_features
    
    rf = TreeModelTrainer.train_random_forest(X_static[train_idx], y_delay[train_idx], tune=False)
    xgb, _ = TreeModelTrainer.train_xgboost(X_static[train_idx], y_delay[train_idx], X_static[val_idx], y_delay[val_idx], tune=False)
    lstm, _, _ = LSTMTrainer.train_lstm(X_temporal[train_idx], y_delay[train_idx], X_temporal[val_idx], y_delay[val_idx], tune=False)
    
    rf_tr = rf.predict_proba(X_static[train_idx])[:, 1]
    xgb_tr = xgb.predict_proba(X_static[train_idx])[:, 1]
    lstm_tr = lstm.predict(X_temporal[train_idx], verbose=0).flatten()
    
    meta = MetaEnsembleTrainer.train_meta_ensemble(rf_tr, xgb_tr, lstm_tr, y_delay[train_idx])
    
    rf_te = rf.predict_proba(X_static[test_idx])[:, 1]
    xgb_te = xgb.predict_proba(X_static[test_idx])[:, 1]
    lstm_te = lstm.predict(X_temporal[test_idx], verbose=0).flatten()
    
    meta_preds, meta_probs = MetaEnsembleTrainer.predict_meta(meta, rf_te, xgb_te, lstm_te)
    
    assert meta_probs.shape == (len(test_idx), 2)
    assert meta_preds.shape[0] == len(test_idx)
