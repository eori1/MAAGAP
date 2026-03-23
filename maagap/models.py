"""Multi-stage predictive framework: RF, XGBoost, LSTM, and Meta-ensemble.

Stage 1 -- Ensemble classifiers (Random Forest + XGBoost) on static features
Stage 2 -- LSTM on temporal quarterly monitoring sequences
Meta    -- Stacking classifier that fuses Stage 1 & Stage 2 probabilities
"""

import numpy as np
import os, warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
import joblib

from .config import (
    SEED, MODELS_DIR,
    RF_N_ESTIMATORS, RF_MAX_DEPTH,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE,
    LSTM_UNITS, LSTM_EPOCHS, LSTM_BATCH_SIZE, LSTM_MAX_TIMESTEPS,
)

_RF_PARAM_DIST = {
    "n_estimators": [200, 300, 400, 500],
    "max_depth": [10, 15, 20, 25, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.5, 0.7],
}

_XGB_PARAM_DIST = {
    "n_estimators": [200, 300, 400, 500],
    "max_depth": [6, 8, 10, 12],
    "learning_rate": [0.01, 0.05, 0.08, 0.1, 0.15],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "reg_alpha": [0, 0.1, 0.5, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0],
    "min_child_weight": [1, 3, 5],
}


# ---------------------------------------------------------------------------
# Stage 1 -- Tree-based ensemble classifiers
# ---------------------------------------------------------------------------

def train_random_forest(X_train, y_train, task="binary", tune=True):
    n_classes = len(np.unique(y_train))
    scoring = "f1" if n_classes == 2 else "f1_macro"

    if tune:
        base = RandomForestClassifier(
            class_weight="balanced", random_state=SEED, n_jobs=-1,
        )
        search = RandomizedSearchCV(
            base, _RF_PARAM_DIST,
            n_iter=40, cv=5, scoring=scoring,
            random_state=SEED, n_jobs=-1, verbose=0,
        )
        search.fit(X_train, y_train)
        rf = search.best_estimator_
        print(f"    RF best params: {search.best_params_}")
    else:
        rf = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
            class_weight="balanced", random_state=SEED, n_jobs=-1,
        )
        rf.fit(X_train, y_train)

    joblib.dump(rf, os.path.join(MODELS_DIR, f"rf_{task}.pkl"))
    return rf


def train_xgboost(X_train, y_train, task="binary", tune=True):
    n_classes = len(np.unique(y_train))
    scoring = "f1" if n_classes == 2 else "f1_macro"

    base_params = dict(
        random_state=SEED, eval_metric="logloss",
        use_label_encoder=False, n_jobs=-1,
    )
    if n_classes > 2:
        base_params["objective"] = "multi:softprob"
        base_params["num_class"] = n_classes
    else:
        base_params["objective"] = "binary:logistic"
        pos = (y_train == 1).sum()
        neg = (y_train == 0).sum()
        base_params["scale_pos_weight"] = neg / max(pos, 1)

    if tune:
        base = XGBClassifier(**base_params)
        search = RandomizedSearchCV(
            base, _XGB_PARAM_DIST,
            n_iter=40, cv=5, scoring=scoring,
            random_state=SEED, n_jobs=-1, verbose=0,
        )
        search.fit(X_train, y_train)
        xgb = search.best_estimator_
        print(f"    XGB best params: {search.best_params_}")
    else:
        base_params.update(
            n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
        )
        xgb = XGBClassifier(**base_params)
        xgb.fit(X_train, y_train)

    joblib.dump(xgb, os.path.join(MODELS_DIR, f"xgb_{task}.pkl"))
    return xgb


# ---------------------------------------------------------------------------
# Stage 2 -- LSTM for temporal dependencies
# ---------------------------------------------------------------------------

_LSTM_PARAM_CONFIGS = [
    {"units_1": 128, "units_2": 64,  "dropout": 0.35, "lr": 1e-3,  "batch_size": 32},
    {"units_1": 64,  "units_2": 32,  "dropout": 0.30, "lr": 1e-3,  "batch_size": 32},
    {"units_1": 128, "units_2": 64,  "dropout": 0.25, "lr": 5e-4,  "batch_size": 32},
    {"units_1": 96,  "units_2": 48,  "dropout": 0.35, "lr": 5e-4,  "batch_size": 64},
    {"units_1": 128, "units_2": 64,  "dropout": 0.40, "lr": 2e-3,  "batch_size": 64},
    {"units_1": 64,  "units_2": 32,  "dropout": 0.20, "lr": 1e-3,  "batch_size": 64},
    {"units_1": 96,  "units_2": 48,  "dropout": 0.30, "lr": 1e-3,  "batch_size": 32},
    {"units_1": 128, "units_2": 32,  "dropout": 0.35, "lr": 1e-3,  "batch_size": 32},
]


def _build_lstm(n_features, n_classes=2, units_1=None, units_2=None,
                dropout=0.35, lr=1e-3):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        LSTM as KerasLSTM, Dense, Dropout, Masking, BatchNormalization,
    )
    from tensorflow.keras.optimizers import Adam

    u1 = units_1 or LSTM_UNITS
    u2 = units_2 or (LSTM_UNITS // 2)

    model = Sequential([
        Masking(mask_value=0.0, input_shape=(LSTM_MAX_TIMESTEPS, n_features)),
        KerasLSTM(u1, return_sequences=True),
        Dropout(dropout),
        KerasLSTM(u2, return_sequences=False),
        Dropout(dropout),
        BatchNormalization(),
        Dense(32, activation="relu"),
        Dropout(max(0.1, dropout - 0.10)),
        Dense(1, activation="sigmoid") if n_classes == 2
            else Dense(n_classes, activation="softmax"),
    ])

    loss = "binary_crossentropy" if n_classes == 2 else "sparse_categorical_crossentropy"
    model.compile(optimizer=Adam(learning_rate=lr), loss=loss, metrics=["accuracy"])
    return model


def train_lstm(X_train, y_train, X_val, y_val, task="binary", tune=True):
    n_features = X_train.shape[2]
    n_classes = len(np.unique(y_train))
    nc = n_classes if n_classes > 2 else 2

    from sklearn.utils.class_weight import compute_class_weight
    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    cw_dict = dict(enumerate(cw))

    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    def _train_one(params, verbose_label=None):
        if verbose_label:
            print(f"    [{verbose_label}] units=({params['units_1']},{params['units_2']}), "
                  f"dropout={params['dropout']}, lr={params['lr']}, "
                  f"batch={params['batch_size']}")
        model = _build_lstm(n_features, nc,
                            units_1=params["units_1"], units_2=params["units_2"],
                            dropout=params["dropout"], lr=params["lr"])
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True,
        )
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=LSTM_EPOCHS,
            batch_size=params["batch_size"],
            class_weight=cw_dict,
            callbacks=[early_stop],
            verbose=0,
        )
        best_val_loss = min(history.history["val_loss"])
        best_val_acc = max(history.history["val_accuracy"])
        if verbose_label:
            print(f"           val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f}")
        return model, history, best_val_loss

    if tune:
        print(f"    LSTM hyperparameter search ({len(_LSTM_PARAM_CONFIGS)} configurations)...")
        best_model, best_history, best_loss = None, None, float("inf")
        best_params = None
        for i, params in enumerate(_LSTM_PARAM_CONFIGS):
            model, history, val_loss = _train_one(params, verbose_label=f"{i+1}/{len(_LSTM_PARAM_CONFIGS)}")
            if val_loss < best_loss:
                best_loss = val_loss
                best_model, best_history = model, history
                best_params = params

        print(f"    LSTM best params: {best_params} (val_loss={best_loss:.4f})")
        best_model.save(os.path.join(MODELS_DIR, f"lstm_{task}.keras"))
        return best_model, best_history, best_params
    else:
        default_params = {
            "units_1": LSTM_UNITS, "units_2": LSTM_UNITS // 2,
            "dropout": 0.35, "lr": 1e-3, "batch_size": LSTM_BATCH_SIZE,
        }
        model, history, _ = _train_one(default_params)
        model.save(os.path.join(MODELS_DIR, f"lstm_{task}.keras"))
        return model, history, default_params


# ---------------------------------------------------------------------------
# Meta-ensemble -- Stacking classifier
# ---------------------------------------------------------------------------

def train_meta_ensemble(rf_proba, xgb_proba, lstm_proba, y_train, artifact_name="meta_ensemble.pkl"):
    """Logistic-regression meta-learner on stacked Stage 1 + Stage 2 outputs.

    artifact_name: filename under MODELS_DIR (e.g. meta_ensemble.pkl or meta_ensemble_baseline.pkl).
    """
    meta_X = np.column_stack([rf_proba, xgb_proba, lstm_proba])
    meta = LogisticRegression(max_iter=500, random_state=SEED)
    meta.fit(meta_X, y_train)
    joblib.dump(meta, os.path.join(MODELS_DIR, artifact_name))
    return meta


def predict_meta(meta, rf_proba, xgb_proba, lstm_proba):
    meta_X = np.column_stack([rf_proba, xgb_proba, lstm_proba])
    return meta.predict(meta_X), meta.predict_proba(meta_X)
