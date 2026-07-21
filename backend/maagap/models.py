"""Multi-stage predictive framework: RF, XGBoost, LSTM, and Meta-ensemble.

Stage 1 -- Ensemble classifiers (Random Forest + XGBoost) on static features
Stage 2 -- LSTM on temporal quarterly monitoring sequences
Meta    -- Stacking classifier that fuses Stage 1 & Stage 2 probabilities
"""

import numpy as np
import os
import warnings
from typing import Tuple, Dict, Any, Optional, List

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
import joblib

from .config import (
    SEED, MODELS_DIR,
    model_params,
)
from .logger import get_logger

logger = get_logger(__name__)


def _xgb_gpu_supported() -> bool:
    """Return True if this xgboost build supports CUDA tree_method."""
    try:
        from xgboost import XGBClassifier
        import numpy as _np
        X = _np.random.randn(32, 4)
        y = (X[:, 0] > 0).astype(int)
        m = XGBClassifier(
            n_estimators=1,
            max_depth=2,
            learning_rate=0.1,
            eval_metric="logloss",
            tree_method="gpu_hist",
            predictor="gpu_predictor",
        )
        m.fit(X, y)
        return True
    except Exception:
        return False


class TreeModelTrainer:
    """Trainer for Stage 1 models (Random Forest, XGBoost)."""

    RF_PARAM_DIST = {
        "n_estimators": [200, 300, 400, 500],
        "max_depth": [10, 15, 20, 25, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5, 0.7],
    }

    XGB_PARAM_DIST = {
        "n_estimators": [200, 300, 400, 500],
        "max_depth": [6, 8, 10, 12],
        "learning_rate": [0.01, 0.05, 0.08, 0.1, 0.15],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.1, 0.5, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0],
        "min_child_weight": [1, 3, 5],
    }

    @classmethod
    def train_random_forest(cls, X_train: np.ndarray, y_train: np.ndarray, task: str = "binary", tune: bool = True, models_dir: Optional[str] = None) -> RandomForestClassifier:
        n_classes = len(np.unique(y_train))
        scoring = "f1" if n_classes == 2 else "f1_macro"
        
        logger.info(f"Training Random Forest (task={task}, tune={tune})...")

        if tune:
            base = RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1)
            search = RandomizedSearchCV(
                base, cls.RF_PARAM_DIST,
                n_iter=model_params.random_search_n_iter, 
                cv=model_params.random_search_cv, 
                scoring=scoring,
                random_state=SEED, n_jobs=-1, verbose=0,
            )
            search.fit(X_train, y_train)
            rf = search.best_estimator_
            logger.info(f"RF best params: {search.best_params_}")
        else:
            rf = RandomForestClassifier(
                n_estimators=model_params.rf_n_estimators, 
                max_depth=model_params.rf_max_depth,
                class_weight="balanced", random_state=SEED, n_jobs=-1,
            )
            rf.fit(X_train, y_train)

        joblib.dump(rf, os.path.join(models_dir or MODELS_DIR, f"rf_{task}.pkl"))
        return rf

    @classmethod
    def train_xgboost(cls, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None, task: str = "binary", tune: bool = True, models_dir: Optional[str] = None) -> Tuple[XGBClassifier, Dict[str, Any]]:
        n_classes = len(np.unique(y_train))
        scoring = "f1" if n_classes == 2 else "f1_macro"

        logger.info(f"Training XGBoost (task={task}, tune={tune})...")

        base_params = dict(
            random_state=SEED, eval_metric="logloss",
            use_label_encoder=False, n_jobs=-1,
        )
        if model_params.xgb_try_gpu and _xgb_gpu_supported():
            base_params.update(tree_method="gpu_hist", predictor="gpu_predictor")
            
        if n_classes > 2:
            base_params["objective"] = "multi:softprob"
            base_params["num_class"] = n_classes
        else:
            base_params["objective"] = "binary:logistic"
            pos = (y_train == 1).sum()
            neg = (y_train == 0).sum()
            base_params["scale_pos_weight"] = neg / max(pos, 1)

        def _fallback_fit(base_model, search_dist=None):
            logger.warning("GPU fit failed or unavailable, falling back to CPU...")
            cpu_params = dict(base_params)
            cpu_params.pop("tree_method", None)
            cpu_params.pop("predictor", None)
            base_cpu = XGBClassifier(**cpu_params)
            
            if search_dist:
                search_cpu = RandomizedSearchCV(
                    base_cpu, search_dist,
                    n_iter=model_params.random_search_n_iter, 
                    cv=model_params.random_search_cv, 
                    scoring=scoring,
                    random_state=SEED, n_jobs=-1, verbose=0,
                )
                search_cpu.fit(X_train, y_train)
                return search_cpu.best_params_
            else:
                return None

        best_params_found = None
        if tune:
            base = XGBClassifier(**base_params)
            search = RandomizedSearchCV(
                base, cls.XGB_PARAM_DIST,
                n_iter=model_params.random_search_n_iter, 
                cv=model_params.random_search_cv, 
                scoring=scoring,
                random_state=SEED, n_jobs=-1, verbose=0,
            )
            try:
                search.fit(X_train, y_train)
                best_params_found = search.best_params_
                logger.info(f"XGB best params: {best_params_found}")
            except Exception:
                best_params_found = _fallback_fit(base, cls.XGB_PARAM_DIST)
                if best_params_found:
                    logger.info(f"XGB best params (CPU): {best_params_found}")
        else:
            best_params_found = {
                "n_estimators": model_params.xgb_n_estimators, 
                "max_depth": model_params.xgb_max_depth,
                "learning_rate": model_params.xgb_learning_rate,
            }

        # Train final model with eval_set to capture history
        final_params = dict(base_params)
        if best_params_found:
            final_params.update(best_params_found)
            
        xgb = XGBClassifier(**final_params)
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            
        try:
            xgb.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        except Exception:
            # Fallback to CPU if GPU failed on final fit
            final_params.pop("tree_method", None)
            final_params.pop("predictor", None)
            xgb = XGBClassifier(**final_params)
            xgb.fit(X_train, y_train, eval_set=eval_set, verbose=False)

        evals_result = xgb.evals_result() if hasattr(xgb, "evals_result") else {}

        joblib.dump(xgb, os.path.join(models_dir or MODELS_DIR, f"xgb_{task}.pkl"))
        return xgb, evals_result


class RegressionModelTrainer:
    """XGBoost regressors quantifying magnitudes (Objective 2 MAE metric):
    delay duration in days and cost overrun percentage."""

    @staticmethod
    def train_xgboost_regressor(X_train: np.ndarray, y_train: np.ndarray, task: str = "delay_days") -> Any:
        from xgboost import XGBRegressor
        logger.info(f"Training XGBoost Regressor (task={task})...")
        reg = XGBRegressor(
            n_estimators=model_params.xgb_n_estimators,
            max_depth=model_params.xgb_max_depth,
            learning_rate=model_params.xgb_learning_rate,
            objective="reg:absoluteerror",
            random_state=SEED, n_jobs=-1,
        )
        reg.fit(X_train, y_train)
        joblib.dump(reg, os.path.join(MODELS_DIR, f"xgb_reg_{task}.pkl"))
        return reg


class LSTMTrainer:
    """Trainer for Stage 2 LSTM model."""

    LSTM_PARAM_CONFIGS = [
        {"units_1": 128, "units_2": 64,  "dropout": 0.35, "lr": 1e-3,  "batch_size": 32},
        {"units_1": 64,  "units_2": 32,  "dropout": 0.30, "lr": 1e-3,  "batch_size": 32},
        {"units_1": 128, "units_2": 64,  "dropout": 0.25, "lr": 5e-4,  "batch_size": 32},
        {"units_1": 96,  "units_2": 48,  "dropout": 0.35, "lr": 5e-4,  "batch_size": 64},
        {"units_1": 128, "units_2": 64,  "dropout": 0.40, "lr": 2e-3,  "batch_size": 64},
        {"units_1": 64,  "units_2": 32,  "dropout": 0.20, "lr": 1e-3,  "batch_size": 64},
        {"units_1": 96,  "units_2": 48,  "dropout": 0.30, "lr": 1e-3,  "batch_size": 32},
        {"units_1": 128, "units_2": 32,  "dropout": 0.35, "lr": 1e-3,  "batch_size": 32},
    ]

    @staticmethod
    def _build_lstm(n_features: int, n_classes: int = 2, units_1: int = None, units_2: int = None, dropout: float = 0.35, lr: float = 1e-3) -> Any:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        try:
            gpus = tf.config.list_physical_devices("GPU")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
            
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM as KerasLSTM, Dense, Dropout, Masking, BatchNormalization
        from tensorflow.keras.optimizers import Adam

        u1 = units_1 or model_params.lstm_units
        u2 = units_2 or (model_params.lstm_units // 2)

        model = Sequential([
            Masking(mask_value=0.0, input_shape=(model_params.lstm_max_timesteps, n_features)),
            KerasLSTM(u1, return_sequences=True),
            Dropout(dropout),
            KerasLSTM(u2, return_sequences=False),
            Dropout(dropout),
            BatchNormalization(),
            Dense(32, activation="relu"),
            Dropout(max(0.1, dropout - 0.10)),
            Dense(1, activation="sigmoid") if n_classes == 2 else Dense(n_classes, activation="softmax"),
        ])

        loss = "binary_crossentropy" if n_classes == 2 else "sparse_categorical_crossentropy"
        model.compile(optimizer=Adam(learning_rate=lr), loss=loss, metrics=["accuracy"])
        return model

    @classmethod
    def train_lstm(cls, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, task: str = "binary", tune: bool = True, models_dir: Optional[str] = None) -> Tuple[Any, Any, Dict[str, Any]]:
        n_features = X_train.shape[2]
        n_classes = len(np.unique(y_train))
        nc = n_classes if n_classes > 2 else 2

        from sklearn.utils.class_weight import compute_class_weight
        import tensorflow as tf
        
        cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        cw_dict = dict(enumerate(cw))
        tf.get_logger().setLevel("ERROR")

        def _train_one(params: Dict[str, Any], verbose_label: Optional[str] = None) -> Tuple[Any, Any, float]:
            if verbose_label:
                logger.info(f"    [{verbose_label}] units=({params['units_1']},{params['units_2']}), dropout={params['dropout']}, lr={params['lr']}, batch={params['batch_size']}")
            
            model = cls._build_lstm(n_features, nc, units_1=params["units_1"], units_2=params["units_2"], dropout=params["dropout"], lr=params["lr"])
            early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=model_params.lstm_epochs,
                batch_size=params["batch_size"],
                class_weight=cw_dict,
                callbacks=[early_stop],
                verbose=0,
            )
            best_val_loss = min(history.history["val_loss"])
            best_val_acc = max(history.history["val_accuracy"])
            if verbose_label:
                logger.info(f"           val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f}")
            return model, history, best_val_loss

        logger.info(f"Training LSTM (task={task}, tune={tune})...")
        
        if tune:
            logger.info(f"    LSTM hyperparameter search ({len(cls.LSTM_PARAM_CONFIGS)} configurations)...")
            best_model, best_history, best_loss = None, None, float("inf")
            best_params = {}
            for i, params in enumerate(cls.LSTM_PARAM_CONFIGS):
                model, history, val_loss = _train_one(params, verbose_label=f"{i+1}/{len(cls.LSTM_PARAM_CONFIGS)}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model, best_history = model, history
                    best_params = params

            logger.info(f"    LSTM best params: {best_params} (val_loss={best_loss:.4f})")
            best_model.save(os.path.join(models_dir or MODELS_DIR, f"lstm_{task}.keras"))
            return best_model, best_history, best_params
        else:
            default_params = {
                "units_1": model_params.lstm_units, "units_2": model_params.lstm_units // 2,
                "dropout": 0.35, "lr": 1e-3, "batch_size": model_params.lstm_batch_size,
            }
            model, history, _ = _train_one(default_params)
            model.save(os.path.join(models_dir or MODELS_DIR, f"lstm_{task}.keras"))
            return model, history, default_params


class MetaEnsembleTrainer:
    """Logistic-regression meta-learner on stacked Stage 1 + Stage 2 outputs."""

    @staticmethod
    def train_meta_ensemble(rf_proba: np.ndarray, xgb_proba: np.ndarray, lstm_proba: np.ndarray, y_train: np.ndarray, artifact_name: str = "meta_ensemble.pkl", models_dir: Optional[str] = None) -> LogisticRegression:
        logger.info(f"Training Meta-Ensemble (artifact={artifact_name})...")
        meta_X = np.column_stack([rf_proba, xgb_proba, lstm_proba])
        meta = LogisticRegression(max_iter=500, random_state=SEED)
        meta.fit(meta_X, y_train)
        joblib.dump(meta, os.path.join(models_dir or MODELS_DIR, artifact_name))
        return meta

    @staticmethod
    def meta_ensemble_percent_contributions(meta: LogisticRegression, rf_proba: np.ndarray, xgb_proba: np.ndarray, lstm_proba: np.ndarray, names: Optional[List[str]] = None) -> Dict[str, float]:
        if names is None:
            names = ["Random Forest", "XGBoost", "LSTM"]

        meta_X = np.column_stack([rf_proba, xgb_proba, lstm_proba])
        coef = getattr(meta, "coef_", None)
        if coef is None:
            raise ValueError("Meta model has no coef_ attribute; expected LogisticRegression.")

        coef = np.asarray(coef)
        if coef.ndim == 2 and coef.shape[0] > 1:
            w = np.mean(np.abs(coef), axis=0)
        else:
            w = np.abs(coef).reshape(-1)

        scale = np.std(meta_X, axis=0, ddof=0)
        contrib = w * scale

        total = float(np.sum(contrib))
        if not np.isfinite(total) or total <= 0:
            contrib = w
            total = float(np.sum(contrib))
            if total <= 0:
                return {n: 0.0 for n in names}

        pct = 100.0 * contrib / total
        return {n: float(round(v, 2)) for n, v in zip(names, pct)}

    @staticmethod
    def predict_meta(meta: LogisticRegression, rf_proba: np.ndarray, xgb_proba: np.ndarray, lstm_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        meta_X = np.column_stack([rf_proba, xgb_proba, lstm_proba])
        return meta.predict(meta_X), meta.predict_proba(meta_X)

# Backward compatibility wrappers
def train_random_forest(X_train, y_train, task="binary", tune=True):
    return TreeModelTrainer.train_random_forest(X_train, y_train, task, tune)

def train_xgboost(X_train, y_train, X_val=None, y_val=None, task="binary", tune=True):
    return TreeModelTrainer.train_xgboost(X_train, y_train, X_val, y_val, task, tune)

def train_lstm(X_train, y_train, X_val, y_val, task="binary", tune=True):
    return LSTMTrainer.train_lstm(X_train, y_train, X_val, y_val, task, tune)

def train_meta_ensemble(rf_proba, xgb_proba, lstm_proba, y_train, artifact_name="meta_ensemble.pkl"):
    return MetaEnsembleTrainer.train_meta_ensemble(rf_proba, xgb_proba, lstm_proba, y_train, artifact_name)

def meta_ensemble_percent_contributions(meta, rf_proba, xgb_proba, lstm_proba, names=None):
    return MetaEnsembleTrainer.meta_ensemble_percent_contributions(meta, rf_proba, xgb_proba, lstm_proba, names)

def predict_meta(meta, rf_proba, xgb_proba, lstm_proba):
    return MetaEnsembleTrainer.predict_meta(meta, rf_proba, xgb_proba, lstm_proba)
