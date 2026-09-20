"""
train.py — Model training for ML-Based NIDS (Mode 1).

Responsibilities
----------------
1. Load preprocessed feature matrices and target arrays from data/processed/.
2. Train three binary classification models:
   - Logistic Regression (on X_train_scaled) with class_weight='balanced'
   - Random Forest (on X_train_unscaled) with class_weight='balanced_subsample'
   - XGBoost (on X_train_unscaled) with scale_pos_weight derived from training set
3. Record training times, hyperparameters, seeds, and library versions.
4. Save fitted models to models/ via joblib.
5. Provide backward-compatible train_binary() and train_multiclass() interfaces.

Usage
-----
    python -m src.nids.train --task binary
    # or:
    python -m nids.train --task binary
    # or as a module:
    from nids.train import train_binary_models
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import platform
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

try:
    from nids.config import (
        BINARY_CLF_PARAMS,
        BINARY_CLF_PATH,
        BINARY_LR_PATH,
        BINARY_RF_PATH,
        BINARY_XGB_PATH,
        METRICS_DIR,
        MODELS_DIR,
        MULTI_CLF_PARAMS,
        MULTI_CLF_PATH,
        MULTICLASS_LR_PATH,
        MULTICLASS_RF_PATH,
        MULTICLASS_XGB_PATH,
        MULTICLASS_BEST_MODEL_PATH,
        PROCESSED_DIR,
        RANDOM_STATE,
    )
    from nids.preprocessing import load_splits
except ImportError:
    from src.nids.config import (
        BINARY_CLF_PARAMS,
        BINARY_CLF_PATH,
        BINARY_LR_PATH,
        BINARY_RF_PATH,
        BINARY_XGB_PATH,
        METRICS_DIR,
        MODELS_DIR,
        MULTI_CLF_PARAMS,
        MULTI_CLF_PATH,
        MULTICLASS_LR_PATH,
        MULTICLASS_RF_PATH,
        MULTICLASS_XGB_PATH,
        MULTICLASS_BEST_MODEL_PATH,
        PROCESSED_DIR,
        RANDOM_STATE,
    )
    from src.nids.preprocessing import load_splits

logger = logging.getLogger(__name__)

TRAINING_META_PATH = METRICS_DIR / "binary_training_meta.json"


def get_library_versions() -> Dict[str, str]:
    """Capture runtime package versions for reproducibility."""
    pkgs = ["scikit-learn", "xgboost", "numpy", "pandas", "joblib"]
    versions = {}
    for pkg in pkgs:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            versions[pkg] = "unknown"
    versions["python"] = platform.python_version()
    return versions


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[LogisticRegression, float, Dict[str, Any]]:
    """
    Train a Logistic Regression model on scaled features.

    Parameters
    ----------
    X_train : np.ndarray (scaled features)
    y_train : np.ndarray (binary labels)
    params  : Optional custom hyperparameters
    """
    default_params: Dict[str, Any] = {
        "max_iter": 1000,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "random_state": RANDOM_STATE,
    }
    if params:
        default_params.update(params)

    logger.info("Training Logistic Regression (scaled features, shape %s)...", X_train.shape)
    model = LogisticRegression(**default_params)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    logger.info("Logistic Regression training completed in %.2fs", elapsed)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, BINARY_LR_PATH)
    logger.info("Saved Logistic Regression -> %s", BINARY_LR_PATH)
    return model, elapsed, default_params


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[RandomForestClassifier, float, Dict[str, Any]]:
    """
    Train a Random Forest classifier on unscaled features.

    Parameters
    ----------
    X_train : np.ndarray (unscaled features)
    y_train : np.ndarray (binary labels)
    params  : Optional custom hyperparameters
    """
    default_params: Dict[str, Any] = {
        "n_estimators": 200,
        "max_depth": 20,
        "class_weight": "balanced_subsample",
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    }
    if params:
        default_params.update(params)

    logger.info("Training Random Forest (unscaled features, shape %s)...", X_train.shape)
    model = RandomForestClassifier(**default_params)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    logger.info("Random Forest training completed in %.2fs", elapsed)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, BINARY_RF_PATH)
    logger.info("Saved Random Forest -> %s", BINARY_RF_PATH)
    return model, elapsed, default_params


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[XGBClassifier, float, Dict[str, Any]]:
    """
    Train an XGBoost classifier on unscaled features with scale_pos_weight.

    Parameters
    ----------
    X_train : np.ndarray (unscaled features)
    y_train : np.ndarray (binary labels)
    params  : Optional custom hyperparameters
    """
    neg_count = float(np.sum(y_train == 0))
    pos_count = float(np.sum(y_train == 1))
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    default_params: Dict[str, Any] = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "tree_method": "hist",
        "scale_pos_weight": round(scale_pos_weight, 5),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    if params:
        default_params.update(params)

    logger.info("Training XGBoost (unscaled features, scale_pos_weight=%.4f)...", scale_pos_weight)
    model = XGBClassifier(**default_params)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    logger.info("XGBoost training completed in %.2fs", elapsed)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, BINARY_XGB_PATH)
    logger.info("Saved XGBoost -> %s", BINARY_XGB_PATH)
    return model, elapsed, default_params


def train_binary_models(
    splits: Optional[Dict[str, Any]] = None,
    save_meta: bool = True,
) -> Dict[str, Any]:
    """
    Train all 3 candidate binary models and persist training metadata.
    """
    if splits is None:
        logger.info("Loading preprocessed arrays from %s...", PROCESSED_DIR)
        splits = load_splits(PROCESSED_DIR)

    X_train_scaled = splits["X_train_scaled"]
    X_train_unscaled = splits["X_train_unscaled"]
    y_train = splits["y_bin_train"]

    logger.info("Beginning binary models training suite...")

    # 1. Logistic Regression
    lr_model, lr_time, lr_params = train_logistic_regression(X_train_scaled, y_train)

    # 2. Random Forest
    rf_model, rf_time, rf_params = train_random_forest(X_train_unscaled, y_train)

    # 3. XGBoost
    xgb_model, xgb_time, xgb_params = train_xgboost(X_train_unscaled, y_train)

    training_meta = {
        "project": "ML-Based-NIDS-and-Attack-Classification-System",
        "task": "binary",
        "mode": 1,
        "timestamp": pd.Timestamp.now().isoformat(),
        "random_state": RANDOM_STATE,
        "train_records_count": int(len(y_train)),
        "models": {
            "logistic_regression": {
                "artifact_path": str(BINARY_LR_PATH.resolve()),
                "input_array": "X_train_scaled",
                "training_time_seconds": round(lr_time, 3),
                "parameters": lr_params,
                "class_balance_strategy": "class_weight='balanced'",
            },
            "random_forest": {
                "artifact_path": str(BINARY_RF_PATH.resolve()),
                "input_array": "X_train_unscaled",
                "training_time_seconds": round(rf_time, 3),
                "parameters": rf_params,
                "class_balance_strategy": "class_weight='balanced_subsample'",
            },
            "xgboost": {
                "artifact_path": str(BINARY_XGB_PATH.resolve()),
                "input_array": "X_train_unscaled",
                "training_time_seconds": round(xgb_time, 3),
                "parameters": xgb_params,
                "class_balance_strategy": f"scale_pos_weight={xgb_params.get('scale_pos_weight')}",
            },
        },
        "library_versions": get_library_versions(),
    }

    if save_meta:
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        with open(TRAINING_META_PATH, "w", encoding="utf-8") as f:
            json.dump(training_meta, f, indent=2)
        logger.info("Saved binary training metadata -> %s", TRAINING_META_PATH)

    return {
        "models": {
            "logistic_regression": lr_model,
            "random_forest": rf_model,
            "xgboost": xgb_model,
        },
        "training_meta": training_meta,
    }


# ─── Multiclass training functions ───────────────────────────────────────────

MULTICLASS_META_PATH = METRICS_DIR / "multiclass_training_meta.json"


def train_multiclass_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[LogisticRegression, float, Dict[str, Any]]:
    """
    Train Logistic Regression for multiclass attack-category prediction.
    Uses scaled features and class_weight='balanced'.
    """
    params: Dict[str, Any] = {
        "max_iter": 1000,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "random_state": RANDOM_STATE,
    }
    logger.info("Training Multiclass Logistic Regression (scaled, shape %s)...", X_train.shape)
    model = LogisticRegression(**params)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    logger.info("Multiclass LR training completed in %.2fs", elapsed)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MULTICLASS_LR_PATH)
    logger.info("Saved multiclass LR -> %s", MULTICLASS_LR_PATH)
    return model, elapsed, params


def train_multiclass_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[RandomForestClassifier, float, Dict[str, Any]]:
    """
    Train Random Forest for multiclass attack-category prediction.
    Uses unscaled features and class_weight='balanced_subsample'.
    """
    params: Dict[str, Any] = {
        "n_estimators": 200,
        "class_weight": "balanced_subsample",
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    }
    logger.info("Training Multiclass Random Forest (unscaled, shape %s)...", X_train.shape)
    model = RandomForestClassifier(**params)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    logger.info("Multiclass RF training completed in %.2fs", elapsed)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MULTICLASS_RF_PATH)
    logger.info("Saved multiclass RF -> %s", MULTICLASS_RF_PATH)
    return model, elapsed, params


def train_multiclass_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[XGBClassifier, float, Dict[str, Any]]:
    """
    Train XGBoost for multiclass attack-category prediction.
    Uses unscaled features with per-sample weights derived from training labels
    to handle class imbalance (XGBoost does not support class_weight directly).
    """
    num_classes = int(len(np.unique(y_train)))
    params: Dict[str, Any] = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    # Compute per-sample weights from training labels only (no test leakage)
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    logger.info(
        "Training Multiclass XGBoost (unscaled, shape %s, num_class=%d, sample_weights computed)...",
        X_train.shape, num_classes,
    )
    model = XGBClassifier(**params)
    t0 = time.perf_counter()
    model.fit(X_train, y_train, sample_weight=sample_weights)
    elapsed = time.perf_counter() - t0
    logger.info("Multiclass XGBoost training completed in %.2fs", elapsed)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MULTICLASS_XGB_PATH)
    logger.info("Saved multiclass XGBoost -> %s", MULTICLASS_XGB_PATH)
    return model, elapsed, params


def train_multiclass_models(
    splits: Optional[Dict[str, Any]] = None,
    save_meta: bool = True,
) -> Dict[str, Any]:
    """
    Train all 3 candidate multiclass models and persist training metadata.
    No test data is touched during this function.
    """
    if splits is None:
        logger.info("Loading preprocessed arrays from %s...", PROCESSED_DIR)
        splits = load_splits(PROCESSED_DIR)

    X_train_scaled = splits["X_train_scaled"]
    X_train_unscaled = splits["X_train_unscaled"]
    y_train = splits["y_multi_train"]

    logger.info(
        "Beginning multiclass models training suite. Train shape: %s, classes: %s",
        X_train_scaled.shape, sorted(np.unique(y_train).tolist()),
    )

    # 1. Logistic Regression (scaled)
    lr_model, lr_time, lr_params = train_multiclass_logistic_regression(X_train_scaled, y_train)

    # 2. Random Forest (unscaled)
    rf_model, rf_time, rf_params = train_multiclass_random_forest(X_train_unscaled, y_train)

    # 3. XGBoost (unscaled, sample_weight)
    xgb_model, xgb_time, xgb_params = train_multiclass_xgboost(X_train_unscaled, y_train)

    training_meta = {
        "project": "ML-Based-NIDS-and-Attack-Classification-System",
        "task": "multiclass",
        "mode": 1,
        "timestamp": pd.Timestamp.now().isoformat(),
        "random_state": RANDOM_STATE,
        "train_records_count": int(len(y_train)),
        "num_classes": int(len(np.unique(y_train))),
        "models": {
            "logistic_regression": {
                "artifact_path": str(MULTICLASS_LR_PATH.resolve()),
                "input_array": "X_train_scaled",
                "training_time_seconds": round(lr_time, 3),
                "parameters": lr_params,
                "class_balance_strategy": "class_weight='balanced'",
            },
            "random_forest": {
                "artifact_path": str(MULTICLASS_RF_PATH.resolve()),
                "input_array": "X_train_unscaled",
                "training_time_seconds": round(rf_time, 3),
                "parameters": rf_params,
                "class_balance_strategy": "class_weight='balanced_subsample'",
            },
            "xgboost": {
                "artifact_path": str(MULTICLASS_XGB_PATH.resolve()),
                "input_array": "X_train_unscaled",
                "training_time_seconds": round(xgb_time, 3),
                "parameters": xgb_params,
                "class_balance_strategy": "compute_sample_weight('balanced', y_train)",
            },
        },
        "library_versions": get_library_versions(),
    }

    if save_meta:
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        with open(MULTICLASS_META_PATH, "w", encoding="utf-8") as f:
            json.dump(training_meta, f, indent=2)
        logger.info("Saved multiclass training metadata -> %s", MULTICLASS_META_PATH)

    return {
        "models": {
            "logistic_regression": lr_model,
            "random_forest": rf_model,
            "xgboost": xgb_model,
        },
        "training_meta": training_meta,
    }


# ─── Backward compatibility wrappers ─────────────────────────────────────────

def train_binary(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
) -> object:
    """Backward-compatible train_binary entry point (trains XGBoost by default)."""
    model, _, _ = train_xgboost(X_train, y_train, params=params)
    joblib.dump(model, BINARY_CLF_PATH)
    return model


def train_multiclass(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
) -> object:
    """Backward-compatible train_multiclass entry point (trains XGBoost by default)."""
    model, _, _ = train_multiclass_xgboost(X_train, y_train)
    joblib.dump(model, MULTI_CLF_PATH)
    return model


# ─── CLI Entrypoint ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train NIDS machine learning models (Mode 1)")
    parser.add_argument(
        "--task",
        choices=["binary", "multiclass", "all"],
        default="binary",
        help="Target classification task to train (default: binary)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.task in ("binary", "all"):
        train_binary_models()
        print("\n" + "=" * 70)
        print("BINARY CLASSIFICATION MODELS TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"1. Logistic Regression : {BINARY_LR_PATH}")
        print(f"2. Random Forest        : {BINARY_RF_PATH}")
        print(f"3. XGBoost              : {BINARY_XGB_PATH}")
        print(f"Training Metadata       : {TRAINING_META_PATH}")
        print("=" * 70 + "\n")

    if args.task in ("multiclass", "all"):
        results = train_multiclass_models()
        meta = results["training_meta"]
        print("\n" + "=" * 70)
        print("MULTICLASS CLASSIFICATION MODELS TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"1. Logistic Regression : {MULTICLASS_LR_PATH}")
        print(f"2. Random Forest        : {MULTICLASS_RF_PATH}")
        print(f"3. XGBoost              : {MULTICLASS_XGB_PATH}")
        for mname, minfo in meta["models"].items():
            print(f"   {mname:<22}: {minfo['training_time_seconds']:.2f}s")
        print(f"Training Metadata       : {MULTICLASS_META_PATH}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
