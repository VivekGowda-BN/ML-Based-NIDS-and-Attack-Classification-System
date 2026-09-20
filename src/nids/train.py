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
    """Placeholder for multiclass training phase."""
    logger.info("train_multiclass called (scheduled for multiclass phase).")
    return None


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
    else:
        logger.info("Multiclass training is not scheduled in this phase.")


if __name__ == "__main__":
    main()
