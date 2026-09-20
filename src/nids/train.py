"""
train.py — Train the binary and multiclass NIDS classifiers.

Model choices (both are swappable TODO items):
  Binary:     XGBoostClassifier  (Normal vs Attack)
  Multiclass: XGBoostClassifier  (attack category — 10 classes)

Both models are saved as .joblib files in models/.

Usage
-----
    python -m nids.train
    # or:
    from nids.train import train_binary, train_multiclass
    binary_model  = train_binary(X_train, y_bin_train)
    multi_model   = train_multiclass(X_train, y_multi_train)
"""

from __future__ import annotations

import logging
import time

import joblib
import numpy as np

from nids.config import (
    BINARY_CLF_PARAMS,
    BINARY_CLF_PATH,
    MULTI_CLF_PARAMS,
    MULTI_CLF_PATH,
)

logger = logging.getLogger(__name__)

# ─── Lazy import so the package doesn't hard-fail if XGBoost isn't installed ──
def _get_xgb_classifier():
    try:
        from xgboost import XGBClassifier  # type: ignore
        return XGBClassifier
    except ImportError:
        logger.warning("XGBoost not found — falling back to RandomForestClassifier.")
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier


def train_binary(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict | None = None,
) -> object:
    """
    Train a binary (Normal=0, Attack=1) classifier.

    Parameters
    ----------
    X_train : np.ndarray  — scaled feature matrix
    y_train : np.ndarray  — binary labels {0, 1}
    params  : dict        — model hyperparameters (defaults to config.BINARY_CLF_PARAMS)

    Returns
    -------
    Fitted classifier object.
    """
    params = params or BINARY_CLF_PARAMS
    Clf = _get_xgb_classifier()

    # TODO: add class_weight / scale_pos_weight to handle class imbalance.
    #       Alternatively, apply SMOTE in preprocessing.py.
    model = Clf(**params)

    logger.info("Training binary classifier  …  X_train shape: %s", X_train.shape)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    logger.info("Binary training complete in %.1fs", elapsed)

    joblib.dump(model, BINARY_CLF_PATH)
    logger.info("Binary model saved → %s", BINARY_CLF_PATH)
    return model


def train_multiclass(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict | None = None,
) -> object:
    """
    Train a multiclass classifier for UNSW-NB15 attack categories.

    Parameters
    ----------
    X_train : np.ndarray  — scaled feature matrix
    y_train : np.ndarray  — integer-encoded attack category labels
    params  : dict        — model hyperparameters (defaults to config.MULTI_CLF_PARAMS)

    Returns
    -------
    Fitted classifier object.
    """
    params = params or MULTI_CLF_PARAMS
    Clf = _get_xgb_classifier()

    n_classes = len(np.unique(y_train))
    logger.info(
        "Training multiclass classifier  …  X_train: %s, classes: %d",
        X_train.shape, n_classes,
    )

    # TODO: for XGBoost set objective='multi:softprob', eval_metric='mlogloss'
    model = Clf(**params)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    logger.info("Multiclass training complete in %.1fs", elapsed)

    joblib.dump(model, MULTI_CLF_PATH)
    logger.info("Multiclass model saved → %s", MULTI_CLF_PATH)
    return model


# ─── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from nids.preprocessing import load_splits

    splits = load_splits()
    train_binary(splits.X_train, splits.y_bin_train)
    train_multiclass(splits.X_train, splits.y_multi_train)
    print("Training complete.")
