"""
features.py — Feature selection and engineering for the UNSW-NB15 dataset.

This module is intentionally kept separate from preprocessing.py so that
feature engineering experiments can be iterated on independently.

Planned steps
-------------
1. Remove near-zero variance features.
2. Remove highly correlated feature pairs (Pearson |r| > threshold).
3. Select top-k features by mutual information.
4. (Optional) Create interaction features.

Usage
-----
    from nids.features import select_features
    X_train_sel, X_test_sel, selected_names = select_features(
        X_train, X_test, y_bin_train, feature_names
    )
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import VarianceThreshold

logger = logging.getLogger(__name__)

# ─── Tuneable knobs ───────────────────────────────────────────────────────────
# TODO: optimise these via cross-validation.
VARIANCE_THRESHOLD: float = 0.01        # drop features with variance below this
CORRELATION_THRESHOLD: float = 0.95     # drop one of each highly-correlated pair
TOP_K_FEATURES: Optional[int] = 40      # None → keep all after variance/corr filter


def select_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Apply feature selection fitted **only on training data**.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Scaled train and test feature matrices.
    y_train : np.ndarray
        Binary training labels.
    feature_names : list[str]
        Column names corresponding to X_train columns.

    Returns
    -------
    (X_train_sel, X_test_sel, selected_names)
    """
    names = list(feature_names)

    # Step 1 — Variance filter
    X_train, X_test, names = _variance_filter(X_train, X_test, names)

    # Step 2 — Correlation filter
    X_train, X_test, names = _correlation_filter(X_train, X_test, names)

    # Step 3 — Mutual information top-k
    if TOP_K_FEATURES is not None:
        X_train, X_test, names = _mutual_info_filter(X_train, X_test, y_train, names)

    logger.info("Final feature count after selection: %d", len(names))
    return X_train, X_test, names


# ─── Individual steps ─────────────────────────────────────────────────────────

def _variance_filter(
    X_tr: np.ndarray, X_te: np.ndarray, names: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    sel = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    X_tr = sel.fit_transform(X_tr)
    X_te = sel.transform(X_te)
    mask = sel.get_support()
    names = [n for n, m in zip(names, mask) if m]
    logger.info(
        "After variance filter (threshold=%.3f): %d features", VARIANCE_THRESHOLD, len(names)
    )
    return X_tr, X_te, names


def _correlation_filter(
    X_tr: np.ndarray, X_te: np.ndarray, names: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Drop one feature from each highly correlated pair."""
    # TODO: consider using the feature with higher MI instead of an arbitrary one.
    df = pd.DataFrame(X_tr, columns=names)
    corr = df.corr(method="pearson").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [col for col in upper.columns if any(upper[col] > CORRELATION_THRESHOLD)]
    keep_mask = [n not in drop for n in names]
    X_tr = X_tr[:, keep_mask]
    X_te = X_te[:, keep_mask]
    names = [n for n, k in zip(names, keep_mask) if k]
    logger.info(
        "After correlation filter (threshold=%.2f): %d features, dropped %d",
        CORRELATION_THRESHOLD, len(names), len(drop),
    )
    return X_tr, X_te, names


def _mutual_info_filter(
    X_tr: np.ndarray, X_te: np.ndarray, y_tr: np.ndarray, names: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    k = min(TOP_K_FEATURES, X_tr.shape[1])  # can't select more than available
    sel = SelectKBest(score_func=mutual_info_classif, k=k)
    X_tr = sel.fit_transform(X_tr, y_tr)
    X_te = sel.transform(X_te)
    mask = sel.get_support()
    names = [n for n, m in zip(names, mask) if m]
    logger.info("After MI top-%d filter: %d features", k, len(names))
    return X_tr, X_te, names
