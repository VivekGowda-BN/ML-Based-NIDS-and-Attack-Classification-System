"""
preprocessing.py — Encode, scale, and split the cleaned UNSW-NB15 data.

Steps
-----
1. Separate features (X) from binary and multiclass targets (y_bin, y_multi).
2. Drop non-predictive identifier columns (IP addresses, timestamps, etc.).
3. One-hot encode low-cardinality categorical columns.
4. Impute remaining nulls (median for numeric, mode for categorical).
5. Fit a StandardScaler on the training fold and apply to train + test.
6. Encode attack category labels with LabelEncoder.
7. Persist train/test arrays and fitted transformers.

Usage
-----
    python -m nids.preprocessing
    # or:
    from nids.preprocessing import run_preprocessing
    splits = run_preprocessing(df_clean)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from nids.config import (
    LABEL_ENCODER_PATH,
    PROCESSED_DIR,
    RANDOM_STATE,
    SCALER_PATH,
    TARGET_BINARY,
    TARGET_MULTI,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)

# ─── Columns to drop before training ─────────────────────────────────────────
# TODO: review this list against the full UNSW-NB15 feature set.
DROP_COLS: list[str] = [
    "srcip",   # raw IP — not a generalisable feature
    "dstip",
    "Stime",   # absolute timestamps
    "Ltime",
    "attack_cat",  # target — handled separately
    "label",       # target — handled separately
]

# TODO: confirm which UNSW-NB15 columns are categorical (should be OHE).
CATEGORICAL_COLS: list[str] = ["proto", "state", "service"]


class DataSplits(NamedTuple):
    """Container for all preprocessed train/test arrays."""
    X_train: np.ndarray
    X_test: np.ndarray
    y_bin_train: np.ndarray
    y_bin_test: np.ndarray
    y_multi_train: np.ndarray
    y_multi_test: np.ndarray
    feature_names: list[str]


def run_preprocessing(df: pd.DataFrame) -> DataSplits:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned UNSW-NB15 DataFrame (output of cleaning.clean).

    Returns
    -------
    DataSplits
        NamedTuple holding train/test splits for both tasks.
    """
    # ── Targets ──────────────────────────────────────────────────────────────
    y_bin = df[TARGET_BINARY].astype(int).values

    if TARGET_MULTI in df.columns:
        le = LabelEncoder()
        y_multi = le.fit_transform(df[TARGET_MULTI].fillna("Normal"))
        joblib.dump(le, LABEL_ENCODER_PATH)
        logger.info("LabelEncoder saved → %s  (classes: %s)", LABEL_ENCODER_PATH, list(le.classes_))
    else:
        logger.warning("'%s' column missing — using binary label as multiclass target.", TARGET_MULTI)
        y_multi = y_bin
        le = None

    # ── Features ─────────────────────────────────────────────────────────────
    drop = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=drop)

    # TODO: consider whether to keep IP as a hashed / bucketed feature.

    # ── Impute nulls ─────────────────────────────────────────────────────────
    # TODO: replace simple imputation with a proper strategy per column.
    for col in X.select_dtypes(include="number").columns:
        X[col] = X[col].fillna(X[col].median())
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "unknown")

    # ── One-hot encode categoricals ───────────────────────────────────────────
    cat_cols_present = [c for c in CATEGORICAL_COLS if c in X.columns]
    X = pd.get_dummies(X, columns=cat_cols_present, drop_first=False)
    logger.info("Shape after OHE: %s", X.shape)

    feature_names: list[str] = X.columns.tolist()

    # ── Train / test split ────────────────────────────────────────────────────
    X_arr = X.values.astype(np.float32)
    X_train, X_test, y_bin_tr, y_bin_te, y_multi_tr, y_multi_te = train_test_split(
        X_arr, y_bin, y_multi,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_bin,   # preserve class imbalance ratio
    )

    # ── Scale ─────────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)
    logger.info("StandardScaler saved → %s", SCALER_PATH)

    # ── Persist processed arrays ──────────────────────────────────────────────
    _save_splits(X_train, X_test, y_bin_tr, y_bin_te, y_multi_tr, y_multi_te, feature_names)

    return DataSplits(
        X_train, X_test,
        y_bin_tr, y_bin_te,
        y_multi_tr, y_multi_te,
        feature_names,
    )


def _save_splits(
    X_train, X_test, y_bin_tr, y_bin_te, y_multi_tr, y_multi_te, feature_names
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DIR / "X_train.npy", X_train)
    np.save(PROCESSED_DIR / "X_test.npy", X_test)
    np.save(PROCESSED_DIR / "y_bin_train.npy", y_bin_tr)
    np.save(PROCESSED_DIR / "y_bin_test.npy", y_bin_te)
    np.save(PROCESSED_DIR / "y_multi_train.npy", y_multi_tr)
    np.save(PROCESSED_DIR / "y_multi_test.npy", y_multi_te)
    pd.Series(feature_names).to_csv(PROCESSED_DIR / "feature_names.csv", index=False, header=False)
    logger.info("Processed arrays saved → %s", PROCESSED_DIR)


def load_splits() -> DataSplits:
    """Load previously saved train/test splits from data/processed/."""
    base = PROCESSED_DIR
    feature_names = pd.read_csv(base / "feature_names.csv", header=None)[0].tolist()
    return DataSplits(
        X_train=np.load(base / "X_train.npy"),
        X_test=np.load(base / "X_test.npy"),
        y_bin_train=np.load(base / "y_bin_train.npy"),
        y_bin_test=np.load(base / "y_bin_test.npy"),
        y_multi_train=np.load(base / "y_multi_train.npy"),
        y_multi_test=np.load(base / "y_multi_test.npy"),
        feature_names=feature_names,
    )


# ─── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from nids.cleaning import clean, save_clean, CLEAN_PATH  # noqa: F401
    from nids.dataset import load_raw

    raw = load_raw()
    cleaned = clean(raw)
    splits = run_preprocessing(cleaned)
    print(
        f"Done.\n"
        f"  X_train: {splits.X_train.shape}\n"
        f"  X_test:  {splits.X_test.shape}\n"
        f"  Features: {len(splits.feature_names)}"
    )
