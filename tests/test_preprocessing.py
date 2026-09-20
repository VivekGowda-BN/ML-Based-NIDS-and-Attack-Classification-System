"""
tests/test_preprocessing.py — Unit and integration tests for leakage-safe preprocessing.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from nids.config import (
    CLEAN_TEST_PARQUET,
    CLEAN_TRAIN_PARQUET,
    FEATURE_NAMES_JSON,
    LABEL_ENCODER_PATH,
    NON_FEATURE_COLUMNS,
    PREPROCESSING_REPORT_MD,
    PREPROCESSING_SUMMARY_JSON,
    PREPROCESSOR_SCALED_PATH,
    PREPROCESSOR_UNSCALED_PATH,
    PROCESSED_DIR,
    TARGET_BINARY,
    TARGET_MULTI,
)
from nids.preprocessing import (
    build_preprocessor,
    detect_feature_types,
    load_splits,
)


@pytest.fixture(scope="module")
def train_df():
    return pd.read_parquet(CLEAN_TRAIN_PARQUET)


@pytest.fixture(scope="module")
def test_df():
    return pd.read_parquet(CLEAN_TEST_PARQUET)


def test_labels_and_id_excluded(train_df):
    """Ensure id, label, and attack_cat are excluded from model features."""
    num_cols, cat_cols = detect_feature_types(train_df, exclude_cols=NON_FEATURE_COLUMNS)
    all_features = set(num_cols + cat_cols)

    assert "id" not in all_features
    assert "label" not in all_features
    assert "attack_cat" not in all_features
    assert len(all_features) == 42
    assert len(num_cols) == 39
    assert len(cat_cols) == 3


def test_preprocessor_fitted_only_on_training(train_df, test_df):
    """Verify preprocessor is fitted strictly on training data and knows only training categories."""
    num_cols, cat_cols = detect_feature_types(train_df)
    preprocessor = build_preprocessor(num_cols, cat_cols, scale_numeric=True)
    preprocessor.fit(train_df[num_cols + cat_cols])

    # Check categories learned by OneHotEncoder for 'state' (index 2 in cat_cols)
    state_idx = cat_cols.index("state")
    learned_states = set(preprocessor.named_transformers_["cat"]["ohe"].categories_[state_idx])

    # ACC and CLO only exist in test_df
    assert "ACC" not in learned_states
    assert "CLO" not in learned_states
    assert len(learned_states) == train_df["state"].nunique()


def test_unseen_test_categories_do_not_crash(train_df):
    """Verify unseen categories in test data (e.g. ACC, CLO) do not raise errors."""
    num_cols, cat_cols = detect_feature_types(train_df)
    preprocessor = build_preprocessor(num_cols, cat_cols, scale_numeric=True)
    preprocessor.fit(train_df[num_cols + cat_cols])

    # Construct test sample with unseen state and unknown protocol
    sample_test = train_df[num_cols + cat_cols].iloc[:5].copy()
    sample_test.loc[0, "state"] = "ACC"
    sample_test.loc[1, "state"] = "CLO"
    sample_test.loc[2, "proto"] = "unknown_proto_xyz"

    transformed = preprocessor.transform(sample_test)
    assert transformed.shape == (5, 194)
    assert not np.isnan(transformed).any()


def test_missing_numeric_and_categorical_handling(train_df):
    """Verify imputers handle missing numeric and categorical entries seamlessly."""
    num_cols, cat_cols = detect_feature_types(train_df)
    preprocessor = build_preprocessor(num_cols, cat_cols, scale_numeric=True)
    preprocessor.fit(train_df[num_cols + cat_cols])

    sample = train_df[num_cols + cat_cols].iloc[:5].copy()
    sample.loc[0, num_cols[0]] = np.nan
    sample.loc[1, cat_cols[0]] = None

    transformed = preprocessor.transform(sample)
    assert transformed.shape == (5, 194)
    assert not np.isnan(transformed).any()


def test_train_test_transformed_columns_consistency():
    """Verify transformed train and test matrices have the exact same column count."""
    splits = load_splits()

    assert splits["X_train_scaled"].shape[1] == splits["X_test_scaled"].shape[1] == 194
    assert splits["X_train_unscaled"].shape[1] == splits["X_test_unscaled"].shape[1] == 194
    assert len(splits["feature_names"]) == 194

    # Verify rows match official dataset counts
    assert splits["X_train_scaled"].shape[0] == 175341
    assert splits["X_test_scaled"].shape[0] == 82332
    assert splits["y_bin_train"].shape[0] == 175341
    assert splits["y_bin_test"].shape[0] == 82332


def test_preprocessor_serialization_roundtrip(train_df):
    """Verify saved preprocessors can be loaded and produce identical output."""
    assert PREPROCESSOR_SCALED_PATH.exists()
    assert PREPROCESSOR_UNSCALED_PATH.exists()
    assert LABEL_ENCODER_PATH.exists()

    loaded_scaled = joblib.load(PREPROCESSOR_SCALED_PATH)
    loaded_le = joblib.load(LABEL_ENCODER_PATH)

    sample = train_df.iloc[:10]
    num_cols, cat_cols = detect_feature_types(train_df)
    transformed_sample = loaded_scaled.transform(sample[num_cols + cat_cols])

    assert transformed_sample.shape == (10, 194)
    assert len(loaded_le.classes_) == 10


def test_preprocessing_artifacts_and_reports_exist():
    """Verify metadata json, summary json, and markdown report exist."""
    assert FEATURE_NAMES_JSON.exists()
    assert PREPROCESSING_SUMMARY_JSON.exists()
    assert PREPROCESSING_REPORT_MD.exists()

    with open(FEATURE_NAMES_JSON, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["transformed_feature_count"] == 194
    assert len(meta["numeric_features"]) == 39
    assert len(meta["categorical_features"]) == 3

    with open(PREPROCESSING_SUMMARY_JSON, "r", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["phase"] == "preprocessing"
    assert summary["shapes"]["transformed_features_count"] == 194
