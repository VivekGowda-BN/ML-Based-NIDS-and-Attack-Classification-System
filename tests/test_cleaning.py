"""
tests/test_cleaning.py — Unit and regression tests for data cleaning (Mode 1).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nids.cleaning import clean_split, get_feature_columns
from nids.config import (
    CLEAN_TEST_PARQUET,
    CLEAN_TRAIN_PARQUET,
    CLEANING_REPORT_MD,
    CLEANING_SUMMARY_JSON,
    RAW_TEST_PATH,
    RAW_TRAIN_PATH,
    TARGET_BINARY,
    TARGET_MULTI,
)
from nids.dataset import compute_sha256

EXPECTED_TRAIN_SHA = "bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa"
EXPECTED_TEST_SHA = "734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559"
EXPECTED_TRAIN_ROWS = 175341
EXPECTED_TEST_ROWS = 82332


def test_raw_file_hashes_unchanged():
    """Verify raw dataset CSV files have never been modified."""
    train_hash = compute_sha256(RAW_TRAIN_PATH)
    test_hash = compute_sha256(RAW_TEST_PATH)
    assert train_hash == EXPECTED_TRAIN_SHA, "Raw training CSV hash altered!"
    assert test_hash == EXPECTED_TEST_SHA, "Raw testing CSV hash altered!"


def test_cleaned_parquet_files_exist():
    """Verify cleaned Parquet artifacts exist and have non-zero size."""
    assert CLEAN_TRAIN_PARQUET.exists(), f"Missing: {CLEAN_TRAIN_PARQUET}"
    assert CLEAN_TEST_PARQUET.exists(), f"Missing: {CLEAN_TEST_PARQUET}"
    assert CLEAN_TRAIN_PARQUET.stat().st_size > 0
    assert CLEAN_TEST_PARQUET.stat().st_size > 0


def test_row_counts_preserved():
    """Verify no rows were dropped during cleaning."""
    df_train = pd.read_parquet(CLEAN_TRAIN_PARQUET)
    df_test = pd.read_parquet(CLEAN_TEST_PARQUET)
    assert len(df_train) == EXPECTED_TRAIN_ROWS, f"Expected {EXPECTED_TRAIN_ROWS}, got {len(df_train)}"
    assert len(df_test) == EXPECTED_TEST_ROWS, f"Expected {EXPECTED_TEST_ROWS}, got {len(df_test)}"
    assert df_train.shape[1] == 45
    assert df_test.shape[1] == 45


def test_label_normalization():
    """Verify label normalization on cleaned Parquet and in clean_split function."""
    df_train = pd.read_parquet(CLEAN_TRAIN_PARQUET)
    df_test = pd.read_parquet(CLEAN_TEST_PARQUET)

    for df in (df_train, df_test):
        assert pd.api.types.is_integer_dtype(df[TARGET_BINARY])
        assert set(df[TARGET_BINARY].unique()) == {0, 1}

    # Synthetic test of clean_split
    sample_df = pd.DataFrame({
        "id": [1, 2],
        "label": ["0", 1.0],
        "attack_cat": [" Normal ", "Exploits"],
        "rate": [10.5, np.inf],
    })
    cleaned = clean_split(sample_df)
    assert pd.api.types.is_integer_dtype(cleaned["label"])
    assert list(cleaned["label"]) == [0, 1]
    assert np.isnan(cleaned.loc[1, "rate"])


def test_normal_attack_category_handling():
    """Verify for every row where label == 0, attack_cat is strictly 'Normal'."""
    df_train = pd.read_parquet(CLEAN_TRAIN_PARQUET)
    df_test = pd.read_parquet(CLEAN_TEST_PARQUET)

    for df in (df_train, df_test):
        normal_rows = df[df[TARGET_BINARY] == 0]
        assert (normal_rows[TARGET_MULTI] == "Normal").all()

    # Synthetic test
    sample_df = pd.DataFrame({
        "id": [1, 2],
        "label": [0, 1],
        "attack_cat": ["Generic", "Generic"],
    })
    cleaned = clean_split(sample_df)
    assert cleaned.loc[0, TARGET_MULTI] == "Normal"
    assert cleaned.loc[1, TARGET_MULTI] == "Generic"


def test_preservation_of_backdoor():
    """Verify 'Backdoor' spelling is preserved and 'Backdoors' is not introduced."""
    df_train = pd.read_parquet(CLEAN_TRAIN_PARQUET)
    df_test = pd.read_parquet(CLEAN_TEST_PARQUET)

    for df in (df_train, df_test):
        cats = set(df[TARGET_MULTI].unique())
        assert "Backdoor" in cats, "'Backdoor' missing from categories"
        assert "Backdoors" not in cats, "'Backdoors' should not be present"


def test_exclusion_of_id_label_attack_cat():
    """Verify model features exclude id, label, and attack_cat."""
    df_train = pd.read_parquet(CLEAN_TRAIN_PARQUET)
    features = get_feature_columns(df_train)

    assert "id" not in features
    assert "label" not in features
    assert "attack_cat" not in features
    assert len(features) == 42
    assert len(df_train.columns) == 45


def test_cleaning_reports_exist_and_valid():
    """Verify cleaning summary JSON and markdown report exist with valid contents."""
    assert CLEANING_SUMMARY_JSON.exists()
    assert CLEANING_REPORT_MD.exists()

    with open(CLEANING_SUMMARY_JSON, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["phase"] == "cleaning"
    assert summary["model_features_count"] == 42
    assert summary["training_split"]["after"]["row_count"] == EXPECTED_TRAIN_ROWS
    assert summary["testing_split"]["after"]["row_count"] == EXPECTED_TEST_ROWS

    report_content = CLEANING_REPORT_MD.read_text(encoding="utf-8")
    assert "UNSW-NB15 Data Cleaning Report" in report_content
    assert "Backdoor" in report_content
