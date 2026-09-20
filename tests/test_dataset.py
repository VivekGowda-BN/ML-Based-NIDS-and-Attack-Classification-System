"""
tests/test_dataset.py — Unit and integration tests for dataset verification and profiling.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nids.config import (
    ATTACK_CATEGORIES,
    DATA_PROFILE_PATH,
    FIGURES_DIR,
    MANIFEST_PATH,
    RAW_TEST_PATH,
    RAW_TRAIN_PATH,
    TARGET_BINARY,
    TARGET_MULTI,
)
from nids.dataset import compute_sha256, load_raw


def test_raw_files_exist():
    """Ensure raw dataset files exist and have non-zero size."""
    assert RAW_TRAIN_PATH.exists(), f"Missing training file: {RAW_TRAIN_PATH}"
    assert RAW_TEST_PATH.exists(), f"Missing testing file: {RAW_TEST_PATH}"
    assert RAW_TRAIN_PATH.stat().st_size > 0
    assert RAW_TEST_PATH.stat().st_size > 0


def test_sha256_computation():
    """Verify SHA-256 calculation on raw files."""
    train_sha = compute_sha256(RAW_TRAIN_PATH)
    test_sha = compute_sha256(RAW_TEST_PATH)
    assert len(train_sha) == 64
    assert len(test_sha) == 64
    assert train_sha == "bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa"
    assert test_sha == "734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559"


def test_manifest_file_structure():
    """Verify data/raw/manifest.json contains required profiling sections."""
    assert MANIFEST_PATH.exists(), f"Manifest file missing: {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for key in ("project", "dataset", "mode", "train_split", "test_split", "schema_comparison", "overlap_analysis"):
        assert key in manifest, f"Key '{key}' missing from manifest"

    assert manifest["train_split"]["rows"] == 175341
    assert manifest["test_split"]["rows"] == 82332
    assert manifest["train_split"]["columns_count"] == 45
    assert manifest["test_split"]["columns_count"] == 45
    assert manifest["schema_comparison"]["schemas_match"] is True


def test_load_raw():
    """Verify load_raw() loads train and test sets with correct dimensions and columns."""
    df_train, df_test = load_raw()

    assert df_train.shape == (175341, 45)
    assert df_test.shape == (82332, 45)
    assert list(df_train.columns) == list(df_test.columns)

    # Check targets exist
    assert TARGET_BINARY in df_train.columns
    assert TARGET_MULTI in df_train.columns

    # Check attack categories match config
    unique_cats = set(df_train[TARGET_MULTI].unique())
    assert unique_cats == set(ATTACK_CATEGORIES)


def test_data_quality_cleanliness():
    """Verify absence of nulls and infinite values in raw data."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["train_split"]["total_missing_values"] == 0
    assert manifest["test_split"]["total_missing_values"] == 0
    assert manifest["train_split"]["total_infinite_values"] == 0
    assert manifest["test_split"]["total_infinite_values"] == 0
    assert manifest["train_split"]["exact_duplicate_rows"] == 0
    assert manifest["test_split"]["exact_duplicate_rows"] == 0


def test_reports_and_figures_generated():
    """Verify generated report and distribution figures exist."""
    assert DATA_PROFILE_PATH.exists()
    assert DATA_PROFILE_PATH.stat().st_size > 0

    bin_fig = FIGURES_DIR / "binary_label_distribution.png"
    cat_fig = FIGURES_DIR / "attack_category_distribution.png"
    assert bin_fig.exists() and bin_fig.stat().st_size > 0
    assert cat_fig.exists() and cat_fig.stat().st_size > 0
