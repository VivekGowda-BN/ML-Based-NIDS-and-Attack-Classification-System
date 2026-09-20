"""
tests/test_eda.py — Unit and integration tests for Exploratory Data Analysis module.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nids.config import EDA_REPORT_MD, FIGURES_DIR
from nids.eda import (
    analyze_categorical_features,
    analyze_correlations,
    analyze_numeric_features,
    analyze_rare_categories,
    analyze_target_distributions,
    load_clean_data,
)


@pytest.fixture(scope="module")
def clean_datasets():
    """Load clean datasets once for the module tests."""
    return load_clean_data()


def test_load_clean_data(clean_datasets):
    """Verify clean Parquet loader produces expected splits and column counts."""
    df_train, df_test = clean_datasets
    assert len(df_train) == 175341
    assert len(df_test) == 82332
    assert df_train.shape[1] == 45
    assert df_test.shape[1] == 45


def test_target_distributions_analysis(clean_datasets):
    """Verify accurate calculation of target frequencies."""
    df_train, df_test = clean_datasets
    targets = analyze_target_distributions(df_train, df_test)

    assert targets["binary"]["train"]["normal_count"] == 56000
    assert targets["binary"]["train"]["attack_count"] == 119341
    assert targets["binary"]["test"]["normal_count"] == 37000
    assert targets["binary"]["test"]["attack_count"] == 45332

    # Check 10 multiclass categories
    assert len(targets["multiclass"]) == 10
    assert "Backdoor" in targets["multiclass"]
    assert "Backdoors" not in targets["multiclass"]


def test_categorical_domain_shift(clean_datasets):
    """Verify categorical profiling identifies test-only states."""
    df_train, df_test = clean_datasets
    cat_analysis = analyze_categorical_features(df_train, df_test)

    assert "proto" in cat_analysis
    assert "service" in cat_analysis
    assert "state" in cat_analysis

    # Crucial domain shift test: ACC and CLO exist in test only
    test_only_states = set(cat_analysis["state"]["test_only_values"])
    assert "ACC" in test_only_states
    assert "CLO" in test_only_states


def test_numeric_analysis(clean_datasets):
    """Verify numeric feature profiling."""
    df_train, _ = clean_datasets
    num_analysis = analyze_numeric_features(df_train)

    assert num_analysis["numeric_features_count"] == 39
    assert "is_ftp_login" in num_analysis["lowest_variance_features"] or "ackdat" in num_analysis["lowest_variance_features"]


def test_correlation_analysis(clean_datasets):
    """Verify collinearity detection."""
    df_train, _ = clean_datasets
    corr_results = analyze_correlations(df_train, threshold=0.90)

    pairs = corr_results["high_correlation_pairs"]
    assert len(pairs) > 0

    # is_ftp_login and ct_ftp_cmd have r = 1.0
    ftp_pair = [p for p in pairs if {"is_ftp_login", "ct_ftp_cmd"} == {p["feature_1"], p["feature_2"]}]
    assert len(ftp_pair) == 1
    assert ftp_pair[0]["correlation"] == 1.0


def test_rare_attacks_analysis(clean_datasets):
    """Verify rare attack analysis isolates low-frequency attacks."""
    df_train, df_test = clean_datasets
    rare_stats = analyze_rare_categories(df_train, df_test)

    assert set(rare_stats.keys()) == {"Analysis", "Backdoor", "Shellcode", "Worms"}
    assert rare_stats["Worms"]["train_count"] == 130
    assert rare_stats["Worms"]["test_count"] == 44


def test_eda_artifacts_exist():
    """Verify markdown report and all generated figures exist."""
    assert EDA_REPORT_MD.exists()
    assert EDA_REPORT_MD.stat().st_size > 0

    expected_figures = [
        "eda_binary_distribution.png",
        "eda_attack_categories.png",
        "eda_numeric_correlations.png",
        "eda_top_protocols.png",
        "eda_top_services.png",
        "eda_rare_attacks_breakdown.png",
    ]
    for fig_name in expected_figures:
        fig_path = FIGURES_DIR / fig_name
        assert fig_path.exists(), f"Figure missing: {fig_path}"
        assert fig_path.stat().st_size > 0
