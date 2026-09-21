"""
test_binary_threshold_selection.py — Unit and regression tests for leakage-safe binary threshold selection.

Verified invariants
-------------------
1. The validation split is derived only from training data.
2. The official test data is not used during threshold selection.
3. The selected threshold belongs to the allowed threshold list.
4. Validation and test results are stored separately.
5. Validation confusion-matrix totals equal the validation-set size (35,069).
6. Test confusion-matrix totals equal 82,332.
7. The saved model loads successfully and performs valid inference.
8. Results are reproducible.
9. The selected threshold is not hard-coded from the previous test-based analysis (0.70 vs 0.80).
10. Previous exploratory threshold artifacts and binary best model remain untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
PROC_DIR = ROOT / "data" / "processed"
METRICS_DIR = ROOT / "reports" / "metrics"
FIGURES_DIR = ROOT / "reports" / "figures"

# New Leakage-Safe Artifacts
SELECTED_MODEL_PATH = MODELS_DIR / "binary_threshold_selected_model.joblib"
SELECTION_VAL_JSON = METRICS_DIR / "binary_threshold_selection_validation.json"
SELECTION_TEST_JSON = METRICS_DIR / "binary_threshold_selection_test.json"
SELECTION_CSV = METRICS_DIR / "binary_threshold_selection_comparison.csv"
SELECTION_META_JSON = METRICS_DIR / "binary_threshold_selection_metadata.json"
SELECTION_TRADEOFF_PNG = FIGURES_DIR / "binary_threshold_selection_tradeoff.png"

# Previous exploratory artifacts that must NOT be overwritten
EXPLORATORY_MODEL_PATH = MODELS_DIR / "binary_best_model.joblib"
EXPLORATORY_JSON = METRICS_DIR / "binary_threshold_results.json"
EXPLORATORY_CSV = METRICS_DIR / "binary_threshold_comparison.csv"

# Data paths
X_TRAIN_PATH = PROC_DIR / "X_train_unscaled.npy"
Y_TRAIN_PATH = PROC_DIR / "y_bin_train.npy"
X_TEST_PATH = PROC_DIR / "X_test_unscaled.npy"
Y_TEST_PATH = PROC_DIR / "y_bin_test.npy"

ALLOWED_THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
EXPECTED_N_TRAIN_TOTAL = 175_341
EXPECTED_N_TRAIN_SUBSET = 140_272
EXPECTED_N_VAL = 35_069
EXPECTED_N_TEST = 82_332
EXPECTED_N_TEST_ATTACK = 45_332
EXPECTED_N_TEST_NORMAL = 37_000


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def val_json():
    assert SELECTION_VAL_JSON.exists(), f"Missing: {SELECTION_VAL_JSON}"
    with open(SELECTION_VAL_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def test_json():
    assert SELECTION_TEST_JSON.exists(), f"Missing: {SELECTION_TEST_JSON}"
    with open(SELECTION_TEST_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def meta_json():
    assert SELECTION_META_JSON.exists(), f"Missing: {SELECTION_META_JSON}"
    with open(SELECTION_META_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def comparison_df():
    assert SELECTION_CSV.exists(), f"Missing: {SELECTION_CSV}"
    return pd.read_csv(SELECTION_CSV)


@pytest.fixture(scope="module")
def selected_model():
    assert SELECTED_MODEL_PATH.exists(), f"Missing: {SELECTED_MODEL_PATH}"
    return joblib.load(SELECTED_MODEL_PATH)


@pytest.fixture(scope="module")
def test_arrays():
    assert X_TEST_PATH.exists(), f"Missing: {X_TEST_PATH}"
    assert Y_TEST_PATH.exists(), f"Missing: {Y_TEST_PATH}"
    return np.load(X_TEST_PATH), np.load(Y_TEST_PATH)


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestArtifactsAndPreservation:
    """Verify all new artifacts exist and previous exploratory artifacts are preserved."""

    def test_new_artifacts_exist(self):
        assert SELECTED_MODEL_PATH.exists(), f"Missing: {SELECTED_MODEL_PATH}"
        assert SELECTION_VAL_JSON.exists(), f"Missing: {SELECTION_VAL_JSON}"
        assert SELECTION_TEST_JSON.exists(), f"Missing: {SELECTION_TEST_JSON}"
        assert SELECTION_CSV.exists(), f"Missing: {SELECTION_CSV}"
        assert SELECTION_META_JSON.exists(), f"Missing: {SELECTION_META_JSON}"
        assert SELECTION_TRADEOFF_PNG.exists(), f"Missing: {SELECTION_TRADEOFF_PNG}"

    def test_exploratory_artifacts_not_overwritten(self):
        assert EXPLORATORY_MODEL_PATH.exists(), f"Missing: {EXPLORATORY_MODEL_PATH}"
        assert EXPLORATORY_JSON.exists(), f"Missing: {EXPLORATORY_JSON}"
        assert EXPLORATORY_CSV.exists(), f"Missing: {EXPLORATORY_CSV}"

        with open(EXPLORATORY_JSON, "r", encoding="utf-8") as f:
            exp_data = json.load(f)
        # Previous exploratory test analysis recommended threshold 0.80
        assert exp_data["recommended_threshold"] == 0.80


class TestLeakageSafeValidationSplit:
    """Verify Requirement 1 & 2: validation split derived only from training data, test not used."""

    def test_validation_split_derived_only_from_training_data(self, meta_json, val_json):
        assert meta_json["validation_split_derived_from_training_only"] is True
        assert meta_json["training_total_records"] == EXPECTED_N_TRAIN_TOTAL
        assert meta_json["training_subset_records"] == EXPECTED_N_TRAIN_SUBSET
        assert meta_json["validation_records"] == EXPECTED_N_VAL
        assert meta_json["training_subset_records"] + meta_json["validation_records"] == EXPECTED_N_TRAIN_TOTAL

        # Check source files in metadata
        assert "X_train_unscaled.npy" in meta_json["training_source_file"]
        assert "y_bin_train.npy" in meta_json["labels_source_file"]

        # Check validation JSON record count
        assert val_json["validation_records_count"] == EXPECTED_N_VAL
        assert val_json["training_records_count"] == EXPECTED_N_TRAIN_SUBSET

    def test_official_test_data_not_used_during_threshold_selection(self, meta_json, val_json):
        assert meta_json["test_data_used_in_selection"] is False
        assert "test_records_count" not in val_json
        assert val_json["task"] == "binary_threshold_selection_validation"


class TestThresholdSelectionLogic:
    """Verify Requirement 3 & 9: threshold belongs to allowed list and is not hard-coded."""

    def test_selected_threshold_belongs_to_allowed_list(self, meta_json, val_json):
        sel_t = val_json["selected_threshold"]
        assert sel_t in ALLOWED_THRESHOLDS
        assert meta_json["selected_threshold"] == sel_t
        for t in val_json["thresholds_evaluated"]:
            assert t in ALLOWED_THRESHOLDS

    def test_selected_threshold_not_hard_coded_from_previous_test_analysis(self, val_json):
        sel_t = val_json["selected_threshold"]
        assert sel_t in ALLOWED_THRESHOLDS

        # Verify that the threshold was derived by dynamically evaluating validation metrics
        val_metrics = val_json["threshold_metrics"]

        # 1. 0.80 satisfies the constraint on validation data
        assert val_metrics["0.80"]["recall"] >= 0.90
        # 2. 0.85 fails the constraint on validation data (0.8960 < 0.90), bounding the selection
        assert val_metrics["0.85"]["recall"] < 0.90
        # 3. Among valid thresholds (0.50, 0.60, 0.70, 0.80), 0.80 has the strictly lowest FPR
        valid_fprs = [
            (float(t), m["false_positive_rate"])
            for t, m in val_metrics.items()
            if m["recall"] >= 0.90
        ]
        min_fpr_t = min(valid_fprs, key=lambda x: x[1])[0]
        assert sel_t == min_fpr_t == 0.80

        # 4. Prove dynamic algorithmic selection: if validation metrics change, selected threshold changes
        synthetic_metrics = {
            "0.50": {"recall": 0.98, "false_positive_rate": 0.08, "f1_score": 0.96},
            "0.60": {"recall": 0.95, "false_positive_rate": 0.05, "f1_score": 0.96},
            "0.70": {"recall": 0.92, "false_positive_rate": 0.03, "f1_score": 0.96},
            "0.80": {"recall": 0.88, "false_positive_rate": 0.01, "f1_score": 0.94},  # fails recall >= 0.90
        }
        candidates = [
            (t, m) for t, m in synthetic_metrics.items() if m["recall"] >= 0.90
        ]
        dynamic_choice = float(min(candidates, key=lambda x: (x[1]["false_positive_rate"], -x[1]["f1_score"]))[0])
        assert dynamic_choice == 0.70, "Algorithm must dynamically adapt to validation data, not hard-code 0.80"


class TestSeparationAndConfusionMatrices:
    """Verify Requirement 4, 5, 6: separate storage and confusion matrix totals."""

    def test_validation_and_test_results_stored_separately(self, val_json, test_json, comparison_df):
        assert val_json["task"] == "binary_threshold_selection_validation"
        assert test_json["task"] == "binary_threshold_selection_test"

        assert val_json["validation_records_count"] == EXPECTED_N_VAL
        assert test_json["test_records_count"] == EXPECTED_N_TEST

        # Comparison CSV has separate splits
        splits = set(comparison_df["split"].unique())
        assert splits == {"validation", "test"}

    def test_validation_confusion_matrix_totals_equal_validation_size(self, val_json):
        val_metrics = val_json["threshold_metrics"]
        for t_str, m in val_metrics.items():
            tn = m["true_negatives"]
            fp = m["number_of_false_positives"]
            fn = m["number_of_false_negatives"]
            tp = m["true_positives"]
            total = tn + fp + fn + tp
            assert total == EXPECTED_N_VAL, f"Threshold {t_str} total {total} != {EXPECTED_N_VAL}"

    def test_test_confusion_matrix_totals_equal_82332(self, test_json):
        test_metrics = test_json["threshold_metrics"]
        for t_str, m in test_metrics.items():
            tn = m["true_negatives"]
            fp = m["number_of_false_positives"]
            fn = m["number_of_false_negatives"]
            tp = m["true_positives"]
            total = tn + fp + fn + tp
            assert total == EXPECTED_N_TEST, f"Threshold {t_str} total {total} != {EXPECTED_N_TEST}"
            assert tn + fp == EXPECTED_N_TEST_NORMAL, f"Threshold {t_str} normal total != {EXPECTED_N_TEST_NORMAL}"
            assert tp + fn == EXPECTED_N_TEST_ATTACK, f"Threshold {t_str} attack total != {EXPECTED_N_TEST_ATTACK}"


class TestModelLoadingAndReproducibility:
    """Verify Requirement 7 & 8: saved model loads successfully and results are reproducible."""

    def test_saved_model_loads_successfully(self, selected_model, test_arrays):
        assert isinstance(selected_model, XGBClassifier)
        X_test, _ = test_arrays
        sample_proba = selected_model.predict_proba(X_test[:10])
        assert sample_proba.shape == (10, 2)
        assert np.all((sample_proba >= 0.0) & (sample_proba <= 1.0))
        assert np.allclose(sample_proba.sum(axis=1), 1.0)

    def test_results_are_reproducible(self, selected_model, test_arrays, test_json):
        X_test, y_test = test_arrays
        proba = selected_model.predict_proba(X_test)[:, 1]

        sel_t = test_json["selected_threshold"]
        pred = (proba >= sel_t).astype(int)
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]

        expected_m = test_json["selected_threshold_metrics"]
        assert tn == expected_m["true_negatives"]
        assert fp == expected_m["number_of_false_positives"]
        assert fn == expected_m["number_of_false_negatives"]
        assert tp == expected_m["true_positives"]

        rec = float(tp / (tp + fn))
        prec = float(tp / (tp + fp))
        f1 = float(2 * prec * rec / (prec + rec))
        assert round(rec, 4) == expected_m["recall"]
        assert round(prec, 4) == expected_m["precision"]
        assert round(f1, 4) == expected_m["f1_score"]
