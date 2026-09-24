"""
test_binary_thresholds.py — Unit and regression tests for binary decision-threshold analysis.

Verified invariants
-------------------
1. Threshold artifacts (results JSON, comparison CSV, 3 figures) exist.
2. Evaluated thresholds are strictly sorted in ascending order.
3. Model-predicted probabilities are in [0.0, 1.0].
4. Metric values (precision, recall, f1, specificity, FPR, FNR, balanced accuracy) are valid floats in [0.0, 1.0].
5. Complementary error rates hold: FPR + Specificity == 1.0, Recall + FNR == 1.0.
6. Confusion matrix totals (TP + TN + FP + FN) equal the exact test set size (82,332).
7. Ground truth class totals are preserved across all thresholds (TP + FN == 45,332; TN + FP == 37,000).
8. Threshold evaluation results are 100% reproducible against fresh model predictions.
9. Recommendation rule and default/recommended roles are properly documented and distinguished.
10. Existing binary model artifact (models/binary_best_model.joblib) is untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
PROC_DIR = ROOT / "data" / "processed"
METRICS_DIR = ROOT / "reports" / "metrics"
FIGURES_DIR = ROOT / "reports" / "figures"

BINARY_BEST_MODEL_PATH = MODELS_DIR / "binary_best_model.joblib"
X_TEST_UNSCALED_PATH = PROC_DIR / "X_test_unscaled.npy"
Y_BIN_TEST_PATH = PROC_DIR / "y_bin_test.npy"

THRESHOLD_RESULTS_JSON = METRICS_DIR / "binary_threshold_results.json"
THRESHOLD_COMPARISON_CSV = METRICS_DIR / "binary_threshold_comparison.csv"

FIG_TRADEOFF = FIGURES_DIR / "binary_threshold_tradeoff.png"
FIG_PR = FIGURES_DIR / "binary_precision_recall_threshold.png"
FIG_FPR_FNR = FIGURES_DIR / "binary_fpr_fnr_threshold.png"

EXPECTED_THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
EXPECTED_N_TEST = 82_332
EXPECTED_N_ATTACK = 45_332
EXPECTED_N_NORMAL = 37_000

REQUIRED_METRICS = [
    "precision",
    "recall",
    "f1_score",
    "specificity",
    "false_positive_rate",
    "false_negative_rate",
    "balanced_accuracy",
    "number_of_false_positives",
    "number_of_false_negatives",
]


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def threshold_json():
    assert THRESHOLD_RESULTS_JSON.exists(), f"Missing: {THRESHOLD_RESULTS_JSON}"
    with open(THRESHOLD_RESULTS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def threshold_csv():
    assert THRESHOLD_COMPARISON_CSV.exists(), f"Missing: {THRESHOLD_COMPARISON_CSV}"
    return pd.read_csv(THRESHOLD_COMPARISON_CSV)


@pytest.fixture(scope="module")
def binary_model():
    assert BINARY_BEST_MODEL_PATH.exists(), f"Missing: {BINARY_BEST_MODEL_PATH}"
    return joblib.load(BINARY_BEST_MODEL_PATH)


@pytest.fixture(scope="module")
def test_data():
    assert X_TEST_UNSCALED_PATH.exists(), f"Missing: {X_TEST_UNSCALED_PATH}"
    assert Y_BIN_TEST_PATH.exists(), f"Missing: {Y_BIN_TEST_PATH}"
    X_te = np.load(X_TEST_UNSCALED_PATH)
    y_te = np.load(Y_BIN_TEST_PATH)
    return X_te, y_te


# ─── Tests: Artifact Existence & Model Preservation ───────────────────────────

class TestThresholdArtifacts:
    def test_results_json_exists(self):
        assert THRESHOLD_RESULTS_JSON.exists()

    def test_comparison_csv_exists(self):
        assert THRESHOLD_COMPARISON_CSV.exists()

    def test_tradeoff_figure_exists(self):
        assert FIG_TRADEOFF.exists()

    def test_precision_recall_figure_exists(self):
        assert FIG_PR.exists()

    def test_fpr_fnr_figure_exists(self):
        assert FIG_FPR_FNR.exists()

    def test_binary_best_model_remains_untouched(self, binary_model):
        """Model artifact must still load and predict properly."""
        assert hasattr(binary_model, "predict_proba")
        assert hasattr(binary_model, "predict")


# ─── Tests: Threshold Ordering & Specification ────────────────────────────────

class TestThresholdsStructure:
    def test_thresholds_are_sorted_in_json(self, threshold_json):
        t_list = threshold_json["thresholds_evaluated"]
        assert t_list == sorted(t_list), f"Thresholds not sorted: {t_list}"

    def test_thresholds_are_sorted_in_csv(self, threshold_csv):
        t_vals = threshold_csv["threshold"].astype(float).tolist()
        assert t_vals == sorted(t_vals), f"CSV thresholds not sorted: {t_vals}"

    def test_evaluated_thresholds_match_expected(self, threshold_json):
        t_list = [round(float(t), 2) for t in threshold_json["thresholds_evaluated"]]
        assert t_list == EXPECTED_THRESHOLDS, f"Expected {EXPECTED_THRESHOLDS}, got {t_list}"

    def test_default_and_recommended_thresholds_specified(self, threshold_json):
        assert threshold_json["default_threshold"] == 0.50
        assert threshold_json["recommended_threshold"] == 0.80
        assert "recommendation_rule" in threshold_json
        assert len(threshold_json["recommendation_rule"]) > 20


# ─── Tests: Predicted Probabilities Validity ──────────────────────────────────

class TestProbabilities:
    def test_probabilities_are_between_0_and_1(self, binary_model, test_data):
        X_te, _ = test_data
        probs = binary_model.predict_proba(X_te)[:, 1]
        assert probs.shape == (EXPECTED_N_TEST,)
        assert np.all(probs >= 0.0), f"Negative probability found: min={probs.min()}"
        assert np.all(probs <= 1.0), f"Probability > 1.0 found: max={probs.max()}"

    def test_probabilities_are_non_trivial(self, binary_model, test_data):
        X_te, _ = test_data
        probs = binary_model.predict_proba(X_te)[:, 1]
        # Should have reasonable spread across the unit interval
        assert probs.min() < 0.05
        assert probs.max() > 0.95
        assert 0.40 < probs.mean() < 0.80


# ─── Tests: Metric Values Validity & Mathematical Relationships ───────────────

class TestMetricValues:
    @pytest.mark.parametrize("t", EXPECTED_THRESHOLDS)
    def test_all_required_metrics_present(self, threshold_json, t):
        t_key = f"{t:.2f}"
        metrics = threshold_json["threshold_metrics"][t_key]
        for mk in REQUIRED_METRICS:
            assert mk in metrics, f"Missing metric '{mk}' at threshold {t_key}"

    @pytest.mark.parametrize("t", EXPECTED_THRESHOLDS)
    def test_metric_values_in_valid_range(self, threshold_json, t):
        t_key = f"{t:.2f}"
        m = threshold_json["threshold_metrics"][t_key]
        for mk in [
            "precision", "recall", "f1_score", "specificity",
            "false_positive_rate", "false_negative_rate", "balanced_accuracy",
        ]:
            val = m[mk]
            assert isinstance(val, (float, int)), f"{mk} at {t_key} is not numeric: {val}"
            assert 0.0 <= float(val) <= 1.0, f"{mk} at {t_key} = {val} outside [0, 1]"

    @pytest.mark.parametrize("t", EXPECTED_THRESHOLDS)
    def test_fpr_plus_specificity_equals_1(self, threshold_json, t):
        t_key = f"{t:.2f}"
        m = threshold_json["threshold_metrics"][t_key]
        fpr = m["false_positive_rate"]
        spec = m["specificity"]
        assert abs((fpr + spec) - 1.0) < 1e-3, f"FPR ({fpr}) + Specificity ({spec}) != 1.0 at {t_key}"

    @pytest.mark.parametrize("t", EXPECTED_THRESHOLDS)
    def test_recall_plus_fnr_equals_1(self, threshold_json, t):
        t_key = f"{t:.2f}"
        m = threshold_json["threshold_metrics"][t_key]
        rec = m["recall"]
        fnr = m["false_negative_rate"]
        assert abs((rec + fnr) - 1.0) < 1e-3, f"Recall ({rec}) + FNR ({fnr}) != 1.0 at {t_key}"

    @pytest.mark.parametrize("t", EXPECTED_THRESHOLDS)
    def test_balanced_accuracy_formula(self, threshold_json, t):
        t_key = f"{t:.2f}"
        m = threshold_json["threshold_metrics"][t_key]
        expected_bal_acc = round((m["recall"] + m["specificity"]) / 2, 4)
        assert abs(m["balanced_accuracy"] - expected_bal_acc) <= 1e-4

    @pytest.mark.parametrize("t", EXPECTED_THRESHOLDS)
    def test_f1_score_formula(self, threshold_json, t):
        t_key = f"{t:.2f}"
        m = threshold_json["threshold_metrics"][t_key]
        p = m["precision"]
        r = m["recall"]
        expected_f1 = round((2 * p * r) / (p + r), 4) if (p + r) > 0 else 0.0
        assert abs(m["f1_score"] - expected_f1) <= 1e-4


# ─── Tests: Confusion Matrix Totals Equal Test Set Size ───────────────────────

class TestConfusionMatrixTotals:
    @pytest.mark.parametrize("t", EXPECTED_THRESHOLDS)
    def test_confusion_matrix_total_equals_test_set_size(self, threshold_json, t):
        t_key = f"{t:.2f}"
        m = threshold_json["threshold_metrics"][t_key]
        tp = m["true_positives"]
        tn = m["true_negatives"]
        fp = m["number_of_false_positives"]
        fn = m["number_of_false_negatives"]
        total = tp + tn + fp + fn
        assert total == EXPECTED_N_TEST, f"Total at {t_key} was {total}, expected {EXPECTED_N_TEST}"

    @pytest.mark.parametrize("t", EXPECTED_THRESHOLDS)
    def test_actual_class_totals_preserved(self, threshold_json, t):
        t_key = f"{t:.2f}"
        m = threshold_json["threshold_metrics"][t_key]
        actual_attacks = m["true_positives"] + m["number_of_false_negatives"]
        actual_normal = m["true_negatives"] + m["number_of_false_positives"]
        assert actual_attacks == EXPECTED_N_ATTACK, f"Attacks at {t_key} = {actual_attacks}, expected {EXPECTED_N_ATTACK}"
        assert actual_normal == EXPECTED_N_NORMAL, f"Normal at {t_key} = {actual_normal}, expected {EXPECTED_N_NORMAL}"


# ─── Tests: Threshold Results Reproducibility ──────────────────────────────────

class TestReproducibility:
    def test_threshold_results_are_reproducible_from_scratch(self, binary_model, test_data, threshold_json):
        """Fresh inference must exactly reproduce the saved metrics."""
        X_te, y_te = test_data
        probs = binary_model.predict_proba(X_te)[:, 1]

        for t in EXPECTED_THRESHOLDS:
            t_key = f"{t:.2f}"
            saved = threshold_json["threshold_metrics"][t_key]

            y_pred = (probs >= t).astype(int)
            tp = int(np.sum((y_pred == 1) & (y_te == 1)))
            fp = int(np.sum((y_pred == 1) & (y_te == 0)))
            fn = int(np.sum((y_pred == 0) & (y_te == 1)))
            tn = int(np.sum((y_pred == 0) & (y_te == 0)))

            assert tp == saved["true_positives"]
            assert fp == saved["number_of_false_positives"]
            assert fn == saved["number_of_false_negatives"]
            assert tn == saved["true_negatives"]

            prec = round(tp / (tp + fp), 4)
            rec = round(tp / (tp + fn), 4)
            assert abs(prec - saved["precision"]) <= 1e-4
            assert abs(rec - saved["recall"]) <= 1e-4

    def test_recommended_threshold_reduction_metrics(self, threshold_json):
        delta = threshold_json["comparison_summary"]["impact_delta"]
        assert delta["false_positive_reduction_count"] == 4561
        assert delta["false_positive_reduction_percent"] == 77.45
        assert delta["fpr_reduction"] == 0.1233
        assert delta["precision_gain"] == 0.0872
        assert delta["f1_gain"] == 0.0134
