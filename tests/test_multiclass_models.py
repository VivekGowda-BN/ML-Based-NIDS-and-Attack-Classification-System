"""
test_multiclass_models.py — Regression tests for the multiclass attack-classification phase.

Verified invariants
-------------------
- All three model artifacts load without error.
- Predictions have exactly 82 332 rows (test-partition size).
- Decoded predictions are subset of the 10 verified attack categories.
- Probability matrices are (82332, 10) and every row sums to ≈ 1.0.
- Confusion matrices stored in the results JSON are 10 × 10.
- Required aggregate metrics keys exist in the results JSON.
- Rare-class metrics (Analysis, Backdoor, Shellcode, Worms) exist per model.
- Neither 'attack_cat' nor 'label' appear in feature_names.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
METRICS_DIR = ROOT / "reports" / "metrics"
FIGURES_DIR = ROOT / "reports" / "figures"

LR_MODEL_PATH   = MODELS_DIR / "multiclass_logistic_regression.joblib"
RF_MODEL_PATH   = MODELS_DIR / "multiclass_random_forest.joblib"
XGB_MODEL_PATH  = MODELS_DIR / "multiclass_xgboost.joblib"
BEST_MODEL_PATH = MODELS_DIR / "multiclass_best_model.joblib"
LE_PATH         = MODELS_DIR / "label_encoder.joblib"

X_TEST_SCALED_PATH   = PROCESSED_DIR / "X_test_scaled.npy"
X_TEST_UNSCALED_PATH = PROCESSED_DIR / "X_test_unscaled.npy"
Y_MULTI_TEST_PATH    = PROCESSED_DIR / "y_multi_test.npy"
FEATURE_NAMES_PATH   = PROCESSED_DIR / "feature_names.json"

RESULTS_JSON   = METRICS_DIR / "multiclass_results.json"
COMPARISON_CSV = METRICS_DIR / "multiclass_model_comparison.csv"
BEST_JSON      = METRICS_DIR / "multiclass_best_model.json"

EXPECTED_N_TEST   = 82_332
EXPECTED_N_CLASSES = 10
VALID_CATEGORIES  = {
    "Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers",
    "Generic", "Normal", "Reconnaissance", "Shellcode", "Worms",
}
RARE_CLASSES = ["Analysis", "Backdoor", "Shellcode", "Worms"]

REQUIRED_AGGREGATE_METRICS = [
    "accuracy", "balanced_accuracy",
    "macro_precision", "macro_recall", "macro_f1",
    "weighted_precision", "weighted_recall", "weighted_f1",
]

MODEL_KEYS = ["logistic_regression", "random_forest", "xgboost"]


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def label_encoder():
    assert LE_PATH.exists(), f"Label encoder not found: {LE_PATH}"
    return joblib.load(LE_PATH)


@pytest.fixture(scope="module")
def X_test_scaled():
    assert X_TEST_SCALED_PATH.exists(), f"Missing: {X_TEST_SCALED_PATH}"
    return np.load(X_TEST_SCALED_PATH)


@pytest.fixture(scope="module")
def X_test_unscaled():
    assert X_TEST_UNSCALED_PATH.exists(), f"Missing: {X_TEST_UNSCALED_PATH}"
    return np.load(X_TEST_UNSCALED_PATH)


@pytest.fixture(scope="module")
def y_multi_test():
    assert Y_MULTI_TEST_PATH.exists(), f"Missing: {Y_MULTI_TEST_PATH}"
    return np.load(Y_MULTI_TEST_PATH)


@pytest.fixture(scope="module")
def results_json():
    assert RESULTS_JSON.exists(), f"Results JSON not found: {RESULTS_JSON}"
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def feature_names():
    assert FEATURE_NAMES_PATH.exists(), f"Missing: {FEATURE_NAMES_PATH}"
    with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    # feature_names.json is a dict; the flat list of 194 post-transform names
    # lives under 'transformed_feature_names'.
    if isinstance(data, dict):
        return data["transformed_feature_names"]
    return data  # fallback: already a list


# ─── Tests: model artifacts ────────────────────────────────────────────────────

class TestModelArtifacts:
    def test_lr_model_loads(self):
        assert LR_MODEL_PATH.exists(), f"Missing: {LR_MODEL_PATH}"
        model = joblib.load(LR_MODEL_PATH)
        assert hasattr(model, "predict"), "LR model missing predict()"

    def test_rf_model_loads(self):
        assert RF_MODEL_PATH.exists(), f"Missing: {RF_MODEL_PATH}"
        model = joblib.load(RF_MODEL_PATH)
        assert hasattr(model, "predict"), "RF model missing predict()"

    def test_xgb_model_loads(self):
        assert XGB_MODEL_PATH.exists(), f"Missing: {XGB_MODEL_PATH}"
        model = joblib.load(XGB_MODEL_PATH)
        assert hasattr(model, "predict"), "XGB model missing predict()"

    def test_best_model_loads(self):
        assert BEST_MODEL_PATH.exists(), f"Missing: {BEST_MODEL_PATH}"
        model = joblib.load(BEST_MODEL_PATH)
        assert hasattr(model, "predict"), "Best model missing predict()"


# ─── Tests: prediction shape ───────────────────────────────────────────────────

class TestPredictionShape:
    def test_lr_prediction_row_count(self, X_test_scaled, y_multi_test):
        model = joblib.load(LR_MODEL_PATH)
        preds = model.predict(X_test_scaled)
        assert len(preds) == EXPECTED_N_TEST, (
            f"LR: expected {EXPECTED_N_TEST} predictions, got {len(preds)}"
        )

    def test_rf_prediction_row_count(self, X_test_unscaled, y_multi_test):
        model = joblib.load(RF_MODEL_PATH)
        preds = model.predict(X_test_unscaled)
        assert len(preds) == EXPECTED_N_TEST, (
            f"RF: expected {EXPECTED_N_TEST} predictions, got {len(preds)}"
        )

    def test_xgb_prediction_row_count(self, X_test_unscaled, y_multi_test):
        model = joblib.load(XGB_MODEL_PATH)
        preds = model.predict(X_test_unscaled)
        assert len(preds) == EXPECTED_N_TEST, (
            f"XGB: expected {EXPECTED_N_TEST} predictions, got {len(preds)}"
        )


# ─── Tests: decoded predictions are valid categories ──────────────────────────

class TestDecodedPredictions:
    def test_lr_decoded_predictions_valid(self, X_test_scaled, label_encoder):
        model = joblib.load(LR_MODEL_PATH)
        encoded_preds = model.predict(X_test_scaled)
        decoded = set(label_encoder.inverse_transform(encoded_preds))
        assert decoded.issubset(VALID_CATEGORIES), (
            f"LR decoded predictions contain unknown categories: {decoded - VALID_CATEGORIES}"
        )

    def test_rf_decoded_predictions_valid(self, X_test_unscaled, label_encoder):
        model = joblib.load(RF_MODEL_PATH)
        encoded_preds = model.predict(X_test_unscaled)
        decoded = set(label_encoder.inverse_transform(encoded_preds))
        assert decoded.issubset(VALID_CATEGORIES), (
            f"RF decoded predictions contain unknown categories: {decoded - VALID_CATEGORIES}"
        )

    def test_xgb_decoded_predictions_valid(self, X_test_unscaled, label_encoder):
        model = joblib.load(XGB_MODEL_PATH)
        encoded_preds = model.predict(X_test_unscaled)
        decoded = set(label_encoder.inverse_transform(encoded_preds))
        assert decoded.issubset(VALID_CATEGORIES), (
            f"XGB decoded predictions contain unknown categories: {decoded - VALID_CATEGORIES}"
        )


# ─── Tests: probability rows sum to 1 ─────────────────────────────────────────

class TestProbabilityRows:
    def test_lr_probability_shape_and_sum(self, X_test_scaled):
        model = joblib.load(LR_MODEL_PATH)
        proba = model.predict_proba(X_test_scaled)
        assert proba.shape == (EXPECTED_N_TEST, EXPECTED_N_CLASSES), (
            f"LR proba shape mismatch: {proba.shape}"
        )
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5,
                                   err_msg="LR probability rows do not sum to 1")

    def test_rf_probability_shape_and_sum(self, X_test_unscaled):
        model = joblib.load(RF_MODEL_PATH)
        proba = model.predict_proba(X_test_unscaled)
        assert proba.shape == (EXPECTED_N_TEST, EXPECTED_N_CLASSES), (
            f"RF proba shape mismatch: {proba.shape}"
        )
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5,
                                   err_msg="RF probability rows do not sum to 1")

    def test_xgb_probability_shape_and_sum(self, X_test_unscaled):
        model = joblib.load(XGB_MODEL_PATH)
        proba = model.predict_proba(X_test_unscaled)
        assert proba.shape == (EXPECTED_N_TEST, EXPECTED_N_CLASSES), (
            f"XGB proba shape mismatch: {proba.shape}"
        )
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-4,
                                   err_msg="XGB probability rows do not sum to 1")


# ─── Tests: confusion matrices are 10x10 ──────────────────────────────────────

class TestConfusionMatrices:
    @pytest.mark.parametrize("model_key", MODEL_KEYS)
    def test_confusion_matrix_is_10x10(self, results_json, model_key):
        cm = results_json["models"][model_key]["confusion_matrix"]
        assert len(cm) == EXPECTED_N_CLASSES, (
            f"{model_key}: confusion matrix has {len(cm)} rows, expected {EXPECTED_N_CLASSES}"
        )
        for i, row in enumerate(cm):
            assert len(row) == EXPECTED_N_CLASSES, (
                f"{model_key}: confusion matrix row {i} has {len(row)} cols, expected {EXPECTED_N_CLASSES}"
            )


# ─── Tests: required metrics exist ────────────────────────────────────────────

class TestRequiredMetrics:
    @pytest.mark.parametrize("model_key", MODEL_KEYS)
    def test_aggregate_metrics_exist(self, results_json, model_key):
        metrics = results_json["models"][model_key]["metrics"]
        for metric in REQUIRED_AGGREGATE_METRICS:
            assert metric in metrics, (
                f"{model_key}: missing required metric '{metric}'"
            )

    @pytest.mark.parametrize("model_key", MODEL_KEYS)
    def test_aggregate_metrics_are_finite(self, results_json, model_key):
        metrics = results_json["models"][model_key]["metrics"]
        for metric in REQUIRED_AGGREGATE_METRICS:
            value = metrics[metric]
            assert isinstance(value, (int, float)), (
                f"{model_key}.{metric}: expected numeric, got {type(value)}"
            )
            assert 0.0 <= float(value) <= 1.0, (
                f"{model_key}.{metric}: value {value} outside [0, 1]"
            )


# ─── Tests: rare-class metrics exist ──────────────────────────────────────────

class TestRareClassMetrics:
    @pytest.mark.parametrize("model_key", MODEL_KEYS)
    @pytest.mark.parametrize("rare_cls", RARE_CLASSES)
    def test_rare_class_metrics_exist(self, results_json, model_key, rare_cls):
        per_class = results_json["models"][model_key]["per_class_metrics"]
        assert rare_cls in per_class, (
            f"{model_key}: missing per-class entry for rare class '{rare_cls}'"
        )
        for metric in ("precision", "recall", "f1", "support"):
            assert metric in per_class[rare_cls], (
                f"{model_key}.{rare_cls}: missing key '{metric}'"
            )


# ─── Tests: feature names exclude target columns ──────────────────────────────

class TestFeatureNames:
    def test_attack_cat_absent_from_features(self, feature_names):
        assert "attack_cat" not in feature_names, (
            "'attack_cat' found in feature_names.json — leakage risk!"
        )

    def test_label_absent_from_features(self, feature_names):
        assert "label" not in feature_names, (
            "'label' found in feature_names.json — leakage risk!"
        )

    def test_feature_count_is_194(self, feature_names):
        assert len(feature_names) == 194, (
            f"Expected 194 features, got {len(feature_names)}"
        )


# ─── Tests: output artifacts exist ────────────────────────────────────────────

class TestOutputArtifacts:
    def test_results_json_exists(self):
        assert RESULTS_JSON.exists(), f"Missing: {RESULTS_JSON}"

    def test_comparison_csv_exists(self):
        assert COMPARISON_CSV.exists(), f"Missing: {COMPARISON_CSV}"

    def test_best_model_json_exists(self):
        assert BEST_JSON.exists(), f"Missing: {BEST_JSON}"

    @pytest.mark.parametrize("fig_name", [
        "multiclass_confusion_matrix_logistic_regression.png",
        "multiclass_confusion_matrix_random_forest.png",
        "multiclass_confusion_matrix_xgboost.png",
        "multiclass_model_comparison.png",
        "multiclass_per_class_f1.png",
    ])
    def test_figure_exists(self, fig_name):
        fig_path = FIGURES_DIR / fig_name
        assert fig_path.exists(), f"Missing figure: {fig_path}"

    def test_best_model_json_contains_selected_model(self):
        with open(BEST_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "selected_model" in data, "best_model.json missing 'selected_model'"
        assert data["selected_model"] in MODEL_KEYS, (
            f"selected_model '{data['selected_model']}' not a valid model key"
        )

    def test_results_json_class_names(self, results_json):
        assert "class_names" in results_json, "results JSON missing 'class_names'"
        assert set(results_json["class_names"]) == VALID_CATEGORIES, (
            f"class_names mismatch: {set(results_json['class_names'])}"
        )
