"""
test_attack_only_multiclass.py — Regression tests for the attack-only 9-class experiment.

Verified invariants
-------------------
- All four model artifacts load without error.
- The 9-class label encoder contains exactly 9 attack categories (no Normal).
- Label encoder maps to the expected ATTACK_CLASSES list.
- Predictions have exactly 45 332 rows (attack-only test partition).
- Decoded predictions are a subset of the 9 valid attack categories.
- Probability matrices are (45332, 9) with every row summing to ≈ 1.
- Confusion matrices in the results JSON are 9 × 9.
- Required aggregate metric keys exist and are in [0, 1].
- Rare-class per-class metrics exist for Analysis, Backdoor, Shellcode, Worms.
- No Normal class appears in predictions or class_names.
- No target columns (attack_cat, label) in feature_names.
- All required output artifacts (JSON, CSV, figures) exist.
- Baseline artifacts are untouched (multiclass_best_model.joblib still exists).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
MODELS_DIR  = ROOT / "models"
PROC_DIR    = ROOT / "data" / "processed"
METRICS_DIR = ROOT / "reports" / "metrics"
FIGURES_DIR = ROOT / "reports" / "figures"

AO_LR_PATH   = MODELS_DIR / "attack_only_multiclass_logistic_regression.joblib"
AO_RF_PATH   = MODELS_DIR / "attack_only_multiclass_random_forest.joblib"
AO_XGB_PATH  = MODELS_DIR / "attack_only_multiclass_xgboost.joblib"
AO_BEST_PATH = MODELS_DIR / "attack_only_multiclass_best_model.joblib"
AO_LE_PATH   = MODELS_DIR / "attack_only_label_encoder.joblib"

# Baseline artifact that must NOT be overwritten
BASELINE_BEST  = MODELS_DIR / "multiclass_best_model.joblib"
BASELINE_JSON  = METRICS_DIR / "multiclass_results.json"

AO_RESULTS_JSON   = METRICS_DIR / "attack_only_multiclass_results.json"
AO_BEST_JSON      = METRICS_DIR / "attack_only_multiclass_best_model.json"
AO_COMPARISON_CSV = METRICS_DIR / "attack_only_multiclass_comparison.csv"

X_TEST_SCALED_PATH   = PROC_DIR / "X_test_scaled.npy"
X_TEST_UNSCALED_PATH = PROC_DIR / "X_test_unscaled.npy"
Y_BIN_TEST_PATH      = PROC_DIR / "y_bin_test.npy"
FEATURE_NAMES_PATH   = PROC_DIR / "feature_names.json"

EXPECTED_N_TEST   = 45_332      # attack-only test records
EXPECTED_N_CLASSES = 9
ATTACK_CLASSES = [
    "Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers",
    "Generic", "Reconnaissance", "Shellcode", "Worms",
]
RARE_CLASSES = ["Analysis", "Backdoor", "Shellcode", "Worms"]
MODEL_KEYS   = ["logistic_regression", "random_forest", "xgboost"]

REQUIRED_METRICS = [
    "accuracy", "balanced_accuracy",
    "macro_precision", "macro_recall", "macro_f1",
    "weighted_precision", "weighted_recall", "weighted_f1",
]


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ao_label_encoder():
    assert AO_LE_PATH.exists(), f"Missing: {AO_LE_PATH}"
    return joblib.load(AO_LE_PATH)


@pytest.fixture(scope="module")
def X_test_scaled():
    arr = np.load(X_TEST_SCALED_PATH)
    y_bin = np.load(Y_BIN_TEST_PATH)
    return arr[y_bin == 1]   # attack-only


@pytest.fixture(scope="module")
def X_test_unscaled():
    arr = np.load(X_TEST_UNSCALED_PATH)
    y_bin = np.load(Y_BIN_TEST_PATH)
    return arr[y_bin == 1]   # attack-only


@pytest.fixture(scope="module")
def results_json():
    assert AO_RESULTS_JSON.exists(), f"Missing: {AO_RESULTS_JSON}"
    with open(AO_RESULTS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def feature_names():
    assert FEATURE_NAMES_PATH.exists(), f"Missing: {FEATURE_NAMES_PATH}"
    with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data["transformed_feature_names"]
    return data


# ─── Tests: label encoder ─────────────────────────────────────────────────────

class TestLabelEncoder:
    def test_encoder_loads(self, ao_label_encoder):
        assert hasattr(ao_label_encoder, "classes_")

    def test_encoder_has_9_classes(self, ao_label_encoder):
        assert len(ao_label_encoder.classes_) == EXPECTED_N_CLASSES, (
            f"Expected 9 classes, got {len(ao_label_encoder.classes_)}"
        )

    def test_encoder_classes_match_attack_classes(self, ao_label_encoder):
        encoded_classes = sorted(str(c) for c in ao_label_encoder.classes_)
        assert encoded_classes == sorted(ATTACK_CLASSES), (
            f"Encoder classes mismatch: {encoded_classes}"
        )

    def test_encoder_does_not_contain_normal(self, ao_label_encoder):
        classes = [str(c) for c in ao_label_encoder.classes_]
        assert "Normal" not in classes, (
            "Normal class found in attack-only label encoder — leakage!"
        )

    def test_encoder_mapping_is_alphabetical(self, ao_label_encoder):
        """LabelEncoder should map classes in alphabetical order (0=Analysis, …)."""
        classes = [str(c) for c in ao_label_encoder.classes_]
        assert classes == sorted(classes), (
            f"Encoder classes not alphabetical: {classes}"
        )

    def test_encoder_analysis_is_index_0(self, ao_label_encoder):
        idx = ao_label_encoder.transform(["Analysis"])[0]
        assert idx == 0, f"Expected Analysis=0, got {idx}"

    def test_encoder_worms_is_last(self, ao_label_encoder):
        idx = ao_label_encoder.transform(["Worms"])[0]
        assert idx == EXPECTED_N_CLASSES - 1, f"Expected Worms={EXPECTED_N_CLASSES-1}, got {idx}"


# ─── Tests: model artifacts ────────────────────────────────────────────────────

class TestModelArtifacts:
    def test_lr_model_loads(self):
        assert AO_LR_PATH.exists(), f"Missing: {AO_LR_PATH}"
        m = joblib.load(AO_LR_PATH)
        assert hasattr(m, "predict")

    def test_rf_model_loads(self):
        assert AO_RF_PATH.exists(), f"Missing: {AO_RF_PATH}"
        m = joblib.load(AO_RF_PATH)
        assert hasattr(m, "predict")

    def test_xgb_model_loads(self):
        assert AO_XGB_PATH.exists(), f"Missing: {AO_XGB_PATH}"
        m = joblib.load(AO_XGB_PATH)
        assert hasattr(m, "predict")

    def test_best_model_loads(self):
        assert AO_BEST_PATH.exists(), f"Missing: {AO_BEST_PATH}"
        m = joblib.load(AO_BEST_PATH)
        assert hasattr(m, "predict")

    def test_baseline_best_model_untouched(self):
        """Baseline multiclass_best_model.joblib must not be overwritten."""
        assert BASELINE_BEST.exists(), (
            f"Baseline best model was deleted: {BASELINE_BEST}"
        )

    def test_baseline_results_json_untouched(self):
        assert BASELINE_JSON.exists(), (
            f"Baseline results JSON was deleted: {BASELINE_JSON}"
        )


# ─── Tests: prediction shape ───────────────────────────────────────────────────

class TestPredictionShape:
    def test_lr_prediction_count(self, X_test_scaled):
        m = joblib.load(AO_LR_PATH)
        assert len(m.predict(X_test_scaled)) == EXPECTED_N_TEST

    def test_rf_prediction_count(self, X_test_unscaled):
        m = joblib.load(AO_RF_PATH)
        assert len(m.predict(X_test_unscaled)) == EXPECTED_N_TEST

    def test_xgb_prediction_count(self, X_test_unscaled):
        m = joblib.load(AO_XGB_PATH)
        assert len(m.predict(X_test_unscaled)) == EXPECTED_N_TEST


# ─── Tests: decoded predictions valid & no Normal ─────────────────────────────

class TestDecodedPredictions:
    def test_lr_predictions_are_valid_attack_classes(self, X_test_scaled, ao_label_encoder):
        m = joblib.load(AO_LR_PATH)
        decoded = set(str(c) for c in ao_label_encoder.inverse_transform(m.predict(X_test_scaled)))
        assert decoded.issubset(set(ATTACK_CLASSES)), f"Unknown categories: {decoded - set(ATTACK_CLASSES)}"
        assert "Normal" not in decoded, "Normal in LR predictions — label mapping error!"

    def test_rf_predictions_are_valid_attack_classes(self, X_test_unscaled, ao_label_encoder):
        m = joblib.load(AO_RF_PATH)
        decoded = set(str(c) for c in ao_label_encoder.inverse_transform(m.predict(X_test_unscaled)))
        assert decoded.issubset(set(ATTACK_CLASSES)), f"Unknown categories: {decoded - set(ATTACK_CLASSES)}"
        assert "Normal" not in decoded, "Normal in RF predictions — label mapping error!"

    def test_xgb_predictions_are_valid_attack_classes(self, X_test_unscaled, ao_label_encoder):
        m = joblib.load(AO_XGB_PATH)
        decoded = set(str(c) for c in ao_label_encoder.inverse_transform(m.predict(X_test_unscaled)))
        assert decoded.issubset(set(ATTACK_CLASSES)), f"Unknown categories: {decoded - set(ATTACK_CLASSES)}"
        assert "Normal" not in decoded, "Normal in XGB predictions — label mapping error!"


# ─── Tests: probability matrix ────────────────────────────────────────────────

class TestProbabilityRows:
    def test_lr_proba_shape_and_sum(self, X_test_scaled):
        proba = joblib.load(AO_LR_PATH).predict_proba(X_test_scaled)
        assert proba.shape == (EXPECTED_N_TEST, EXPECTED_N_CLASSES), f"Shape: {proba.shape}"
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_rf_proba_shape_and_sum(self, X_test_unscaled):
        proba = joblib.load(AO_RF_PATH).predict_proba(X_test_unscaled)
        assert proba.shape == (EXPECTED_N_TEST, EXPECTED_N_CLASSES), f"Shape: {proba.shape}"
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_xgb_proba_shape_and_sum(self, X_test_unscaled):
        proba = joblib.load(AO_XGB_PATH).predict_proba(X_test_unscaled)
        assert proba.shape == (EXPECTED_N_TEST, EXPECTED_N_CLASSES), f"Shape: {proba.shape}"
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-4)


# ─── Tests: confusion matrices are 9×9 ────────────────────────────────────────

class TestConfusionMatrices:
    @pytest.mark.parametrize("model_key", MODEL_KEYS)
    def test_cm_is_9x9(self, results_json, model_key):
        cm = results_json["models"][model_key]["confusion_matrix"]
        assert len(cm) == EXPECTED_N_CLASSES, f"{model_key}: CM rows={len(cm)}"
        for i, row in enumerate(cm):
            assert len(row) == EXPECTED_N_CLASSES, f"{model_key}: CM[{i}] cols={len(row)}"


# ─── Tests: aggregate metrics exist and are in range ─────────────────────────

class TestRequiredMetrics:
    @pytest.mark.parametrize("model_key", MODEL_KEYS)
    def test_all_required_metrics_present(self, results_json, model_key):
        m = results_json["models"][model_key]["metrics"]
        for key in REQUIRED_METRICS:
            assert key in m, f"{model_key}: missing metric '{key}'"

    @pytest.mark.parametrize("model_key", MODEL_KEYS)
    def test_metrics_are_in_range(self, results_json, model_key):
        m = results_json["models"][model_key]["metrics"]
        for key in REQUIRED_METRICS:
            v = m[key]
            assert isinstance(v, (int, float)), f"{model_key}.{key}: not numeric"
            assert 0.0 <= float(v) <= 1.0, f"{model_key}.{key}={v} outside [0,1]"


# ─── Tests: rare-class per-class metrics ─────────────────────────────────────

class TestRareClassMetrics:
    @pytest.mark.parametrize("model_key", MODEL_KEYS)
    @pytest.mark.parametrize("rare_cls", RARE_CLASSES)
    def test_rare_class_metrics_exist(self, results_json, model_key, rare_cls):
        pc = results_json["models"][model_key]["per_class_metrics"]
        assert rare_cls in pc, f"{model_key}: missing rare class '{rare_cls}'"
        for k in ("precision", "recall", "f1", "support"):
            assert k in pc[rare_cls], f"{model_key}.{rare_cls}: missing '{k}'"

    @pytest.mark.parametrize("model_key", MODEL_KEYS)
    @pytest.mark.parametrize("rare_cls", RARE_CLASSES)
    def test_rare_class_support_matches_known(self, results_json, model_key, rare_cls):
        """Verify test partition sizes match the verified dataset profile."""
        expected = {"Analysis": 677, "Backdoor": 583, "Shellcode": 378, "Worms": 44}
        pc = results_json["models"][model_key]["per_class_metrics"]
        actual = pc[rare_cls]["support"]
        assert actual == expected[rare_cls], (
            f"{model_key}.{rare_cls}: support={actual}, expected={expected[rare_cls]}"
        )


# ─── Tests: class_names in results JSON ───────────────────────────────────────

class TestClassNames:
    def test_class_names_are_9_attack_classes(self, results_json):
        cnames = results_json["class_names"]
        assert set(cnames) == set(ATTACK_CLASSES), f"class_names mismatch: {cnames}"
        assert len(cnames) == EXPECTED_N_CLASSES

    def test_normal_absent_from_class_names(self, results_json):
        assert "Normal" not in results_json["class_names"], (
            "Normal class present in attack-only results — mapping error!"
        )

    def test_class_names_are_plain_strings(self, results_json):
        for c in results_json["class_names"]:
            assert type(c) is str, f"class_name '{c}' is not a plain Python str (type={type(c)})"


# ─── Tests: feature leakage guards ────────────────────────────────────────────

class TestFeatureLeakage:
    def test_attack_cat_absent(self, feature_names):
        assert "attack_cat" not in feature_names

    def test_label_absent(self, feature_names):
        assert "label" not in feature_names


# ─── Tests: output artifacts exist ────────────────────────────────────────────

class TestOutputArtifacts:
    def test_results_json_exists(self):
        assert AO_RESULTS_JSON.exists()

    def test_comparison_csv_exists(self):
        assert AO_COMPARISON_CSV.exists()

    def test_attack_only_label_encoder_exists(self):
        assert AO_LE_PATH.exists()

    @pytest.mark.parametrize("fig_name", [
        "attack_only_multiclass_confusion_matrix_logistic_regression.png",
        "attack_only_multiclass_confusion_matrix_random_forest.png",
        "attack_only_multiclass_confusion_matrix_xgboost.png",
        "attack_only_multiclass_model_comparison.png",
        "attack_only_multiclass_per_class_f1.png",
        "attack_only_multiclass_rare_class_recall.png",
    ])
    def test_figure_exists(self, fig_name):
        assert (FIGURES_DIR / fig_name).exists(), f"Missing figure: {fig_name}"

    def test_best_model_json_has_selected_model(self):
        with open(AO_RESULTS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "selected_model" in data["selection"]
        assert data["selection"]["selected_model"] in MODEL_KEYS

    def test_results_json_experiment_key(self):
        with open(AO_RESULTS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("experiment") == "attack_only_multiclass"


# ─── Tests: model selection methodology (leakage-free) ─────────────────────────

class TestModelSelectionMethodology:
    def test_best_model_json_artifact_exists(self):
        assert AO_BEST_JSON.exists(), f"Missing: {AO_BEST_JSON}"

    def test_validation_data_derived_only_from_training_data(self):
        """Confirm validation set is drawn strictly from y_bin_train == 1."""
        try:
            from nids.preprocessing import load_splits
            from nids.attack_only_multiclass import build_attack_only_splits
        except ImportError:
            from src.nids.preprocessing import load_splits
            from src.nids.attack_only_multiclass import build_attack_only_splits

        splits = load_splits(PROC_DIR)
        ao_splits = build_attack_only_splits(splits)

        n_train_atk = int((splits["y_bin_train"] == 1).sum())
        assert len(ao_splits["y_train"]) + len(ao_splits["y_val"]) == n_train_atk, (
            f"Train ({len(ao_splits['y_train'])}) + Val ({len(ao_splits['y_val'])}) "
            f"!= Total training attacks ({n_train_atk})"
        )
        # Verify test split count is completely independent
        n_test_atk = int((splits["y_bin_test"] == 1).sum())
        assert len(ao_splits["y_test"]) == n_test_atk == EXPECTED_N_TEST

    def test_test_data_not_used_during_model_selection(self):
        """Confirm selection metadata explicitly records test_data_used_for_selection == False."""
        with open(AO_BEST_JSON, "r", encoding="utf-8") as f:
            best_data = json.load(f)
        assert best_data["test_data_used_for_selection"] is False
        assert best_data["selection_stage"] == "validation"

        with open(AO_RESULTS_JSON, "r", encoding="utf-8") as f:
            results_data = json.load(f)
        assert results_data["selection"]["test_data_used_for_selection"] is False

    def test_composite_scores_match_validation_metrics_not_test_metrics(self):
        """Confirm composite scores used for selection match validation scores, not test scores."""
        with open(AO_RESULTS_JSON, "r", encoding="utf-8") as f:
            results_data = json.load(f)

        val_xgb = results_data["validation_results"]["xgboost"]
        rare_recalls = [
            val_xgb["per_class_metrics"][c]["recall"]
            for c in RARE_CLASSES
        ]
        mean_rare = float(np.mean(rare_recalls))
        m = val_xgb["metrics"]
        expected_val_comp = round(
            m["macro_f1"] * 0.40
            + m["macro_recall"] * 0.25
            + mean_rare * 0.25
            + m["balanced_accuracy"] * 0.10,
            4,
        )
        actual_comp = results_data["selection"]["composite_scores"]["xgboost"]
        assert abs(actual_comp - expected_val_comp) < 1e-3, (
            f"Composite score ({actual_comp}) does not match validation calculation ({expected_val_comp})"
        )

    def test_both_validation_and_final_test_metrics_present(self):
        """Confirm both validation and final test metrics are present in best_model and results JSON."""
        with open(AO_BEST_JSON, "r", encoding="utf-8") as f:
            best_data = json.load(f)
        assert "validation_metrics" in best_data
        assert "final_test_metrics" in best_data

        for mk in REQUIRED_METRICS:
            assert mk in best_data["validation_metrics"], f"Missing val metric: {mk}"
            assert mk in best_data["final_test_metrics"], f"Missing test metric: {mk}"

        with open(AO_RESULTS_JSON, "r", encoding="utf-8") as f:
            results_data = json.load(f)
        assert "validation_results" in results_data
        assert "final_test_evaluation" in results_data
        assert "metrics" in results_data["final_test_evaluation"]

        for mk in REQUIRED_METRICS:
            assert mk in results_data["final_test_evaluation"]["metrics"], f"Missing final test metric: {mk}"

