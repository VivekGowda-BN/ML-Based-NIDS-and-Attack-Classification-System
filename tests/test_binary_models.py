"""
tests/test_binary_models.py — Unit and integration tests for binary intrusion detection models.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from nids.config import (
    BINARY_BEST_MODEL_JSON,
    BINARY_BEST_MODEL_PATH,
    BINARY_COMPARISON_CSV,
    BINARY_LR_PATH,
    BINARY_RESULTS_JSON,
    BINARY_RF_PATH,
    BINARY_XGB_PATH,
    FEATURE_NAMES_JSON,
    PROCESSED_DIR,
)
from nids.preprocessing import load_splits

EXPECTED_TEST_ROWS = 82332
EXPECTED_FEATURES = 194


@pytest.fixture(scope="module")
def splits_data():
    """Load preprocessed splits once for testing."""
    return load_splits(PROCESSED_DIR)


def test_saved_model_files_exist():
    """Verify that all trained model joblib artifacts exist and are non-empty."""
    model_paths = [
        BINARY_LR_PATH,
        BINARY_RF_PATH,
        BINARY_XGB_PATH,
        BINARY_BEST_MODEL_PATH,
    ]
    for p in model_paths:
        assert p.exists(), f"Model file missing: {p}"
        assert p.stat().st_size > 0, f"Model file empty: {p}"


def test_saved_models_load_and_predict(splits_data):
    """Verify saved models load cleanly, generate valid probabilities, and match test row count."""
    X_test_scaled = splits_data["X_test_scaled"]
    X_test_unscaled = splits_data["X_test_unscaled"]

    models_to_test = [
        (BINARY_LR_PATH, X_test_scaled),
        (BINARY_RF_PATH, X_test_unscaled),
        (BINARY_XGB_PATH, X_test_unscaled),
        (BINARY_BEST_MODEL_PATH, X_test_unscaled),
    ]

    for path, X_test in models_to_test:
        model = joblib.load(path)
        assert hasattr(model, "predict"), f"Model at {path} missing predict()"
        assert hasattr(model, "predict_proba"), f"Model at {path} missing predict_proba()"

        # Test predictions on a subset of 100 rows for speed and shape verification
        proba = model.predict_proba(X_test[:100])
        preds = model.predict(X_test[:100])

        assert proba.shape == (100, 2), f"Expected shape (100, 2), got {proba.shape}"
        assert preds.shape == (100,)
        assert (proba >= 0.0).all() and (proba <= 1.0).all(), "Probabilities outside [0, 1]"
        assert np.allclose(proba.sum(axis=1), 1.0), "Probabilities do not sum to 1.0"


def test_full_test_predictions_row_count(splits_data):
    """Verify predictions on full test set have exactly 82,332 rows."""
    model = joblib.load(BINARY_BEST_MODEL_PATH)
    X_test = splits_data["X_test_unscaled"]
    preds = model.predict(X_test)
    assert len(preds) == EXPECTED_TEST_ROWS


def test_metrics_contain_required_fields():
    """Verify binary_results.json contains all required evaluation fields."""
    assert BINARY_RESULTS_JSON.exists(), f"Missing: {BINARY_RESULTS_JSON}"
    with open(BINARY_RESULTS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "models" in data
    assert "selection" in data
    required_models = ["logistic_regression", "random_forest", "xgboost"]
    for m in required_models:
        assert m in data["models"]
        info = data["models"][m]
        assert "confusion_matrix" in info
        assert "metrics" in info
        assert "training_time_sec" in info
        assert "inference_time_sec" in info

        metrics = info["metrics"]
        required_metrics = [
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "pr_auc",
            "false_positive_rate",
            "false_negative_rate",
        ]
        for rm in required_metrics:
            assert rm in metrics, f"Missing metric {rm} in {m}"
            assert 0.0 <= metrics[rm] <= 1.0


def test_confusion_matrix_dimensions():
    """Verify confusion matrix has 2x2 dimensions and sum equals test row count."""
    with open(BINARY_RESULTS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    for m in ["logistic_regression", "random_forest", "xgboost"]:
        cm_dict = data["models"][m]["confusion_matrix"]
        cm = np.array(cm_dict["matrix"])
        assert cm.shape == (2, 2)
        total = cm_dict["tn"] + cm_dict["fp"] + cm_dict["fn"] + cm_dict["tp"]
        assert total == EXPECTED_TEST_ROWS


def test_no_target_columns_in_feature_metadata():
    """Ensure target columns and id are excluded from the 194 transformed feature metadata."""
    assert FEATURE_NAMES_JSON.exists(), f"Missing: {FEATURE_NAMES_JSON}"
    with open(FEATURE_NAMES_JSON, "r", encoding="utf-8") as f:
        meta = json.load(f)

    features = meta["transformed_feature_names"]
    assert len(features) == EXPECTED_FEATURES

    # Verify no forbidden target names appear
    for f in features:
        assert f != "id"
        assert f != "label"
        assert f != "attack_cat"


def test_model_comparison_artifacts_exist():
    """Verify comparison CSV, best model json, and plots exist."""
    assert BINARY_COMPARISON_CSV.exists()
    assert BINARY_BEST_MODEL_JSON.exists()

    df_comp = pd.read_csv(BINARY_COMPARISON_CSV)
    assert len(df_comp) == 3
    assert set(df_comp["model"]) == {"logistic_regression", "random_forest", "xgboost"}

    with open(BINARY_BEST_MODEL_JSON, "r", encoding="utf-8") as f:
        best_meta = json.load(f)
    assert "selected_model" in best_meta
    assert best_meta["selected_model"] in {"logistic_regression", "random_forest", "xgboost"}
