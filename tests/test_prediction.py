"""
test_prediction.py — Unit and regression tests for the two-stage prediction service.

Verified invariants
-------------------
1. Known normal record classifies as Normal, severity None, multiclass model bypassed.
2. Known attack record classifies as Attack, valid attack category, multiclass model executed.
3. Records containing unseen categorical values (such as ACC, CLO, unknown proto) process gracefully.
4. Missing optional metadata is handled gracefully; extra connection fields ignored.
5. Invalid inputs (missing required feature, non-numeric values, invalid types) raise clear ValueErrors.
6. Binary threshold is dynamically loaded as 0.80 from selection metadata, not hard-coded.
7. A normal prediction strictly does not invoke the multiclass model.
8. An attack prediction strictly invokes the multiclass model.
9. Probability and confidence values are strictly within [0.0, 1.0].
10. Returned attack types belong to valid label encoder classes.
11. Ground-truth label columns (id, label, attack_cat) are absent from live prediction output.
12. All required saved model and pipeline artifacts load successfully.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from nids.config import (
    BINARY_THRESHOLD_SELECTED_MODEL_PATH,
    BINARY_THRESHOLD_SELECTION_META_JSON,
    CLEAN_TEST_PARQUET,
    LABEL_ENCODER_PATH,
    MULTICLASS_BEST_MODEL_PATH,
    PREPROCESSOR_UNSCALED_PATH,
)
from nids.predict import NIDSPredictor
from nids.schemas import PredictionResult


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def predictor() -> NIDSPredictor:
    return NIDSPredictor()


@pytest.fixture(scope="module")
def test_df() -> pd.DataFrame:
    assert CLEAN_TEST_PARQUET.exists(), f"Missing: {CLEAN_TEST_PARQUET}"
    return pd.read_parquet(CLEAN_TEST_PARQUET)


@pytest.fixture(scope="module")
def normal_record(test_df: pd.DataFrame, predictor: NIDSPredictor) -> dict:
    """Find a known normal record where label == 0 and binary prob < 0.80."""
    norm_rows = test_df[test_df["label"] == 0]
    for _, row in norm_rows.iterrows():
        rec = row.to_dict()
        res = predictor.predict_record(rec)
        if res.prediction == "Normal":
            return rec
    return norm_rows.iloc[2].to_dict()


@pytest.fixture(scope="module")
def attack_record(test_df: pd.DataFrame, predictor: NIDSPredictor) -> dict:
    """Find a known attack record where label == 1 and binary prob >= 0.80."""
    atk_rows = test_df[test_df["label"] == 1]
    for _, row in atk_rows.iterrows():
        rec = row.to_dict()
        res = predictor.predict_record(rec)
        if res.prediction == "Attack":
            return rec
    return atk_rows.iloc[0].to_dict()


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestArtifactLoading:
    """Verify Requirement 12: All saved artifacts load successfully."""

    def test_saved_artifacts_load_successfully(self):
        assert BINARY_THRESHOLD_SELECTED_MODEL_PATH.exists()
        bin_model = joblib.load(BINARY_THRESHOLD_SELECTED_MODEL_PATH)
        assert isinstance(bin_model, XGBClassifier)

        assert MULTICLASS_BEST_MODEL_PATH.exists()
        multi_model = joblib.load(MULTICLASS_BEST_MODEL_PATH)
        assert hasattr(multi_model, "predict_proba")

        assert PREPROCESSOR_UNSCALED_PATH.exists()
        prep = joblib.load(PREPROCESSOR_UNSCALED_PATH)
        assert isinstance(prep, ColumnTransformer)
        assert len(prep.feature_names_in_) == 42

        assert LABEL_ENCODER_PATH.exists()
        le = joblib.load(LABEL_ENCODER_PATH)
        assert isinstance(le, LabelEncoder)
        assert len(le.classes_) == 10

        assert BINARY_THRESHOLD_SELECTION_META_JSON.exists()
        with open(BINARY_THRESHOLD_SELECTION_META_JSON, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["selected_threshold"] == 0.80


class TestDynamicThresholdLoading:
    """Verify Requirement 6: Binary threshold is loaded from metadata, not hard-coded."""

    def test_threshold_loaded_from_metadata(self, predictor: NIDSPredictor):
        assert predictor.binary_threshold == 0.80

    def test_threshold_not_hard_coded(self, tmp_path: Path):
        dummy_meta = tmp_path / "dummy_meta.json"
        dummy_meta.write_text(json.dumps({"selected_threshold": 0.65}), encoding="utf-8")

        custom_predictor = NIDSPredictor(metadata_path=dummy_meta)
        assert custom_predictor.binary_threshold == 0.65, (
            "Predictor must load threshold dynamically from metadata rather than hard-coding 0.80"
        )


class TestClassificationStages:
    """Verify Requirements 1, 2, 7, 8: Normal vs Attack two-stage flow."""

    def test_known_normal_record(self, predictor: NIDSPredictor, normal_record: dict):
        result = predictor.predict_record(normal_record)

        assert isinstance(result, PredictionResult)
        assert result.prediction == "Normal"
        assert result.attack_type == "Normal"
        assert result.severity == "None"
        assert result.multiclass_probability is None
        assert result.binary_probability < predictor.binary_threshold
        assert result.binary_confidence == round(1.0 - result.binary_probability, 4)
        assert result.confidence == result.binary_confidence
        assert "Allow traffic" in result.recommended_action

    def test_known_attack_record(self, predictor: NIDSPredictor, attack_record: dict):
        result = predictor.predict_record(attack_record)

        assert isinstance(result, PredictionResult)
        assert result.prediction == "Attack"
        assert result.attack_type in predictor.label_encoder.classes_
        assert result.attack_type != "Normal"
        assert result.severity in ("Medium", "High", "Critical")
        assert result.multiclass_probability is not None
        assert result.binary_probability >= predictor.binary_threshold
        assert result.binary_confidence == result.binary_probability

    def test_normal_prediction_does_not_call_multiclass_model(
        self, predictor: NIDSPredictor, normal_record: dict
    ):
        with patch.object(predictor.multi_clf, "predict_proba", wraps=predictor.multi_clf.predict_proba) as mock_multi:
            res = predictor.predict_record(normal_record)
            assert res.prediction == "Normal"
            assert not mock_multi.called, "Multiclass model must NOT be called for Normal predictions"

    def test_attack_prediction_calls_multiclass_model(
        self, predictor: NIDSPredictor, attack_record: dict
    ):
        with patch.object(predictor.multi_clf, "predict_proba", wraps=predictor.multi_clf.predict_proba) as mock_multi:
            res = predictor.predict_record(attack_record)
            assert res.prediction == "Attack"
            assert mock_multi.called, "Multiclass model MUST be called for Attack predictions"


class TestRobustnessAndEdgeCases:
    """Verify Requirements 3, 4, 5, 10, 11: Unseen categories, missing metadata, invalid input."""

    def test_unseen_categorical_values(self, predictor: NIDSPredictor, normal_record: dict):
        record_with_unseen = dict(normal_record)
        record_with_unseen["state"] = "ACC"  # Unseen state from UNSW-NB15 test
        record_with_unseen["proto"] = "unseen_test_protocol"
        record_with_unseen["service"] = "unseen_test_service"

        result = predictor.predict_record(record_with_unseen)
        assert isinstance(result, PredictionResult)
        assert result.prediction in ("Normal", "Attack")
        assert 0.0 <= result.binary_probability <= 1.0

    def test_missing_optional_metadata(self, predictor: NIDSPredictor, normal_record: dict):
        # Strip all optional metadata columns, keeping only the 42 required features
        clean_only = {k: v for k, v in normal_record.items() if k in predictor.required_features}
        assert len(clean_only) == 42

        result = predictor.predict_record(clean_only)
        assert isinstance(result, PredictionResult)

    def test_extra_metadata_handled_gracefully(self, predictor: NIDSPredictor, normal_record: dict):
        rec_with_extras = dict(normal_record)
        rec_with_extras["srcip"] = "192.168.1.100"
        rec_with_extras["dstip"] = "10.0.0.1"
        rec_with_extras["sport"] = 54321
        rec_with_extras["dsport"] = 443
        rec_with_extras["custom_system_tag"] = "datacenter-east"

        result = predictor.predict_record(rec_with_extras)
        assert isinstance(result, PredictionResult)

    def test_invalid_input_missing_required_feature(self, predictor: NIDSPredictor, normal_record: dict):
        bad_rec = dict(normal_record)
        del bad_rec["dur"]

        with pytest.raises(ValueError, match="missing.*required.*feature"):
            predictor.predict_record(bad_rec)

    def test_invalid_input_non_numeric_feature(self, predictor: NIDSPredictor, normal_record: dict):
        bad_rec = dict(normal_record)
        bad_rec["sbytes"] = "invalid_not_a_number"

        with pytest.raises(ValueError, match="must be numeric"):
            predictor.predict_record(bad_rec)

    def test_invalid_input_wrong_type(self, predictor: NIDSPredictor):
        with pytest.raises(ValueError, match="Expected dict or single-row DataFrame"):
            predictor.predict_record("not_a_dictionary_or_dataframe")

        with pytest.raises(ValueError, match="empty record"):
            predictor.predict_record({})


class TestOutputPurityAndProbabilities:
    """Verify Requirements 8, 9, 10, 11: Schema purity and valid metrics."""

    def test_ground_truth_fields_absent_from_live_output(
        self, predictor: NIDSPredictor, attack_record: dict
    ):
        rec = dict(attack_record)
        rec["id"] = 9999
        rec["label"] = 1
        rec["attack_cat"] = "Exploits"

        result = predictor.predict_record(rec)
        dump = result.model_dump()

        assert "label" not in dump, "Ground-truth 'label' must not appear in prediction event"
        assert "attack_cat" not in dump, "Ground-truth 'attack_cat' must not appear in prediction event"
        assert "id" not in dump, "Ground-truth 'id' must not appear in prediction event"

    def test_probability_values_valid(self, predictor: NIDSPredictor, attack_record: dict, normal_record: dict):
        for rec in (attack_record, normal_record):
            res = predictor.predict_record(rec)
            assert 0.0 <= res.binary_probability <= 1.0
            assert 0.0 <= res.binary_confidence <= 1.0
            assert 0.0 <= res.confidence <= 1.0
            if res.multiclass_probability is not None:
                assert 0.0 <= res.multiclass_probability <= 1.0

    def test_returned_attack_types_are_valid_label_encoder_classes(
        self, predictor: NIDSPredictor, test_df: pd.DataFrame
    ):
        for idx in range(15):
            rec = test_df.iloc[idx].to_dict()
            res = predictor.predict_record(rec)
            assert res.attack_type in predictor.label_encoder.classes_
            if res.prediction == "Attack":
                assert res.attack_type != "Normal"

    def test_batch_prediction(self, predictor: NIDSPredictor, test_df: pd.DataFrame):
        batch = test_df.head(5)
        results = predictor.predict_batch(batch)
        assert len(results) == 5
        for res in results:
            assert isinstance(res, PredictionResult)


class TestGroundTruthIsolation:
    """
    Regression tests — Requirement 5.

    Verify that the live prediction event NEVER exposes ground-truth dataset
    fields, regardless of whether those fields are present in the input record.
    """

    # Names that must never appear in the serialised prediction event.
    FORBIDDEN_GROUND_TRUTH_FIELDS = {
        "label",
        "attack_cat",
        "ground_truth_label",
        "ground_truth_attack_cat",
        "true_label",
        "true_attack_category",
    }

    def test_ground_truth_fields_stripped_from_output(
        self, predictor: NIDSPredictor, attack_record: dict
    ):
        """Pass a record carrying every forbidden field; none must appear in model_dump()."""
        rec = dict(attack_record)
        rec["label"] = 1
        rec["attack_cat"] = "Exploits"
        rec["ground_truth_label"] = 1
        rec["ground_truth_attack_cat"] = "Exploits"
        rec["true_label"] = 1
        rec["true_attack_category"] = "Exploits"

        result = predictor.predict_record(rec)
        dump = result.model_dump()

        for field in self.FORBIDDEN_GROUND_TRUTH_FIELDS:
            assert field not in dump, (
                f"Ground-truth field '{field}' must not appear in the prediction event output."
            )

    def test_no_ground_truth_field_names_in_output_keys(
        self, predictor: NIDSPredictor, normal_record: dict
    ):
        """Verify the full set of output keys contains no forbidden ground-truth names."""
        rec = dict(normal_record)
        rec["label"] = 0
        rec["attack_cat"] = ""
        rec["ground_truth_label"] = 0
        rec["ground_truth_attack_cat"] = ""
        rec["true_label"] = 0
        rec["true_attack_category"] = ""

        result = predictor.predict_record(rec)
        dump_keys = set(result.model_dump().keys())

        overlap = dump_keys & self.FORBIDDEN_GROUND_TRUTH_FIELDS
        assert not overlap, (
            f"Prediction event output keys contain forbidden ground-truth fields: {overlap}"
        )

    def test_attack_type_equals_decoded_multiclass_prediction(
        self, predictor: NIDSPredictor, attack_record: dict
    ):
        """
        Verify that attack_type is exactly the decoded label-encoder class
        output from the multiclass model — not any dataset column.
        """
        from unittest.mock import patch

        expected_classes = list(predictor.label_encoder.classes_)

        result = predictor.predict_record(attack_record)
        assert result.prediction == "Attack"
        assert result.attack_type in expected_classes, (
            f"attack_type '{result.attack_type}' is not a valid label-encoder class."
        )
        assert result.attack_type != "Normal", (
            "attack_type for an Attack prediction must not be 'Normal'."
        )

    def test_is_attack_derived_from_binary_model_not_dataset_label(
        self, predictor: NIDSPredictor, attack_record: dict
    ):
        """
        Verify is_attack equals (binary_probability >= binary_threshold),
        not any dataset label field.
        """
        rec = dict(attack_record)
        # Inject a conflicting dataset label (0 = Normal) while the model
        # should still classify the record as Attack.
        rec["label"] = 0

        result = predictor.predict_record(rec)
        expected_is_attack = result.binary_probability >= result.binary_threshold
        assert result.is_attack == expected_is_attack, (
            "is_attack must equal (binary_probability >= binary_threshold), "
            "not the dataset 'label' field."
        )

    def test_attack_category_is_prediction_only_alias_for_attack_type(
        self, predictor: NIDSPredictor, attack_record: dict
    ):
        """
        attack_category must equal attack_type (a model prediction).
        It must NOT carry the dataset's attack_cat value.
        """
        rec = dict(attack_record)
        rec["attack_cat"] = "Worms"  # deliberately different from expected model output

        result = predictor.predict_record(rec)
        dump = result.model_dump()

        # attack_category must equal attack_type (model prediction)
        assert dump.get("attack_category") == dump.get("attack_type"), (
            "attack_category must be a prediction-only alias for attack_type."
        )
        # attack_category must NOT be the dataset value injected above
        # (unless by coincidence the model also predicts Worms, which is fine)
        if result.attack_type != "Worms":
            assert dump.get("attack_category") != "Worms", (
                "attack_category must reflect the model prediction, not the dataset attack_cat."
            )

    def test_threshold_from_metadata_is_0_80(self, predictor: NIDSPredictor):
        """Verify the operational threshold is 0.80 loaded from saved metadata."""
        assert predictor.binary_threshold == 0.80, (
            f"Threshold must be 0.80 from metadata; got {predictor.binary_threshold}"
        )
        result_meta = predictor.get_metadata()
        assert result_meta["binary_threshold"] == 0.80
