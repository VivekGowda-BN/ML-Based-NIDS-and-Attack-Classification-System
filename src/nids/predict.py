"""
predict.py — Two-stage production-ready inference service for ML-Based NIDS.

Workflow
--------
1. Accepts a single raw network-flow record as a dictionary or one-row DataFrame.
2. Removes or ignores ground-truth labels (id, label, attack_cat) and connection metadata.
3. Validates required schema features and numeric types; rejects malformed inputs with clear errors.
4. Preprocesses the record using the fitted unscaled pipeline (handles unseen categorical values).
5. Evaluates Stage 1 (Binary XGBoost model) against operational threshold (0.80 loaded from metadata).
6. If binary probability < 0.80:
   - Returns prediction="Normal", attack_type="Normal", severity="None".
   - Stage 2 (Multiclass model) is strictly bypassed.
7. If binary probability >= 0.80:
   - Returns prediction="Attack".
   - Executes Stage 2 (Multiclass model) and decodes attack category via label encoder.
8. Returns a strongly typed PredictionResult containing event_id, timestamp, prediction, attack_type,
   binary_probability, binary_threshold, binary_confidence, multiclass_probability, confidence,
   severity, recommended_action, and metadata.
9. Ground truth fields are strictly excluded from live prediction events.

CLI Usage
---------
    python -m src.nids.predict --smoke-test --limit 10
"""

from __future__ import annotations

import argparse
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

try:
    from nids.config import (
        BINARY_THRESHOLD_SELECTED_MODEL_PATH,
        BINARY_THRESHOLD_SELECTION_META_JSON,
        CLEAN_TEST_PARQUET,
        FEATURE_NAMES_JSON,
        LABEL_ENCODER_PATH,
        MULTICLASS_BEST_MODEL_PATH,
        NON_FEATURE_COLUMNS,
        PREPROCESSOR_UNSCALED_PATH,
        RECOMMENDED_ACTIONS,
        SEVERITY_MAPPING,
    )
    from nids.schemas import PredictionResult
except ImportError:
    from src.nids.config import (
        BINARY_THRESHOLD_SELECTED_MODEL_PATH,
        BINARY_THRESHOLD_SELECTION_META_JSON,
        CLEAN_TEST_PARQUET,
        FEATURE_NAMES_JSON,
        LABEL_ENCODER_PATH,
        MULTICLASS_BEST_MODEL_PATH,
        NON_FEATURE_COLUMNS,
        PREPROCESSOR_UNSCALED_PATH,
        RECOMMENDED_ACTIONS,
        SEVERITY_MAPPING,
    )
    from src.nids.schemas import PredictionResult

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD: float = 0.80
CATEGORICAL_FEATURES: tuple[str, ...] = ("proto", "service", "state")


class NIDSPredictor:
    """
    Two-stage Network Intrusion Detection prediction service.

    Stage 1: Binary classification (Normal vs. Attack) with operational threshold loaded from metadata.
    Stage 2: Multiclass classification executed only when Stage 1 flags an Attack.
    """

    def __init__(
        self,
        binary_model_path: Optional[Union[str, Path]] = None,
        multiclass_model_path: Optional[Union[str, Path]] = None,
        preprocessor_path: Optional[Union[str, Path]] = None,
        label_encoder_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
        feature_names_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.binary_model_path = Path(binary_model_path or BINARY_THRESHOLD_SELECTED_MODEL_PATH)
        self.multiclass_model_path = Path(multiclass_model_path or MULTICLASS_BEST_MODEL_PATH)
        self.preprocessor_path = Path(preprocessor_path or PREPROCESSOR_UNSCALED_PATH)
        self.label_encoder_path = Path(label_encoder_path or LABEL_ENCODER_PATH)
        self.metadata_path = Path(metadata_path or BINARY_THRESHOLD_SELECTION_META_JSON)
        self.feature_names_path = Path(feature_names_path or FEATURE_NAMES_JSON)

        # 1. Load operational binary threshold from metadata
        self.binary_threshold = self._load_threshold_from_metadata(self.metadata_path)

        # 2. Load trained model artifacts
        self.binary_clf = self._load_artifact(self.binary_model_path, "Binary XGBoost model")
        self.multi_clf = self._load_artifact(self.multiclass_model_path, "Multiclass model")
        self.preprocessor = self._load_artifact(self.preprocessor_path, "Unscaled preprocessor")
        self.label_encoder = self._load_artifact(self.label_encoder_path, "Label encoder")

        # 3. Determine expected input feature schema
        if hasattr(self.preprocessor, "feature_names_in_"):
            self.required_features: List[str] = list(self.preprocessor.feature_names_in_)
        else:
            self.required_features = [
                "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl", "sload",
                "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit", "swin", "stcpb",
                "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean", "dmean", "trans_depth",
                "response_body_len", "ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm",
                "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd",
                "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
                "proto", "service", "state",
            ]

        self.numeric_features: List[str] = [
            c for c in self.required_features if c not in CATEGORICAL_FEATURES
        ]

        logger.info(
            "NIDSPredictor initialised: Binary Threshold=%.2f, Required Features=%d, Multiclass Classes=%d",
            self.binary_threshold,
            len(self.required_features),
            len(self.label_encoder.classes_),
        )

    @staticmethod
    def _load_threshold_from_metadata(meta_path: Path) -> float:
        """Load selected binary threshold dynamically from selection metadata."""
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                thresh = float(meta["selected_threshold"])
                logger.info("Loaded binary decision threshold %.2f from %s", thresh, meta_path)
                return thresh
            except Exception as exc:
                logger.warning("Failed reading threshold from %s: %s. Using default %.2f", meta_path, exc, DEFAULT_THRESHOLD)
                return DEFAULT_THRESHOLD
        logger.warning("Metadata file %s not found. Using default threshold %.2f", meta_path, DEFAULT_THRESHOLD)
        return DEFAULT_THRESHOLD

    @staticmethod
    def _load_artifact(path: Path, name: str) -> Any:
        """Load a serialised artifact via joblib."""
        if not path.exists():
            raise FileNotFoundError(f"{name} artifact not found at: {path}")
        logger.debug("Loading %s from %s", name, path)
        return joblib.load(path)

    def validate_record(self, record: Union[Dict[str, Any], pd.DataFrame, pd.Series]) -> Dict[str, Any]:
        """
        Validate and sanitize a single input record.

        - Rejects non-dictionary / non-DataFrame inputs.
        - Verifies that all 42 required model features are present.
        - Verifies that numeric features contain numeric or numeric-castable values.
        - Strips ground-truth label columns (id, label, attack_cat).
        """
        if isinstance(record, pd.DataFrame):
            if len(record) != 1:
                raise ValueError(f"Expected single-row DataFrame, got DataFrame with {len(record)} rows.")
            rec = record.iloc[0].to_dict()
        elif isinstance(record, pd.Series):
            rec = record.to_dict()
        elif isinstance(record, dict):
            rec = dict(record)
        else:
            raise ValueError(f"Invalid input type: Expected dict or single-row DataFrame, got {type(record).__name__}.")

        if not rec:
            raise ValueError("Invalid input: empty record provided.")

        # Check for missing required features
        missing = [f for f in self.required_features if f not in rec]
        if missing:
            raise ValueError(f"Input record is missing {len(missing)} required model feature(s): {missing}")

        # Validate numeric features
        for col in self.numeric_features:
            val = rec[col]
            if val is None or val == "" or pd.isna(val):
                continue
            try:
                float(val)
            except (ValueError, TypeError):
                raise ValueError(f"Feature '{col}' must be numeric, got invalid value: {val!r}")

        # Sanitize by stripping ground truth columns
        clean_rec = {
            col: rec[col] for col in self.required_features
        }
        return clean_rec

    def predict_record(
        self,
        record: Union[Dict[str, Any], pd.DataFrame, pd.Series],
        record_index: Optional[int] = None,
        binary_threshold: Optional[float] = None,
    ) -> PredictionResult:
        """
        Execute two-stage prediction on a single network flow record.

        Parameters
        ----------
        record : dict or single-row DataFrame
            Raw network flow record containing the 42 model features.
        record_index : int, optional
            Index of the record for bookkeeping.
        binary_threshold : float, optional
            Override for the operational binary decision threshold.

        Returns
        -------
        PredictionResult
            Strongly typed prediction event containing classification, probabilities,
            severity, and recommended action.
        """
        # Step 1: Validate input and ignore ground-truth columns
        feat_dict = self.validate_record(record)

        # Step 2: Construct single-row DataFrame in exact feature order
        df_row = pd.DataFrame([{col: feat_dict[col] for col in self.required_features}])

        # Step 3: Preprocess unscaled features (handles unseen categories gracefully)
        X_unscaled = self.preprocessor.transform(df_row)

        # Step 4: Stage 1 — Binary Classification
        threshold = self.binary_threshold if binary_threshold is None else float(binary_threshold)
        binary_proba = float(self.binary_clf.predict_proba(X_unscaled)[0, 1])
        is_attack = binary_proba >= threshold

        # Step 5: Stage 2 — Multiclass Classification (conditioned on Stage 1 decision)
        if not is_attack:
            prediction = "Normal"
            attack_type = "Normal"
            severity = "None"
            binary_confidence = round(1.0 - binary_proba, 4)
            multiclass_probability = None
            confidence = binary_confidence
            recommended_action = RECOMMENDED_ACTIONS["Normal"]
        else:
            prediction = "Attack"
            multi_proba = self.multi_clf.predict_proba(X_unscaled)[0]
            pred_idx = int(np.argmax(multi_proba))
            pred_cat = str(self.label_encoder.classes_[pred_idx])

            # If multiclass argmax is Normal despite binary flagging Attack, select top attack class
            if pred_cat == "Normal":
                atk_indices = [i for i, c in enumerate(self.label_encoder.classes_) if c != "Normal"]
                atk_probs = multi_proba[atk_indices]
                best_sub_idx = int(np.argmax(atk_probs))
                pred_idx = atk_indices[best_sub_idx]
                pred_cat = str(self.label_encoder.classes_[pred_idx])

            attack_type = pred_cat
            multiclass_probability = round(float(multi_proba[pred_idx]), 4)
            binary_confidence = round(binary_proba, 4)
            confidence = multiclass_probability
            severity = SEVERITY_MAPPING.get(attack_type, "Critical")
            recommended_action = RECOMMENDED_ACTIONS.get(
                attack_type,
                "CRITICAL ALERT: Isolate affected host and alert SOC security operations."
            )

        # Step 6: Construct structured prediction event
        event = PredictionResult(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            prediction=prediction,
            attack_type=attack_type,
            binary_probability=round(binary_proba, 4),
            binary_threshold=round(threshold, 2),
            binary_confidence=binary_confidence,
            multiclass_probability=multiclass_probability,
            confidence=confidence,
            severity=severity,
            recommended_action=recommended_action,
            model_version="1.0.0",
            record_index=record_index,
        )
        return event

    def predict(
        self,
        record: Union[Dict[str, Any], pd.DataFrame, pd.Series],
        record_index: Optional[int] = None,
        binary_threshold: Optional[float] = None,
    ) -> PredictionResult:
        """Alias for predict_record."""
        return self.predict_record(record, record_index=record_index, binary_threshold=binary_threshold)

    def predict_batch(
        self,
        df: pd.DataFrame,
        binary_threshold: Optional[float] = None,
    ) -> List[PredictionResult]:
        """
        Run two-stage inference on every row of a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame of network flow records.
        binary_threshold : float, optional
            Override for the binary decision threshold.

        Returns
        -------
        list[PredictionResult]
        """
        results: List[PredictionResult] = []
        for idx, row in df.iterrows():
            rec_idx = int(idx) if isinstance(idx, (int, np.integer)) else None
            event = self.predict_record(row.to_dict(), record_index=rec_idx, binary_threshold=binary_threshold)
            results.append(event)
        return results

    def get_metadata(self) -> Dict[str, Any]:
        """Return comprehensive metadata about models, thresholds, and feature schemas."""
        return {
            "model_version": "1.0.0",
            "binary_model_artifact": str(self.binary_model_path),
            "multiclass_model_artifact": str(self.multiclass_model_path),
            "preprocessor_artifact": str(self.preprocessor_path),
            "label_encoder_artifact": str(self.label_encoder_path),
            "binary_threshold": self.binary_threshold,
            "threshold_metadata_source": str(self.metadata_path),
            "required_features_count": len(self.required_features),
            "required_features": self.required_features,
            "classes": list(self.label_encoder.classes_),
            "severity_mapping": SEVERITY_MAPPING,
        }


# ─── Smoke-Test CLI ───────────────────────────────────────────────────────────

def run_smoke_test(limit: int = 10) -> None:
    """
    Run an offline smoke test using a small sample of held-out test records.

    Safety:
    - Reads solely from local processed/interim files.
    - Never generates network packets or connects to external systems.
    """
    print("\n" + "=" * 115)
    print("NIDS TWO-STAGE PREDICTION SERVICE — OFFLINE SMOKE TEST")
    print("=" * 115)
    print(f"Test data source : {CLEAN_TEST_PARQUET}")
    print(f"Sample limit     : {limit} records\n")

    if not CLEAN_TEST_PARQUET.exists():
        raise FileNotFoundError(f"Clean test dataset not found at {CLEAN_TEST_PARQUET}")

    predictor = NIDSPredictor()
    meta = predictor.get_metadata()
    print(f"Loaded Binary Model     : {meta['binary_model_artifact']}")
    print(f"Loaded Multiclass Model : {meta['multiclass_model_artifact']}")
    print(f"Loaded Preprocessor     : {meta['preprocessor_artifact']}")
    print(f"Loaded Threshold        : {meta['binary_threshold']:.2f} (from {meta['threshold_metadata_source']})")
    print(f"Required Features Count : {meta['required_features_count']}")
    print("-" * 115)

    df_test = pd.read_parquet(CLEAN_TEST_PARQUET)
    sample_df = df_test.head(limit)

    print(f"{'Idx':>3} | {'Prediction':<8} | {'Attack Type':<16} | {'Bin Prob':>8} | {'Thresh':>6} | {'Multi Prob':>10} | {'Severity':<8} | {'Recommended Action':<36}")
    print("-" * 115)

    events: List[PredictionResult] = []
    for idx, row in sample_df.iterrows():
        rec_dict = row.to_dict()
        event = predictor.predict_record(rec_dict, record_index=int(idx))
        events.append(event)

        m_prob_str = f"{event.multiclass_probability:.4f}" if event.multiclass_probability is not None else "N/A"
        print(
            f"{int(idx):3d} | "
            f"{event.prediction:<8} | "
            f"{event.attack_type:<16} | "
            f"{event.binary_probability:8.4f} | "
            f"{event.binary_threshold:6.2f} | "
            f"{m_prob_str:>10} | "
            f"{event.severity:<8} | "
            f"{event.recommended_action[:36]:<36}"
        )

    print("-" * 115)
    print("SAMPLE PREDICTION EVENT (JSON Serialized):")
    print(json.dumps(events[0].model_dump(), indent=2))
    print("=" * 115 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="NIDS Two-Stage Prediction Service")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Execute offline smoke-test on a sample of held-out test records.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of records to test during smoke-test (default: 10).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.smoke_test:
        run_smoke_test(limit=args.limit)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
