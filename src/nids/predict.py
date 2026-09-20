"""
predict.py — Single-record and batch inference.

Loads the trained binary and multiclass models (and the StandardScaler)
from models/ and produces structured PredictionResult objects.

Usage
-----
    from nids.predict import NIDSPredictor
    predictor = NIDSPredictor()
    result = predictor.predict_record({"srcip": "...", "proto": "tcp", ...})
    # or batch:
    results = predictor.predict_batch(df_rows)
"""

from __future__ import annotations

import logging
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from nids.config import (
    BINARY_CLF_PATH,
    LABEL_ENCODER_PATH,
    MULTI_CLF_PATH,
    PROCESSED_DIR,
    SCALER_PATH,
)
from nids.schemas import PredictionResult

logger = logging.getLogger(__name__)


class NIDSPredictor:
    """
    Wraps the trained binary + multiclass models for inference.

    Attributes
    ----------
    binary_clf   : fitted binary classifier
    multi_clf    : fitted multiclass classifier
    scaler       : fitted StandardScaler
    label_encoder: fitted LabelEncoder (attack categories)
    feature_names: list of feature column names (from preprocessing)
    """

    def __init__(self) -> None:
        self.binary_clf = self._load(BINARY_CLF_PATH, "binary classifier")
        self.multi_clf  = self._load(MULTI_CLF_PATH,  "multiclass classifier")
        self.scaler     = self._load(SCALER_PATH,      "scaler")
        self.label_encoder = (
            self._load(LABEL_ENCODER_PATH, "label encoder")
            if LABEL_ENCODER_PATH.exists()
            else None
        )
        fn_path = PROCESSED_DIR / "feature_names.csv"
        self.feature_names: list[str] = (
            pd.read_csv(fn_path, header=None)[0].tolist()
            if fn_path.exists()
            else []
        )

    @staticmethod
    def _load(path, name: str):
        if not path.exists():
            raise FileNotFoundError(
                f"{name} not found at {path}. Run `make train` first."
            )
        logger.debug("Loading %s from %s", name, path)
        return joblib.load(path)

    def _to_feature_row(self, record: dict) -> np.ndarray:
        """
        Convert a raw record dict to a scaled 1-D numpy array.

        TODO: replicate the exact OHE and imputation logic from preprocessing.py.
              Right now this is a stub that returns zeros for any unknown feature.
        """
        row = pd.Series(record)
        # Align to expected feature columns; fill missing with 0
        row = row.reindex(self.feature_names, fill_value=0)
        arr = row.values.astype(np.float32).reshape(1, -1)
        arr = self.scaler.transform(arr)
        return arr

    def predict_record(
        self,
        record: dict,
        record_index: Optional[int] = None,
        binary_threshold: float = 0.5,
    ) -> PredictionResult:
        """
        Run full inference on a single network flow record.

        Parameters
        ----------
        record           : dict mapping feature names to values
        record_index     : optional row index for bookkeeping
        binary_threshold : probability threshold for attack classification

        Returns
        -------
        PredictionResult
        """
        X = self._to_feature_row(record)

        # Binary decision
        proba_attack = float(self.binary_clf.predict_proba(X)[0, 1])
        is_attack = proba_attack >= binary_threshold

        # Multiclass decision (only if binary says attack)
        attack_category: Optional[str] = None
        multi_conf: Optional[float] = None
        if is_attack:
            multi_proba = self.multi_clf.predict_proba(X)[0]
            pred_idx = int(np.argmax(multi_proba))
            multi_conf = float(multi_proba[pred_idx])
            if self.label_encoder is not None:
                attack_category = str(self.label_encoder.classes_[pred_idx])
            else:
                attack_category = str(pred_idx)

        return PredictionResult(
            is_attack=is_attack,
            binary_confidence=proba_attack,
            attack_category=attack_category,
            multi_confidence=multi_conf,
            record_index=record_index,
        )

    def predict_batch(self, df: pd.DataFrame) -> list[PredictionResult]:
        """
        Run inference on every row of a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Each row is a network flow record.

        Returns
        -------
        list[PredictionResult]
        """
        results: list[PredictionResult] = []
        for idx, row in df.iterrows():
            result = self.predict_record(row.to_dict(), record_index=int(idx))
            results.append(result)
        return results
