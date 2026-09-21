"""
schemas.py — Pydantic data models for input validation and typed interfaces.

These schemas are used to validate:
  - Individual network flow records before inference.
  - Two-stage prediction events produced by NIDSPredictor.
  - Alert records polled by the dashboard.
  - Replay scenario records.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

try:
    from nids.config import RECOMMENDED_ACTIONS, SEVERITY_MAPPING
except ImportError:
    from src.nids.config import RECOMMENDED_ACTIONS, SEVERITY_MAPPING


# ─── Attack category enum ─────────────────────────────────────────────────────

class AttackCategory(IntEnum):
    """Numeric labels matching the UNSW-NB15 attack_cat column."""
    NORMAL = 0
    FUZZERS = 1
    ANALYSIS = 2
    BACKDOORS = 3
    DOS = 4
    EXPLOITS = 5
    GENERIC = 6
    RECONNAISSANCE = 7
    SHELLCODE = 8
    WORMS = 9


# ─── Raw network flow record ──────────────────────────────────────────────────

class NetworkFlowRecord(BaseModel):
    """
    Represents a single network flow record as it arrives from raw sources.
    Accepts connection identifiers and model features, allowing extra fields.
    """

    # Connection identifiers (optional in processed UNSW-NB15 flow arrays)
    srcip: Optional[str] = Field(None, description="Source IP address")
    sport: Optional[int] = Field(None, ge=0, le=65535, description="Source port")
    dstip: Optional[str] = Field(None, description="Destination IP address")
    dsport: Optional[int] = Field(None, ge=0, le=65535, description="Destination port")
    proto: str = Field(..., description="Transport-layer protocol (tcp/udp/…)")

    # Basic traffic statistics
    dur: float = Field(..., ge=0.0, description="Connection duration (seconds)")
    sbytes: int = Field(..., ge=0, description="Source-to-dest bytes")
    dbytes: int = Field(..., ge=0, description="Dest-to-source bytes")
    spkts: int = Field(..., ge=0, description="Source-to-dest packet count")
    dpkts: int = Field(..., ge=0, description="Dest-to-source packet count")

    # Ground truth (optional — present in test data, absent in live replay)
    label: Optional[int] = Field(None, ge=0, le=1)
    attack_cat: Optional[str] = None

    model_config = {"extra": "allow"}  # accept all 49 UNSW-NB15 columns

    @field_validator("proto")
    @classmethod
    def proto_lower(cls, v: str) -> str:
        return v.strip().lower()


# ─── Two-Stage Prediction Result ──────────────────────────────────────────────

class PredictionResult(BaseModel):
    """
    Structured output produced by nids.predict for a single network flow record.
    Implements the complete two-stage NIDS prediction event schema.

    Data-Isolation Guarantee
    ------------------------
    This schema NEVER contains ground-truth dataset fields.  The following
    names are not defined here and are rejected by ``extra='forbid'``:

        label, attack_cat, ground_truth_label, ground_truth_attack_cat,
        true_label, true_attack_category

    ``is_attack``    — derived solely from the binary-model decision and
                       the operational threshold; never from the dataset label.
    ``attack_category`` — a prediction-only backwards-compatible alias for
                          ``attack_type`` (the decoded multiclass model output).
                          It is NOT the dataset column ``attack_cat``.
    """

    # Event identification and timing
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique event identifier")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="ISO-8601 UTC timestamp")

    # Two-stage classification decisions
    prediction: str = Field(..., description="Binary classification decision ('Normal' or 'Attack')")
    attack_type: str = Field(..., description="Decoded multiclass model prediction ('Normal' or specific attack name)")

    # Probabilities and confidence metrics
    binary_probability: float = Field(..., ge=0.0, le=1.0, description="Stage-1 model probability for the attack class")
    binary_threshold: float = Field(..., ge=0.0, le=1.0, description="Operational binary decision threshold (loaded from metadata)")
    binary_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the binary decision")
    multiclass_probability: Optional[float] = Field(None, ge=0.0, le=1.0, description="Stage-2 model class probability (None when bypassed)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall decision confidence")

    # Operational triage
    severity: str = Field(..., description="Operational severity: None, Medium, High, Critical")
    recommended_action: str = Field(..., description="Actionable SOC mitigation instruction")

    # Provenance metadata
    model_version: Optional[str] = Field("1.0.0", description="Model pipeline version")
    record_index: Optional[int] = Field(None, description="Row index in source data")

    # Backwards-compatibility fields — all are PREDICTION-ONLY, never dataset ground truth
    is_attack: bool = Field(
        False,
        description=(
            "Prediction-only boolean: True when binary_probability >= binary_threshold. "
            "NOT the dataset 'label' column."
        ),
    )
    attack_category: Optional[str] = Field(
        None,
        description=(
            "Prediction-only alias for attack_type (decoded multiclass model output). "
            "NOT the dataset 'attack_cat' column."
        ),
    )
    multi_confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Backwards-compatible alias for multiclass_probability.",
    )

    # extra='forbid' ensures ground-truth field names (label, attack_cat, etc.)
    # are never silently accepted into this output schema.
    model_config = {"extra": "forbid"}

    @model_validator(mode="before")
    @classmethod
    def _harmonize_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        d = dict(data)

        # Harmonize is_attack and prediction
        is_attack = d.get("is_attack")
        prediction = d.get("prediction")
        if prediction is None and is_attack is not None:
            prediction = "Attack" if is_attack else "Normal"
            d["prediction"] = prediction
        elif prediction is not None and is_attack is None:
            is_attack = (prediction == "Attack")
            d["is_attack"] = is_attack
        elif is_attack is not None:
            d["is_attack"] = bool(is_attack)
        else:
            d["is_attack"] = (d.get("prediction") == "Attack")

        # Harmonize attack_type and attack_category
        attack_type = d.get("attack_type")
        attack_category = d.get("attack_category")
        if attack_type is None:
            attack_type = attack_category if attack_category is not None else ("Normal" if not d.get("is_attack") else "Attack")
            d["attack_type"] = attack_type
        if attack_category is None:
            d["attack_category"] = attack_type if d.get("is_attack") else None

        # Harmonize binary_probability and binary_confidence
        bin_prob = d.get("binary_probability")
        bin_conf = d.get("binary_confidence")
        if bin_prob is None and bin_conf is not None:
            d["binary_probability"] = bin_conf
            bin_prob = bin_conf
        if bin_conf is None and bin_prob is not None:
            bin_conf = bin_prob if d.get("is_attack") else round(1.0 - bin_prob, 4)
            d["binary_confidence"] = bin_conf

        # Harmonize multiclass_probability and multi_confidence
        multi_prob = d.get("multiclass_probability")
        multi_conf = d.get("multi_confidence")
        if multi_prob is None and multi_conf is not None:
            d["multiclass_probability"] = multi_conf
            multi_prob = multi_conf
        if multi_conf is None and multi_prob is not None:
            d["multi_confidence"] = multi_prob

        # Binary threshold
        if d.get("binary_threshold") is None:
            d["binary_threshold"] = 0.80

        # Confidence
        if d.get("confidence") is None:
            d["confidence"] = d.get("multiclass_probability") if d.get("is_attack") else d.get("binary_confidence")
            if d["confidence"] is None:
                d["confidence"] = d.get("binary_confidence", 0.0)

        # Severity
        if d.get("severity") is None:
            atk_key = d.get("attack_type", "Normal")
            d["severity"] = SEVERITY_MAPPING.get(atk_key, "None" if not d.get("is_attack") else "Critical")

        # Recommended action
        if d.get("recommended_action") is None:
            atk_key = d.get("attack_type", "Normal")
            d["recommended_action"] = RECOMMENDED_ACTIONS.get(atk_key, "Allow traffic / Normal operation")

        # Event ID & Timestamp
        if d.get("event_id") is None:
            d["event_id"] = str(uuid.uuid4())
        if d.get("timestamp") is None:
            d["timestamp"] = datetime.now(timezone.utc).isoformat()

        return d

    @field_validator(
        "binary_probability",
        "binary_confidence",
        "multiclass_probability",
        "multi_confidence",
        "confidence",
        mode="before",
    )
    @classmethod
    def round_probabilities(cls, v: Optional[float]) -> Optional[float]:
        return round(v, 4) if v is not None else None


# ─── Alert record (written to the dashboard) ─────────────────────────────────

class AlertRecord(BaseModel):
    """One row in the dashboard alert table."""

    timestamp: str
    src_ip: str
    dst_ip: str
    proto: str
    is_attack: bool
    attack_category: Optional[str]
    confidence: float
