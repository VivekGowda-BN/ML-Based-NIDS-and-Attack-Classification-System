"""
schemas.py — Pydantic data models for input validation and typed interfaces.

These schemas are used to validate:
  - Individual network flow records before inference.
  - Prediction results before they are displayed.
  - Replay scenario records.
"""

from __future__ import annotations

from enum import IntEnum, auto
from typing import Optional

from pydantic import BaseModel, Field, field_validator


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
    Represents a single network flow record as it arrives from a CSV row.

    TODO: Add all 49 UNSW-NB15 feature fields with correct types and ranges.
          For now, only the most important meta-fields are declared; the rest
          are captured in `extra_features`.
    """

    # Connection identifiers
    srcip: str = Field(..., description="Source IP address")
    sport: int = Field(..., ge=0, le=65535, description="Source port")
    dstip: str = Field(..., description="Destination IP address")
    dsport: int = Field(..., ge=0, le=65535, description="Destination port")
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


# ─── Prediction result ────────────────────────────────────────────────────────

class PredictionResult(BaseModel):
    """Structured output produced by nids.predict for a single record."""

    # Binary decision
    is_attack: bool = Field(..., description="True if the flow is classified as an attack")
    binary_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Model probability for the 'attack' class"
    )

    # Multiclass decision (only populated when is_attack=True)
    attack_category: Optional[str] = Field(
        None, description="Predicted attack category (UNSW-NB15 label)"
    )
    multi_confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Model probability for the predicted category"
    )

    # Metadata
    record_index: Optional[int] = Field(None, description="Row index in the source file")

    @field_validator("binary_confidence", "multi_confidence", mode="before")
    @classmethod
    def round_confidence(cls, v: Optional[float]) -> Optional[float]:
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

    # TODO: add SHAP explanation snippet once explain.py is implemented
