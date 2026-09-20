"""
tests/test_schemas.py — Unit tests for Pydantic schemas in nids.schemas.
"""

import pytest
from pydantic import ValidationError

from nids.schemas import AlertRecord, NetworkFlowRecord, PredictionResult


class TestNetworkFlowRecord:
    def _valid_record(self):
        return dict(
            srcip="192.168.1.1",
            sport=1234,
            dstip="10.0.0.1",
            dsport=80,
            proto="TCP",
            dur=0.5,
            sbytes=512,
            dbytes=256,
            spkts=4,
            dpkts=3,
        )

    def test_valid_record_parses(self):
        r = NetworkFlowRecord(**self._valid_record())
        assert r.proto == "tcp"  # proto_lower validator

    def test_proto_is_lowercased(self):
        r = NetworkFlowRecord(**{**self._valid_record(), "proto": "  UDP  "})
        assert r.proto == "udp"

    def test_invalid_port_raises(self):
        with pytest.raises(ValidationError):
            NetworkFlowRecord(**{**self._valid_record(), "sport": -1})

    def test_extra_fields_allowed(self):
        """UNSW-NB15 has 49 columns — all extras should be accepted."""
        r = NetworkFlowRecord(**{**self._valid_record(), "sload": 1000.0, "dload": 500.0})
        assert r.model_extra.get("sload") == 1000.0


class TestPredictionResult:
    def test_attack_result(self):
        r = PredictionResult(
            is_attack=True,
            binary_confidence=0.87654,
            attack_category="DoS",
            multi_confidence=0.91234,
        )
        assert r.binary_confidence == 0.8765  # rounded to 4dp
        assert r.multi_confidence == 0.9123

    def test_normal_result(self):
        r = PredictionResult(is_attack=False, binary_confidence=0.12)
        assert r.attack_category is None
        assert r.multi_confidence is None

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            PredictionResult(is_attack=True, binary_confidence=1.5)


class TestAlertRecord:
    def test_valid_alert(self):
        a = AlertRecord(
            timestamp="2024-01-01T00:00:00",
            src_ip="1.2.3.4",
            dst_ip="5.6.7.8",
            proto="tcp",
            is_attack=True,
            attack_category="Exploits",
            confidence=0.95,
        )
        assert a.is_attack is True

    def test_alert_serialises_to_dict(self):
        a = AlertRecord(
            timestamp="2024-01-01T00:00:00",
            src_ip="1.2.3.4",
            dst_ip="5.6.7.8",
            proto="tcp",
            is_attack=True,
            attack_category=None,
            confidence=0.75,
        )
        d = a.model_dump()
        assert "timestamp" in d
        assert d["attack_category"] is None
