"""
test_dashboard.py — Unit tests for the Mode 1 Streamlit dashboard.

Tested invariants
-----------------
1.  Dashboard module imports successfully (helper functions importable).
2.  Scenario list contains the four expected scenarios.
3.  Missing-file handling returns safe defaults (empty dict / empty DataFrame).
4.  Summary-card calculations are correct for known event lists.
5.  Severity calculations are correct.
6.  Ground-truth fields are excluded from alert display DataFrames.
7.  Exported CSV/JSONL data contains only prediction-only fields.
8.  Metrics files load correctly and have expected structure.
9.  Empty replay state is handled (compute_summary returns zeros).
10. Dashboard helper functions do not create network connections.
"""

from __future__ import annotations

import json
import socket
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ─── Import pure helpers from dashboard.app ───────────────────────────────────
# Patch streamlit before importing so the module doesn't require a running
# Streamlit server.  The pure helper functions do not call st.* at all.
_st_mock = MagicMock()
sys.modules.setdefault("streamlit", _st_mock)
# Also mock plotly if not installed in test env
sys.modules.setdefault("plotly", MagicMock())
sys.modules.setdefault("plotly.express", MagicMock())
sys.modules.setdefault("plotly.graph_objects", MagicMock())

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))

# Import only the pure utility layer — no Streamlit calls happen
from app import (  # noqa: E402
    compute_summary,
    events_to_display_df,
    load_json,
    load_csv_safe,
    get_scenario_files,
    FORBIDDEN_FIELDS,
    ALERT_DISPLAY_FIELDS,
    EXPECTED_SCENARIOS,
    SCENARIOS_DIR,
    METRICS_DIR,
    BINARY_RESULTS_JSON,
    BINARY_COMPARISON_CSV,
    MULTICLASS_RESULTS_JSON,
    SEVERITY_COLORS,
)


# ─── Sample prediction events (pure model output — no ground truth) ────────────

def _make_event(
    prediction: str = "Attack",
    attack_type: str = "DoS",
    severity: str = "Critical",
    confidence: float = 0.95,
    binary_prob: float = 0.97,
    multi_prob: float | None = 0.90,
) -> dict:
    return {
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prediction": prediction,
        "attack_type": attack_type,
        "binary_probability": binary_prob,
        "binary_threshold": 0.80,
        "binary_confidence": binary_prob,
        "multiclass_probability": multi_prob if prediction == "Attack" else None,
        "confidence": confidence,
        "severity": severity,
        "recommended_action": "Block source IP",
        "model_version": "1.0.0",
        "record_index": 0,
    }


def _make_normal_event() -> dict:
    return _make_event(
        prediction="Normal", attack_type="Normal",
        severity="None", confidence=0.72,
        binary_prob=0.35, multi_prob=None,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  1. Module imports
# ─────────────────────────────────────────────────────────────────────────────

class TestModuleImport:

    def test_dashboard_helpers_importable(self):
        assert callable(compute_summary)
        assert callable(events_to_display_df)
        assert callable(load_json)
        assert callable(load_csv_safe)
        assert callable(get_scenario_files)

    def test_expected_constants_defined(self):
        assert isinstance(FORBIDDEN_FIELDS, frozenset)
        assert isinstance(ALERT_DISPLAY_FIELDS, list)
        assert isinstance(EXPECTED_SCENARIOS, list)
        assert len(EXPECTED_SCENARIOS) == 4
        assert len(SEVERITY_COLORS) >= 4


# ─────────────────────────────────────────────────────────────────────────────
#  2. Scenario list
# ─────────────────────────────────────────────────────────────────────────────

class TestScenarioList:

    def test_expected_scenario_names_defined(self):
        for name in [
            "mixed_demo.parquet", "normal_demo.parquet",
            "reconnaissance_demo.parquet", "dos_demo.parquet",
        ]:
            assert name in EXPECTED_SCENARIOS

    def test_scenario_files_exist_on_disk(self):
        files = get_scenario_files()
        for name in EXPECTED_SCENARIOS:
            assert name in files, f"Expected scenario file missing: {name}"

    def test_scenario_files_are_parquet(self):
        for name in get_scenario_files():
            assert name.endswith(".parquet"), f"Non-parquet file in scenarios: {name}"


# ─────────────────────────────────────────────────────────────────────────────
#  3. Missing-file handling
# ─────────────────────────────────────────────────────────────────────────────

class TestMissingFileHandling:

    def test_load_json_returns_empty_dict_for_missing_file(self, tmp_path):
        assert load_json(tmp_path / "nonexistent.json") == {}

    def test_load_json_returns_empty_dict_for_corrupt_file(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("NOT VALID {{{", encoding="utf-8")
        assert load_json(bad) == {}

    def test_load_csv_safe_returns_empty_df_for_missing_file(self, tmp_path):
        df = load_csv_safe(tmp_path / "nonexistent.csv")
        assert isinstance(df, pd.DataFrame) and df.empty

    def test_load_csv_safe_returns_empty_df_for_corrupt_file(self, tmp_path):
        bad = tmp_path / "bad.csv"
        bad.write_bytes(b"\x00\x01\x02\x03")
        df = load_csv_safe(bad)
        assert isinstance(df, pd.DataFrame) and df.empty

    def test_load_json_loads_valid_file(self, tmp_path):
        f = tmp_path / "ok.json"
        f.write_text('{"key": 42}', encoding="utf-8")
        assert load_json(f) == {"key": 42}


# ─────────────────────────────────────────────────────────────────────────────
#  4. Summary-card calculations
# ─────────────────────────────────────────────────────────────────────────────

class TestSummaryCards:

    def test_empty_returns_zeros_and_dash(self):
        s = compute_summary([])
        assert s["total"] == 0
        assert s["attacks"] == 0
        assert s["normal"] == 0
        assert s["critical"] == 0
        assert s["avg_confidence"] == 0.0
        assert s["risk_level"] == "—"

    def test_mixed_events_correct_counts(self):
        events = [
            _make_event("Attack", "DoS", "Critical", 0.95),
            _make_event("Attack", "Fuzzers", "High", 0.85),
            _make_normal_event(),
            _make_normal_event(),
        ]
        s = compute_summary(events)
        assert s["total"] == 4
        assert s["attacks"] == 2
        assert s["normal"] == 2
        assert s["critical"] == 1

    def test_avg_confidence_computed_correctly(self):
        events = [
            _make_event(confidence=0.80),
            _make_event(confidence=0.60),
        ]
        s = compute_summary(events)
        assert abs(s["avg_confidence"] - 0.70) < 1e-6

    def test_risk_critical_when_critical_alert(self):
        s = compute_summary([_make_event("Attack", "Shellcode", "Critical", 0.99)])
        assert "Critical" in s["risk_level"]

    def test_risk_normal_when_all_normal(self):
        s = compute_summary([_make_normal_event(), _make_normal_event()])
        assert "Normal" in s["risk_level"]

    def test_risk_high_when_only_high_severity(self):
        s = compute_summary([_make_event("Attack", "Fuzzers", "High", 0.85)])
        assert "High" in s["risk_level"]

    def test_risk_medium_when_only_medium_severity(self):
        ev = _make_event("Attack", "Analysis", "Medium", 0.82)
        s = compute_summary([ev])
        assert "Medium" in s["risk_level"]


# ─────────────────────────────────────────────────────────────────────────────
#  5. Severity values
# ─────────────────────────────────────────────────────────────────────────────

class TestSeverityValues:

    def test_all_severity_levels_have_colours(self):
        for level in ("None", "Medium", "High", "Critical"):
            assert level in SEVERITY_COLORS

    def test_critical_events_counted_correctly(self):
        events = [
            _make_event("Attack", "Shellcode", "Critical"),
            _make_event("Attack", "Worms", "Critical"),
            _make_event("Attack", "Fuzzers", "High"),
        ]
        assert compute_summary(events)["critical"] == 2


# ─────────────────────────────────────────────────────────────────────────────
#  6. Ground-truth field exclusion
# ─────────────────────────────────────────────────────────────────────────────

class TestGroundTruthExclusion:

    def test_forbidden_fields_not_in_display_df(self):
        events = [_make_event(), _make_normal_event()]
        for ev in events:
            for f in FORBIDDEN_FIELDS:
                ev[f] = "SHOULD_NOT_APPEAR"
        df = events_to_display_df(events)
        for field in FORBIDDEN_FIELDS:
            assert field not in df.columns, \
                f"Forbidden field '{field}' found in display DataFrame"

    def test_display_df_contains_prediction_fields(self):
        df = events_to_display_df([_make_event()])
        for field in ["prediction", "attack_type", "severity", "confidence"]:
            assert field in df.columns

    def test_display_df_columns_subset_of_alert_display_fields(self):
        df = events_to_display_df([_make_event()])
        for col in df.columns:
            assert col in ALERT_DISPLAY_FIELDS, \
                f"Unexpected column in display DataFrame: {col}"

    def test_all_six_forbidden_names_stripped(self):
        forbidden = [
            "label", "attack_cat", "ground_truth_label",
            "ground_truth_attack_cat", "true_label", "true_attack_category",
        ]
        ev = _make_event()
        for f in forbidden:
            ev[f] = "LEAKED"
        df = events_to_display_df([ev])
        for f in forbidden:
            assert f not in df.columns


# ─────────────────────────────────────────────────────────────────────────────
#  7. Exported data has prediction-only fields
# ─────────────────────────────────────────────────────────────────────────────

class TestExportedData:

    def test_csv_export_has_no_ground_truth(self):
        import io
        events = [_make_event(), _make_normal_event()]
        for ev in events:
            ev["label"] = 1
            ev["attack_cat"] = "DoS"
        df = events_to_display_df(events)
        reader = pd.read_csv(io.StringIO(df.to_csv(index=False)))
        for field in FORBIDDEN_FIELDS:
            assert field not in reader.columns

    def test_jsonl_export_has_no_ground_truth(self):
        events = [_make_event(), _make_normal_event()]
        for ev in events:
            ev["ground_truth_label"] = 99
        df = events_to_display_df(events)
        for row in df.to_dict(orient="records"):
            for field in FORBIDDEN_FIELDS:
                assert field not in row


# ─────────────────────────────────────────────────────────────────────────────
#  8. Metrics files load correctly
# ─────────────────────────────────────────────────────────────────────────────

class TestMetricsLoading:

    def test_binary_results_json_loads(self):
        if not BINARY_RESULTS_JSON.exists():
            pytest.skip("binary_results.json not present")
        d = load_json(BINARY_RESULTS_JSON)
        assert isinstance(d, dict) and d

    def test_binary_comparison_csv_loads(self):
        if not BINARY_COMPARISON_CSV.exists():
            pytest.skip("binary_model_comparison.csv not present")
        df = load_csv_safe(BINARY_COMPARISON_CSV)
        assert not df.empty and "model" in df.columns

    def test_multiclass_results_json_loads(self):
        if not MULTICLASS_RESULTS_JSON.exists():
            pytest.skip("multiclass_results.json not present")
        d = load_json(MULTICLASS_RESULTS_JSON)
        assert isinstance(d, dict) and d

    def test_threshold_metadata_has_selected_threshold(self):
        meta_path = METRICS_DIR / "binary_threshold_selection_metadata.json"
        if not meta_path.exists():
            pytest.skip("threshold metadata not present")
        d = load_json(meta_path)
        assert "selected_threshold" in d
        assert d["selected_threshold"] == 0.80


# ─────────────────────────────────────────────────────────────────────────────
#  9. Empty replay state
# ─────────────────────────────────────────────────────────────────────────────

class TestEmptyReplayState:

    def test_empty_summary(self):
        s = compute_summary([])
        assert s["total"] == 0 and s["risk_level"] == "—"

    def test_empty_display_df(self):
        assert events_to_display_df([]).empty

    def test_display_df_with_none_list(self):
        df = events_to_display_df([] or [])
        assert isinstance(df, pd.DataFrame) and df.empty


# ─────────────────────────────────────────────────────────────────────────────
#  10. No network connections
# ─────────────────────────────────────────────────────────────────────────────

class TestNoNetworkConnections:

    def test_load_json_does_not_open_socket(self, tmp_path):
        f = tmp_path / "test.json"
        f.write_text('{"k": 1}', encoding="utf-8")

        def fail_connect(self, *a, **kw):
            raise AssertionError("socket.connect called during load_json")

        with patch.object(socket.socket, "connect", fail_connect):
            result = load_json(f)
        assert result == {"k": 1}

    def test_compute_summary_no_socket(self):
        events = [_make_event(), _make_normal_event()]

        def fail_connect(self, *a, **kw):
            raise AssertionError("socket.connect called during compute_summary")

        with patch.object(socket.socket, "connect", fail_connect):
            s = compute_summary(events)
        assert s["total"] == 2

    def test_events_to_display_df_no_socket(self):
        def fail_connect(self, *a, **kw):
            raise AssertionError("socket.connect called during events_to_display_df")

        with patch.object(socket.socket, "connect", fail_connect):
            df = events_to_display_df([_make_event()])
        assert not df.empty
