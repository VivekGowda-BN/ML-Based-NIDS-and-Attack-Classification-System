"""
test_replay.py — Unit and regression tests for the safe offline replay engine.

Verified invariants
-------------------
1.  Replay with limit=10 processes exactly 10 records.
2.  Replay with interval=0 runs without sleeping.
3.  JSONL output written and parseable.
4.  CSV output written with correct header and row count.
5.  SQLite output written with correct table and row count.
6.  Row-limit behaviour is exact.
7.  Scenario sampling is deterministic (same seed → same events).
8.  Empty scenario raises ValueError.
9.  Missing scenario raises FileNotFoundError.
10. Prediction event output contains no ground-truth fields.
11. Output schema matches OUTPUT_FIELDS.
12. No real network connections are created during replay.
13. Threshold is 0.80 loaded from metadata, not hard-coded.
14. Replay does not modify the held-out source file.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

try:
    from nids.config import CLEAN_TEST_PARQUET, SCENARIOS_DIR
    from nids.predict import NIDSPredictor
    from nids.replay import (
        OUTPUT_FIELDS,
        _FORBIDDEN_OUTPUT_FIELDS,
        _event_to_output_dict,
        load_scenario,
        run_replay,
        write_csv,
        write_jsonl,
        write_sqlite,
    )
    from nids.schemas import PredictionResult
except ImportError:
    from src.nids.config import CLEAN_TEST_PARQUET, SCENARIOS_DIR
    from src.nids.predict import NIDSPredictor
    from src.nids.replay import (
        OUTPUT_FIELDS,
        _FORBIDDEN_OUTPUT_FIELDS,
        _event_to_output_dict,
        load_scenario,
        run_replay,
        write_csv,
        write_jsonl,
        write_sqlite,
    )
    from src.nids.schemas import PredictionResult

# ─── Fixtures ─────────────────────────────────────────────────────────────────

MIXED_SCENARIO: Path = SCENARIOS_DIR / "mixed_demo.parquet"
NORMAL_SCENARIO: Path = SCENARIOS_DIR / "normal_demo.parquet"


@pytest.fixture(scope="module")
def predictor() -> NIDSPredictor:
    return NIDSPredictor()


@pytest.fixture(scope="module")
def ten_events(predictor: NIDSPredictor) -> list[PredictionResult]:
    """Run replay on the mixed scenario with limit=10."""
    return run_replay(
        scenario_path=MIXED_SCENARIO,
        limit=10,
        interval=0.0,
        predictor=predictor,
    )


# ─── 1. Row-limit behaviour ───────────────────────────────────────────────────

class TestRowLimit:

    def test_limit_10_produces_exactly_10_events(self, ten_events):
        assert len(ten_events) == 10, \
            f"Expected exactly 10 events with limit=10; got {len(ten_events)}"

    def test_limit_5_produces_exactly_5_events(self, predictor):
        events = run_replay(MIXED_SCENARIO, limit=5, interval=0.0, predictor=predictor)
        assert len(events) == 5

    def test_no_limit_processes_all_rows(self, predictor):
        df = pd.read_parquet(MIXED_SCENARIO)
        events = run_replay(MIXED_SCENARIO, limit=None, interval=0.0, predictor=predictor)
        assert len(events) == len(df)


# ─── 2. Interval = 0 runs instantly ──────────────────────────────────────────

class TestIntervalZero:

    def test_interval_zero_does_not_sleep(self, predictor):
        """Replay with interval=0 must not call time.sleep."""
        import nids.replay as _replay_mod
        with patch.object(_replay_mod.time, "sleep") as mock_sleep:
            run_replay(MIXED_SCENARIO, limit=5, interval=0.0, predictor=predictor)
            mock_sleep.assert_not_called()

    def test_interval_nonzero_calls_sleep(self, predictor, tmp_path):
        """Replay with interval > 0 must call time.sleep for each record."""
        import nids.replay as _replay_mod
        with patch.object(_replay_mod.time, "sleep") as mock_sleep:
            run_replay(MIXED_SCENARIO, limit=3, interval=0.001, predictor=predictor)
            assert mock_sleep.call_count == 3


# ─── 3–5. Output format tests ─────────────────────────────────────────────────

class TestOutputFormats:

    def test_jsonl_output_written_and_parseable(self, predictor, tmp_path):
        out = tmp_path / "test_events.jsonl"
        events = run_replay(
            MIXED_SCENARIO, limit=10, output_path=out, fmt="jsonl",
            interval=0.0, predictor=predictor,
        )
        assert out.exists(), "JSONL output file not created"
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == len(events) == 10
        for line in lines:
            obj = json.loads(line)
            assert "event_id" in obj
            assert "prediction" in obj

    def test_csv_output_written_with_correct_structure(self, predictor, tmp_path):
        out = tmp_path / "test_events.csv"
        events = run_replay(
            MIXED_SCENARIO, limit=10, output_path=out, fmt="csv",
            interval=0.0, predictor=predictor,
        )
        assert out.exists(), "CSV output file not created"
        with open(out, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == len(events) == 10
        for field in OUTPUT_FIELDS:
            assert field in reader.fieldnames, f"Expected column '{field}' in CSV header"

    def test_sqlite_output_written_with_correct_table(self, predictor, tmp_path):
        out = tmp_path / "test_events.sqlite"
        events = run_replay(
            MIXED_SCENARIO, limit=10, output_path=out, fmt="sqlite",
            interval=0.0, predictor=predictor,
        )
        assert out.exists(), "SQLite output file not created"
        with sqlite3.connect(str(out)) as conn:
            rows = conn.execute("SELECT * FROM replay_events").fetchall()
            cols = [desc[0] for desc in conn.execute(
                "SELECT * FROM replay_events LIMIT 0"
            ).description]
        assert len(rows) == len(events) == 10
        for field in OUTPUT_FIELDS:
            assert field in cols, f"Expected column '{field}' in SQLite table"

    def test_invalid_format_raises(self, predictor, tmp_path):
        out = tmp_path / "bad.txt"
        with pytest.raises(ValueError, match="Unsupported output format"):
            run_replay(
                MIXED_SCENARIO, limit=3, output_path=out, fmt="xml",
                interval=0.0, predictor=predictor,
            )


# ─── 8–9. Error handling ──────────────────────────────────────────────────────

class TestErrorHandling:

    def test_missing_scenario_raises_file_not_found(self, predictor):
        with pytest.raises(FileNotFoundError, match="Scenario file not found"):
            run_replay(
                Path("data/scenarios/nonexistent_scenario.parquet"),
                predictor=predictor,
            )

    def test_empty_scenario_raises_value_error(self, predictor, tmp_path):
        # Create an empty parquet file
        empty_path = tmp_path / "empty.parquet"
        pd.DataFrame().to_parquet(empty_path)
        with pytest.raises(ValueError, match="empty"):
            run_replay(empty_path, predictor=predictor)

    def test_load_scenario_raises_on_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_scenario(Path("does/not/exist.parquet"))


# ─── 10–11. Ground-truth isolation and output schema ─────────────────────────

class TestOutputPurityAndSchema:

    def test_output_contains_no_ground_truth_fields(self, ten_events):
        for event in ten_events:
            out = _event_to_output_dict(event)
            for forbidden in _FORBIDDEN_OUTPUT_FIELDS:
                assert forbidden not in out, \
                    f"Forbidden ground-truth field '{forbidden}' found in output event"

    def test_output_schema_matches_output_fields(self, ten_events):
        for event in ten_events:
            out = _event_to_output_dict(event)
            for field in OUTPUT_FIELDS:
                assert field in out, \
                    f"Expected output field '{field}' is missing from event"
            extra = set(out.keys()) - set(OUTPUT_FIELDS)
            assert not extra, f"Unexpected extra fields in output: {extra}"

    def test_records_with_ground_truth_injected_are_stripped(self, predictor):
        """
        Inject all six forbidden field names into the input record;
        none must appear in the output event.
        """
        df = pd.read_parquet(MIXED_SCENARIO).head(3)
        for _, row in df.iterrows():
            rec = row.to_dict()
            rec["ground_truth_label"] = 99
            rec["ground_truth_attack_cat"] = "Exploits"
            rec["true_label"] = 1
            rec["true_attack_category"] = "DoS"
            # label and attack_cat already present in parquet row
            event = predictor.predict_record(rec)
            out = _event_to_output_dict(event)
            for forbidden in _FORBIDDEN_OUTPUT_FIELDS:
                assert forbidden not in out, \
                    f"Forbidden field '{forbidden}' leaked into output"

    def test_all_events_are_prediction_result_instances(self, ten_events):
        for event in ten_events:
            assert isinstance(event, PredictionResult)


# ─── 12. No network connections ──────────────────────────────────────────────

class TestNoNetworkConnections:

    def test_replay_does_not_open_socket(self, predictor):
        """Verify no socket.connect or similar calls occur during replay."""
        import socket as _socket
        original_connect = _socket.socket.connect

        def fail_connect(self, *args, **kwargs):
            raise AssertionError(
                f"SAFETY VIOLATION: socket.connect was called during replay: args={args}"
            )

        with patch.object(_socket.socket, "connect", fail_connect):
            # Should complete without triggering the patched connect
            run_replay(MIXED_SCENARIO, limit=5, interval=0.0, predictor=predictor)


# ─── 13. Threshold from metadata ─────────────────────────────────────────────

class TestThresholdFromMetadata:

    def test_predictor_threshold_is_0_80(self, predictor):
        assert predictor.binary_threshold == 0.80, \
            f"Expected threshold 0.80 from metadata; got {predictor.binary_threshold}"

    def test_replay_events_carry_correct_threshold(self, ten_events):
        for event in ten_events:
            assert event.binary_threshold == 0.80, \
                f"Event carries wrong threshold: {event.binary_threshold}"


# ─── 14. Source file not modified ────────────────────────────────────────────

class TestSourceFileIntegrity:

    def test_replay_does_not_modify_source_file(self, predictor):
        source = CLEAN_TEST_PARQUET
        assert source.exists(), f"Source testing file missing: {source}"
        before = source.stat().st_mtime
        run_replay(MIXED_SCENARIO, limit=10, interval=0.0, predictor=predictor)
        after = source.stat().st_mtime
        assert before == after, \
            "Replay modified the source testing file (mtime changed)"

    def test_source_file_shape_unchanged(self):
        df = pd.read_parquet(CLEAN_TEST_PARQUET)
        assert df.shape == (82332, 45), \
            f"Source testing file shape changed: {df.shape}"


# ─── 7. Determinism ──────────────────────────────────────────────────────────

class TestDeterminism:

    def test_same_scenario_produces_identical_events(self, predictor):
        """Two consecutive replays of the same scenario with interval=0 must agree."""
        ev_a = run_replay(MIXED_SCENARIO, limit=10, interval=0.0, predictor=predictor)
        ev_b = run_replay(MIXED_SCENARIO, limit=10, interval=0.0, predictor=predictor)
        for a, b in zip(ev_a, ev_b):
            assert a.prediction == b.prediction
            assert a.attack_type == b.attack_type
            assert a.binary_probability == b.binary_probability
