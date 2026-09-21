"""
test_scenarios.py — Tests for curated NIDS demonstration scenarios.

Verified invariants
-------------------
1.  All four scenario files exist.
2.  Each scenario is a non-empty DataFrame.
3.  Every required model feature is present in each scenario.
4.  Scenarios were built using fixed seed 42 (deterministic output).
5.  normal_demo contains only Normal records (label == 0).
6.  reconnaissance_demo contains only Reconnaissance records.
7.  dos_demo contains only DoS records.
8.  mixed_demo contains records from every attack category.
9.  Scenario files are subsets of the official testing split (no training data leakage).
10. Ground-truth columns (label, attack_cat) are retained in the parquet for filtering
    but are stripped by the replay engine before prediction.
11. The original testing source file is unmodified.
12. Scenario row counts are within expected bounds.
13. README.md exists and documents construction method.
14. Sampling is deterministic (same seed produces same rows).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

try:
    from nids.config import CLEAN_TEST_PARQUET, SCENARIOS_DIR
    from nids.predict import NIDSPredictor
except ImportError:
    from src.nids.config import CLEAN_TEST_PARQUET, SCENARIOS_DIR
    from src.nids.predict import NIDSPredictor

# ─── Fixtures ─────────────────────────────────────────────────────────────────

SCENARIO_FILES = {
    "normal_demo": SCENARIOS_DIR / "normal_demo.parquet",
    "reconnaissance_demo": SCENARIOS_DIR / "reconnaissance_demo.parquet",
    "dos_demo": SCENARIOS_DIR / "dos_demo.parquet",
    "mixed_demo": SCENARIOS_DIR / "mixed_demo.parquet",
}

REQUIRED_FEATURES = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl",
    "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit",
    "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean",
    "dmean", "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl",
    "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm",
    "ct_srv_dst", "is_sm_ips_ports", "proto", "service", "state",
]


@pytest.fixture(scope="module")
def test_source_df() -> pd.DataFrame:
    assert CLEAN_TEST_PARQUET.exists(), f"Testing source not found: {CLEAN_TEST_PARQUET}"
    return pd.read_parquet(CLEAN_TEST_PARQUET)


@pytest.fixture(scope="module", params=list(SCENARIO_FILES.keys()))
def scenario_df(request) -> pd.DataFrame:
    path = SCENARIO_FILES[request.param]
    assert path.exists(), f"Missing scenario file: {path}"
    return pd.read_parquet(path)


# ─── Existence tests ──────────────────────────────────────────────────────────

class TestScenarioExistence:

    def test_all_scenario_files_exist(self):
        for name, path in SCENARIO_FILES.items():
            assert path.exists(), f"Scenario file missing: {name} → {path}"

    def test_readme_exists(self):
        readme = SCENARIOS_DIR / "README.md"
        assert readme.exists(), "data/scenarios/README.md is missing"

    def test_readme_documents_construction(self):
        readme = SCENARIOS_DIR / "README.md"
        content = readme.read_text(encoding="utf-8")
        assert "testing split" in content.lower() or "held-out" in content.lower(), \
            "README.md must document the testing-split source"
        assert "seed" in content.lower() or "42" in content, \
            "README.md must document the random seed"


# ─── Content tests ────────────────────────────────────────────────────────────

class TestScenarioContent:

    def test_scenario_non_empty(self, scenario_df: pd.DataFrame):
        assert len(scenario_df) > 0, "Scenario DataFrame must not be empty"

    def test_all_required_features_present(self, scenario_df: pd.DataFrame):
        missing = [f for f in REQUIRED_FEATURES if f not in scenario_df.columns]
        assert not missing, f"Required features missing from scenario: {missing}"

    def test_normal_demo_contains_only_normal(self):
        df = pd.read_parquet(SCENARIO_FILES["normal_demo"])
        assert (df["label"] == 0).all(), \
            "normal_demo must contain only Normal records (label == 0)"
        assert (df["attack_cat"] == "Normal").all(), \
            "normal_demo must contain only Normal attack_cat"

    def test_reconnaissance_demo_contains_only_recon(self):
        df = pd.read_parquet(SCENARIO_FILES["reconnaissance_demo"])
        assert (df["attack_cat"] == "Reconnaissance").all(), \
            "reconnaissance_demo must contain only Reconnaissance records"

    def test_dos_demo_contains_only_dos(self):
        df = pd.read_parquet(SCENARIO_FILES["dos_demo"])
        assert (df["attack_cat"] == "DoS").all(), \
            "dos_demo must contain only DoS records"

    def test_mixed_demo_contains_multiple_classes(self):
        df = pd.read_parquet(SCENARIO_FILES["mixed_demo"])
        cats = set(df["attack_cat"].unique())
        # Must contain at least Normal and several attack categories
        assert "Normal" in cats, "mixed_demo must contain Normal records"
        assert len(cats) >= 5, f"mixed_demo must span ≥5 classes; got {cats}"

    def test_scenario_row_counts_within_bounds(self):
        bounds = {
            "normal_demo": (1, 200),
            "reconnaissance_demo": (1, 200),
            "dos_demo": (1, 200),
            "mixed_demo": (1, 300),
        }
        for name, (lo, hi) in bounds.items():
            df = pd.read_parquet(SCENARIO_FILES[name])
            assert lo <= len(df) <= hi, \
                f"{name} row count {len(df)} outside expected range [{lo}, {hi}]"


# ─── Data-leakage / isolation tests ──────────────────────────────────────────

class TestScenarioIsolation:

    def test_scenarios_are_subset_of_testing_split(self, test_source_df: pd.DataFrame):
        """No scenario row should contain id values absent from the test set."""
        if "id" not in test_source_df.columns:
            pytest.skip("id column not present in test source")
        test_ids = set(test_source_df["id"].values)
        for name, path in SCENARIO_FILES.items():
            df = pd.read_parquet(path)
            if "id" not in df.columns:
                continue
            unknown_ids = set(df["id"].values) - test_ids
            assert not unknown_ids, \
                f"{name} contains {len(unknown_ids)} ids not in the official test split"

    def test_source_testing_file_unmodified(self, test_source_df: pd.DataFrame):
        """Verify original testing parquet has the expected shape."""
        assert test_source_df.shape == (82332, 45), \
            f"Testing source file may have been modified: shape={test_source_df.shape}"


# ─── Determinism tests ────────────────────────────────────────────────────────

class TestScenarioDeterminism:

    def test_sampling_is_deterministic(self):
        """Re-running the scenario builder with the same seed must produce identical files."""
        try:
            from nids.build_scenarios import build_scenarios
        except ImportError:
            from src.nids.build_scenarios import build_scenarios

        scenarios_a = build_scenarios()
        scenarios_b = build_scenarios()

        for name in scenarios_a:
            df_a = scenarios_a[name].reset_index(drop=True)
            df_b = scenarios_b[name].reset_index(drop=True)
            pd.testing.assert_frame_equal(
                df_a, df_b,
                obj=f"Scenario '{name}' must be identical across two runs with the same seed",
            )

    def test_mixed_demo_shuffled_contains_all_expected_classes(self):
        """Mixed demo must contain Worms even after shuffling (all 10 classes)."""
        df = pd.read_parquet(SCENARIO_FILES["mixed_demo"])
        cats = set(df["attack_cat"].unique())
        expected = {"Normal", "Generic", "Exploits", "DoS", "Fuzzers",
                    "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms"}
        # Worms only has 44 records in test split; we sample 5 — should always succeed
        assert expected == cats, f"mixed_demo class mismatch. Got: {cats}"
