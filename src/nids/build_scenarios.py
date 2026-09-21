"""
build_scenarios.py — Create curated demonstration scenarios from the held-out test split.

Selection rules
---------------
1. Uses ONLY records from the official testing split (UNSW_NB15_testing_clean.parquet).
2. Uses a fixed random seed (RANDOM_SEED = 42) for reproducibility.
3. Ground-truth columns (label, attack_cat) are used ONLY for filtering during
   scenario construction.  They are retained in the parquet files so that the
   replay engine can later strip them before passing records to the predictor.
4. The original testing file is never modified.
5. Scenario files are small enough for quick demonstration.

Scenario definitions
--------------------
normal_demo         : 200 randomly sampled Normal traffic records  (label == 0)
reconnaissance_demo : All Reconnaissance attack records (≤ 200)
dos_demo            : 200 randomly sampled DoS attack records      (attack_cat == DoS)
mixed_demo          : 50 Normal + 50 Generic + 50 Exploits + 30 DoS +
                      20 Fuzzers + 20 Reconnaissance +
                      10 Analysis + 10 Backdoor + 10 Shellcode + 5 Worms
                      → up to 255 records representing every class

Usage
-----
    python src/nids/build_scenarios.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR: Path = Path(__file__).resolve().parents[2]
TEST_PARQUET: Path = ROOT_DIR / "data" / "interim" / "UNSW_NB15_testing_clean.parquet"
SCENARIOS_DIR: Path = ROOT_DIR / "data" / "scenarios"
RANDOM_SEED: int = 42


def _sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Sample up to n rows without replacement."""
    return df.sample(n=min(n, len(df)), random_state=RANDOM_SEED)


def build_scenarios(test_parquet: Path = TEST_PARQUET) -> dict[str, pd.DataFrame]:
    """
    Build all four demonstration scenarios from the held-out testing split.

    Returns
    -------
    dict mapping scenario name -> DataFrame
    """
    logger.info("Loading held-out test data from %s", test_parquet)
    df = pd.read_parquet(test_parquet)
    logger.info("Loaded %d records with columns: %s", len(df), list(df.columns))

    scenarios: dict[str, pd.DataFrame] = {}

    # 1. Normal-only demonstration
    normal_df = df[df["label"] == 0]
    scenarios["normal_demo"] = _sample(normal_df, 200).reset_index(drop=True)

    # 2. Reconnaissance-only demonstration
    recon_df = df[df["attack_cat"] == "Reconnaissance"]
    scenarios["reconnaissance_demo"] = _sample(recon_df, 200).reset_index(drop=True)

    # 3. DoS-only demonstration
    dos_df = df[df["attack_cat"] == "DoS"]
    scenarios["dos_demo"] = _sample(dos_df, 200).reset_index(drop=True)

    # 4. Mixed scenario — representative sample of every class
    mix_parts = [
        _sample(df[df["attack_cat"] == "Normal"],         50),
        _sample(df[df["attack_cat"] == "Generic"],        50),
        _sample(df[df["attack_cat"] == "Exploits"],       50),
        _sample(df[df["attack_cat"] == "DoS"],            30),
        _sample(df[df["attack_cat"] == "Fuzzers"],        20),
        _sample(df[df["attack_cat"] == "Reconnaissance"], 20),
        _sample(df[df["attack_cat"] == "Analysis"],       10),
        _sample(df[df["attack_cat"] == "Backdoor"],       10),
        _sample(df[df["attack_cat"] == "Shellcode"],      10),
        _sample(df[df["attack_cat"] == "Worms"],           5),
    ]
    mixed_df = pd.concat(mix_parts, ignore_index=True)
    # Shuffle the mixed scenario so classes are interleaved
    mixed_df = mixed_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    scenarios["mixed_demo"] = mixed_df

    return scenarios


def save_scenarios(
    scenarios: dict[str, pd.DataFrame],
    out_dir: Path = SCENARIOS_DIR,
) -> None:
    """Write each scenario DataFrame to a parquet file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in scenarios.items():
        out_path = out_dir / f"{name}.parquet"
        df.to_parquet(out_path, index=False)
        logger.info("Saved scenario %-25s  %d rows  → %s", name, len(df), out_path)


def write_readme(scenarios: dict[str, pd.DataFrame], out_dir: Path = SCENARIOS_DIR) -> None:
    """Write data/scenarios/README.md documenting scenario construction."""
    lines = [
        "# NIDS Demonstration Scenarios",
        "",
        "All scenario files in this directory are derived **exclusively** from the",
        "official UNSW-NB15 **held-out testing split**",
        "(`data/interim/UNSW_NB15_testing_clean.parquet`).",
        "",
        "## Construction Method",
        "",
        "| Step | Detail |",
        "|------|--------|",
        "| Source | `UNSW_NB15_testing_clean.parquet` (official test split only) |",
        "| Random seed | `42` (fixed for reproducibility) |",
        "| Selection | Ground-truth `label` and `attack_cat` used **only** during file creation |",
        "| Prediction input | Ground-truth columns stripped by `NIDSPredictor.validate_record()` before inference |",
        "| Output events | `label`, `attack_cat`, and all other ground-truth fields never appear in replay output |",
        "| Source file | Original testing parquet is **never modified** |",
        "",
        "## Scenarios",
        "",
    ]

    descs = {
        "normal_demo": "200 randomly sampled Normal-traffic records (`label == 0`).",
        "reconnaissance_demo": "Up to 200 Reconnaissance attack records (`attack_cat == Reconnaissance`).",
        "dos_demo": "Up to 200 DoS attack records (`attack_cat == DoS`).",
        "mixed_demo": (
            "Representative mix of every class: 50 Normal, 50 Generic, 50 Exploits, "
            "30 DoS, 20 Fuzzers, 20 Reconnaissance, 10 Analysis, 10 Backdoor, "
            "10 Shellcode, 5 Worms — shuffled with seed 42."
        ),
    }

    for name, df in scenarios.items():
        lines += [
            f"### `{name}.parquet`",
            "",
            f"**Rows**: {len(df)}",
            "",
            descs.get(name, ""),
            "",
            "**Class distribution:**",
            "",
            "| attack_cat | count |",
            "|------------|-------|",
        ]
        for cat, cnt in df["attack_cat"].value_counts().items():
            lines.append(f"| {cat} | {cnt} |")
        lines.append("")

    lines += [
        "## Safety Notes",
        "",
        "* Scenario files contain only network-flow **feature records** from the",
        "  UNSW-NB15 dataset.  They do **not** contain live traffic.",
        "* The replay engine (`src/nids/replay.py`) never sends packets,",
        "  never opens network connections, and never scans or attacks any host.",
        "* Ground-truth fields (`label`, `attack_cat`) present in the parquet",
        "  are stripped before any record reaches the prediction models.",
    ]

    readme_path = out_dir / "README.md"
    readme_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote README to %s", readme_path)


if __name__ == "__main__":
    scenarios = build_scenarios()
    save_scenarios(scenarios)
    write_readme(scenarios)
    logger.info("All scenarios created successfully.")
