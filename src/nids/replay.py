"""
replay.py — Offline scenario replay loop.

Reads a held-out or hand-crafted scenario CSV file row-by-row,
passes each record through NIDSPredictor, and writes alerts to a
JSON Lines file that the Streamlit dashboard can poll.

Safety note
-----------
  ✓  Reads only from local CSV files.
  ✗  Does NOT send any network traffic.
  ✗  Does NOT scan, probe, or attack any system.

Usage
-----
    python -m nids.replay --scenario data/scenarios/example.csv
    # or programmatically:
    from nids.replay import replay_scenario
    replay_scenario(Path("data/scenarios/example.csv"))
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

from nids.predict import NIDSPredictor
from nids.schemas import AlertRecord

logger = logging.getLogger(__name__)

# ─── Output file (dashboard polls this) ───────────────────────────────────────
ALERTS_FILE: Path = Path("reports") / "metrics" / "alerts.jsonl"

# ─── Replay speed ─────────────────────────────────────────────────────────────
# Simulated inter-record delay (seconds).  Set to 0 for maximum speed.
REPLAY_DELAY_SECONDS: float = 0.05


def replay_scenario(
    scenario_path: Path,
    alerts_file: Path = ALERTS_FILE,
    delay: float = REPLAY_DELAY_SECONDS,
    binary_threshold: float = 0.5,
) -> list[AlertRecord]:
    """
    Replay every row of a scenario CSV through the NIDS pipeline.

    Parameters
    ----------
    scenario_path    : Path to the input CSV (must exist).
    alerts_file      : Path where alerts are appended (JSON Lines).
    delay            : Sleep between records (seconds) to simulate streaming.
    binary_threshold : Passed to NIDSPredictor.predict_record().

    Returns
    -------
    list[AlertRecord] — all alerts raised during replay.
    """
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario file not found: {scenario_path}")

    predictor = NIDSPredictor()
    df = pd.read_csv(scenario_path, low_memory=False)
    logger.info("Replaying %d records from %s", len(df), scenario_path)

    alerts: list[AlertRecord] = []
    alerts_file.parent.mkdir(parents=True, exist_ok=True)

    with open(alerts_file, "a", encoding="utf-8") as fh:
        for idx, row in df.iterrows():
            record = row.to_dict()
            result = predictor.predict_record(record, record_index=int(idx),
                                              binary_threshold=binary_threshold)

            if result.is_attack:
                # TODO: derive a proper timestamp field from the scenario data.
                timestamp = pd.Timestamp.now().isoformat()
                alert = AlertRecord(
                    timestamp=timestamp,
                    src_ip=str(record.get("srcip", "N/A")),
                    dst_ip=str(record.get("dstip", "N/A")),
                    proto=str(record.get("proto", "N/A")),
                    is_attack=True,
                    attack_category=result.attack_category,
                    confidence=result.binary_confidence,
                )
                alerts.append(alert)
                fh.write(json.dumps(alert.model_dump()) + "\n")
                fh.flush()
                logger.warning(
                    "ALERT  row=%-5d  cat=%-16s  conf=%.3f",
                    idx, result.attack_category, result.binary_confidence,
                )

            if delay > 0:
                time.sleep(delay)

    logger.info(
        "Replay complete. %d / %d records flagged as attacks.",
        len(alerts), len(df),
    )
    return alerts


# ─── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Replay a NIDS scenario CSV.")
    parser.add_argument(
        "--scenario",
        type=Path,
        default=Path("data/scenarios/example.csv"),
        help="Path to the scenario CSV file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binary attack probability threshold (default: 0.5).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=REPLAY_DELAY_SECONDS,
        help="Inter-record sleep delay in seconds (default: 0.05).",
    )
    args = parser.parse_args()

    alerts = replay_scenario(
        scenario_path=args.scenario,
        binary_threshold=args.threshold,
        delay=args.delay,
    )
    print(f"Replay done. Alerts raised: {len(alerts)}")
