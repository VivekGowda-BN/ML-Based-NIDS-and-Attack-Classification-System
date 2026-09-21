"""
replay.py — Safe offline dataset replay engine for ML-Based NIDS.

Safety Guarantees
-----------------
* Reads only from local parquet files (held-out testing split or curated scenarios).
* Never sends packets, opens network connections, scans hosts, or executes attack commands.
* Ground-truth fields (label, attack_cat, ground_truth_label, ground_truth_attack_cat,
  true_label, true_attack_category) are stripped before prediction and never written to output.
* Binary decision threshold is loaded from saved metadata; never hard-coded.

Supported output formats: jsonl, csv, sqlite

CLI Usage
---------
    python -m src.nids.replay --help

    python -m src.nids.replay \\
        --scenario data/scenarios/mixed_demo.parquet \\
        --limit 10 --interval 0 \\
        --output reports/replay_smoke_test.jsonl --format jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import pandas as pd

try:
    from nids.predict import NIDSPredictor
    from nids.schemas import PredictionResult
    from nids.config import CLEAN_TEST_PARQUET, SCENARIOS_DIR
except ImportError:
    from src.nids.predict import NIDSPredictor
    from src.nids.schemas import PredictionResult
    from src.nids.config import CLEAN_TEST_PARQUET, SCENARIOS_DIR

logger = logging.getLogger(__name__)

# Ground-truth field names that must never appear in output events.
_FORBIDDEN_OUTPUT_FIELDS: frozenset = frozenset({
    "label",
    "attack_cat",
    "ground_truth_label",
    "ground_truth_attack_cat",
    "true_label",
    "true_attack_category",
})

# Ordered list of fields written to every output format.
OUTPUT_FIELDS: List[str] = [
    "event_id",
    "timestamp",
    "prediction",
    "attack_type",
    "binary_probability",
    "binary_threshold",
    "binary_confidence",
    "multiclass_probability",
    "confidence",
    "severity",
    "recommended_action",
    "model_version",
    "record_index",
]


# ─── Scenario loader ──────────────────────────────────────────────────────────

def load_scenario(scenario_path: Path, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Load a scenario parquet file and optionally cap the number of rows.

    Parameters
    ----------
    scenario_path : Path
        Path to a .parquet scenario or the full testing parquet.
    limit : int, optional
        Maximum number of rows to return.  None means all rows.

    Returns
    -------
    pd.DataFrame
        Raw records; may still contain ground-truth columns which the predictor
        will strip before inference.

    Raises
    ------
    FileNotFoundError
        If the scenario file does not exist.
    ValueError
        If the loaded DataFrame is empty.
    """
    path = Path(scenario_path)
    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")

    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"Scenario file is empty: {path}")

    if limit is not None and limit > 0:
        df = df.head(limit)

    logger.info("Loaded %d records from %s", len(df), path.name)
    return df


# ─── Event stream ─────────────────────────────────────────────────────────────

def replay_stream(
    df: pd.DataFrame,
    predictor: NIDSPredictor,
    interval: float = 0.0,
) -> Iterator[PredictionResult]:
    """
    Iterate over a DataFrame, yield one PredictionResult per row.

    Parameters
    ----------
    df : pd.DataFrame
        Records to replay.  Ground-truth columns (label, attack_cat, …) are
        stripped inside ``NIDSPredictor.validate_record``; they never reach the
        model or the output event.
    predictor : NIDSPredictor
        Initialised two-stage predictor with threshold loaded from metadata.
    interval : float
        Simulated inter-event delay in seconds (0 = as fast as possible).
    """
    for idx, row in df.iterrows():
        rec = row.to_dict()
        rec_idx = int(idx) if isinstance(idx, int) else None
        event = predictor.predict_record(rec, record_index=rec_idx)
        yield event
        if interval > 0.0:
            time.sleep(interval)


# ─── Output helpers ───────────────────────────────────────────────────────────

def _event_to_output_dict(event: PredictionResult) -> dict:
    """
    Extract the allowed output fields from a PredictionResult.

    Raises RuntimeError if any forbidden ground-truth field is present
    (defence-in-depth; should never happen given schema-level forbid).
    """
    full = event.model_dump()
    out = {k: full[k] for k in OUTPUT_FIELDS if k in full}
    leaked = set(out.keys()) & _FORBIDDEN_OUTPUT_FIELDS
    if leaked:
        raise RuntimeError(
            f"SAFETY VIOLATION: forbidden ground-truth fields in output: {leaked}"
        )
    return out


def write_jsonl(events: Sequence[PredictionResult], output_path: Path) -> int:
    """Write prediction events to a JSONL file.  Returns row count."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(_event_to_output_dict(event)) + "\n")
            count += 1
    logger.info("Wrote %d JSONL events to %s", count, output_path)
    return count


def write_csv(events: Sequence[PredictionResult], output_path: Path) -> int:
    """Write prediction events to a CSV file.  Returns row count."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for event in events:
            writer.writerow(_event_to_output_dict(event))
            count += 1
    logger.info("Wrote %d CSV rows to %s", count, output_path)
    return count


def write_sqlite(events: Sequence[PredictionResult], output_path: Path) -> int:
    """Write prediction events to a SQLite database.  Returns row count."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    col_defs = (
        "event_id TEXT PRIMARY KEY, "
        "timestamp TEXT, "
        "prediction TEXT, "
        "attack_type TEXT, "
        "binary_probability REAL, "
        "binary_threshold REAL, "
        "binary_confidence REAL, "
        "multiclass_probability REAL, "
        "confidence REAL, "
        "severity TEXT, "
        "recommended_action TEXT, "
        "model_version TEXT, "
        "record_index INTEGER"
    )
    create_sql = f"CREATE TABLE IF NOT EXISTS replay_events ({col_defs})"
    placeholders = ", ".join(["?"] * len(OUTPUT_FIELDS))
    insert_sql = (
        f"INSERT OR REPLACE INTO replay_events ({', '.join(OUTPUT_FIELDS)}) "
        f"VALUES ({placeholders})"
    )
    count = 0
    with sqlite3.connect(str(output_path)) as conn:
        conn.execute(create_sql)
        for event in events:
            out = _event_to_output_dict(event)
            conn.execute(insert_sql, [out.get(k) for k in OUTPUT_FIELDS])
            count += 1
        conn.commit()
    logger.info("Wrote %d SQLite rows to %s", count, output_path)
    return count


# ─── High-level replay runner ─────────────────────────────────────────────────

def run_replay(
    scenario_path: Path,
    limit: Optional[int] = None,
    interval: float = 0.0,
    output_path: Optional[Path] = None,
    fmt: str = "jsonl",
    seed: Optional[int] = None,  # noqa: ARG001  (reserved for future use)
    predictor: Optional[NIDSPredictor] = None,
) -> List[PredictionResult]:
    """
    Load a scenario and run the two-stage prediction service on every record.

    Parameters
    ----------
    scenario_path : Path
        Scenario parquet file.
    limit : int, optional
        Maximum number of records to process.
    interval : float
        Simulated inter-event delay (seconds).
    output_path : Path, optional
        If provided, write events to this path in the given format.
    fmt : str
        Output format: 'jsonl', 'csv', or 'sqlite'.
    seed : int, optional
        Reserved for future use (scenario sampling uses its own seed).
    predictor : NIDSPredictor, optional
        Pre-initialised predictor; loaded fresh if None.

    Returns
    -------
    list[PredictionResult]
        All prediction events generated during the replay.
    """
    if predictor is None:
        predictor = NIDSPredictor()

    df = load_scenario(scenario_path, limit=limit)
    events: List[PredictionResult] = list(replay_stream(df, predictor, interval=interval))

    if output_path is not None:
        fmt_lower = fmt.lower().strip()
        if fmt_lower == "jsonl":
            write_jsonl(events, output_path)
        elif fmt_lower == "csv":
            write_csv(events, output_path)
        elif fmt_lower == "sqlite":
            write_sqlite(events, output_path)
        else:
            raise ValueError(
                f"Unsupported output format: '{fmt}'. Choose jsonl, csv, or sqlite."
            )

    return events


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.nids.replay",
        description=(
            "Safe offline NIDS dataset replay engine.\n\n"
            "Feeds held-out test records through the two-stage prediction service\n"
            "and writes prediction-only events to JSONL, CSV, or SQLite.\n\n"
            "Safety: never sends packets or opens network connections."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--scenario",
        type=Path,
        default=SCENARIOS_DIR / "mixed_demo.parquet",
        metavar="PATH",
        help="Path to the scenario parquet file (default: data/scenarios/mixed_demo.parquet).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of records to replay (default: all).",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=0.0,
        metavar="SECS",
        help="Simulated inter-event delay in seconds (default: 0).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Output file path.  If omitted, events are printed to stdout.",
    )
    p.add_argument(
        "--format",
        dest="fmt",
        choices=["jsonl", "csv", "sqlite"],
        default="jsonl",
        help="Output format (default: jsonl).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="INT",
        help="Random seed for reproducibility (default: None).",
    )
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = _build_parser()
    args = parser.parse_args()

    predictor = NIDSPredictor()
    meta = predictor.get_metadata()

    print(f"\nNIDS Replay Engine")
    print(f"  Binary model   : {meta['binary_model_artifact']}")
    print(f"  Multiclass     : {meta['multiclass_model_artifact']}")
    print(f"  Threshold      : {meta['binary_threshold']:.2f}  (metadata-sourced)")
    print(f"  Scenario       : {args.scenario}")
    print(f"  Limit          : {args.limit if args.limit else 'all'}")
    print(f"  Interval (s)   : {args.interval}")
    print(f"  Output         : {args.output if args.output else 'stdout'}")
    print(f"  Format         : {args.fmt}")
    print()

    events = run_replay(
        scenario_path=args.scenario,
        limit=args.limit,
        interval=args.interval,
        output_path=args.output,
        fmt=args.fmt,
        seed=args.seed,
        predictor=predictor,
    )

    print(f"Replay complete: {len(events)} event(s) processed.")

    if args.output is None:
        for ev in events:
            print(json.dumps(_event_to_output_dict(ev)))
    else:
        if events:
            print("\nFirst event (JSON):")
            print(json.dumps(_event_to_output_dict(events[0]), indent=2))


if __name__ == "__main__":
    main()
