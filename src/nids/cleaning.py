"""
cleaning.py — Data quality cleaning and normalization for UNSW-NB15 (Mode 1).

Responsibilities
----------------
1. Load raw UNSW-NB15 training and testing CSVs separately (never modifying raw files).
2. Replace +/- infinity with NaN values defensively across numeric columns.
3. Strip leading and trailing whitespace from string/categorical columns.
4. Normalize and validate `label` (0 = Normal, 1 = Attack).
5. Normalize and validate `attack_cat` (strip whitespace, set to "Normal" where label == 0,
   and strictly preserve "Backdoor" spelling).
6. Define model feature columns by excluding identifier and target columns (id, label, attack_cat).
7. Preserve all flow rows (no silent row deletion; no removal of repeated flow records).
8. Persist cleaned splits as Parquet files in data/interim/.
9. Output before-and-after audit reports (JSON summary and Markdown report).

Usage
-----
    python -m src.nids.cleaning
    # or:
    python -m nids.cleaning
    # or as a module:
    from nids.cleaning import clean_split, run_cleaning, get_feature_columns
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from nids.config import (
        CLEAN_TEST_PARQUET,
        CLEAN_TRAIN_PARQUET,
        CLEANING_REPORT_MD,
        CLEANING_SUMMARY_JSON,
        ID_COLUMN,
        NON_FEATURE_COLUMNS,
        RAW_TEST_PATH,
        RAW_TRAIN_PATH,
        TARGET_BINARY,
        TARGET_MULTI,
    )
except ImportError:
    from src.nids.config import (
        CLEAN_TEST_PARQUET,
        CLEAN_TRAIN_PARQUET,
        CLEANING_REPORT_MD,
        CLEANING_SUMMARY_JSON,
        ID_COLUMN,
        NON_FEATURE_COLUMNS,
        RAW_TEST_PATH,
        RAW_TRAIN_PATH,
        TARGET_BINARY,
        TARGET_MULTI,
    )

logger = logging.getLogger(__name__)


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Return list of model feature column names by excluding non-feature columns.

    Excludes:
      - id (identifier)
      - label (binary target)
      - attack_cat (multiclass target)
    """
    excluded = set(NON_FEATURE_COLUMNS)
    return [col for col in df.columns if col not in excluded]


def summarize_split(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute statistical and schema summary of a split for before/after comparison.
    """
    row_count, col_count = df.shape

    # Missing and infinite values
    null_counts_series = df.isnull().sum()
    null_counts = {col: int(cnt) for col, cnt in null_counts_series.items() if cnt > 0}
    total_nulls = int(null_counts_series.sum())

    numeric_cols = df.select_dtypes(include=[np.number])
    if not numeric_cols.empty:
        inf_counts_series = np.isinf(numeric_cols).sum()
        inf_counts = {col: int(cnt) for col, cnt in inf_counts_series.items() if cnt > 0}
        total_infs = int(inf_counts_series.sum())
    else:
        inf_counts = {}
        total_infs = 0

    # Duplicates
    exact_duplicates = int(df.duplicated().sum())

    # Target distributions
    binary_dist = {}
    if TARGET_BINARY in df.columns:
        b_counts = df[TARGET_BINARY].value_counts(dropna=False).to_dict()
        binary_dist = {
            str(k): {
                "count": int(v),
                "percentage": round(float(v) / row_count * 100, 3) if row_count > 0 else 0.0,
            }
            for k, v in b_counts.items()
        }

    multi_dist = {}
    if TARGET_MULTI in df.columns:
        m_counts = df[TARGET_MULTI].value_counts(dropna=False).to_dict()
        multi_dist = {
            str(k): {
                "count": int(v),
                "percentage": round(float(v) / row_count * 100, 3) if row_count > 0 else 0.0,
            }
            for k, v in m_counts.items()
        }

    return {
        "row_count": row_count,
        "column_count": col_count,
        "total_missing_values": total_nulls,
        "missing_values_by_column": null_counts,
        "total_infinite_values": total_infs,
        "infinite_values_by_column": inf_counts,
        "exact_duplicate_rows": exact_duplicates,
        "binary_label_distribution": binary_dist,
        "attack_category_distribution": multi_dist,
    }


def clean_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply quality cleaning and defensive normalizations to a dataset split.

    Steps
    -----
    1. Create a defensive copy.
    2. Replace positive and negative infinity with NaN in numeric columns.
    3. Strip leading/trailing whitespace from string/object columns.
    4. Normalize `label` to standard integer values (0 = Normal, 1 = Attack).
    5. Normalize `attack_cat`:
       - Strip whitespace.
       - Enforce 'Normal' for all records where label == 0.
       - Strictly preserve 'Backdoor' category (do not rename to 'Backdoors').
    6. Ensure row count remains identical (no rows dropped).
    """
    initial_rows = len(df)
    df_clean = df.copy()

    # 1. Replace +/- inf with NaN in numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df_clean[numeric_cols] = df_clean[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # 2. Normalize binary label (ensure integer 0/1)
    if TARGET_BINARY in df_clean.columns:
        df_clean[TARGET_BINARY] = pd.to_numeric(df_clean[TARGET_BINARY], errors="raise").astype(int)
        invalid_labels = set(df_clean[TARGET_BINARY].unique()) - {0, 1}
        if invalid_labels:
            raise ValueError(f"Unexpected binary labels found: {invalid_labels}")

    # 3. Strip whitespace from string/categorical columns (excluding label)
    str_cols = [c for c in df_clean.select_dtypes(include=["object", "string"]).columns if c != TARGET_BINARY]
    for col in str_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip()

    # 4. Normalize attack category
    if TARGET_MULTI in df_clean.columns:
        df_clean[TARGET_MULTI] = df_clean[TARGET_MULTI].astype(str).str.strip()

        # Enforce consistency: where label == 0, attack_cat must be "Normal"
        if TARGET_BINARY in df_clean.columns:
            df_clean.loc[df_clean[TARGET_BINARY] == 0, TARGET_MULTI] = "Normal"

        # Explicitly verify 'Backdoor' is intact if present
        if "Backdoor" not in df_clean[TARGET_MULTI].unique() and "Backdoors" in df_clean[TARGET_MULTI].unique():
            df_clean[TARGET_MULTI] = df_clean[TARGET_MULTI].replace({"Backdoors": "Backdoor"})

    # 5. Guard against accidental row deletion
    if len(df_clean) != initial_rows:
        raise RuntimeError(
            f"Row count mismatch during cleaning! Initial: {initial_rows}, Cleaned: {len(df_clean)}"
        )

    return df_clean


def build_markdown_report(summary: Dict[str, Any]) -> str:
    """Generate comprehensive before-and-after markdown report."""
    tr_b = summary["training_split"]["before"]
    tr_a = summary["training_split"]["after"]
    te_b = summary["testing_split"]["before"]
    te_a = summary["testing_split"]["after"]
    features = summary["model_features"]

    report = f"""# UNSW-NB15 Data Cleaning Report (Mode 1)

**Generated:** {pd.Timestamp.now().isoformat()}  
**Target Columns:** Binary: `{TARGET_BINARY}` | Multiclass: `{TARGET_MULTI}`  
**Non-Feature Columns Excluded:** `{', '.join(NON_FEATURE_COLUMNS)}` ({len(NON_FEATURE_COLUMNS)} columns)  
**Model Feature Columns Defined:** {len(features)} columns

---

## 1. Before vs After Summary

| Metric | Training (Before) | Training (After) | Testing (Before) | Testing (After) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Row Count** | {tr_b['row_count']:,} | {tr_a['row_count']:,} | {te_b['row_count']:,} | {te_a['row_count']:,} | Exact Preserved |
| **Column Count** | {tr_b['column_count']} | {tr_a['column_count']} | {te_b['column_count']} | {te_a['column_count']} | Preserved (45 cols) |
| **Missing / NaN Values** | {tr_b['total_missing_values']} | {tr_a['total_missing_values']} | {te_b['total_missing_values']} | {te_a['total_missing_values']} | Clean |
| **Infinite Values (+/- inf)** | {tr_b['total_infinite_values']} | {tr_a['total_infinite_values']} | {te_b['total_infinite_values']} | {te_a['total_infinite_values']} | Replaced |
| **Exact Duplicate Rows** | {tr_b['exact_duplicate_rows']} | {tr_a['exact_duplicate_rows']} | {te_b['exact_duplicate_rows']} | {te_a['exact_duplicate_rows']} | Preserved |

---

## 2. Target Distributions (Cleaned)

### 2.1 Binary Target (`{TARGET_BINARY}`)

| Label | Meaning | Training Count | Training % | Testing Count | Testing % |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for label_val, name in [("0", "Normal"), ("1", "Attack")]:
        tr_info = tr_a["binary_label_distribution"].get(label_val, {"count": 0, "percentage": 0.0})
        te_info = te_a["binary_label_distribution"].get(label_val, {"count": 0, "percentage": 0.0})
        report += f"| `{label_val}` | {name} | {tr_info['count']:,} | {tr_info['percentage']}% | {te_info['count']:,} | {te_info['percentage']}% |\n"

    report += f"""
### 2.2 Multiclass Target (`{TARGET_MULTI}`)

| Attack Category | Training Count | Training % | Testing Count | Testing % |
| :--- | :--- | :--- | :--- | :--- |
"""
    all_cats = sorted(
        set(tr_a["attack_category_distribution"].keys()) | set(te_a["attack_category_distribution"].keys())
    )
    for cat in all_cats:
        tr_cat = tr_a["attack_category_distribution"].get(cat, {"count": 0, "percentage": 0.0})
        te_cat = te_a["attack_category_distribution"].get(cat, {"count": 0, "percentage": 0.0})
        report += f"| **{cat}** | {tr_cat['count']:,} | {tr_cat['percentage']}% | {te_cat['count']:,} | {te_cat['percentage']}% |\n"

    report += f"""
---

## 3. Model Feature Column Specification

The following **{len(features)}** columns are designated as input features for subsequent encoding, scaling, and classification:

`{', '.join(features)}`

- Excluded `{ID_COLUMN}` to prevent spurious correlation with record index.
- Excluded `{TARGET_BINARY}` and `{TARGET_MULTI}` from input space to prevent target leakage.

---

## 4. Persisted Interim Artifacts

- Training Clean Parquet: `{summary['artifacts']['clean_train_parquet']}`
- Testing Clean Parquet: `{summary['artifacts']['clean_test_parquet']}`
- Summary Audit JSON: `{summary['artifacts']['summary_json']}`
- Audit Report Markdown: `{summary['artifacts']['report_md']}`
"""
    return report


def run_cleaning(
    raw_train_path: Path = RAW_TRAIN_PATH,
    raw_test_path: Path = RAW_TEST_PATH,
    clean_train_path: Path = CLEAN_TRAIN_PARQUET,
    clean_test_path: Path = CLEAN_TEST_PARQUET,
    summary_json_path: Path = CLEANING_SUMMARY_JSON,
    report_md_path: Path = CLEANING_REPORT_MD,
) -> Dict[str, Any]:
    """
    Execute end-to-end dataset cleaning and output audited Parquet and reports.
    """
    logger.info("Loading raw training file: %s", raw_train_path)
    df_train_raw = pd.read_csv(raw_train_path, low_memory=False)

    logger.info("Loading raw testing file: %s", raw_test_path)
    df_test_raw = pd.read_csv(raw_test_path, low_memory=False)

    logger.info("Summarizing raw splits (before cleaning)...")
    train_before = summarize_split(df_train_raw)
    test_before = summarize_split(df_test_raw)

    logger.info("Applying cleaning and normalization to training split...")
    df_train_clean = clean_split(df_train_raw)

    logger.info("Applying cleaning and normalization to testing split...")
    df_test_clean = clean_split(df_test_raw)

    logger.info("Summarizing cleaned splits (after cleaning)...")
    train_after = summarize_split(df_train_clean)
    test_after = summarize_split(df_test_clean)

    feature_cols = get_feature_columns(df_train_clean)
    logger.info("Identified %d model feature columns.", len(feature_cols))

    # Persist cleaned Parquet files
    clean_train_path.parent.mkdir(parents=True, exist_ok=True)
    clean_test_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Writing cleaned training split -> %s", clean_train_path)
    df_train_clean.to_parquet(clean_train_path, index=False, engine="pyarrow")

    logger.info("Writing cleaned testing split -> %s", clean_test_path)
    df_test_clean.to_parquet(clean_test_path, index=False, engine="pyarrow")

    summary = {
        "project": "ML-Based-NIDS-and-Attack-Classification-System",
        "dataset": "UNSW-NB15",
        "phase": "cleaning",
        "mode": 1,
        "timestamp": pd.Timestamp.now().isoformat(),
        "model_features_count": len(feature_cols),
        "model_features": feature_cols,
        "excluded_columns": NON_FEATURE_COLUMNS,
        "training_split": {
            "before": train_before,
            "after": train_after,
        },
        "testing_split": {
            "before": test_before,
            "after": test_after,
        },
        "artifacts": {
            "clean_train_parquet": str(clean_train_path.resolve()),
            "clean_test_parquet": str(clean_test_path.resolve()),
            "summary_json": str(summary_json_path.resolve()),
            "report_md": str(report_md_path.resolve()),
        },
    }

    # Save summary JSON
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Cleaning summary saved -> %s", summary_json_path)

    # Save markdown report
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    report_md_content = build_markdown_report(summary)
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(report_md_content)
    logger.info("Cleaning report saved -> %s", report_md_path)

    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    summary = run_cleaning()

    tr = summary["training_split"]["after"]
    te = summary["testing_split"]["after"]

    print("\n" + "=" * 70)
    print("UNSW-NB15 CLEANING SUMMARY (MODE 1)")
    print("=" * 70)
    print(f"Training Rows Preserved : {tr['row_count']:,} (Columns: {tr['column_count']})")
    print(f"Testing Rows Preserved  : {te['row_count']:,} (Columns: {te['column_count']})")
    print(f"Model Feature Columns   : {summary['model_features_count']} columns")
    print(f"Excluded Columns        : {summary['excluded_columns']}")
    print(f"Missing / Infinite In Train : {tr['total_missing_values']} / {tr['total_infinite_values']}")
    print(f"Missing / Infinite In Test  : {te['total_missing_values']} / {te['total_infinite_values']}")
    print("\n--- BINARY LABEL DISTRIBUTION ---")
    print(f"Train : {tr['binary_label_distribution']}")
    print(f"Test  : {te['binary_label_distribution']}")
    print("\n--- ATTACK CATEGORIES IN TRAIN ---")
    for cat, data in tr["attack_category_distribution"].items():
        print(f"  {cat:<15}: {data['count']:>7,} ({data['percentage']:>5.2f}%)")
    print("\n--- ARTIFACTS GENERATED ---")
    for k, v in summary["artifacts"].items():
        print(f"  {k:<20}: {v}")
    print("=" * 70 + "\n")
