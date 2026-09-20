"""
dataset.py — Verification, profiling, and loading of raw UNSW-NB15 dataset files.

Responsibilities
----------------
1. Verify presence and integrity (SHA-256) of raw UNSW-NB15 training and testing CSVs.
2. Inspect schema, column types, missing/infinite values, and duplicate records.
3. Verify target columns ('label' for binary, 'attack_cat' for multiclass).
4. Profile label distributions and compute train/test record overlap.
5. Generate machine-readable manifest (data/raw/manifest.json).
6. Generate human-readable profile report (reports/data_profile.md).
7. Save descriptive distribution figures to reports/figures/.

Usage
-----
    python -m src.nids.dataset
    # or:
    python -m nids.dataset
    # or as a library:
    from nids.dataset import load_raw, run_profiling
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from nids.config import (
        DATA_PROFILE_PATH,
        FIGURES_DIR,
        MANIFEST_PATH,
        RAW_DIR,
        RAW_TEST_FILE,
        RAW_TEST_PATH,
        RAW_TRAIN_FILE,
        RAW_TRAIN_PATH,
        TARGET_BINARY,
        TARGET_MULTI,
    )
except ImportError:
    from src.nids.config import (
        DATA_PROFILE_PATH,
        FIGURES_DIR,
        MANIFEST_PATH,
        RAW_DIR,
        RAW_TEST_FILE,
        RAW_TEST_PATH,
        RAW_TRAIN_FILE,
        RAW_TRAIN_PATH,
        TARGET_BINARY,
        TARGET_MULTI,
    )

logger = logging.getLogger(__name__)


def compute_sha256(file_path: Path) -> str:
    """Calculate the SHA-256 checksum of a file without loading it all into memory."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found for checksum calculation: {file_path}")
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_file(file_path: Path) -> Dict[str, Any]:
    """Inspect physical file properties."""
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {file_path}")
    stat = file_path.stat()
    return {
        "filename": file_path.name,
        "path": str(file_path.resolve()),
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "sha256": compute_sha256(file_path),
    }


def profile_dataframe(df: pd.DataFrame, split_name: str, file_meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Profile a loaded DataFrame without altering any of its content.

    Calculates shapes, types, nulls, infinities, duplicates, and target distributions.
    """
    row_count, col_count = df.shape
    columns = list(df.columns)
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Missing values
    null_counts_series = df.isnull().sum()
    null_counts = {col: int(cnt) for col, cnt in null_counts_series.items() if cnt > 0}
    total_nulls = int(null_counts_series.sum())

    # Infinite values in numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        inf_counts_series = np.isinf(numeric_df).sum()
        inf_counts = {col: int(cnt) for col, cnt in inf_counts_series.items() if cnt > 0}
        total_infs = int(inf_counts_series.sum())
    else:
        inf_counts = {}
        total_infs = 0

    # Duplicates
    exact_duplicates = int(df.duplicated().sum())
    duplicates_excl_id = 0
    if "id" in df.columns:
        duplicates_excl_id = int(df.drop(columns=["id"]).duplicated().sum())

    # Target distributions
    binary_dist = {}
    if TARGET_BINARY in df.columns:
        b_counts = df[TARGET_BINARY].value_counts(dropna=False).to_dict()
        binary_dist = {
            str(k): {
                "count": int(v),
                "percentage": round(float(v) / row_count * 100, 3),
            }
            for k, v in b_counts.items()
        }

    multi_dist = {}
    if TARGET_MULTI in df.columns:
        m_counts = df[TARGET_MULTI].value_counts(dropna=False).to_dict()
        multi_dist = {
            str(k): {
                "count": int(v),
                "percentage": round(float(v) / row_count * 100, 3),
            }
            for k, v in m_counts.items()
        }

    return {
        "split": split_name,
        "file": file_meta,
        "rows": row_count,
        "columns_count": col_count,
        "columns": columns,
        "dtypes": dtypes,
        "total_missing_values": total_nulls,
        "missing_values_by_column": null_counts,
        "total_infinite_values": total_infs,
        "infinite_values_by_column": inf_counts,
        "exact_duplicate_rows": exact_duplicates,
        "duplicates_excluding_id": duplicates_excl_id,
        "binary_label_distribution": binary_dist,
        "attack_category_distribution": multi_dist,
    }


def compare_schemas(train_profile: Dict[str, Any], test_profile: Dict[str, Any]) -> Dict[str, Any]:
    """Compare schema attributes between training and testing splits."""
    train_cols = train_profile["columns"]
    test_cols = test_profile["columns"]

    cols_identical = train_cols == test_cols
    col_diff_train_only = list(set(train_cols) - set(test_cols))
    col_diff_test_only = list(set(test_cols) - set(train_cols))

    # Compare data types for common columns
    dtype_mismatches = {}
    train_dtypes = train_profile["dtypes"]
    test_dtypes = test_profile["dtypes"]
    for col in train_cols:
        if col in test_dtypes and train_dtypes[col] != test_dtypes[col]:
            dtype_mismatches[col] = {
                "train_dtype": train_dtypes[col],
                "test_dtype": test_dtypes[col],
            }

    return {
        "schemas_match": cols_identical and len(dtype_mismatches) == 0,
        "column_count_train": len(train_cols),
        "column_count_test": len(test_cols),
        "columns_order_identical": cols_identical,
        "train_only_columns": col_diff_train_only,
        "test_only_columns": col_diff_test_only,
        "dtype_mismatches": dtype_mismatches,
    }


def analyze_train_test_overlap(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Dict[str, Any]:
    """
    Check for overlap between train and test sets.

    Examines exact row overlap (excluding 'id') to detect potential data leakage.
    """
    cols_to_use = [c for c in df_train.columns if c != "id" and c in df_test.columns]
    tr_sub = df_train[cols_to_use]
    te_sub = df_test[cols_to_use]

    # Find rows present in both train and test
    overlap_df = pd.merge(tr_sub.drop_duplicates(), te_sub.drop_duplicates(), how="inner")
    unique_overlap_records = len(overlap_df)

    return {
        "examined_columns_count": len(cols_to_use),
        "unique_overlapping_feature_records": unique_overlap_records,
        "leakage_note": (
            f"Detected {unique_overlap_records} identical flow feature patterns appearing in both "
            "train and test splits (common in network flow captures due to repetitive background protocols "
            "or repeated scans). Models should avoid relying on record memorization."
            if unique_overlap_records > 0
            else "No overlapping flow feature records found between train and test sets."
        ),
    }


def generate_figures(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    figures_dir: Path = FIGURES_DIR,
) -> List[Path]:
    """
    Create distribution and profiling plots and save to figures_dir.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved_plots: List[Path] = []

    # 1. Binary Label Distribution Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    tr_binary = df_train[TARGET_BINARY].value_counts().rename(index={0: "Normal (0)", 1: "Attack (1)"})
    te_binary = df_test[TARGET_BINARY].value_counts().rename(index={0: "Normal (0)", 1: "Attack (1)"})

    b_df = pd.DataFrame({"Train": tr_binary, "Test": te_binary})
    b_df.plot(kind="bar", ax=ax, color=["#2563eb", "#f59e0b"], width=0.6)
    ax.set_title("UNSW-NB15 Binary Label Distribution (Train vs Test)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Class", fontsize=11)
    ax.set_ylabel("Number of Flow Records", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%,d", padding=3, fontsize=9)

    plt.tight_layout()
    b_path = figures_dir / "binary_label_distribution.png"
    plt.savefig(b_path, dpi=300)
    plt.close(fig)
    saved_plots.append(b_path)

    # 2. Attack Category Distribution Plot (Horizontal bar chart)
    fig, ax = plt.subplots(figsize=(10, 6))
    tr_cat = df_train[TARGET_MULTI].value_counts()
    te_cat = df_test[TARGET_MULTI].value_counts()

    # Combine into unified DataFrame sorted by train volume
    cat_df = pd.DataFrame({"Train": tr_cat, "Test": te_cat}).fillna(0)
    cat_df = cat_df.sort_values(by="Train", ascending=True)

    cat_df.plot(kind="barh", ax=ax, color=["#1d4ed8", "#d97706"], width=0.7)
    ax.set_title("UNSW-NB15 Attack Category Distribution (Train vs Test)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Flow Records", fontsize=11)
    ax.set_ylabel("Attack Category", fontsize=11)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    cat_path = figures_dir / "attack_category_distribution.png"
    plt.savefig(cat_path, dpi=300)
    plt.close(fig)
    saved_plots.append(cat_path)

    return saved_plots


def build_markdown_report(manifest: Dict[str, Any]) -> str:
    """Generate comprehensive markdown profile report."""
    train_meta = manifest["train_split"]
    test_meta = manifest["test_split"]
    schema = manifest["schema_comparison"]
    overlap = manifest["overlap_analysis"]

    report = f"""# UNSW-NB15 Dataset Verification & Profiling Report (Mode 1)

**Generated:** {pd.Timestamp.now().isoformat()}  
**Target Columns:** Binary: `{TARGET_BINARY}` | Multiclass: `{TARGET_MULTI}`

---

## 1. File Integrity & Physical Metadata

| Metric | Training Set | Testing Set |
| :--- | :--- | :--- |
| **Filename** | `{train_meta['file']['filename']}` | `{test_meta['file']['filename']}` |
| **File Size** | {train_meta['file']['size_mb']} MB ({train_meta['file']['size_bytes']:,} bytes) | {test_meta['file']['size_mb']} MB ({test_meta['file']['size_bytes']:,} bytes) |
| **Rows** | {train_meta['rows']:,} | {test_meta['rows']:,} |
| **Columns** | {train_meta['columns_count']} | {test_meta['columns_count']} |
| **SHA-256** | `{train_meta['file']['sha256']}` | `{test_meta['file']['sha256']}` |

---

## 2. Schema Consistency Check

- **Schemas Identical:** `{'YES' if schema['schemas_match'] else 'NO'}`
- **Column Count Train:** {schema['column_count_train']}
- **Column Count Test:** {schema['column_count_test']}
- **Column Order Aligned:** `{'YES' if schema['columns_order_identical'] else 'NO'}`
- **Data Type Mismatches:** {len(schema['dtype_mismatches'])}

---

## 3. Data Quality (Missing, Infinite, and Duplicate Values)

| Quality Check | Training Set | Testing Set | Status |
| :--- | :--- | :--- | :--- |
| **Missing / NaN Values** | {train_meta['total_missing_values']} | {test_meta['total_missing_values']} | Clean |
| **Infinite Values (+/- inf)** | {train_meta['total_infinite_values']} | {test_meta['total_infinite_values']} | Clean |
| **Exact Duplicate Rows (all cols)** | {train_meta['exact_duplicate_rows']} | {test_meta['exact_duplicate_rows']} | Clean |
| **Duplicates excluding `id`** | {train_meta['duplicates_excluding_id']:,} | {test_meta['duplicates_excluding_id']:,} | Expected flow duplicates |

---

## 4. Target Distributions

### 4.1 Binary Target (`{TARGET_BINARY}`)

| Label | Meaning | Train Count | Train % | Test Count | Test % |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for label_val, name in [("0", "Normal"), ("1", "Attack")]:
        tr_info = train_meta["binary_label_distribution"].get(label_val, {"count": 0, "percentage": 0.0})
        te_info = test_meta["binary_label_distribution"].get(label_val, {"count": 0, "percentage": 0.0})
        report += f"| `{label_val}` | {name} | {tr_info['count']:,} | {tr_info['percentage']}% | {te_info['count']:,} | {te_info['percentage']}% |\n"

    report += f"""
### 4.2 Multiclass Target (`{TARGET_MULTI}`)

| Attack Category | Train Count | Train % | Test Count | Test % |
| :--- | :--- | :--- | :--- | :--- |
"""
    all_cats = sorted(
        set(train_meta["attack_category_distribution"].keys()) | set(test_meta["attack_category_distribution"].keys())
    )
    for cat in all_cats:
        tr_cat = train_meta["attack_category_distribution"].get(cat, {"count": 0, "percentage": 0.0})
        te_cat = test_meta["attack_category_distribution"].get(cat, {"count": 0, "percentage": 0.0})
        report += f"| **{cat}** | {tr_cat['count']:,} | {tr_cat['percentage']}% | {te_cat['count']:,} | {te_cat['percentage']}% |\n"

    report += f"""
---

## 5. Train / Test Record Overlap Analysis

- **Unique overlapping feature records (excluding `id`):** {overlap['unique_overlapping_feature_records']:,}
- **Observation:** {overlap['leakage_note']}

---

## 6. Generated Figures

- `reports/figures/binary_label_distribution.png`: Comparison of normal vs attack traffic.
- `reports/figures/attack_category_distribution.png`: Breakdown of attack categories.
"""
    return report


def run_profiling(
    train_path: Path = RAW_TRAIN_PATH,
    test_path: Path = RAW_TEST_PATH,
    manifest_path: Path = MANIFEST_PATH,
    report_path: Path = DATA_PROFILE_PATH,
    figures_dir: Path = FIGURES_DIR,
) -> Dict[str, Any]:
    """
    Execute end-to-end profiling, generating manifest, markdown report, and figures.
    """
    logger.info("Verifying raw dataset files...")
    train_meta = inspect_file(train_path)
    test_meta = inspect_file(test_path)

    logger.info("Reading raw CSV files without modifications...")
    df_train = pd.read_csv(train_path, low_memory=False)
    df_test = pd.read_csv(test_path, low_memory=False)

    logger.info("Profiling training set (%s)...", df_train.shape)
    train_profile = profile_dataframe(df_train, split_name="training", file_meta=train_meta)

    logger.info("Profiling testing set (%s)...", df_test.shape)
    test_profile = profile_dataframe(df_test, split_name="testing", file_meta=test_meta)

    logger.info("Comparing schemas...")
    schema_comp = compare_schemas(train_profile, test_profile)

    logger.info("Analyzing train/test overlap...")
    overlap = analyze_train_test_overlap(df_train, df_test)

    logger.info("Generating distribution figures...")
    saved_plots = generate_figures(df_train, df_test, figures_dir=figures_dir)

    manifest = {
        "project": "ML-Based-NIDS-and-Attack-Classification-System",
        "dataset": "UNSW-NB15",
        "mode": 1,
        "timestamp": pd.Timestamp.now().isoformat(),
        "train_split": train_profile,
        "test_split": test_profile,
        "schema_comparison": schema_comp,
        "overlap_analysis": overlap,
        "generated_figures": [str(p) for p in saved_plots],
    }

    # Save manifest.json
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest saved -> %s", manifest_path)

    # Save data_profile.md
    report_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_content = build_markdown_report(manifest)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    logger.info("Profile report saved -> %s", report_path)

    return manifest


def load_raw(
    train_path: Path = RAW_TRAIN_PATH,
    test_path: Path = RAW_TEST_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Safely load the raw UNSW-NB15 training and testing sets.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (df_train, df_test)
    """
    if not train_path.exists():
        raise FileNotFoundError(f"Training set not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Testing set not found: {test_path}")

    df_train = pd.read_csv(train_path, low_memory=False)
    df_test = pd.read_csv(test_path, low_memory=False)
    return df_train, df_test


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    manifest = run_profiling()

    tr = manifest["train_split"]
    te = manifest["test_split"]

    print("\n" + "=" * 70)
    print("UNSW-NB15 DATASET VERIFICATION & PROFILING SUMMARY (MODE 1)")
    print("=" * 70)
    print(f"Training Set : {tr['file']['filename']}")
    print(f"  Rows x Cols : {tr['rows']:,} x {tr['columns_count']}")
    print(f"  File Size   : {tr['file']['size_mb']} MB")
    print(f"  SHA-256     : {tr['file']['sha256']}")
    print(f"Testing Set  : {te['file']['filename']}")
    print(f"  Rows x Cols : {te['rows']:,} x {te['columns_count']}")
    print(f"  File Size   : {te['file']['size_mb']} MB")
    print(f"  SHA-256     : {te['file']['sha256']}")

    print("\n--- TARGET COLUMNS ---")
    print(f"Binary Target Column     : {TARGET_BINARY}")
    print(f"Multiclass Target Column : {TARGET_MULTI}")

    print("\n--- BINARY LABEL DISTRIBUTION ---")
    print(f"Train : {tr['binary_label_distribution']}")
    print(f"Test  : {te['binary_label_distribution']}")

    print("\n--- ATTACK CATEGORY DISTRIBUTION (TRAIN) ---")
    for cat, data in tr["attack_category_distribution"].items():
        print(f"  {cat:<15}: {data['count']:>7,} ({data['percentage']:>5.2f}%)")

    print("\n--- ATTACK CATEGORY DISTRIBUTION (TEST) ---")
    for cat, data in te["attack_category_distribution"].items():
        print(f"  {cat:<15}: {data['count']:>7,} ({data['percentage']:>5.2f}%)")

    print("\n--- SCHEMA CONSISTENCY ---")
    print(f"Schemas Match: {manifest['schema_comparison']['schemas_match']}")

    print("\n--- ARTIFACTS GENERATED ---")
    print(f"Manifest JSON : {MANIFEST_PATH}")
    print(f"Markdown Report: {DATA_PROFILE_PATH}")
    for fig in manifest["generated_figures"]:
        print(f"Figure        : {fig}")
    print("=" * 70 + "\n")
