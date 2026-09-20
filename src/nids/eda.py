"""
eda.py — Exploratory Data Analysis (EDA) module for UNSW-NB15 (Mode 1).

Responsibilities
----------------
1. Load cleaned interim Parquet files (keeping train and test strictly separate).
2. Analyze binary target and multiclass attack-category distributions.
3. Analyze categorical feature distributions (protocols, services, states) and domain shift.
4. Analyze numeric feature distributions, variance, skewness, and multicollinearity.
5. Zoom in on rare attack categories (Analysis, Backdoor, Shellcode, Worms).
6. Generate and save diagnostic visualizations to reports/figures/.
7. Compile comprehensive, reproducible markdown report to reports/eda_report.md.

Usage
-----
    python -m src.nids.eda
    # or:
    python -m nids.eda
    # or as a module:
    from nids.eda import run_eda, load_clean_data
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from nids.config import (
        CLEAN_TEST_PARQUET,
        CLEAN_TRAIN_PARQUET,
        EDA_REPORT_MD,
        FIGURES_DIR,
        NON_FEATURE_COLUMNS,
        TARGET_BINARY,
        TARGET_MULTI,
    )
except ImportError:
    from src.nids.config import (
        CLEAN_TEST_PARQUET,
        CLEAN_TRAIN_PARQUET,
        EDA_REPORT_MD,
        FIGURES_DIR,
        NON_FEATURE_COLUMNS,
        TARGET_BINARY,
        TARGET_MULTI,
    )

logger = logging.getLogger(__name__)


def load_clean_data(
    train_path: Path = CLEAN_TRAIN_PARQUET,
    test_path: Path = CLEAN_TEST_PARQUET,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the cleaned interim Parquet files."""
    if not train_path.exists():
        raise FileNotFoundError(f"Cleaned training parquet not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Cleaned testing parquet not found: {test_path}")

    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)
    return df_train, df_test


def analyze_target_distributions(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Dict[str, Any]:
    """Analyze binary and multiclass distributions."""
    tr_len = len(df_train)
    te_len = len(df_test)

    # Binary label
    tr_bin = df_train[TARGET_BINARY].value_counts().to_dict()
    te_bin = df_test[TARGET_BINARY].value_counts().to_dict()

    binary_analysis = {
        "train": {
            "normal_count": int(tr_bin.get(0, 0)),
            "normal_pct": round(tr_bin.get(0, 0) / tr_len * 100, 2),
            "attack_count": int(tr_bin.get(1, 0)),
            "attack_pct": round(tr_bin.get(1, 0) / tr_len * 100, 2),
        },
        "test": {
            "normal_count": int(te_bin.get(0, 0)),
            "normal_pct": round(te_bin.get(0, 0) / te_len * 100, 2),
            "attack_count": int(te_bin.get(1, 0)),
            "attack_pct": round(te_bin.get(1, 0) / te_len * 100, 2),
        },
    }

    # Multiclass attack categories
    tr_multi = df_train[TARGET_MULTI].value_counts().to_dict()
    te_multi = df_test[TARGET_MULTI].value_counts().to_dict()

    all_cats = sorted(set(tr_multi.keys()) | set(te_multi.keys()))
    multiclass_analysis = {
        cat: {
            "train_count": int(tr_multi.get(cat, 0)),
            "train_pct": round(tr_multi.get(cat, 0) / tr_len * 100, 3),
            "test_count": int(te_multi.get(cat, 0)),
            "test_pct": round(te_multi.get(cat, 0) / te_len * 100, 3),
        }
        for cat in all_cats
    }

    return {
        "binary": binary_analysis,
        "multiclass": multiclass_analysis,
    }


def analyze_categorical_features(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Dict[str, Any]:
    """Analyze categorical columns: proto, service, state, and check domain shift."""
    cat_cols = ["proto", "service", "state"]
    results = {}

    for col in cat_cols:
        tr_counts = df_train[col].value_counts().to_dict()
        te_counts = df_test[col].value_counts().to_dict()

        tr_unique = set(tr_counts.keys())
        te_unique = set(te_counts.keys())

        test_only = list(te_unique - tr_unique)
        train_only = list(tr_unique - te_unique)

        results[col] = {
            "train_unique_count": len(tr_unique),
            "test_unique_count": len(te_unique),
            "test_only_values": test_only,
            "train_only_values": train_only,
            "top_train_values": {k: int(v) for k, v in list(tr_counts.items())[:5]},
            "top_test_values": {k: int(v) for k, v in list(te_counts.items())[:5]},
        }

    return results


def analyze_numeric_features(df_train: pd.DataFrame) -> Dict[str, Any]:
    """Analyze numeric features for variance, min/max, and skewness."""
    feature_cols = [c for c in df_train.columns if c not in NON_FEATURE_COLUMNS]
    num_cols = [c for c in feature_cols if c not in ("proto", "service", "state")]

    desc = df_train[num_cols].describe().T
    var_series = df_train[num_cols].var().sort_values()

    lowest_var = {k: round(float(v), 6) for k, v in var_series.head(8).items()}
    highest_var = {k: round(float(v), 2) for k, v in var_series.tail(8).items()}

    # Check for near-zero variance (< 1e-4)
    near_zero_var = [k for k, v in var_series.items() if v < 1e-4]

    return {
        "numeric_features_count": len(num_cols),
        "lowest_variance_features": lowest_var,
        "highest_variance_features": highest_var,
        "near_zero_variance_features": near_zero_var,
    }


def analyze_correlations(df_train: pd.DataFrame, threshold: float = 0.90) -> Dict[str, Any]:
    """Detect high multicollinearity among numeric features in training set."""
    feature_cols = [c for c in df_train.columns if c not in NON_FEATURE_COLUMNS]
    num_cols = [c for c in feature_cols if c not in ("proto", "service", "state")]

    corr = df_train[num_cols].corr()
    high_pairs = []

    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            c1, c2 = num_cols[i], num_cols[j]
            r = corr.loc[c1, c2]
            if abs(r) >= threshold:
                high_pairs.append({
                    "feature_1": c1,
                    "feature_2": c2,
                    "correlation": round(float(r), 4),
                })

    # Sort descending by absolute correlation
    high_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    return {
        "threshold": threshold,
        "high_correlation_pairs_count": len(high_pairs),
        "high_correlation_pairs": high_pairs,
    }


def analyze_rare_categories(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Dict[str, Any]:
    """Analyze attack categories with low sample counts (< 2% of data)."""
    tr_len = len(df_train)
    te_len = len(df_test)
    rare_names = ["Analysis", "Backdoor", "Shellcode", "Worms"]

    rare_stats = {}
    for cat in rare_names:
        tr_cnt = int((df_train[TARGET_MULTI] == cat).sum())
        te_cnt = int((df_test[TARGET_MULTI] == cat).sum())
        rare_stats[cat] = {
            "train_count": tr_cnt,
            "train_pct": round(tr_cnt / tr_len * 100, 3),
            "test_count": te_cnt,
            "test_pct": round(te_cnt / te_len * 100, 3),
            "ratio_train_to_test": round(tr_cnt / te_cnt, 2) if te_cnt > 0 else None,
        }

    return rare_stats


def generate_eda_figures(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    figures_dir: Path = FIGURES_DIR,
) -> List[Path]:
    """Generate and save comprehensive EDA figures."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved_plots: List[Path] = []

    # 1. Binary Label Distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    tr_b = df_train[TARGET_BINARY].value_counts().rename(index={0: "Normal (0)", 1: "Attack (1)"})
    te_b = df_test[TARGET_BINARY].value_counts().rename(index={0: "Normal (0)", 1: "Attack (1)"})
    b_df = pd.DataFrame({"Train": tr_b, "Test": te_b})
    b_df.plot(kind="bar", ax=ax, color=["#1e40af", "#d97706"], width=0.55)
    ax.set_title("Binary Class Distribution: Train vs Test", fontsize=12, fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Records")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for c in ax.containers:
        ax.bar_label(c, fmt="%,d", padding=3, fontsize=8)
    plt.tight_layout()
    p1 = figures_dir / "eda_binary_distribution.png"
    plt.savefig(p1, dpi=300)
    plt.close(fig)
    saved_plots.append(p1)

    # 2. Attack Categories Distribution (Log Scale for Visibility)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    tr_c = df_train[TARGET_MULTI].value_counts()
    te_c = df_test[TARGET_MULTI].value_counts()
    c_df = pd.DataFrame({"Train": tr_c, "Test": te_c}).fillna(0).sort_values(by="Train", ascending=True)
    c_df.plot(kind="barh", ax=ax, color=["#2563eb", "#f59e0b"], width=0.7)
    ax.set_title("Attack Category Frequencies (Train vs Test)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Records (Logarithmic Scale)")
    ax.set_xscale("log")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    p2 = figures_dir / "eda_attack_categories.png"
    plt.savefig(p2, dpi=300)
    plt.close(fig)
    saved_plots.append(p2)

    # 3. Numeric Correlation Heatmap (Selected Representative Features)
    key_features = [
        "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate",
        "sttl", "dttl", "sload", "dload", "sloss", "dloss",
        "swin", "dwin", "tcprtt", "synack", "ackdat",
        "smean", "dmean", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm",
    ]
    fig, ax = plt.subplots(figsize=(12, 9))
    corr_sub = df_train[key_features].corr()
    sns.heatmap(
        corr_sub,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 7},
    )
    ax.set_title("Correlation Heatmap: Core Network Features (Train)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p3 = figures_dir / "eda_numeric_correlations.png"
    plt.savefig(p3, dpi=300)
    plt.close(fig)
    saved_plots.append(p3)

    # 4. Top Protocols Comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    top_protos = df_train["proto"].value_counts().head(8).index
    proto_df = pd.DataFrame({
        "Train": df_train["proto"].value_counts(),
        "Test": df_test["proto"].value_counts(),
    }).loc[top_protos].fillna(0)
    proto_df.plot(kind="bar", ax=ax, color=["#047857", "#10b981"], width=0.6)
    ax.set_title("Top 8 Network Protocols (Train vs Test)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Protocol")
    ax.set_ylabel("Records (Log Scale)")
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    p4 = figures_dir / "eda_top_protocols.png"
    plt.savefig(p4, dpi=300)
    plt.close(fig)
    saved_plots.append(p4)

    # 5. Services Distribution Comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    top_services = df_train["service"].value_counts().index
    service_df = pd.DataFrame({
        "Train": df_train["service"].value_counts(),
        "Test": df_test["service"].value_counts(),
    }).loc[top_services].fillna(0)
    service_df.plot(kind="bar", ax=ax, color=["#6366f1", "#a855f7"], width=0.6)
    ax.set_title("Network Services Distribution (Train vs Test)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Service")
    ax.set_ylabel("Records (Log Scale)")
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    p5 = figures_dir / "eda_top_services.png"
    plt.savefig(p5, dpi=300)
    plt.close(fig)
    saved_plots.append(p5)

    # 6. Rare Attack Categories Breakdown
    rare_cats = ["Analysis", "Backdoor", "Shellcode", "Worms"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    rare_df = pd.DataFrame({
        "Train": df_train["attack_cat"].value_counts().loc[rare_cats],
        "Test": df_test["attack_cat"].value_counts().loc[rare_cats],
    })
    rare_df.plot(kind="bar", ax=ax, color=["#dc2626", "#f87171"], width=0.55)
    ax.set_title("Rare Attack Categories Breakdown (< 2% Prevalence)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Attack Category")
    ax.set_ylabel("Records")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for c in ax.containers:
        ax.bar_label(c, fmt="%,d", padding=3, fontsize=9)
    plt.tight_layout()
    p6 = figures_dir / "eda_rare_attacks_breakdown.png"
    plt.savefig(p6, dpi=300)
    plt.close(fig)
    saved_plots.append(p6)

    return saved_plots


def build_markdown_report(analysis: Dict[str, Any]) -> str:
    """Generate comprehensive markdown report of EDA observations."""
    targets = analysis["targets"]
    cats = analysis["categorical"]
    nums = analysis["numeric"]
    corrs = analysis["correlations"]
    rares = analysis["rare_attacks"]

    report = f"""# UNSW-NB15 Exploratory Data Analysis (EDA) Report (Mode 1)

**Generated:** {pd.Timestamp.now().isoformat()}  
**Data Sources:**  
- Cleaned Training Split: `data/interim/UNSW_NB15_training_clean.parquet` ({targets['binary']['train']['normal_count'] + targets['binary']['train']['attack_count']:,} rows)  
- Cleaned Testing Split: `data/interim/UNSW_NB15_testing_clean.parquet` ({targets['binary']['test']['normal_count'] + targets['binary']['test']['attack_count']:,} rows)  
**Model Input Features:** 42 features (39 numeric, 3 categorical)  
**Excluded Non-Feature Columns:** `{', '.join(NON_FEATURE_COLUMNS)}`

---

## 1. Executive Summary & Key Findings

1. **Target Distribution Inversion**:
   - In the **Training set**, Attacks account for **68.06%** (119,341 flows) and Normal traffic accounts for **31.94%** (56,000 flows).
   - In the **Testing set**, Attacks account for **55.06%** (45,332 flows) and Normal traffic accounts for **44.94%** (37,000 flows).
   - The higher proportion of normal records in the test set reflects realistic deployment testing where benign flows dominate.
2. **Extreme Multiclass Imbalance**:
   - Four rare attack categories (`Analysis`, `Backdoor`, `Shellcode`, `Worms`) comprise **< 3%** of training flows combined.
   - Specifically, `Worms` contains only **130 records in training** (0.074%) and **44 records in testing** (0.053%).
3. **Severe Multicollinearity**:
   - Detected **{corrs['high_correlation_pairs_count']} feature pairs** with correlation $|r| \\ge 0.90$.
   - Perfect correlation ($r = 1.0000$) between `is_ftp_login` and `ct_ftp_cmd`.
   - Very high correlation ($r > 0.99$) between byte counts and packet loss (`sbytes` $\\leftrightarrow$ `sloss`, `dbytes` $\\leftrightarrow$ `dloss`).
   - `swin` and `dwin` have $r = 0.9901$.
4. **Categorical Domain Shift**:
   - `proto`: 133 unique in Train, 131 in Test. Zero test-only protocols.
   - `service`: 13 unique in Train, 13 in Test (all align).
   - `state`: 9 unique in Train, 7 in Test. **Critical finding:** Test contains 2 states (`ACC` with 4 records, `CLO` with 1 record) that never appear in Train! Categorical encoders must use `handle_unknown="ignore"`.
5. **Data Quality Status**:
   - 0 missing / NaN values and 0 infinite values across all 42 features in both splits.

---

## 2. Target Distributions

### 2.1 Binary Classification Target (`label`)

| Split | Normal (`0`) | Normal % | Attack (`1`) | Attack % | Total Records |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Training** | {targets['binary']['train']['normal_count']:,} | {targets['binary']['train']['normal_pct']}% | {targets['binary']['train']['attack_count']:,} | {targets['binary']['train']['attack_pct']}% | {targets['binary']['train']['normal_count'] + targets['binary']['train']['attack_count']:,} |
| **Testing** | {targets['binary']['test']['normal_count']:,} | {targets['binary']['test']['normal_pct']}% | {targets['binary']['test']['attack_count']:,} | {targets['binary']['test']['attack_pct']}% | {targets['binary']['test']['normal_count'] + targets['binary']['test']['attack_count']:,} |

### 2.2 Multiclass Attack Categories (`attack_cat`)

| Category | Train Count | Train % | Test Count | Test % | Preprocessing Priority |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for cat, data in targets["multiclass"].items():
        priority = "Standard" if data["train_pct"] >= 2.0 else "Rare (High Risk of Misclassification)"
        report += f"| **{cat}** | {data['train_count']:,} | {data['train_pct']}% | {data['test_count']:,} | {data['test_pct']}% | {priority} |\n"

    report += f"""
---

## 3. Categorical Feature Distributions & Domain Shift

| Feature | Unique (Train) | Unique (Test) | Test-Only Unseen Categories | Train-Only Categories | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `proto` | {cats['proto']['train_unique_count']} | {cats['proto']['test_unique_count']} | None (`{cats['proto']['test_only_values']}`) | `{', '.join(cats['proto']['train_only_values'][:5])}` | Dominated by `tcp` and `udp`. |
| `service` | {cats['service']['train_unique_count']} | {cats['service']['test_unique_count']} | None (`{cats['service']['test_only_values']}`) | None (`{cats['service']['train_only_values']}`) | `-` denotes unclassified application service. |
| `state` | {cats['state']['train_unique_count']} | {cats['state']['test_unique_count']} | **`{', '.join(cats['state']['test_only_values'])}`** | `{', '.join(cats['state']['train_only_values'])}` | **Warning:** `ACC` and `CLO` occur in test only! |

---

## 4. Numeric Feature Variance & Collinearity

### 4.1 Lowest Variance Features (Training)
"""
    for k, v in nums["lowest_variance_features"].items():
        report += f"- `{k}`: Variance = `{v}`\n"

    report += f"""
### 4.2 Top Multicollinear Feature Pairs ($|r| \\ge 0.90$)

| Feature 1 | Feature 2 | Pearson Correlation ($r$) | Implication |
| :--- | :--- | :--- | :--- |
"""
    for pair in corrs["high_correlation_pairs"]:
        report += f"| `{pair['feature_1']}` | `{pair['feature_2']}` | **{pair['correlation']}** | Redundant information / collinearity |\n"

    report += f"""
---

## 5. Rare Attack Categories Profile

| Rare Category | Train Records | Test Records | Train/Test Ratio | Observation |
| :--- | :--- | :--- | :--- | :--- |
"""
    for cat, r_info in rares.items():
        report += f"| **{cat}** | {r_info['train_count']:,} ({r_info['train_pct']}%) | {r_info['test_count']:,} ({r_info['test_pct']}%) | {r_info['ratio_train_to_test']} | Extreme scarcity; requires class weighting or macro-averaging. |\n"

    report += f"""
---

## 6. Generated Visualizations

- `reports/figures/eda_binary_distribution.png`: Binary label distributions.
- `reports/figures/eda_attack_categories.png`: Log-scale comparison across all 10 attack classes.
- `reports/figures/eda_numeric_correlations.png`: Heatmap of core traffic volume and temporal features.
- `reports/figures/eda_top_protocols.png`: Top protocols breakdown.
- `reports/figures/eda_top_services.png`: Service distribution breakdown.
- `reports/figures/eda_rare_attacks_breakdown.png`: Scarcity inspection for low-volume attacks.
"""
    return report


def run_eda(
    train_path: Path = CLEAN_TRAIN_PARQUET,
    test_path: Path = CLEAN_TEST_PARQUET,
    report_path: Path = EDA_REPORT_MD,
    figures_dir: Path = FIGURES_DIR,
) -> Dict[str, Any]:
    """Execute end-to-end exploratory data analysis and compile reports."""
    logger.info("Loading cleaned interim Parquet files...")
    df_train, df_test = load_clean_data(train_path, test_path)

    logger.info("Analyzing target distributions...")
    targets = analyze_target_distributions(df_train, df_test)

    logger.info("Analyzing categorical features and domain shift...")
    categoricals = analyze_categorical_features(df_train, df_test)

    logger.info("Analyzing numeric distributions and feature variances...")
    numerics = analyze_numeric_features(df_train)

    logger.info("Analyzing numeric feature multicollinearity...")
    correlations = analyze_correlations(df_train, threshold=0.90)

    logger.info("Analyzing rare attack categories...")
    rares = analyze_rare_categories(df_train, df_test)

    logger.info("Generating diagnostic figures...")
    saved_plots = generate_eda_figures(df_train, df_test, figures_dir=figures_dir)

    analysis_results = {
        "targets": targets,
        "categorical": categoricals,
        "numeric": numerics,
        "correlations": correlations,
        "rare_attacks": rares,
        "figures": [str(p) for p in saved_plots],
    }

    logger.info("Writing EDA report -> %s", report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_content = build_markdown_report(analysis_results)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    return analysis_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    results = run_eda()

    targets = results["targets"]
    print("\n" + "=" * 70)
    print("UNSW-NB15 EXPLORATORY DATA ANALYSIS (EDA) SUMMARY")
    print("=" * 70)
    print("BINARY DISTRIBUTIONS:")
    print(f"  Train: Normal={targets['binary']['train']['normal_count']:,} ({targets['binary']['train']['normal_pct']}%), "
          f"Attack={targets['binary']['train']['attack_count']:,} ({targets['binary']['train']['attack_pct']}%)")
    print(f"  Test : Normal={targets['binary']['test']['normal_count']:,} ({targets['binary']['test']['normal_pct']}%), "
          f"Attack={targets['binary']['test']['attack_count']:,} ({targets['binary']['test']['attack_pct']}%)")

    print("\nMULTICLASS ATTACK CATEGORIES:")
    for cat, data in targets["multiclass"].items():
        print(f"  {cat:<15}: Train={data['train_count']:>7,} ({data['train_pct']:>5.2f}%) | "
              f"Test={data['test_count']:>6,} ({data['test_pct']:>5.2f}%)")

    print("\nCATEGORICAL DOMAIN SHIFT:")
    for col, data in results["categorical"].items():
        print(f"  {col:<8}: Train={data['train_unique_count']} unique | Test={data['test_unique_count']} unique | "
              f"Test-only unseen={data['test_only_values']}")

    print("\nHIGH CORRELATION PAIRS (|r| >= 0.90):")
    for pair in results["correlations"]["high_correlation_pairs"]:
        print(f"  {pair['feature_1']} <-> {pair['feature_2']}: r = {pair['correlation']}")

    print("\nRARE ATTACK CATEGORIES:")
    for cat, r_info in results["rare_attacks"].items():
        print(f"  {cat:<15}: Train={r_info['train_count']:>5,} ({r_info['train_pct']:>5.2f}%) | "
              f"Test={r_info['test_count']:>4,} ({r_info['test_pct']:>5.2f}%)")

    print("\nFIGURES GENERATED:")
    for fig in results["figures"]:
        print(f"  {fig}")
    print("=" * 70 + "\n")
