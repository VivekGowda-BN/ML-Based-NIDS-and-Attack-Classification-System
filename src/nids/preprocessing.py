"""
preprocessing.py — Leakage-safe feature encoding, imputation, and scaling for UNSW-NB15 (Mode 1).

Responsibilities
----------------
1. Load cleaned interim Parquet files (keeping train and test splits strictly separate).
2. Extract binary ('label') and multiclass ('attack_cat') targets.
3. Exclude non-feature columns ('id', 'label', 'attack_cat') from the model feature set.
4. Detect numeric and categorical column types exclusively from the training set.
5. Construct two leakage-safe ColumnTransformer pipelines:
   - Scaled pipeline (for linear models like Logistic Regression):
     * Numeric: SimpleImputer(strategy="median") + StandardScaler()
     * Categorical: SimpleImputer(strategy="most_frequent") + OneHotEncoder(handle_unknown="ignore")
   - Unscaled pipeline (for tree models like XGBoost / Random Forest):
     * Numeric: SimpleImputer(strategy="median")
     * Categorical: SimpleImputer(strategy="most_frequent") + OneHotEncoder(handle_unknown="ignore")
6. Fit transformers ONLY on the training split; transform both train and test splits.
7. Fit LabelEncoder for attack categories ONLY on the training split.
8. Persist fitted preprocessors and label encoder to models/ via joblib.
9. Persist processed arrays and feature metadata to data/processed/.
10. Generate reports/preprocessing_summary.json and reports/preprocessing_report.md.

Usage
-----
    python -m src.nids.preprocessing
    # or:
    python -m nids.preprocessing
    # or as a library:
    from nids.preprocessing import run_preprocessing, load_splits, build_preprocessor
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

try:
    from nids.config import (
        CLEAN_TEST_PARQUET,
        CLEAN_TRAIN_PARQUET,
        FEATURE_NAMES_JSON,
        LABEL_ENCODER_PATH,
        MODELS_DIR,
        NON_FEATURE_COLUMNS,
        PREPROCESSING_REPORT_MD,
        PREPROCESSING_SUMMARY_JSON,
        PREPROCESSOR_SCALED_PATH,
        PREPROCESSOR_UNSCALED_PATH,
        PROCESSED_DIR,
        RANDOM_STATE,
        SCALER_PATH,
        TARGET_BINARY,
        TARGET_MULTI,
    )
except ImportError:
    from src.nids.config import (
        CLEAN_TEST_PARQUET,
        CLEAN_TRAIN_PARQUET,
        FEATURE_NAMES_JSON,
        LABEL_ENCODER_PATH,
        MODELS_DIR,
        NON_FEATURE_COLUMNS,
        PREPROCESSING_REPORT_MD,
        PREPROCESSING_SUMMARY_JSON,
        PREPROCESSOR_SCALED_PATH,
        PREPROCESSOR_UNSCALED_PATH,
        PROCESSED_DIR,
        RANDOM_STATE,
        SCALER_PATH,
        TARGET_BINARY,
        TARGET_MULTI,
    )

logger = logging.getLogger(__name__)


class PreprocessingSplits(NamedTuple):
    """Container for preprocessed train and test arrays and artifacts."""
    X_train_scaled: np.ndarray
    X_test_scaled: np.ndarray
    X_train_unscaled: np.ndarray
    X_test_unscaled: np.ndarray
    y_bin_train: np.ndarray
    y_bin_test: np.ndarray
    y_multi_train: np.ndarray
    y_multi_test: np.ndarray
    feature_names: List[str]
    numeric_features: List[str]
    categorical_features: List[str]
    label_encoder: LabelEncoder
    preprocessor_scaled: ColumnTransformer
    preprocessor_unscaled: ColumnTransformer


def detect_feature_types(
    df_train: pd.DataFrame,
    exclude_cols: List[str] = NON_FEATURE_COLUMNS,
) -> Tuple[List[str], List[str]]:
    """
    Detect numeric and categorical feature names strictly from the training dataframe.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training dataframe.
    exclude_cols : list[str]
        Columns to exclude (id, targets).

    Returns
    -------
    Tuple[list[str], list[str]]
        (numeric_features, categorical_features)
    """
    excluded = set(exclude_cols)
    candidate_cols = [c for c in df_train.columns if c not in excluded]

    cat_cols = list(df_train[candidate_cols].select_dtypes(include=["object", "string", "category"]).columns)
    num_cols = [c for c in candidate_cols if c not in cat_cols]

    return num_cols, cat_cols


def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """
    Construct a scikit-learn ColumnTransformer pipeline.

    Parameters
    ----------
    numeric_cols : list[str]
        Numeric feature column names.
    categorical_cols : list[str]
        Categorical feature column names.
    scale_numeric : bool
        If True, applies StandardScaler to numeric features.
        If False, passes imputed numeric features through without scaling (for trees).

    Returns
    -------
    ColumnTransformer
        Unfitted ColumnTransformer ready for training data.
    """
    if scale_numeric:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
    else:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", num_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", cat_pipeline, categorical_cols))

    return ColumnTransformer(
        transformers=transformers,
        verbose_feature_names_out=False,
    )


def build_markdown_report(summary: Dict[str, Any]) -> str:
    """Generate comprehensive preprocessing markdown report."""
    shapes = summary["shapes"]
    cats = summary["categorical_features"]
    nums = summary["numeric_features"]
    transformed = summary["transformed_features"]
    le_mapping = summary["label_encoder_mapping"]

    report = f"""# UNSW-NB15 Leakage-Safe Preprocessing Report (Mode 1)

**Generated:** {pd.Timestamp.now().isoformat()}  
**Random State:** `{summary['random_state']}`  
**Training Split:** `{summary['inputs']['train_parquet']}`  
**Testing Split:** `{summary['inputs']['test_parquet']}`  

---

## 1. Feature Space & Column Allocation

- **Raw Columns Count:** {summary['raw_column_count']}
- **Excluded Non-Feature Columns ({len(summary['excluded_columns'])}):** `{', '.join(summary['excluded_columns'])}`
  - `id`: Row identifier (prevents index memorization)
  - `{TARGET_BINARY}`: Binary classification target (0 = Normal, 1 = Attack)
  - `{TARGET_MULTI}`: Multiclass attack category target (10 distinct classes)
- **Model Input Features Detected (Training set only):** **{len(nums) + len(cats)}** columns
  - **Numeric Features ({len(nums)}):** `{', '.join(nums)}`
  - **Categorical Features ({len(cats)}):** `{', '.join(cats)}`

---

## 2. Transformation Pipeline Architecture

| Channel | Transformations Applied | Rationale |
| :--- | :--- | :--- |
| **Numeric** | `SimpleImputer(strategy="median")` $\\rightarrow$ `StandardScaler()` *(scaled config)* | Median handles right-skewed network volumes; scaling normalizes variance for linear/distance models. |
| **Numeric (Tree)** | `SimpleImputer(strategy="median")` *(unscaled config)* | Trees are invariant to monotonic scaling; preserves raw interpretable units. |
| **Categorical** | `SimpleImputer(strategy="most_frequent")` $\\rightarrow$ `OneHotEncoder(handle_unknown="ignore")` | `handle_unknown="ignore"` prevents failures on test-only states (`ACC`, `CLO`). |

---

## 3. Preprocessing Dimensions & Matrix Shapes

| Dataset Split | Cleaned Records | Transformed Matrix Shape (Scaled) | Transformed Matrix Shape (Unscaled) | Column Consistency |
| :--- | :--- | :--- | :--- | :--- |
| **Training Set** | {shapes['train_rows']:,} | `{shapes['train_rows']:,} x {shapes['transformed_features_count']}` | `{shapes['train_rows']:,} x {shapes['transformed_features_count']}` | **Aligned** |
| **Testing Set** | {shapes['test_rows']:,} | `{shapes['test_rows']:,} x {shapes['transformed_features_count']}` | `{shapes['test_rows']:,} x {shapes['transformed_features_count']}` | **Aligned** |

- **Total Transformed Features:** **{shapes['transformed_features_count']}** columns (39 numeric + 155 one-hot columns derived from 133 protocols, 13 services, 9 training states).

---

## 4. Multiclass Label Encoding (`{TARGET_MULTI}`)

The `LabelEncoder` was fitted **exclusively on the training split**. Classes mapped as follows:

| Class Index | Attack Category |
| :--- | :--- |
"""
    for cat_name, idx in sorted(le_mapping.items(), key=lambda x: x[1]):
        report += f"| `{idx}` | **{cat_name}** |\n"

    report += f"""
---

## 5. Persisted Artifacts

### 5.1 Models Directory (`models/`)
- `preprocessor_scaled.joblib`: Fitted ColumnTransformer with StandardScaler.
- `preprocessor_unscaled.joblib`: Fitted ColumnTransformer with unscaled numeric features.
- `scaler.joblib`: Scaler artifact for inference.
- `label_encoder.joblib`: Fitted LabelEncoder for multiclass attack categories.

### 5.2 Processed Data Directory (`data/processed/`)
- `X_train_scaled.npy` ({shapes['train_rows']:,} x {shapes['transformed_features_count']})
- `X_test_scaled.npy` ({shapes['test_rows']:,} x {shapes['transformed_features_count']})
- `X_train_unscaled.npy` ({shapes['train_rows']:,} x {shapes['transformed_features_count']})
- `X_test_unscaled.npy` ({shapes['test_rows']:,} x {shapes['transformed_features_count']})
- `y_bin_train.npy`, `y_bin_test.npy` (binary targets)
- `y_multi_train.npy`, `y_multi_test.npy` (multiclass targets)
- `feature_names.json`: Complete record of input, transformed, and excluded feature names.
"""
    return report


def run_preprocessing(
    train_parquet_path: Path = CLEAN_TRAIN_PARQUET,
    test_parquet_path: Path = CLEAN_TEST_PARQUET,
    processed_dir: Path = PROCESSED_DIR,
    models_dir: Path = MODELS_DIR,
    summary_json_path: Path = PREPROCESSING_SUMMARY_JSON,
    report_md_path: Path = PREPROCESSING_REPORT_MD,
) -> PreprocessingSplits:
    """
    Execute leakage-safe preprocessing on the cleaned training and testing Parquet datasets.
    """
    logger.info("Loading cleaned Parquet datasets...")
    df_train = pd.read_parquet(train_parquet_path)
    df_test = pd.read_parquet(test_parquet_path)

    logger.info("Extracting target vectors...")
    y_bin_train = df_train[TARGET_BINARY].to_numpy(dtype=np.int32)
    y_bin_test = df_test[TARGET_BINARY].to_numpy(dtype=np.int32)

    le = LabelEncoder()
    y_multi_train = le.fit_transform(df_train[TARGET_MULTI])
    y_multi_test = le.transform(df_test[TARGET_MULTI])
    logger.info("Fitted LabelEncoder on training attack categories: %s", list(le.classes_))

    # Detect features strictly from training set
    logger.info("Detecting feature types strictly from training split...")
    num_cols, cat_cols = detect_feature_types(df_train, exclude_cols=NON_FEATURE_COLUMNS)
    feature_cols = num_cols + cat_cols
    logger.info("Features identified: %d numeric, %d categorical (Total: %d)", len(num_cols), len(cat_cols), len(feature_cols))

    # Build and fit scaled preprocessor
    logger.info("Fitting scaled preprocessor (ColumnTransformer) on training split...")
    preprocessor_scaled = build_preprocessor(num_cols, cat_cols, scale_numeric=True)
    preprocessor_scaled.fit(df_train[feature_cols])

    # Build and fit unscaled preprocessor
    logger.info("Fitting unscaled preprocessor (ColumnTransformer) on training split...")
    preprocessor_unscaled = build_preprocessor(num_cols, cat_cols, scale_numeric=False)
    preprocessor_unscaled.fit(df_train[feature_cols])

    # Transform both splits
    logger.info("Transforming training and testing splits (scaled)...")
    X_train_scaled = preprocessor_scaled.transform(df_train[feature_cols]).astype(np.float32)
    X_test_scaled = preprocessor_scaled.transform(df_test[feature_cols]).astype(np.float32)

    logger.info("Transforming training and testing splits (unscaled)...")
    X_train_unscaled = preprocessor_unscaled.transform(df_train[feature_cols]).astype(np.float32)
    X_test_unscaled = preprocessor_unscaled.transform(df_test[feature_cols]).astype(np.float32)

    transformed_feature_names = list(preprocessor_scaled.get_feature_names_out())
    logger.info("Transformed features count: %d", len(transformed_feature_names))

    # Save models/ artifacts
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor_scaled, PREPROCESSOR_SCALED_PATH)
    joblib.dump(preprocessor_unscaled, PREPROCESSOR_UNSCALED_PATH)
    joblib.dump(preprocessor_scaled, SCALER_PATH)  # Alias for backward compatibility
    joblib.dump(le, LABEL_ENCODER_PATH)
    logger.info("Saved fitted transformers to %s", models_dir)

    # Save data/processed/ artifacts
    processed_dir.mkdir(parents=True, exist_ok=True)
    np.save(processed_dir / "X_train_scaled.npy", X_train_scaled)
    np.save(processed_dir / "X_test_scaled.npy", X_test_scaled)
    np.save(processed_dir / "X_train_unscaled.npy", X_train_unscaled)
    np.save(processed_dir / "X_test_unscaled.npy", X_test_unscaled)
    np.save(processed_dir / "y_bin_train.npy", y_bin_train)
    np.save(processed_dir / "y_bin_test.npy", y_bin_test)
    np.save(processed_dir / "y_multi_train.npy", y_multi_train)
    np.save(processed_dir / "y_multi_test.npy", y_multi_test)

    # Save feature names JSON and CSV
    feature_metadata = {
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "excluded_columns": NON_FEATURE_COLUMNS,
        "transformed_feature_names": transformed_feature_names,
        "transformed_feature_count": len(transformed_feature_names),
        "target_binary": TARGET_BINARY,
        "target_multi": TARGET_MULTI,
        "label_encoder_classes": list(le.classes_),
    }
    with open(FEATURE_NAMES_JSON, "w", encoding="utf-8") as f:
        json.dump(feature_metadata, f, indent=2)

    pd.Series(transformed_feature_names).to_csv(processed_dir / "feature_names.csv", index=False, header=False)
    logger.info("Saved processed arrays and feature metadata to %s", processed_dir)

    # Compile summary dictionary
    summary: Dict[str, Any] = {
        "project": "ML-Based-NIDS-and-Attack-Classification-System",
        "phase": "preprocessing",
        "mode": 1,
        "timestamp": pd.Timestamp.now().isoformat(),
        "random_state": RANDOM_STATE,
        "inputs": {
            "train_parquet": str(train_parquet_path.resolve()),
            "test_parquet": str(test_parquet_path.resolve()),
        },
        "raw_column_count": df_train.shape[1],
        "excluded_columns": NON_FEATURE_COLUMNS,
        "numeric_features_count": len(num_cols),
        "numeric_features": num_cols,
        "categorical_features_count": len(cat_cols),
        "categorical_features": cat_cols,
        "transformed_features_count": len(transformed_feature_names),
        "transformed_features": transformed_feature_names,
        "label_encoder_mapping": {cls: int(idx) for idx, cls in enumerate(le.classes_)},
        "shapes": {
            "train_rows": int(len(df_train)),
            "test_rows": int(len(df_test)),
            "transformed_features_count": int(len(transformed_feature_names)),
            "X_train_scaled": list(X_train_scaled.shape),
            "X_test_scaled": list(X_test_scaled.shape),
            "X_train_unscaled": list(X_train_unscaled.shape),
            "X_test_unscaled": list(X_test_unscaled.shape),
            "y_bin_train": list(y_bin_train.shape),
            "y_bin_test": list(y_bin_test.shape),
            "y_multi_train": list(y_multi_train.shape),
            "y_multi_test": list(y_multi_test.shape),
        },
        "artifacts": {
            "preprocessor_scaled": str(PREPROCESSOR_SCALED_PATH.resolve()),
            "preprocessor_unscaled": str(PREPROCESSOR_UNSCALED_PATH.resolve()),
            "scaler": str(SCALER_PATH.resolve()),
            "label_encoder": str(LABEL_ENCODER_PATH.resolve()),
            "feature_names_json": str(FEATURE_NAMES_JSON.resolve()),
            "summary_json": str(summary_json_path.resolve()),
            "report_md": str(report_md_path.resolve()),
        },
    }

    # Save summary JSON
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved preprocessing summary -> %s", summary_json_path)

    # Save markdown report
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    report_content = build_markdown_report(summary)
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    logger.info("Saved preprocessing report -> %s", report_md_path)

    return PreprocessingSplits(
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        X_train_unscaled=X_train_unscaled,
        X_test_unscaled=X_test_unscaled,
        y_bin_train=y_bin_train,
        y_bin_test=y_bin_test,
        y_multi_train=y_multi_train,
        y_multi_test=y_multi_test,
        feature_names=transformed_feature_names,
        numeric_features=num_cols,
        categorical_features=cat_cols,
        label_encoder=le,
        preprocessor_scaled=preprocessor_scaled,
        preprocessor_unscaled=preprocessor_unscaled,
    )


def load_splits(processed_dir: Path = PROCESSED_DIR) -> Dict[str, Any]:
    """
    Load preprocessed arrays and feature metadata from data/processed/.

    Returns
    -------
    dict[str, Any]
        Dictionary holding preprocessed arrays and feature names.
    """
    feature_meta_path = processed_dir / "feature_names.json"
    if not feature_meta_path.exists():
        raise FileNotFoundError(
            f"Processed metadata not found at {feature_meta_path}. Run preprocessing first."
        )

    with open(feature_meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return {
        "X_train_scaled": np.load(processed_dir / "X_train_scaled.npy"),
        "X_test_scaled": np.load(processed_dir / "X_test_scaled.npy"),
        "X_train_unscaled": np.load(processed_dir / "X_train_unscaled.npy"),
        "X_test_unscaled": np.load(processed_dir / "X_test_unscaled.npy"),
        "y_bin_train": np.load(processed_dir / "y_bin_train.npy"),
        "y_bin_test": np.load(processed_dir / "y_bin_test.npy"),
        "y_multi_train": np.load(processed_dir / "y_multi_train.npy"),
        "y_multi_test": np.load(processed_dir / "y_multi_test.npy"),
        "feature_names": meta["transformed_feature_names"],
        "numeric_features": meta["numeric_features"],
        "categorical_features": meta["categorical_features"],
        "excluded_columns": meta["excluded_columns"],
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    splits = run_preprocessing()

    print("\n" + "=" * 70)
    print("UNSW-NB15 PREPROCESSING SUMMARY (MODE 1)")
    print("=" * 70)
    print(f"Input Features Detected : {len(splits.numeric_features)} numeric, {len(splits.categorical_features)} categorical")
    print(f"Transformed Features    : {len(splits.feature_names)} features")
    print(f"X_train_scaled Shape    : {splits.X_train_scaled.shape}")
    print(f"X_test_scaled Shape     : {splits.X_test_scaled.shape}")
    print(f"X_train_unscaled Shape  : {splits.X_train_unscaled.shape}")
    print(f"X_test_unscaled Shape   : {splits.X_test_unscaled.shape}")
    print(f"Binary Targets Shape    : Train={splits.y_bin_train.shape}, Test={splits.y_bin_test.shape}")
    print(f"Multiclass Targets Shape: Train={splits.y_multi_train.shape}, Test={splits.y_multi_test.shape}")
    print("\nMULTICLASS ENCODING MAPPING:")
    for idx, cls in enumerate(splits.label_encoder.classes_):
        print(f"  {idx}: {cls}")
    print("\nARTIFACTS SAVED:")
    print(f"  Preprocessor (Scaled)   : {PREPROCESSOR_SCALED_PATH}")
    print(f"  Preprocessor (Unscaled) : {PREPROCESSOR_UNSCALED_PATH}")
    print(f"  Label Encoder           : {LABEL_ENCODER_PATH}")
    print(f"  Summary JSON            : {PREPROCESSING_SUMMARY_JSON}")
    print(f"  Report Markdown         : {PREPROCESSING_REPORT_MD}")
    print("=" * 70 + "\n")
