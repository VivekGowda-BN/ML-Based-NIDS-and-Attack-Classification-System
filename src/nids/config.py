"""
config.py — Central configuration for paths, constants, and hyperparameters.

All other modules import from here so that changing a path or a parameter
only requires editing this single file.
"""

from __future__ import annotations

from pathlib import Path

# ─── Repository root ──────────────────────────────────────────────────────────
# Resolves to the project root regardless of where the script is called from.
ROOT_DIR: Path = Path(__file__).resolve().parents[2]

# ─── Data paths ───────────────────────────────────────────────────────────────
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"
SCENARIOS_DIR: Path = DATA_DIR / "scenarios"
FEATURE_NAMES_JSON: Path = PROCESSED_DIR / "feature_names.json"

# ─── Model paths ──────────────────────────────────────────────────────────────
MODELS_DIR: Path = ROOT_DIR / "models"
BINARY_CLF_PATH: Path = MODELS_DIR / "binary_clf.joblib"
MULTI_CLF_PATH: Path = MODELS_DIR / "multi_clf.joblib"
SCALER_PATH: Path = MODELS_DIR / "scaler.joblib"
LABEL_ENCODER_PATH: Path = MODELS_DIR / "label_encoder.joblib"
PREPROCESSOR_SCALED_PATH: Path = MODELS_DIR / "preprocessor_scaled.joblib"
PREPROCESSOR_UNSCALED_PATH: Path = MODELS_DIR / "preprocessor_unscaled.joblib"
BINARY_LR_PATH: Path = MODELS_DIR / "binary_logistic_regression.joblib"
BINARY_RF_PATH: Path = MODELS_DIR / "binary_random_forest.joblib"
BINARY_XGB_PATH: Path = MODELS_DIR / "binary_xgboost.joblib"
BINARY_BEST_MODEL_PATH: Path = MODELS_DIR / "binary_best_model.joblib"
MULTICLASS_LR_PATH: Path = MODELS_DIR / "multiclass_logistic_regression.joblib"
MULTICLASS_RF_PATH: Path = MODELS_DIR / "multiclass_random_forest.joblib"
MULTICLASS_XGB_PATH: Path = MODELS_DIR / "multiclass_xgboost.joblib"
MULTICLASS_BEST_MODEL_PATH: Path = MODELS_DIR / "multiclass_best_model.joblib"

# ─── Report paths ─────────────────────────────────────────────────────────────
REPORTS_DIR: Path = ROOT_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
METRICS_DIR: Path = REPORTS_DIR / "metrics"
EXPLANATIONS_DIR: Path = REPORTS_DIR / "explanations"
PREPROCESSING_SUMMARY_JSON: Path = REPORTS_DIR / "preprocessing_summary.json"
PREPROCESSING_REPORT_MD: Path = REPORTS_DIR / "preprocessing_report.md"
BINARY_RESULTS_JSON: Path = METRICS_DIR / "binary_results.json"
BINARY_COMPARISON_CSV: Path = METRICS_DIR / "binary_model_comparison.csv"
BINARY_BEST_MODEL_JSON: Path = METRICS_DIR / "binary_best_model.json"
BINARY_THRESHOLD_RESULTS_JSON: Path = METRICS_DIR / "binary_threshold_results.json"
BINARY_THRESHOLD_COMPARISON_CSV: Path = METRICS_DIR / "binary_threshold_comparison.csv"
BINARY_THRESHOLD_SELECTED_MODEL_PATH: Path = MODELS_DIR / "binary_threshold_selected_model.joblib"
BINARY_THRESHOLD_SELECTION_VAL_JSON: Path = METRICS_DIR / "binary_threshold_selection_validation.json"
BINARY_THRESHOLD_SELECTION_TEST_JSON: Path = METRICS_DIR / "binary_threshold_selection_test.json"
BINARY_THRESHOLD_SELECTION_CSV: Path = METRICS_DIR / "binary_threshold_selection_comparison.csv"
BINARY_THRESHOLD_SELECTION_META_JSON: Path = METRICS_DIR / "binary_threshold_selection_metadata.json"
BINARY_THRESHOLD_SELECTION_TRADEOFF_PNG: Path = FIGURES_DIR / "binary_threshold_selection_tradeoff.png"
MULTICLASS_RESULTS_JSON: Path = METRICS_DIR / "multiclass_results.json"
MULTICLASS_COMPARISON_CSV: Path = METRICS_DIR / "multiclass_model_comparison.csv"
MULTICLASS_BEST_MODEL_JSON: Path = METRICS_DIR / "multiclass_best_model.json"

# ─── UNSW-NB15 raw file paths ─────────────────────────────────────────────────
RAW_TRAIN_FILE: str = "UNSW_NB15_training-set.csv"
RAW_TEST_FILE: str = "UNSW_NB15_testing-set.csv"
RAW_TRAIN_PATH: Path = RAW_DIR / RAW_TRAIN_FILE
RAW_TEST_PATH: Path = RAW_DIR / RAW_TEST_FILE
MANIFEST_PATH: Path = RAW_DIR / "manifest.json"
DATA_PROFILE_PATH: Path = REPORTS_DIR / "data_profile.md"

# ─── Cleaned interim paths ───────────────────────────────────────────────────
CLEAN_TRAIN_PARQUET: Path = INTERIM_DIR / "UNSW_NB15_training_clean.parquet"
CLEAN_TEST_PARQUET: Path = INTERIM_DIR / "UNSW_NB15_testing_clean.parquet"
CLEANING_SUMMARY_JSON: Path = REPORTS_DIR / "cleaning_summary.json"
CLEANING_REPORT_MD: Path = REPORTS_DIR / "cleaning_report.md"
EDA_REPORT_MD: Path = REPORTS_DIR / "eda_report.md"

# ─── Column definitions ───────────────────────────────────────────────────────
ID_COLUMN: str = "id"
NON_FEATURE_COLUMNS: list[str] = [ID_COLUMN, "attack_cat", "label"]

UNSW_NB15_FILES: list[str] = [
    RAW_TRAIN_FILE,
    RAW_TEST_FILE,
]
FEATURES_FILE: str = "NUSW-NB15_features.csv"

# ─── UNSW-NB15 column names ───────────────────────────────────────────────────
TARGET_BINARY: str = "label"       # 0 = Normal, 1 = Attack
TARGET_MULTI: str = "attack_cat"   # attack category string

# ─── Attack categories ────────────────────────────────────────────────────────
# Matches the exact categories present in UNSW-NB15 raw CSVs
ATTACK_CATEGORIES: list[str] = [
    "Normal",
    "Generic",
    "Exploits",
    "Fuzzers",
    "DoS",
    "Reconnaissance",
    "Analysis",
    "Backdoor",
    "Shellcode",
    "Worms",
]

# ─── Operational Severity & Action Mappings ──────────────────────────────────
SEVERITY_MAPPING: dict[str, str] = {
    "Normal": "None",
    "Analysis": "Medium",
    "Reconnaissance": "Medium",
    "Backdoor": "Critical",
    "DoS": "Critical",
    "Exploits": "Critical",
    "Fuzzers": "High",
    "Generic": "High",
    "Shellcode": "Critical",
    "Worms": "Critical",
}

RECOMMENDED_ACTIONS: dict[str, str] = {
    "Normal": "Allow traffic / Normal operation",
    "Analysis": "Investigate source port scan / port sweeping; monitor host activity",
    "Reconnaissance": "Block scanner IP; review firewall logs and inspect reconnaissance targets",
    "Backdoor": "CRITICAL ALERT: Immediately isolate host from network; revoke credentials; conduct forensic memory analysis",
    "DoS": "CRITICAL ALERT: Activate rate-limiting / anti-DoS ACLs; drop traffic from attacking IP range",
    "Exploits": "CRITICAL ALERT: Block exploit source IP; inspect targeted vulnerability; verify service patches",
    "Fuzzers": "Apply protocol sanitization; rate limit offending client; inspect payload anomalies",
    "Generic": "Enforce strict IPS signature filtering; inspect cryptographic / certificate anomalies",
    "Shellcode": "CRITICAL ALERT: Immediate host quarantine; terminate suspicious execution threads; memory dump required",
    "Worms": "CRITICAL ALERT: Network segment isolation; block lateral propagation ports; inspect neighbor nodes",
}

# ─── Train / test split ───────────────────────────────────────────────────────
TEST_SIZE: float = 0.20
RANDOM_STATE: int = 42

# ─── Model hyperparameters (TODO: tune these) ─────────────────────────────────
BINARY_CLF_PARAMS: dict = {
    # TODO: replace with optimised XGBoost / LightGBM params after tuning
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "random_state": RANDOM_STATE,
}

MULTI_CLF_PARAMS: dict = {
    # TODO: replace with optimised multiclass params after tuning
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "random_state": RANDOM_STATE,
}

# ─── Ensure output directories exist at import time ──────────────────────────
for _dir in (MODELS_DIR, FIGURES_DIR, METRICS_DIR, EXPLANATIONS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
