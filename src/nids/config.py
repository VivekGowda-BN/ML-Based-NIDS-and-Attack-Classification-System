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

# ─── Model paths ──────────────────────────────────────────────────────────────
MODELS_DIR: Path = ROOT_DIR / "models"
BINARY_CLF_PATH: Path = MODELS_DIR / "binary_clf.joblib"
MULTI_CLF_PATH: Path = MODELS_DIR / "multi_clf.joblib"
SCALER_PATH: Path = MODELS_DIR / "scaler.joblib"
LABEL_ENCODER_PATH: Path = MODELS_DIR / "label_encoder.joblib"

# ─── Report paths ─────────────────────────────────────────────────────────────
REPORTS_DIR: Path = ROOT_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
METRICS_DIR: Path = REPORTS_DIR / "metrics"
EXPLANATIONS_DIR: Path = REPORTS_DIR / "explanations"

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
