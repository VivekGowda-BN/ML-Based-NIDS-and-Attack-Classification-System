"""
cleaning.py — Data-quality fixes on the merged UNSW-NB15 DataFrame.

Steps (in order)
-----------------
1. Drop exact duplicate rows.
2. Standardise the attack_cat column (strip whitespace, title-case).
3. Fix known type issues (hex ports → integers, '-' → NaN, etc.).
4. Report and optionally drop remaining null values.
5. Save cleaned data to data/interim/unsw_nb15_clean.csv.

Usage
-----
    python -m nids.cleaning
    # or:
    from nids.cleaning import clean
    df_clean = clean(df_raw)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from nids.config import ATTACK_CATEGORIES, INTERIM_DIR, TARGET_BINARY, TARGET_MULTI

logger = logging.getLogger(__name__)

CLEAN_PATH: Path = INTERIM_DIR / "unsw_nb15_clean.csv"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps and return a cleaned copy.

    Parameters
    ----------
    df : pd.DataFrame
        Raw merged UNSW-NB15 DataFrame (output of dataset.load_raw).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for preprocessing.
    """
    df = df.copy()
    df = _drop_duplicates(df)
    df = _normalise_attack_cat(df)
    df = _fix_port_columns(df)
    df = _replace_placeholder_nulls(df)
    df = _report_nulls(df)
    logger.info("Cleaned shape: %s", df.shape)
    return df


# ─── Individual cleaning steps ────────────────────────────────────────────────

def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d duplicate rows.", dropped)
    return df


def _normalise_attack_cat(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip whitespace and apply consistent capitalisation to attack_cat.
    UNSW-NB15 files sometimes contain ' Backdoors' (leading space) or
    'backdoors' (all lower).
    """
    if TARGET_MULTI not in df.columns:
        return df

    df[TARGET_MULTI] = (
        df[TARGET_MULTI]
        .astype(str)
        .str.strip()
        .str.title()
        .replace({"Nan": "Normal", "": "Normal"})
    )

    # TODO: verify the full list of raw category strings in the downloaded files
    #       and extend this mapping if needed.
    known = set(ATTACK_CATEGORIES)
    unexpected = set(df[TARGET_MULTI].unique()) - known
    if unexpected:
        logger.warning("Unexpected attack_cat values (will be kept as-is): %s", unexpected)

    return df


def _fix_port_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    UNSW-NB15 ports are sometimes stored as hex strings (e.g. '0x0050').
    Convert sport / dsport to integers, coercing errors to NaN.
    """
    for col in ("sport", "dsport"):
        if col in df.columns:
            # TODO: confirm whether all four CSV parts use the same port format.
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def _replace_placeholder_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    UNSW-NB15 uses '-' as a missing-value placeholder in some string columns.
    Replace with NaN so pandas can handle them uniformly.
    """
    # TODO: check if other placeholder strings (e.g. '?', 'NA') appear.
    df = df.replace("-", pd.NA)
    return df


def _report_nulls(df: pd.DataFrame) -> pd.DataFrame:
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if not null_cols.empty:
        logger.info(
            "Remaining null values per column:\n%s",
            null_cols.to_string(),
        )
    # TODO: decide per-column strategy — impute or drop.
    #       For now we leave nulls in; preprocessing.py will handle them.
    return df


def save_clean(df: pd.DataFrame, path: Path = CLEAN_PATH) -> None:
    """Persist the cleaned DataFrame to data/interim/."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved cleaned data → %s", path)


# ─── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from nids.dataset import load_raw  # local import to avoid circular deps

    raw = load_raw()
    cleaned = clean(raw)
    save_clean(cleaned)
    print(f"Done. Cleaned shape: {cleaned.shape}")
