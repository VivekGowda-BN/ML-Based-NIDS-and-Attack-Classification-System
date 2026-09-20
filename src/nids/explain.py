"""
explain.py — Model explainability with a SHAP-first, permutation-importance fallback.

Strategy
--------
1. Try to import ``shap``.  If available, use SHAP TreeExplainer (fast, accurate).
2. If SHAP is not importable (e.g. no prebuilt wheel for the current Python /
   platform), silently fall back to scikit-learn permutation importance.
3. Either code path produces the same output interface:
     - reports/explanations/binary_shap_importance.json  (or _perm_importance.json)
     - reports/explanations/binary_shap_bar.png          (or _perm_bar.png)
     - per-class multiclass importance files

The application **never crashes** solely because SHAP is unavailable.

Installing SHAP on Windows / Python 3.13
-----------------------------------------
SHAP ships only a source distribution (sdist) for Windows / cp313.
Attempting a normal ``pip install shap`` triggers a C-extension build that
requires Microsoft Visual C++ 14.0.  Use the prebuilt-only flag instead:

    pip install --only-binary=shap shap

If no prebuilt wheel is available for your exact platform, skip SHAP and
rely on the built-in permutation importance fallback — the rest of the
project will continue to work.

Usage
-----
    python -m nids.explain
    # or:
    from nids.explain import explain_binary, explain_multiclass
    explain_binary(model, X_test, y_test, feature_names)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance

from nids.config import (
    BINARY_CLF_PATH,
    EXPLANATIONS_DIR,
    LABEL_ENCODER_PATH,
    MULTI_CLF_PATH,
    PROCESSED_DIR,
)

logger = logging.getLogger(__name__)

# ─── Runtime SHAP availability flag ──────────────────────────────────────────
try:
    import shap as _shap  # type: ignore  # noqa: F401
    _SHAP_AVAILABLE = True
    logger.debug("SHAP is available — will use TreeExplainer.")
except Exception:  # ImportError, OSError (missing .pyd), etc.
    _SHAP_AVAILABLE = False
    logger.warning(
        "SHAP is not importable on this platform. "
        "Falling back to scikit-learn permutation importance. "
        "To enable SHAP on Windows/Python 3.13, run: "
        "pip install --only-binary=shap shap"
    )

SHAP_AVAILABLE: bool = _SHAP_AVAILABLE

# ─── Tuneable constants ────────────────────────────────────────────────────────
SHAP_EXPLAIN_SAMPLES: int = 500   # rows used for SHAP computation
PERM_N_REPEATS: int = 10          # repeats for permutation importance
PERM_RANDOM_STATE: int = 42


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def explain_binary(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
) -> dict[str, float]:
    """
    Generate binary-classifier feature importance.

    Uses SHAP TreeExplainer when available; falls back to permutation importance.

    Parameters
    ----------
    model        : fitted binary classifier
    X_test       : scaled test feature matrix
    y_test       : true binary labels (needed only for permutation importance)
    feature_names: column names corresponding to X_test

    Returns
    -------
    dict mapping feature_name → importance_score (sorted descending)
    """
    if _SHAP_AVAILABLE:
        return _shap_binary(model, X_test, feature_names)
    return _perm_binary(model, X_test, y_test, feature_names)


def explain_multiclass(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    label_names: Optional[list[str]] = None,
) -> None:
    """
    Generate multiclass feature importance (one importance set per class with
    SHAP; one global set with permutation importance).

    Parameters
    ----------
    model        : fitted multiclass classifier
    X_test       : scaled test feature matrix
    y_test       : integer-encoded true labels
    feature_names: column names
    label_names  : human-readable class names (loaded from LabelEncoder if None)
    """
    if label_names is None and LABEL_ENCODER_PATH.exists():
        le = joblib.load(LABEL_ENCODER_PATH)
        label_names = list(le.classes_)

    if _SHAP_AVAILABLE:
        _shap_multiclass(model, X_test, feature_names, label_names)
    else:
        _perm_multiclass(model, X_test, y_test, feature_names)


# ═══════════════════════════════════════════════════════════════════════════════
# SHAP implementation
# ═══════════════════════════════════════════════════════════════════════════════

def _shap_binary(
    model: object,
    X_test: np.ndarray,
    feature_names: list[str],
) -> dict[str, float]:
    import shap  # type: ignore

    logger.info("Computing SHAP values for binary classifier …")
    X_sample = X_test[:SHAP_EXPLAIN_SAMPLES]

    # TODO: fall back to shap.KernelExplainer for non-tree-based models.
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Some versions return a list [class-0, class-1]; use class-1 for binary.
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    # ── Bar plot ──────────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv, X_sample, feature_names=feature_names,
                      plot_type="bar", show=False)
    plt.tight_layout()
    _savefig("binary_shap_bar.png")

    # ── Beeswarm ──────────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    _savefig("binary_shap_beeswarm.png")

    # ── JSON importance ───────────────────────────────────────────────────────
    mean_abs = np.abs(sv).mean(axis=0)
    importance = _build_importance_dict(feature_names, mean_abs)
    _save_importance(importance, "binary_shap_importance.json")
    logger.info("Binary SHAP complete.")
    return importance


def _shap_multiclass(
    model: object,
    X_test: np.ndarray,
    feature_names: list[str],
    label_names: Optional[list[str]],
) -> None:
    import shap  # type: ignore

    logger.info("Computing SHAP values for multiclass classifier …")
    X_sample = X_test[:SHAP_EXPLAIN_SAMPLES]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if not isinstance(shap_values, list):
        logger.warning(
            "Expected list of per-class SHAP arrays; got %s — "
            "saving global mean instead.", type(shap_values)
        )
        shap_values = [shap_values]

    for cls_idx, sv in enumerate(shap_values):
        name = (label_names[cls_idx]
                if label_names and cls_idx < len(label_names)
                else str(cls_idx))
        safe_name = name.replace(" ", "_").replace("/", "-")

        plt.figure(figsize=(10, 6))
        shap.summary_plot(sv, X_sample, feature_names=feature_names,
                          plot_type="bar", show=False, max_display=15)
        plt.title(f"SHAP — {name}")
        plt.tight_layout()
        _savefig(f"multi_shap_{safe_name}.png")

        mean_abs = np.abs(sv).mean(axis=0)
        importance = _build_importance_dict(feature_names, mean_abs)
        _save_importance(importance, f"multi_shap_{safe_name}_importance.json")

    logger.info("Multiclass SHAP plots saved → %s", EXPLANATIONS_DIR)


# ═══════════════════════════════════════════════════════════════════════════════
# Permutation-importance fallback
# ═══════════════════════════════════════════════════════════════════════════════

def _perm_binary(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
) -> dict[str, float]:
    logger.info(
        "Computing permutation importance for binary classifier "
        "(SHAP fallback, n_repeats=%d) …", PERM_N_REPEATS
    )
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=PERM_N_REPEATS,
        random_state=PERM_RANDOM_STATE,
        scoring="roc_auc",
    )
    importance = _build_importance_dict(feature_names, result.importances_mean)

    # ── Bar plot ──────────────────────────────────────────────────────────────
    top = list(importance.items())[:20]           # top-20 for readability
    names, scores = zip(*top) if top else ([], [])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(list(names)[::-1], list(scores)[::-1], color="#4C72B0")
    ax.set_xlabel("Mean decrease in ROC-AUC")
    ax.set_title("Binary Classifier — Permutation Importance (top 20)\n"
                 "[SHAP unavailable on this platform]")
    plt.tight_layout()
    _savefig("binary_perm_bar.png")

    _save_importance(importance, "binary_perm_importance.json")
    logger.info("Binary permutation importance complete.")
    return importance


def _perm_multiclass(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
) -> None:
    logger.info(
        "Computing permutation importance for multiclass classifier "
        "(SHAP fallback, n_repeats=%d) …", PERM_N_REPEATS
    )
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=PERM_N_REPEATS,
        random_state=PERM_RANDOM_STATE,
        scoring="accuracy",
    )
    importance = _build_importance_dict(feature_names, result.importances_mean)

    top = list(importance.items())[:20]
    names, scores = zip(*top) if top else ([], [])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(list(names)[::-1], list(scores)[::-1], color="#DD8452")
    ax.set_xlabel("Mean decrease in accuracy")
    ax.set_title("Multiclass Classifier — Permutation Importance (top 20)\n"
                 "[SHAP unavailable on this platform]")
    plt.tight_layout()
    _savefig("multi_perm_bar.png")

    _save_importance(importance, "multi_perm_importance.json")
    logger.info("Multiclass permutation importance complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _build_importance_dict(
    feature_names: list[str],
    scores: np.ndarray,
) -> dict[str, float]:
    """Return {feature: score} sorted by score descending."""
    raw = dict(zip(feature_names, scores.tolist()))
    return dict(sorted(raw.items(), key=lambda x: x[1], reverse=True))


def _save_importance(importance: dict[str, float], filename: str) -> None:
    out = EXPLANATIONS_DIR / filename
    out.write_text(json.dumps(importance, indent=2))
    logger.info("Importance saved → %s", out)


def _savefig(filename: str) -> None:
    path = EXPLANATIONS_DIR / filename
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("Figure saved → %s", path)


# ─── Module-level public flag (importable by dashboard / tests) ───────────────
SHAP_AVAILABLE: bool = _SHAP_AVAILABLE


# ─── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not BINARY_CLF_PATH.exists() or not MULTI_CLF_PATH.exists():
        raise FileNotFoundError("Models not found. Run `make train` first.")

    from nids.preprocessing import load_splits

    splits = load_splits()
    binary_model = joblib.load(BINARY_CLF_PATH)
    multi_model  = joblib.load(MULTI_CLF_PATH)

    backend = "SHAP" if SHAP_AVAILABLE else "permutation importance (SHAP fallback)"
    print(f"Explainability backend: {backend}")

    explain_binary(binary_model, splits.X_test, splits.y_bin_test, splits.feature_names)
    explain_multiclass(multi_model, splits.X_test, splits.y_multi_test, splits.feature_names)
    print("Explanations complete.")
