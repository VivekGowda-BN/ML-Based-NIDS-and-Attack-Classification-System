"""
evaluate.py — Evaluate the trained binary and multiclass models.

Metrics computed
----------------
Binary:
  - Accuracy, Precision, Recall, F1 (macro & weighted)
  - ROC-AUC
  - Confusion matrix
  - Classification report

Multiclass:
  - Accuracy, Macro-F1, Weighted-F1
  - Per-class Precision / Recall / F1
  - Confusion matrix

All results are saved to reports/metrics/.
Confusion-matrix figures are saved to reports/figures/.

Usage
-----
    python -m nids.evaluate
    # or:
    from nids.evaluate import evaluate_binary, evaluate_multiclass
    binary_metrics  = evaluate_binary(model, X_test, y_bin_test)
    multi_metrics   = evaluate_multiclass(model, X_test, y_multi_test, label_names)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from nids.config import (
    ATTACK_CATEGORIES,
    BINARY_CLF_PATH,
    FIGURES_DIR,
    LABEL_ENCODER_PATH,
    METRICS_DIR,
    MULTI_CLF_PATH,
)

logger = logging.getLogger(__name__)


# ─── Binary evaluation ────────────────────────────────────────────────────────

def evaluate_binary(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Compute binary classification metrics and save reports.

    Parameters
    ----------
    model     : fitted binary classifier with predict_proba()
    X_test    : scaled test feature matrix
    y_test    : true binary labels {0, 1}
    threshold : decision threshold (default 0.5)

    Returns
    -------
    dict of metric names → values
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics: dict[str, Any] = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_macro":  round(f1_score(y_test, y_pred, average="macro",    zero_division=0), 4),
        "f1_weighted": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_pred_proba), 4),
        "threshold": threshold,
    }
    logger.info("Binary metrics: %s", metrics)

    # Save JSON
    out_path = METRICS_DIR / "binary_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Binary metrics saved → %s", out_path)

    # Save classification report
    report = classification_report(
        y_test, y_pred, target_names=["Normal", "Attack"], zero_division=0
    )
    (METRICS_DIR / "binary_classification_report.txt").write_text(report)

    # Confusion matrix figure
    _save_confusion_matrix(
        confusion_matrix(y_test, y_pred),
        labels=["Normal", "Attack"],
        title="Binary Classifier — Confusion Matrix",
        filename="binary_confusion_matrix.png",
    )

    return metrics


# ─── Multiclass evaluation ────────────────────────────────────────────────────

def evaluate_multiclass(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compute multiclass metrics and save reports.

    Parameters
    ----------
    model       : fitted multiclass classifier
    X_test      : scaled test feature matrix
    y_test      : integer-encoded true labels
    label_names : human-readable class names (in label order)

    Returns
    -------
    dict of metric names → values
    """
    if label_names is None:
        # Attempt to load from saved LabelEncoder
        if LABEL_ENCODER_PATH.exists():
            le = joblib.load(LABEL_ENCODER_PATH)
            label_names = list(le.classes_)
        else:
            label_names = [str(i) for i in sorted(np.unique(y_test))]

    y_pred = model.predict(X_test)

    metrics: dict[str, Any] = {
        "accuracy":    round(accuracy_score(y_test, y_pred), 4),
        "f1_macro":    round(f1_score(y_test, y_pred, average="macro",    zero_division=0), 4),
        "f1_weighted": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
    }
    logger.info("Multiclass metrics: %s", metrics)

    # Save JSON
    out_path = METRICS_DIR / "multiclass_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Multiclass metrics saved → %s", out_path)

    # Per-class report
    report = classification_report(y_test, y_pred, target_names=label_names, zero_division=0)
    (METRICS_DIR / "multiclass_classification_report.txt").write_text(report)

    # Confusion matrix figure
    cm = confusion_matrix(y_test, y_pred)
    _save_confusion_matrix(
        cm,
        labels=label_names,
        title="Multiclass Classifier — Confusion Matrix",
        filename="multiclass_confusion_matrix.png",
    )

    return metrics


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _save_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    title: str,
    filename: str,
) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    plt.tight_layout()
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix figure saved → %s", path)


# ─── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from nids.preprocessing import load_splits

    if not BINARY_CLF_PATH.exists():
        raise FileNotFoundError("Binary model not found. Run `make train` first.")
    if not MULTI_CLF_PATH.exists():
        raise FileNotFoundError("Multiclass model not found. Run `make train` first.")

    splits = load_splits()
    binary_model = joblib.load(BINARY_CLF_PATH)
    multi_model  = joblib.load(MULTI_CLF_PATH)

    binary_metrics = evaluate_binary(binary_model, splits.X_test, splits.y_bin_test)
    multi_metrics  = evaluate_multiclass(multi_model, splits.X_test, splits.y_multi_test)

    print("Binary  metrics:", binary_metrics)
    print("Multi   metrics:", multi_metrics)
