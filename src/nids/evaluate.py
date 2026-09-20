"""
evaluate.py — Evaluation and model selection for ML-Based NIDS (Mode 1).

Responsibilities
----------------
1. Load trained binary models (Logistic Regression, Random Forest, XGBoost).
2. Evaluate each model on the untouched testing partition.
3. Compute extensive metrics:
   - Confusion Matrix (TN, FP, FN, TP)
   - Precision, Attack Recall, F1-Score, Balanced Accuracy
   - ROC-AUC and PR-AUC (Average Precision)
   - False Positive Rate (FPR), False Negative Rate (FNR)
   - Total inference time and per-sample latency in microseconds
4. Generate diagnostic evaluation plots:
   - Confusion matrix heatmap for each model
   - Combined ROC Curves (ROC-AUC comparison)
   - Combined Precision-Recall Curves (PR-AUC comparison)
   - Comparative Metrics Bar Chart
5. Multi-criteria selection of best model based on PR-AUC, Attack Recall, FPR, and F1-Score.
6. Persist selected model to models/binary_best_model.joblib and metadata reports to reports/metrics/.

Usage
-----
    python -m src.nids.evaluate --task binary
    # or:
    python -m nids.evaluate --task binary
    # or as a module:
    from nids.evaluate import evaluate_binary_models
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

try:
    from nids.config import (
        BINARY_BEST_MODEL_JSON,
        BINARY_BEST_MODEL_PATH,
        BINARY_CLF_PATH,
        BINARY_COMPARISON_CSV,
        BINARY_LR_PATH,
        BINARY_RESULTS_JSON,
        BINARY_RF_PATH,
        BINARY_XGB_PATH,
        FIGURES_DIR,
        LABEL_ENCODER_PATH,
        METRICS_DIR,
        MULTICLASS_BEST_MODEL_JSON,
        MULTICLASS_BEST_MODEL_PATH,
        MULTICLASS_COMPARISON_CSV,
        MULTICLASS_LR_PATH,
        MULTICLASS_RESULTS_JSON,
        MULTICLASS_RF_PATH,
        MULTICLASS_XGB_PATH,
        MULTI_CLF_PATH,
        PROCESSED_DIR,
    )
    from nids.preprocessing import load_splits
except ImportError:
    from src.nids.config import (
        BINARY_BEST_MODEL_JSON,
        BINARY_BEST_MODEL_PATH,
        BINARY_CLF_PATH,
        BINARY_COMPARISON_CSV,
        BINARY_LR_PATH,
        BINARY_RESULTS_JSON,
        BINARY_RF_PATH,
        BINARY_XGB_PATH,
        FIGURES_DIR,
        LABEL_ENCODER_PATH,
        METRICS_DIR,
        MULTICLASS_BEST_MODEL_JSON,
        MULTICLASS_BEST_MODEL_PATH,
        MULTICLASS_COMPARISON_CSV,
        MULTICLASS_LR_PATH,
        MULTICLASS_RESULTS_JSON,
        MULTICLASS_RF_PATH,
        MULTICLASS_XGB_PATH,
        MULTI_CLF_PATH,
        PROCESSED_DIR,
    )
    from src.nids.preprocessing import load_splits

logger = logging.getLogger(__name__)

TRAINING_META_PATH = METRICS_DIR / "binary_training_meta.json"


def evaluate_single_model(
    name: str,
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    training_time: float = 0.0,
    model_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive metrics for a single binary classification model.
    """
    n_samples = len(y_test)

    # Benchmark inference time
    t0 = time.perf_counter()
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    inference_time = time.perf_counter() - t0
    per_sample_us = (inference_time / n_samples) * 1e6 if n_samples > 0 else 0.0

    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Rate calculations
    fpr = float(fp) / float(fp + tn) if (fp + tn) > 0 else 0.0
    fnr = float(fn) / float(tp + fn) if (tp + fn) > 0 else 0.0

    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y_test, y_pred))
    roc_auc = float(roc_auc_score(y_test, y_pred_proba))
    pr_auc = float(average_precision_score(y_test, y_pred_proba))

    cls_report = classification_report(
        y_test, y_pred, target_names=["Normal", "Attack"], output_dict=True, zero_division=0
    )

    return {
        "model_name": name,
        "parameters": model_params or {},
        "training_time_sec": round(training_time, 3),
        "inference_time_sec": round(inference_time, 3),
        "per_sample_latency_us": round(per_sample_us, 2),
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "metrics": {
            "accuracy": round(float((tp + tn) / n_samples), 4),
            "balanced_accuracy": round(balanced_acc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "false_positive_rate": round(fpr, 4),
            "false_negative_rate": round(fnr, 4),
        },
        "classification_report": cls_report,
        "raw_predictions": {
            "y_pred_proba": y_pred_proba,
            "y_pred": y_pred,
        },
    }


def save_confusion_matrix_figure(
    cm: np.ndarray,
    title: str,
    output_path: Path,
) -> Path:
    """Save an annotated confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=",d",
        cmap="Blues",
        xticklabels=["Normal (0)", "Attack (1)"],
        yticklabels=["Normal (0)", "Attack (1)"],
        ax=ax,
        cbar=False,
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Class", fontsize=11)
    ax.set_ylabel("True Class", fontsize=11)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def save_roc_curves_figure(
    models_data: Dict[str, Dict[str, Any]],
    y_test: np.ndarray,
    output_path: Path,
) -> Path:
    """Save combined ROC curves comparing all models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"logistic_regression": "#2563eb", "random_forest": "#10b981", "xgboost": "#f59e0b"}
    display_names = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }

    for key, data in models_data.items():
        y_proba = data["raw_predictions"]["y_pred_proba"]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = data["metrics"]["roc_auc"]
        color = colors.get(key, "#4b5563")
        label = f"{display_names.get(key, key)} (ROC-AUC = {roc_auc:.4f})"
        ax.plot(fpr, tpr, label=label, color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess (AUC = 0.5000)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=11)
    ax.set_ylabel("True Positive Rate (TPR / Attack Recall)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def save_pr_curves_figure(
    models_data: Dict[str, Dict[str, Any]],
    y_test: np.ndarray,
    output_path: Path,
) -> Path:
    """Save combined Precision-Recall curves comparing all models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"logistic_regression": "#2563eb", "random_forest": "#10b981", "xgboost": "#f59e0b"}
    display_names = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }

    baseline = float(np.sum(y_test == 1)) / len(y_test)

    for key, data in models_data.items():
        y_proba = data["raw_predictions"]["y_pred_proba"]
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = data["metrics"]["pr_auc"]
        color = colors.get(key, "#4b5563")
        label = f"{display_names.get(key, key)} (PR-AUC = {pr_auc:.4f})"
        ax.plot(rec, prec, label=label, color=color, linewidth=2)

    ax.axhline(baseline, color="k", linestyle="--", alpha=0.6, label=f"Baseline Attack Ratio ({baseline:.3f})")
    ax.set_title("Precision-Recall (PR) Curves", fontsize=13, fontweight="bold")
    ax.set_xlabel("Attack Recall", fontsize=11)
    ax.set_ylabel("Attack Precision", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def save_model_comparison_bar_chart(
    models_data: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> Path:
    """Save comparative metrics bar chart."""
    metrics_to_plot = ["pr_auc", "roc_auc", "f1_score", "recall", "precision"]
    metric_labels = ["PR-AUC", "ROC-AUC", "F1-Score", "Attack Recall", "Precision"]

    display_names = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }

    records = []
    for model_key, data in models_data.items():
        row = {"Model": display_names.get(model_key, model_key)}
        for m in metrics_to_plot:
            row[m] = data["metrics"][m]
        records.append(row)

    df_comp = pd.DataFrame(records).set_index("Model")
    df_comp.columns = metric_labels

    fig, ax = plt.subplots(figsize=(10, 5))
    df_comp.plot(kind="bar", ax=ax, width=0.8)
    ax.set_title("Binary Intrusion Detection — Model Comparison", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0.70, 1.02)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    plt.xticks(rotation=0)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def select_best_model(models_data: Dict[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """
    Select the superior model based on balanced intrusion-detection objectives:
    High PR-AUC, High Attack Recall, Low False-Positive Rate, High F1-Score.
    """
    scores = {}
    for key, data in models_data.items():
        m = data["metrics"]
        # Composite objective: maximize PR-AUC, Recall, and F1; penalize FPR
        composite = (m["pr_auc"] * 0.35) + (m["recall"] * 0.30) + (m["f1_score"] * 0.25) - (m["false_positive_rate"] * 0.10)
        scores[key] = composite

    best_key = max(scores, key=scores.get)
    best_data = models_data[best_key]

    selection_rationale = {
        "selected_model": best_key,
        "selection_criteria": "Composite prioritization: PR-AUC (35%), Recall (30%), F1-Score (25%), and minimal False-Positive Rate (10% penalty)",
        "composite_scores": {k: round(v, 4) for k, v in scores.items()},
        "winning_metrics": best_data["metrics"],
        "training_time_sec": best_data["training_time_sec"],
        "per_sample_latency_us": best_data["per_sample_latency_us"],
    }
    return best_key, selection_rationale


def evaluate_binary_models(
    splits: Optional[Dict[str, Any]] = None,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate all three binary models on test splits and persist reports and figures.
    """
    if splits is None:
        logger.info("Loading preprocessed test data from %s...", PROCESSED_DIR)
        splits = load_splits(PROCESSED_DIR)

    X_test_scaled = splits["X_test_scaled"]
    X_test_unscaled = splits["X_test_unscaled"]
    y_test = splits["y_bin_test"]

    # Load training metadata if available
    train_meta = {}
    if TRAINING_META_PATH.exists():
        with open(TRAINING_META_PATH, "r", encoding="utf-8") as f:
            train_meta = json.load(f)

    # 1. Evaluate Logistic Regression
    logger.info("Evaluating Logistic Regression on X_test_scaled...")
    lr_model = joblib.load(BINARY_LR_PATH)
    lr_train_time = train_meta.get("models", {}).get("logistic_regression", {}).get("training_time_seconds", 0.0)
    lr_params = train_meta.get("models", {}).get("logistic_regression", {}).get("parameters", {})
    lr_eval = evaluate_single_model("logistic_regression", lr_model, X_test_scaled, y_test, lr_train_time, lr_params)

    # 2. Evaluate Random Forest
    logger.info("Evaluating Random Forest on X_test_unscaled...")
    rf_model = joblib.load(BINARY_RF_PATH)
    rf_train_time = train_meta.get("models", {}).get("random_forest", {}).get("training_time_seconds", 0.0)
    rf_params = train_meta.get("models", {}).get("random_forest", {}).get("parameters", {})
    rf_eval = evaluate_single_model("random_forest", rf_model, X_test_unscaled, y_test, rf_train_time, rf_params)

    # 3. Evaluate XGBoost
    logger.info("Evaluating XGBoost on X_test_unscaled...")
    xgb_model = joblib.load(BINARY_XGB_PATH)
    xgb_train_time = train_meta.get("models", {}).get("xgboost", {}).get("training_time_seconds", 0.0)
    xgb_params = train_meta.get("models", {}).get("xgboost", {}).get("parameters", {})
    xgb_eval = evaluate_single_model("xgboost", xgb_model, X_test_unscaled, y_test, xgb_train_time, xgb_params)

    models_data = {
        "logistic_regression": lr_eval,
        "random_forest": rf_eval,
        "xgboost": xgb_eval,
    }

    # Model Selection
    best_name, selection_meta = select_best_model(models_data)
    logger.info("Selected superior binary model: %s", best_name)

    # Copy best model to models/binary_best_model.joblib and models/binary_clf.joblib
    best_model_obj = {"logistic_regression": lr_model, "random_forest": rf_model, "xgboost": xgb_model}[best_name]
    joblib.dump(best_model_obj, BINARY_BEST_MODEL_PATH)
    joblib.dump(best_model_obj, BINARY_CLF_PATH)
    logger.info("Persisted best model -> %s and %s", BINARY_BEST_MODEL_PATH, BINARY_CLF_PATH)

    # Generate Figures
    logger.info("Generating evaluation figures...")
    cm_lr_path = save_confusion_matrix_figure(
        np.array(lr_eval["confusion_matrix"]["matrix"]),
        "Logistic Regression — Confusion Matrix",
        FIGURES_DIR / "binary_confusion_matrix_logistic_regression.png",
    )
    cm_rf_path = save_confusion_matrix_figure(
        np.array(rf_eval["confusion_matrix"]["matrix"]),
        "Random Forest — Confusion Matrix",
        FIGURES_DIR / "binary_confusion_matrix_random_forest.png",
    )
    cm_xgb_path = save_confusion_matrix_figure(
        np.array(xgb_eval["confusion_matrix"]["matrix"]),
        "XGBoost — Confusion Matrix",
        FIGURES_DIR / "binary_confusion_matrix_xgboost.png",
    )
    roc_path = save_roc_curves_figure(models_data, y_test, FIGURES_DIR / "binary_roc_curves.png")
    pr_path = save_pr_curves_figure(models_data, y_test, FIGURES_DIR / "binary_precision_recall_curves.png")
    comp_bar_path = save_model_comparison_bar_chart(models_data, FIGURES_DIR / "binary_model_comparison.png")

    # Construct clean serializable results dictionary (without raw probability arrays)
    clean_results = {}
    csv_rows = []
    for k, v in models_data.items():
        m = v["metrics"]
        clean_results[k] = {
            "model_name": v["model_name"],
            "parameters": v["parameters"],
            "training_time_sec": v["training_time_sec"],
            "inference_time_sec": v["inference_time_sec"],
            "per_sample_latency_us": v["per_sample_latency_us"],
            "confusion_matrix": v["confusion_matrix"],
            "metrics": m,
            "classification_report": v["classification_report"],
        }
        csv_rows.append({
            "model": k,
            "accuracy": m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1_score": m["f1_score"],
            "roc_auc": m["roc_auc"],
            "pr_auc": m["pr_auc"],
            "false_positive_rate": m["false_positive_rate"],
            "false_negative_rate": m["false_negative_rate"],
            "training_time_sec": v["training_time_sec"],
            "inference_time_sec": v["inference_time_sec"],
            "per_sample_latency_us": v["per_sample_latency_us"],
        })

    # Save CSV comparison
    df_comparison = pd.DataFrame(csv_rows)
    df_comparison.to_csv(BINARY_COMPARISON_CSV, index=False)
    logger.info("Saved comparison CSV -> %s", BINARY_COMPARISON_CSV)

    # Save Results JSON
    full_output = {
        "task": "binary_classification",
        "evaluation_timestamp": pd.Timestamp.now().isoformat(),
        "test_records_count": len(y_test),
        "models": clean_results,
        "selection": selection_meta,
        "figures": [
            str(cm_lr_path), str(cm_rf_path), str(cm_xgb_path),
            str(roc_path), str(pr_path), str(comp_bar_path)
        ],
    }
    with open(BINARY_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    logger.info("Saved binary results JSON -> %s", BINARY_RESULTS_JSON)

    # Save Best Model JSON
    with open(BINARY_BEST_MODEL_JSON, "w", encoding="utf-8") as f:
        json.dump(selection_meta, f, indent=2)
    logger.info("Saved best model metadata -> %s", BINARY_BEST_MODEL_JSON)

    return full_output


# ─── Multiclass evaluation ────────────────────────────────────────────────

MULTICLASS_META_PATH = METRICS_DIR / "multiclass_training_meta.json"
RARE_CLASSES = ["Analysis", "Backdoor", "Shellcode", "Worms"]


def evaluate_multiclass_single_model(
    name: str,
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    training_time: float = 0.0,
) -> Dict[str, Any]:
    """
    Compute comprehensive multiclass metrics for a single model.
    """
    n_samples = len(y_test)
    n_classes = len(class_names)

    # Inference timing
    t0 = time.perf_counter()
    y_pred_proba = model.predict_proba(X_test)  # shape (n_samples, n_classes)
    inference_time = time.perf_counter() - t0
    per_sample_us = (inference_time / n_samples) * 1e6 if n_samples > 0 else 0.0

    y_pred = np.argmax(y_pred_proba, axis=1)

    # Aggregate metrics
    acc = float(np.mean(y_pred == y_test))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    macro_prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    weighted_prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    weighted_rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    # Log loss
    try:
        logloss = float(log_loss(y_test, y_pred_proba, labels=list(range(n_classes))))
    except Exception:
        logloss = None

    # ROC-AUC (OvR macro)
    try:
        roc_auc = float(
            roc_auc_score(y_test, y_pred_proba, multi_class="ovr", average="macro", labels=list(range(n_classes)))
        )
    except Exception:
        roc_auc = None

    # Per-class metrics
    per_class_prec = precision_score(y_test, y_pred, average=None, zero_division=0, labels=list(range(n_classes)))
    per_class_rec = recall_score(y_test, y_pred, average=None, zero_division=0, labels=list(range(n_classes)))
    per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0, labels=list(range(n_classes)))

    per_class = {}
    for i, cls in enumerate(class_names):
        per_class[cls] = {
            "precision": round(float(per_class_prec[i]), 4),
            "recall": round(float(per_class_rec[i]), 4),
            "f1": round(float(per_class_f1[i]), 4),
            "support": int(np.sum(y_test == i)),
        }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=list(range(n_classes)))

    return {
        "model_name": name,
        "training_time_sec": round(training_time, 3),
        "inference_time_sec": round(inference_time, 3),
        "per_sample_latency_us": round(per_sample_us, 2),
        "metrics": {
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "macro_precision": round(macro_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_precision": round(weighted_prec, 4),
            "weighted_recall": round(weighted_rec, 4),
            "weighted_f1": round(weighted_f1, 4),
            "roc_auc_ovr_macro": round(roc_auc, 4) if roc_auc is not None else None,
            "log_loss": round(logloss, 4) if logloss is not None else None,
        },
        "per_class_metrics": per_class,
        "confusion_matrix": cm.tolist(),
        "raw_predictions": {
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
        },
    }


def save_multiclass_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str,
    output_path: Path,
) -> Path:
    """Save an annotated 10x10 confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt=",d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.4,
        linecolor="#e5e7eb",
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Predicted Class", fontsize=11)
    ax.set_ylabel("True Class", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_multiclass_comparison_chart(
    models_data: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> Path:
    """Save comparative macro metric bar chart."""
    metrics_keys = ["macro_f1", "macro_recall", "macro_precision", "balanced_accuracy", "weighted_f1"]
    metric_labels = ["Macro F1", "Macro Recall", "Macro Precision", "Balanced Accuracy", "Weighted F1"]
    display_names = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }

    records = []
    for key, data in models_data.items():
        row = {"Model": display_names.get(key, key)}
        m = data["metrics"]
        for mk in metrics_keys:
            row[mk] = m.get(mk, 0.0) or 0.0
        records.append(row)

    df_comp = pd.DataFrame(records).set_index("Model")
    df_comp.columns = metric_labels

    fig, ax = plt.subplots(figsize=(11, 5))
    df_comp.plot(kind="bar", ax=ax, width=0.75)
    ax.set_title("Multiclass Attack Classification — Model Comparison", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    plt.xticks(rotation=0)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_multiclass_per_class_f1_chart(
    models_data: Dict[str, Dict[str, Any]],
    class_names: List[str],
    output_path: Path,
) -> Path:
    """Save per-class F1 comparison chart highlighting rare classes."""
    display_names = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }
    colors = {"logistic_regression": "#2563eb", "random_forest": "#10b981", "xgboost": "#f59e0b"}

    x = np.arange(len(class_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(15, 6))

    for i, (key, data) in enumerate(models_data.items()):
        f1_values = [data["per_class_metrics"][cls]["f1"] for cls in class_names]
        bars = ax.bar(
            x + i * width,
            f1_values,
            width,
            label=display_names.get(key, key),
            color=colors.get(key, "#4b5563"),
            alpha=0.85,
        )

    # Highlight rare classes
    for i, cls in enumerate(class_names):
        if cls in RARE_CLASSES:
            ax.axvspan(i - 0.3, i + 0.85, alpha=0.07, color="red")

    ax.set_title("Per-Class F1 Score by Model (rare classes highlighted in red)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Attack Category", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=10)
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def select_best_multiclass_model(
    models_data: Dict[str, Dict[str, Any]],
    class_names: List[str],
) -> Tuple[str, Dict[str, Any]]:
    """
    Select the best multiclass model based on:
    1. Macro F1 (primary)
    2. Macro Recall
    3. Mean rare-class recall (Analysis, Backdoor, Shellcode, Worms)
    4. Balanced Accuracy
    """
    scores = {}
    for key, data in models_data.items():
        m = data["metrics"]
        rare_recalls = []
        for cls in RARE_CLASSES:
            if cls in data["per_class_metrics"]:
                rare_recalls.append(data["per_class_metrics"][cls]["recall"])
        mean_rare_recall = float(np.mean(rare_recalls)) if rare_recalls else 0.0
        composite = (
            (m.get("macro_f1") or 0.0) * 0.40
            + (m.get("macro_recall") or 0.0) * 0.25
            + mean_rare_recall * 0.25
            + (m.get("balanced_accuracy") or 0.0) * 0.10
        )
        scores[key] = composite
        logger.info(
            "Model %s composite=%.4f (macro_f1=%.4f, macro_rec=%.4f, rare_rec=%.4f, bal_acc=%.4f)",
            key, composite, m.get("macro_f1", 0), m.get("macro_recall", 0),
            mean_rare_recall, m.get("balanced_accuracy", 0),
        )

    best_key = max(scores, key=scores.get)
    best_data = models_data[best_key]

    rare_class_detail = {}
    for cls in RARE_CLASSES:
        if cls in best_data["per_class_metrics"]:
            rare_class_detail[cls] = best_data["per_class_metrics"][cls]

    selection_rationale = {
        "selected_model": best_key,
        "selection_criteria": (
            "Composite: Macro F1 (40%) + Macro Recall (25%) + "
            "Mean Rare-Class Recall (25%) + Balanced Accuracy (10%)"
        ),
        "composite_scores": {k: round(v, 4) for k, v in scores.items()},
        "winning_metrics": best_data["metrics"],
        "rare_class_metrics": rare_class_detail,
        "training_time_sec": best_data["training_time_sec"],
        "per_sample_latency_us": best_data["per_sample_latency_us"],
    }
    return best_key, selection_rationale


def evaluate_multiclass_models(
    splits: Optional[Dict[str, Any]] = None,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate all three multiclass models on the untouched test split.
    Persists reports, confusion matrices, and comparison figures.
    """
    if splits is None:
        logger.info("Loading preprocessed test data from %s...", PROCESSED_DIR)
        splits = load_splits(PROCESSED_DIR)

    X_test_scaled = splits["X_test_scaled"]
    X_test_unscaled = splits["X_test_unscaled"]
    y_test = splits["y_multi_test"]

    # Load label encoder to get class names
    le = joblib.load(LABEL_ENCODER_PATH)
    class_names: List[str] = list(le.classes_)
    logger.info("Loaded label encoder. Classes: %s", class_names)

    # Load training metadata if available
    train_meta: Dict[str, Any] = {}
    if MULTICLASS_META_PATH.exists():
        with open(MULTICLASS_META_PATH, "r", encoding="utf-8") as f:
            train_meta = json.load(f)

    def _get_train_time(mname: str) -> float:
        return train_meta.get("models", {}).get(mname, {}).get("training_time_seconds", 0.0)

    # 1. Evaluate Logistic Regression (scaled)
    logger.info("Evaluating Multiclass Logistic Regression on X_test_scaled...")
    lr_model = joblib.load(MULTICLASS_LR_PATH)
    lr_eval = evaluate_multiclass_single_model(
        "logistic_regression", lr_model, X_test_scaled, y_test, class_names, _get_train_time("logistic_regression")
    )

    # 2. Evaluate Random Forest (unscaled)
    logger.info("Evaluating Multiclass Random Forest on X_test_unscaled...")
    rf_model = joblib.load(MULTICLASS_RF_PATH)
    rf_eval = evaluate_multiclass_single_model(
        "random_forest", rf_model, X_test_unscaled, y_test, class_names, _get_train_time("random_forest")
    )

    # 3. Evaluate XGBoost (unscaled)
    logger.info("Evaluating Multiclass XGBoost on X_test_unscaled...")
    xgb_model = joblib.load(MULTICLASS_XGB_PATH)
    xgb_eval = evaluate_multiclass_single_model(
        "xgboost", xgb_model, X_test_unscaled, y_test, class_names, _get_train_time("xgboost")
    )

    models_data = {
        "logistic_regression": lr_eval,
        "random_forest": rf_eval,
        "xgboost": xgb_eval,
    }

    # Model selection
    best_name, selection_meta = select_best_multiclass_model(models_data, class_names)
    logger.info("Selected superior multiclass model: %s", best_name)

    # Persist best model
    best_model_obj = {"logistic_regression": lr_model, "random_forest": rf_model, "xgboost": xgb_model}[best_name]
    joblib.dump(best_model_obj, MULTICLASS_BEST_MODEL_PATH)
    joblib.dump(best_model_obj, MULTI_CLF_PATH)
    logger.info("Persisted best model -> %s and %s", MULTICLASS_BEST_MODEL_PATH, MULTI_CLF_PATH)

    # Generate confusion matrix figures
    fig_model_map = {
        "logistic_regression": (lr_eval, "Logistic Regression — Multiclass Confusion Matrix",
                                 FIGURES_DIR / "multiclass_confusion_matrix_logistic_regression.png"),
        "random_forest": (rf_eval, "Random Forest — Multiclass Confusion Matrix",
                          FIGURES_DIR / "multiclass_confusion_matrix_random_forest.png"),
        "xgboost": (xgb_eval, "XGBoost — Multiclass Confusion Matrix",
                    FIGURES_DIR / "multiclass_confusion_matrix_xgboost.png"),
    }
    cm_paths = {}
    for key, (eval_data, title, path) in fig_model_map.items():
        cm_paths[key] = save_multiclass_confusion_matrix(
            np.array(eval_data["confusion_matrix"]), class_names, title, path
        )
        logger.info("Saved confusion matrix -> %s", path)

    comp_chart_path = save_multiclass_comparison_chart(
        models_data, FIGURES_DIR / "multiclass_model_comparison.png"
    )
    per_class_f1_path = save_multiclass_per_class_f1_chart(
        models_data, class_names, FIGURES_DIR / "multiclass_per_class_f1.png"
    )
    logger.info("Saved comparison and per-class F1 charts.")

    # Build serializable results (strip raw prediction arrays)
    clean_results: Dict[str, Any] = {}
    csv_rows = []
    for key, data in models_data.items():
        m = data["metrics"]
        clean_results[key] = {
            "model_name": data["model_name"],
            "training_time_sec": data["training_time_sec"],
            "inference_time_sec": data["inference_time_sec"],
            "per_sample_latency_us": data["per_sample_latency_us"],
            "metrics": m,
            "per_class_metrics": data["per_class_metrics"],
            "confusion_matrix": data["confusion_matrix"],
        }
        csv_row = {"model": key}
        csv_row.update({k: (v if v is not None else "") for k, v in m.items()})
        csv_row["training_time_sec"] = data["training_time_sec"]
        csv_row["inference_time_sec"] = data["inference_time_sec"]
        csv_row["per_sample_latency_us"] = data["per_sample_latency_us"]
        # rare-class columns
        for cls in RARE_CLASSES:
            pc = data["per_class_metrics"].get(cls, {})
            csv_row[f"{cls}_precision"] = pc.get("precision", "")
            csv_row[f"{cls}_recall"] = pc.get("recall", "")
            csv_row[f"{cls}_f1"] = pc.get("f1", "")
        csv_rows.append(csv_row)

    # Save CSV
    df_comparison = pd.DataFrame(csv_rows)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    df_comparison.to_csv(MULTICLASS_COMPARISON_CSV, index=False)
    logger.info("Saved comparison CSV -> %s", MULTICLASS_COMPARISON_CSV)

    # Save full results JSON
    full_output = {
        "task": "multiclass_classification",
        "evaluation_timestamp": pd.Timestamp.now().isoformat(),
        "test_records_count": int(len(y_test)),
        "class_names": class_names,
        "models": clean_results,
        "selection": selection_meta,
        "figures": [
            str(v) for v in list(cm_paths.values()) + [comp_chart_path, per_class_f1_path]
        ],
    }
    with open(MULTICLASS_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    logger.info("Saved multiclass results JSON -> %s", MULTICLASS_RESULTS_JSON)

    # Save best model JSON
    with open(MULTICLASS_BEST_MODEL_JSON, "w", encoding="utf-8") as f:
        json.dump(selection_meta, f, indent=2)
    logger.info("Saved multiclass best model metadata -> %s", MULTICLASS_BEST_MODEL_JSON)

    return full_output


# ─── Backward compatibility wrappers ─────────────────────────────────────────

def evaluate_binary(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Backward-compatible single-model evaluator."""
    res = evaluate_single_model("model", model, X_test, y_test)
    return res["metrics"]


def evaluate_multiclass(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Backward-compatible single-model multiclass evaluator."""
    class_names = label_names or [str(i) for i in range(len(np.unique(y_test)))]
    return evaluate_multiclass_single_model("model", model, X_test, y_test, class_names)


# ─── CLI Entrypoint ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NIDS machine learning models (Mode 1)")
    parser.add_argument(
        "--task",
        choices=["binary", "multiclass", "all"],
        default="binary",
        help="Target classification task to evaluate (default: binary)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.task in ("binary", "all"):
        out = evaluate_binary_models()
        sel = out["selection"]

        print("\n" + "=" * 70)
        print("BINARY MODELS EVALUATION & BENCHMARKING SUMMARY")
        print("=" * 70)
        for model_name, info in out["models"].items():
            m = info["metrics"]
            print(f"\nModel: {model_name.upper()}")
            print(f"  Accuracy          : {m['accuracy']:.4f}")
            print(f"  Balanced Accuracy : {m['balanced_accuracy']:.4f}")
            print(f"  Attack Recall     : {m['recall']:.4f}")
            print(f"  Precision         : {m['precision']:.4f}")
            print(f"  F1-Score          : {m['f1_score']:.4f}")
            print(f"  ROC-AUC           : {m['roc_auc']:.4f}")
            print(f"  PR-AUC            : {m['pr_auc']:.4f}")
            print(f"  False Pos Rate    : {m['false_positive_rate']:.4f}")
            print(f"  Training Time     : {info['training_time_sec']:.2f}s")
            print(f"  Inference Latency : {info['per_sample_latency_us']:.2f} µs/sample")

        print("\n" + "-" * 70)
        print(f"SELECTED SUPERIOR MODEL: {sel['selected_model'].upper()}")
        print(f"Selection Rationale   : {sel['selection_criteria']}")
        print(f"Best Model Artifact   : {BINARY_BEST_MODEL_PATH}")
        print("=" * 70 + "\n")

    if args.task in ("multiclass", "all"):
        out = evaluate_multiclass_models()
        sel = out["selection"]

        print("\n" + "=" * 70)
        print("MULTICLASS MODELS EVALUATION & BENCHMARKING SUMMARY")
        print("=" * 70)
        for model_name, info in out["models"].items():
            m = info["metrics"]
            print(f"\nModel: {model_name.upper()}")
            print(f"  Accuracy          : {m['accuracy']:.4f}")
            print(f"  Balanced Accuracy : {m['balanced_accuracy']:.4f}")
            print(f"  Macro Precision   : {m['macro_precision']:.4f}")
            print(f"  Macro Recall      : {m['macro_recall']:.4f}")
            print(f"  Macro F1          : {m['macro_f1']:.4f}")
            print(f"  Weighted F1       : {m['weighted_f1']:.4f}")
            auc_str = f"{m['roc_auc_ovr_macro']:.4f}" if m.get('roc_auc_ovr_macro') is not None else "N/A"
            ll_str = f"{m['log_loss']:.4f}" if m.get('log_loss') is not None else "N/A"
            print(f"  ROC-AUC (OvR)     : {auc_str}")
            print(f"  Log Loss          : {ll_str}")
            print(f"  Training Time     : {info['training_time_sec']:.2f}s")
            print(f"  Inference Latency : {info['per_sample_latency_us']:.2f} µs/sample")
            print(f"  Rare-Class Recall:")
            for cls in ["Analysis", "Backdoor", "Shellcode", "Worms"]:
                pc = info.get("per_class_metrics", {}).get(cls, {})
                print(f"    {cls:<18}: recall={pc.get('recall', 0):.4f}  f1={pc.get('f1', 0):.4f}")

        print("\n" + "-" * 70)
        print(f"SELECTED SUPERIOR MODEL: {sel['selected_model'].upper()}")
        print(f"Selection Rationale   : {sel['selection_criteria']}")
        print(f"Best Model Artifact   : {MULTICLASS_BEST_MODEL_PATH}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
