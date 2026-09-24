"""
attack_only_multiclass.py — Attack-only 9-class experiment (Mode 1, Phase 5b).

Motivation
----------
The baseline 10-class experiment includes Normal records in training.  This
causes tree-based models to focus heavily on separating Normal vs. Attack, which
hurts within-attack discrimination, especially for rare classes (Worms, Shellcode,
Backdoor, Analysis).

This module trains a dedicated classifier on *attack-only* records (label == 1)
across the 9 attack categories, using a fresh LabelEncoder that maps:

    0=Analysis  1=Backdoor  2=DoS  3=Exploits  4=Fuzzers
    5=Generic   6=Reconnaissance  7=Shellcode  8=Worms

All baselines in models/ and reports/metrics/ are untouched.

Usage
-----
    python -m src.nids.attack_only_multiclass
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

try:
    from nids.config import (
        FIGURES_DIR,
        METRICS_DIR,
        MODELS_DIR,
        PROCESSED_DIR,
        RANDOM_STATE,
    )
    from nids.preprocessing import load_splits
except ImportError:
    from src.nids.config import (
        FIGURES_DIR,
        METRICS_DIR,
        MODELS_DIR,
        PROCESSED_DIR,
        RANDOM_STATE,
    )
    from src.nids.preprocessing import load_splits

logger = logging.getLogger(__name__)

# ─── Artifact paths (all prefixed attack_only_*) ─────────────────────────────
AO_LR_PATH         = MODELS_DIR / "attack_only_multiclass_logistic_regression.joblib"
AO_RF_PATH         = MODELS_DIR / "attack_only_multiclass_random_forest.joblib"
AO_XGB_PATH        = MODELS_DIR / "attack_only_multiclass_xgboost.joblib"
AO_BEST_PATH       = MODELS_DIR / "attack_only_multiclass_best_model.joblib"
AO_LE_PATH         = MODELS_DIR / "attack_only_label_encoder.joblib"

AO_RESULTS_JSON    = METRICS_DIR / "attack_only_multiclass_results.json"
AO_BEST_JSON       = METRICS_DIR / "attack_only_multiclass_best_model.json"
AO_COMPARISON_CSV  = METRICS_DIR / "attack_only_multiclass_comparison.csv"
AO_TRAINING_META   = METRICS_DIR / "attack_only_multiclass_training_meta.json"

# Nine attack categories in alphabetical order (matches LabelEncoder default)
ATTACK_CLASSES = [
    "Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers",
    "Generic", "Reconnaissance", "Shellcode", "Worms",
]
RARE_CLASSES = ["Analysis", "Backdoor", "Shellcode", "Worms"]

VAL_SIZE      = 0.20   # validation fraction from attack training records
N_ATTACK_CLS  = 9


# ─── Data preparation ─────────────────────────────────────────────────────────

def build_attack_only_splits(
    splits: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Filter the preprocessed arrays to attack-only records and build a fresh
    9-class LabelEncoder.

    Parameters
    ----------
    splits : dict returned by load_splits(PROCESSED_DIR)

    Returns
    -------
    dict with keys:
        X_train_scaled, X_train_unscaled, y_train   (attack train, full)
        X_val_scaled,   X_val_unscaled,   y_val     (held-out validation)
        X_test_scaled,  X_test_unscaled,  y_test    (attack test records)
        label_encoder  : fitted 9-class LabelEncoder
        class_names    : list[str] of 9 attack categories
    """
    # ── Training: keep only attack records (y_bin == 1) ──────────────────────
    y_bin_train  = splits["y_bin_train"]
    y_multi_train = splits["y_multi_train"]     # 10-class encoded (Normal=6)

    # The 10-class LE classes: Analysis=0 Backdoor=1 DoS=2 Exploits=3
    # Fuzzers=4 Generic=5 Normal=6 Reconnaissance=7 Shellcode=8 Worms=9
    # "label == 1" rows guaranteed to have attack_cat != Normal
    atk_mask_tr = y_bin_train == 1
    logger.info("Attack train records: %d / %d", atk_mask_tr.sum(), len(y_bin_train))

    X_tr_sc  = splits["X_train_scaled"][atk_mask_tr]
    X_tr_un  = splits["X_train_unscaled"][atk_mask_tr]
    y_tr_raw = y_multi_train[atk_mask_tr]      # still 10-class integer codes

    # ── Build 9-class LabelEncoder from training attack labels only ───────────
    # Decode to strings using the original 10-class encoder then re-encode
    orig_le = joblib.load(MODELS_DIR / "label_encoder.joblib")
    y_tr_names = orig_le.inverse_transform(y_tr_raw)   # array of category strings

    le_9 = LabelEncoder()
    le_9.fit(ATTACK_CLASSES)          # deterministic alphabetical fit
    y_tr = le_9.transform(y_tr_names)
    class_names: List[str] = [str(c) for c in le_9.classes_]  # plain str, not np.str_
    logger.info("9-class label encoder classes: %s", class_names)

    # Verify no Normal leaked into training
    assert "Normal" not in set(y_tr_names), "Normal records leaked into attack train!"

    # ── Validation split (from attack training records only) ──────────────────
    (
        X_tr_sc, X_val_sc,
        X_tr_un, X_val_un,
        y_tr,    y_val,
    ) = _stratified_split(
        X_tr_sc, X_tr_un, y_tr,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
    )

    logger.info(
        "After val split — train: %d, val: %d",
        len(y_tr), len(y_val),
    )

    # ── Test: keep only attack records ────────────────────────────────────────
    y_bin_test   = splits["y_bin_test"]
    y_multi_test = splits["y_multi_test"]
    atk_mask_te  = y_bin_test == 1
    logger.info("Attack test records: %d / %d", atk_mask_te.sum(), len(y_bin_test))

    X_te_sc  = splits["X_test_scaled"][atk_mask_te]
    X_te_un  = splits["X_test_unscaled"][atk_mask_te]
    y_te_raw = y_multi_test[atk_mask_te]
    y_te_names = orig_le.inverse_transform(y_te_raw)
    y_te = le_9.transform(y_te_names)

    assert "Normal" not in set(y_te_names), "Normal records leaked into attack test!"

    # Save the 9-class encoder
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(le_9, AO_LE_PATH)
    logger.info("Saved 9-class label encoder -> %s", AO_LE_PATH)

    return {
        "X_train_scaled":   X_tr_sc,
        "X_train_unscaled": X_tr_un,
        "y_train":          y_tr,
        "X_val_scaled":     X_val_sc,
        "X_val_unscaled":   X_val_un,
        "y_val":            y_val,
        "X_test_scaled":    X_te_sc,
        "X_test_unscaled":  X_te_un,
        "y_test":           y_te,
        "label_encoder":    le_9,
        "class_names":      class_names,
    }


def _stratified_split(
    X_sc: np.ndarray,
    X_un: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/val split returning both scaled and unscaled halves."""
    idx_tr, idx_val = next(
        iter(
            __import__("sklearn.model_selection", fromlist=["StratifiedShuffleSplit"])
            .StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            .split(X_sc, y)
        )
    )
    return (
        X_sc[idx_tr], X_sc[idx_val],
        X_un[idx_tr], X_un[idx_val],
        y[idx_tr],    y[idx_val],
    )


# ─── Training ─────────────────────────────────────────────────────────────────

def _get_versions() -> Dict[str, str]:
    pkgs = ["scikit-learn", "xgboost", "numpy", "pandas", "joblib"]
    v = {}
    for p in pkgs:
        try:
            v[p] = importlib.metadata.version(p)
        except Exception:
            v[p] = "unknown"
    v["python"] = platform.python_version()
    return v


def train_ao_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[LogisticRegression, float, Dict[str, Any]]:
    """LR on scaled features with class_weight='balanced'."""
    params: Dict[str, Any] = {
        "max_iter": 1000,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "random_state": RANDOM_STATE,
    }
    logger.info("Training Attack-Only LR (scaled %s)...", X_train.shape)
    model = LogisticRegression(**params)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    logger.info("LR done in %.2fs", elapsed)
    joblib.dump(model, AO_LR_PATH)
    logger.info("Saved -> %s", AO_LR_PATH)
    return model, elapsed, params


def train_ao_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[RandomForestClassifier, float, Dict[str, Any]]:
    """RF on unscaled features with class_weight='balanced_subsample'."""
    params: Dict[str, Any] = {
        "n_estimators": 200,
        "class_weight": "balanced_subsample",
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    }
    logger.info("Training Attack-Only RF (unscaled %s)...", X_train.shape)
    model = RandomForestClassifier(**params)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    logger.info("RF done in %.2fs", elapsed)
    joblib.dump(model, AO_RF_PATH)
    logger.info("Saved -> %s", AO_RF_PATH)
    return model, elapsed, params


def train_ao_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[XGBClassifier, float, Dict[str, Any]]:
    """XGBoost on unscaled features with per-sample balanced weights."""
    params: Dict[str, Any] = {
        "objective": "multi:softprob",
        "num_class": N_ATTACK_CLS,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    logger.info(
        "Training Attack-Only XGBoost (unscaled %s, num_class=%d)...",
        X_train.shape, N_ATTACK_CLS,
    )
    model = XGBClassifier(**params)
    t0 = time.perf_counter()
    model.fit(X_train, y_train, sample_weight=sample_weights)
    elapsed = time.perf_counter() - t0
    logger.info("XGBoost done in %.2fs", elapsed)
    joblib.dump(model, AO_XGB_PATH)
    logger.info("Saved -> %s", AO_XGB_PATH)
    return model, elapsed, params


def train_attack_only_models(
    ao_splits: Dict[str, Any],
) -> Dict[str, Any]:
    """Train all 3 attack-only models. Returns dict of models + metadata."""
    X_train_scaled   = ao_splits["X_train_scaled"]
    X_train_unscaled = ao_splits["X_train_unscaled"]
    y_train          = ao_splits["y_train"]

    lr_model,  lr_time,  lr_params  = train_ao_logistic_regression(X_train_scaled,   y_train)
    rf_model,  rf_time,  rf_params  = train_ao_random_forest(X_train_unscaled, y_train)
    xgb_model, xgb_time, xgb_params = train_ao_xgboost(X_train_unscaled, y_train)

    meta = {
        "experiment": "attack_only_multiclass",
        "description": "Classifier trained exclusively on attack records (label==1), 9 classes.",
        "timestamp": pd.Timestamp.now().isoformat(),
        "random_state": RANDOM_STATE,
        "val_size": VAL_SIZE,
        "n_attack_classes": N_ATTACK_CLS,
        "attack_classes": ATTACK_CLASSES,
        "train_attack_records": int(len(y_train)),
        "models": {
            "logistic_regression": {
                "artifact": str(AO_LR_PATH),
                "input": "X_train_scaled",
                "time_s": round(lr_time, 3),
                "params": lr_params,
                "balance": "class_weight='balanced'",
            },
            "random_forest": {
                "artifact": str(AO_RF_PATH),
                "input": "X_train_unscaled",
                "time_s": round(rf_time, 3),
                "params": rf_params,
                "balance": "class_weight='balanced_subsample'",
            },
            "xgboost": {
                "artifact": str(AO_XGB_PATH),
                "input": "X_train_unscaled",
                "time_s": round(xgb_time, 3),
                "params": xgb_params,
                "balance": "compute_sample_weight('balanced')",
            },
        },
        "library_versions": _get_versions(),
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(AO_TRAINING_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved training metadata -> %s", AO_TRAINING_META)

    return {
        "models": {
            "logistic_regression": lr_model,
            "random_forest": rf_model,
            "xgboost": xgb_model,
        },
        "training_meta": meta,
    }


# ─── Evaluation helpers ───────────────────────────────────────────────────────

def _eval_single(
    name: str,
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    train_time: float = 0.0,
) -> Dict[str, Any]:
    n = len(y_test)
    n_cls = len(class_names)

    t0 = time.perf_counter()
    proba = model.predict_proba(X_test)
    infer = time.perf_counter() - t0
    lat_us = (infer / n) * 1e6 if n > 0 else 0.0
    y_pred = np.argmax(proba, axis=1)

    acc      = float(np.mean(y_pred == y_test))
    bal_acc  = float(balanced_accuracy_score(y_test, y_pred))
    mac_prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0, labels=list(range(n_cls))))
    mac_rec  = float(recall_score(y_test, y_pred, average="macro", zero_division=0, labels=list(range(n_cls))))
    mac_f1   = float(f1_score(y_test, y_pred, average="macro", zero_division=0, labels=list(range(n_cls))))
    wgt_prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    wgt_rec  = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    wgt_f1   = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    try:
        ll = float(log_loss(y_test, proba, labels=list(range(n_cls))))
    except Exception:
        ll = None

    try:
        roc = float(roc_auc_score(y_test, proba, multi_class="ovr", average="macro", labels=list(range(n_cls))))
    except Exception:
        roc = None

    per_prec = precision_score(y_test, y_pred, average=None, zero_division=0, labels=list(range(n_cls)))
    per_rec  = recall_score(y_test, y_pred, average=None, zero_division=0, labels=list(range(n_cls)))
    per_f1   = f1_score(y_test, y_pred, average=None, zero_division=0, labels=list(range(n_cls)))

    per_class = {}
    for i, cls in enumerate(class_names):
        per_class[cls] = {
            "precision": round(float(per_prec[i]), 4),
            "recall":    round(float(per_rec[i]),  4),
            "f1":        round(float(per_f1[i]),   4),
            "support":   int(np.sum(y_test == i)),
        }

    cm = confusion_matrix(y_test, y_pred, labels=list(range(n_cls)))

    return {
        "model_name":          name,
        "train_time_s":        round(train_time, 3),
        "infer_time_s":        round(infer, 3),
        "latency_us":          round(lat_us, 2),
        "metrics": {
            "accuracy":          round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "macro_precision":   round(mac_prec, 4),
            "macro_recall":      round(mac_rec, 4),
            "macro_f1":          round(mac_f1, 4),
            "weighted_precision": round(wgt_prec, 4),
            "weighted_recall":   round(wgt_rec, 4),
            "weighted_f1":       round(wgt_f1, 4),
            "roc_auc_ovr_macro": round(roc, 4) if roc is not None else None,
            "log_loss":          round(ll, 4)  if ll  is not None else None,
        },
        "per_class_metrics": per_class,
        "confusion_matrix":  cm.tolist(),
        "raw_predictions": {"y_pred": y_pred, "y_pred_proba": proba},
    }


def _save_cm_figure(
    cm: np.ndarray,
    class_names: List[str],
    title: str,
    path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        cm, annot=True, fmt=",d", cmap="YlOrRd",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.4, linecolor="#e5e7eb",
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_comparison_chart(
    models_data: Dict[str, Dict[str, Any]],
    path: Path,
) -> Path:
    keys   = ["macro_f1", "macro_recall", "macro_precision", "balanced_accuracy", "weighted_f1"]
    labels = ["Macro F1", "Macro Recall", "Macro Precision", "Balanced Acc", "Weighted F1"]
    dnames = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest", "xgboost": "XGBoost"}

    rows = []
    for k, d in models_data.items():
        row = {"Model": dnames.get(k, k)}
        for mk in keys:
            row[mk] = d["metrics"].get(mk) or 0.0
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")
    df.columns = labels

    fig, ax = plt.subplots(figsize=(11, 5))
    df.plot(kind="bar", ax=ax, width=0.75)
    ax.set_title("Attack-Only 9-Class Experiment — Model Comparison", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    plt.xticks(rotation=0)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_per_class_f1_chart(
    models_data: Dict[str, Dict[str, Any]],
    class_names: List[str],
    path: Path,
) -> Path:
    dnames = {"logistic_regression": "LR", "random_forest": "RF", "xgboost": "XGBoost"}
    colors = {"logistic_regression": "#2563eb", "random_forest": "#10b981", "xgboost": "#f59e0b"}

    x = np.arange(len(class_names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (k, d) in enumerate(models_data.items()):
        vals = [d["per_class_metrics"][c]["f1"] for c in class_names]
        ax.bar(x + i * w, vals, w, label=dnames.get(k, k),
               color=colors.get(k, "#6b7280"), alpha=0.85)
    for i, c in enumerate(class_names):
        if c in RARE_CLASSES:
            ax.axvspan(i - 0.3, i + 0.85, alpha=0.08, color="red")
    ax.set_title("Attack-Only: Per-Class F1 (rare classes highlighted)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Attack Category"); ax.set_ylabel("F1 Score")
    ax.set_xticks(x + w); ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=10)
    ax.set_ylim(0, 1.05); ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_rare_class_recall_chart(
    models_data: Dict[str, Dict[str, Any]],
    path: Path,
) -> Path:
    """Bar chart focused on the four rare classes."""
    dnames = {"logistic_regression": "LR", "random_forest": "RF", "xgboost": "XGBoost"}
    colors = {"logistic_regression": "#2563eb", "random_forest": "#10b981", "xgboost": "#f59e0b"}

    x = np.arange(len(RARE_CLASSES))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (k, d) in enumerate(models_data.items()):
        vals = [d["per_class_metrics"].get(c, {}).get("recall", 0) for c in RARE_CLASSES]
        ax.bar(x + i * w, vals, w, label=dnames.get(k, k),
               color=colors.get(k, "#6b7280"), alpha=0.85)
    ax.set_title("Rare-Class Recall — Attack-Only Experiment", fontsize=13, fontweight="bold")
    ax.set_xlabel("Rare Attack Category"); ax.set_ylabel("Recall")
    ax.set_xticks(x + w); ax.set_xticklabels(RARE_CLASSES, fontsize=11)
    ax.set_ylim(0, 1.05); ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _select_best(
    val_models_data: Dict[str, Dict[str, Any]],
    class_names: List[str],
) -> Tuple[str, Dict[str, Any]]:
    scores = {}
    for k, d in val_models_data.items():
        m = d["metrics"]
        rare_recalls = [
            d["per_class_metrics"].get(c, {}).get("recall", 0.0)
            for c in RARE_CLASSES if c in d["per_class_metrics"]
        ]
        mean_rare = float(np.mean(rare_recalls)) if rare_recalls else 0.0
        composite = (
            (m.get("macro_f1") or 0.0) * 0.40
            + (m.get("macro_recall") or 0.0) * 0.25
            + mean_rare * 0.25
            + (m.get("balanced_accuracy") or 0.0) * 0.10
        )
        scores[k] = composite
        logger.info(
            "[VALIDATION] %s composite=%.4f (macro_f1=%.4f macro_rec=%.4f rare_rec=%.4f bal_acc=%.4f)",
            k, composite, m.get("macro_f1", 0), m.get("macro_recall", 0),
            mean_rare, m.get("balanced_accuracy", 0),
        )

    best = max(scores, key=scores.get)
    bd   = val_models_data[best]
    return best, {
        "selected_model": best,
        "selection_stage": "validation",
        "test_data_used_for_selection": False,
        "selection_criteria": "Composite: Macro F1 (40%) + Macro Recall (25%) + Mean Rare-Class Recall (25%) + Balanced Accuracy (10%) evaluated strictly on validation split",
        "composite_scores": {k: round(v, 4) for k, v in scores.items()},
        "winning_metrics": bd["metrics"],
        "rare_class_metrics": {
            c: bd["per_class_metrics"][c]
            for c in RARE_CLASSES if c in bd["per_class_metrics"]
        },
        "train_time_s": bd["train_time_s"],
        "latency_us":   bd["latency_us"],
    }


# ─── Evaluation orchestrator ──────────────────────────────────────────────────

def evaluate_attack_only_models(
    ao_splits: Dict[str, Any],
    trained_models: Optional[Dict[str, Any]] = None,
    train_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluate candidate attack-only models on the validation split for model selection,
    select the best model without test leakage, and then evaluate on the official
    attack-only test partition for final unbiased evaluation.
    Saves best_model JSON, results JSON, comparison CSV, and figures.
    Does not touch baseline artifacts.
    """
    X_val_sc = ao_splits["X_val_scaled"]
    X_val_un = ao_splits["X_val_unscaled"]
    y_val    = ao_splits["y_val"]

    X_te_sc  = ao_splits["X_test_scaled"]
    X_te_un  = ao_splits["X_test_unscaled"]
    y_te     = ao_splits["y_test"]
    class_names = ao_splits["class_names"]

    def _tt(mname: str) -> float:
        if train_meta:
            return train_meta.get("models", {}).get(mname, {}).get("time_s", 0.0)
        return 0.0

    # Load models (if not passed in)
    lr_model  = (trained_models or {}).get("logistic_regression") or joblib.load(AO_LR_PATH)
    rf_model  = (trained_models or {}).get("random_forest")       or joblib.load(AO_RF_PATH)
    xgb_model = (trained_models or {}).get("xgboost")             or joblib.load(AO_XGB_PATH)

    # ── Phase 1: Evaluate candidates on VALIDATION SPLIT (used for selection) ──
    logger.info("Evaluating candidate models on validation split (%d records)...", len(y_val))
    val_lr  = _eval_single("logistic_regression", lr_model, X_val_sc, y_val, class_names, _tt("logistic_regression"))
    val_rf  = _eval_single("random_forest",        rf_model, X_val_un, y_val, class_names, _tt("random_forest"))
    val_xgb = _eval_single("xgboost",              xgb_model, X_val_un, y_val, class_names, _tt("xgboost"))

    val_models_data = {
        "logistic_regression": val_lr,
        "random_forest":       val_rf,
        "xgboost":             val_xgb,
    }

    # Model selection occurs STRICTLY on validation data
    best_name, sel_meta = _select_best(val_models_data, class_names)
    logger.info("Selected best attack-only model from validation: %s", best_name)

    best_obj = {"logistic_regression": lr_model, "random_forest": rf_model, "xgboost": xgb_model}[best_name]
    joblib.dump(best_obj, AO_BEST_PATH)
    logger.info("Saved best model -> %s", AO_BEST_PATH)

    # ── Phase 2: Evaluate on OFFICIAL HELD-OUT TEST PARTITION (final evaluation) ──
    logger.info("Evaluating on official attack test partition (%d records)...", len(y_te))
    test_lr  = _eval_single("logistic_regression", lr_model, X_te_sc,  y_te, class_names, _tt("logistic_regression"))
    test_rf  = _eval_single("random_forest",        rf_model, X_te_un, y_te, class_names, _tt("random_forest"))
    test_xgb = _eval_single("xgboost",              xgb_model, X_te_un, y_te, class_names, _tt("xgboost"))

    test_models_data = {
        "logistic_regression": test_lr,
        "random_forest":       test_rf,
        "xgboost":             test_xgb,
    }
    sel_test_eval = test_models_data[best_name]

    # ── Figures ───────────────────────────────────────────────────────────────
    fig_paths = {}
    for key, (ev, suffix) in {
        "logistic_regression": (test_lr,  "logistic_regression"),
        "random_forest":       (test_rf,  "random_forest"),
        "xgboost":             (test_xgb, "xgboost"),
    }.items():
        p = FIGURES_DIR / f"attack_only_multiclass_confusion_matrix_{suffix}.png"
        fig_paths[key] = _save_cm_figure(
            np.array(ev["confusion_matrix"]), class_names,
            f"Attack-Only 9-Class — {suffix.replace('_', ' ').title()} — Confusion Matrix",
            p,
        )
        logger.info("Saved CM -> %s", p)

    comp_path  = _save_comparison_chart(test_models_data,
                                        FIGURES_DIR / "attack_only_multiclass_model_comparison.png")
    f1_path    = _save_per_class_f1_chart(test_models_data, class_names,
                                          FIGURES_DIR / "attack_only_multiclass_per_class_f1.png")
    rare_path  = _save_rare_class_recall_chart(test_models_data,
                                               FIGURES_DIR / "attack_only_multiclass_rare_class_recall.png")
    logger.info("Saved comparison, per-class F1, and rare-class recall charts.")

    # ── Comparison CSV ────────────────────────────────────────────────────────
    csv_rows = []
    # Validation rows (used for model selection)
    for k, d in val_models_data.items():
        m = d["metrics"]
        row = {
            "model": k,
            "split": "validation (used for model selection)",
            "selected": (k == best_name),
            "composite_score": sel_meta["composite_scores"].get(k, ""),
        }
        row.update({kk: (vv if vv is not None else "") for kk, vv in m.items()})
        row["train_time_s"] = d["train_time_s"]
        row["latency_us"]   = d["latency_us"]
        for c in RARE_CLASSES:
            pc = d["per_class_metrics"].get(c, {})
            row[f"{c}_precision"] = pc.get("precision", "")
            row[f"{c}_recall"]    = pc.get("recall", "")
            row[f"{c}_f1"]        = pc.get("f1", "")
        csv_rows.append(row)

    # Final test evaluation row (unbiased evaluation)
    m_test = sel_test_eval["metrics"]
    test_row = {
        "model": f"{best_name} (final unbiased test)",
        "split": "test (held-out final evaluation)",
        "selected": True,
        "composite_score": "",
    }
    test_row.update({kk: (vv if vv is not None else "") for kk, vv in m_test.items()})
    test_row["train_time_s"] = sel_test_eval["train_time_s"]
    test_row["latency_us"]   = sel_test_eval["latency_us"]
    for c in RARE_CLASSES:
        pc = sel_test_eval["per_class_metrics"].get(c, {})
        test_row[f"{c}_precision"] = pc.get("precision", "")
        test_row[f"{c}_recall"]    = pc.get("recall", "")
        test_row[f"{c}_f1"]        = pc.get("f1", "")
    csv_rows.append(test_row)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(csv_rows).to_csv(AO_COMPARISON_CSV, index=False)
    logger.info("Saved comparison CSV -> %s", AO_COMPARISON_CSV)

    # ── Best Model JSON ───────────────────────────────────────────────────────
    best_model_meta = {
        "selected_model": best_name,
        "selection_stage": "validation",
        "selection_criteria": sel_meta["selection_criteria"],
        "test_data_used_for_selection": False,
        "validation_records_count": int(len(y_val)),
        "test_records_count": int(len(y_te)),
        "composite_scores_validation": sel_meta["composite_scores"],
        "validation_metrics": val_models_data[best_name]["metrics"],
        "rare_class_metrics_validation": {
            c: val_models_data[best_name]["per_class_metrics"][c]
            for c in RARE_CLASSES if c in val_models_data[best_name]["per_class_metrics"]
        },
        "final_test_metrics": sel_test_eval["metrics"],
        "rare_class_metrics_test": {
            c: sel_test_eval["per_class_metrics"][c]
            for c in RARE_CLASSES if c in sel_test_eval["per_class_metrics"]
        },
        "training_time_sec": val_models_data[best_name]["train_time_s"],
        "per_sample_latency_us": sel_test_eval["latency_us"],
    }
    with open(AO_BEST_JSON, "w", encoding="utf-8") as f:
        json.dump(best_model_meta, f, indent=2)
    logger.info("Saved best model JSON -> %s", AO_BEST_JSON)

    # ── Clean test and val results for results JSON ───────────────────────────
    val_clean = {}
    for k, d in val_models_data.items():
        val_clean[k] = {
            "model_name":        d["model_name"],
            "split":             "validation",
            "train_time_s":      d["train_time_s"],
            "infer_time_s":      d["infer_time_s"],
            "latency_us":        d["latency_us"],
            "metrics":           d["metrics"],
            "per_class_metrics": d["per_class_metrics"],
            "confusion_matrix":  d["confusion_matrix"],
        }

    test_clean = {}
    for k, d in test_models_data.items():
        test_clean[k] = {
            "model_name":        d["model_name"],
            "split":             "test",
            "train_time_s":      d["train_time_s"],
            "infer_time_s":      d["infer_time_s"],
            "latency_us":        d["latency_us"],
            "metrics":           d["metrics"],
            "per_class_metrics": d["per_class_metrics"],
            "confusion_matrix":  d["confusion_matrix"],
        }

    full_output = {
        "experiment":              "attack_only_multiclass",
        "description":             "9-class attack classifier trained on label==1 records only. Model selection performed strictly on validation split.",
        "evaluation_timestamp":    pd.Timestamp.now().isoformat(),
        "validation_attack_records": int(len(y_val)),
        "test_attack_records":     int(len(y_te)),
        "class_names":             class_names,
        "validation_results":      val_clean,
        "selection":               sel_meta,
        "final_test_evaluation": {
            "selected_model":      best_name,
            "split":               "test",
            "test_attack_records": int(len(y_te)),
            "metrics":             sel_test_eval["metrics"],
            "per_class_metrics":   sel_test_eval["per_class_metrics"],
            "confusion_matrix":    sel_test_eval["confusion_matrix"],
            "latency_us":          sel_test_eval["latency_us"],
        },
        "models":                  test_clean,
        "figures":                 [str(v) for v in list(fig_paths.values()) + [comp_path, f1_path, rare_path]],
    }
    with open(AO_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    logger.info("Saved results JSON -> %s", AO_RESULTS_JSON)

    return full_output


# ─── Full orchestrator ────────────────────────────────────────────────────────

def run_attack_only_experiment() -> Dict[str, Any]:
    """Load data, train, evaluate, return results dict."""
    logger.info("=== Attack-Only Multiclass Experiment START ===")

    logger.info("Loading preprocessed splits from %s...", PROCESSED_DIR)
    splits = load_splits(PROCESSED_DIR)

    logger.info("Building attack-only data splits...")
    ao_splits = build_attack_only_splits(splits)

    logger.info("Training models...")
    train_results = train_attack_only_models(ao_splits)

    logger.info("Evaluating models with leakage-free validation selection...")
    eval_results = evaluate_attack_only_models(
        ao_splits,
        trained_models=train_results["models"],
        train_meta=train_results["training_meta"],
    )

    logger.info("=== Attack-Only Multiclass Experiment COMPLETE ===")
    return eval_results


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _print_summary(out: Dict[str, Any]) -> None:
    sel = out["selection"]
    best_name = sel["selected_model"]
    print("\n" + "=" * 76)
    print("ATTACK-ONLY 9-CLASS EXPERIMENT — MODEL SELECTION (VALIDATION DATA)")
    print("=" * 76)
    print(f"Validation attack records  : {out.get('validation_attack_records', 'N/A')}")
    print(f"Classes                    : {out['class_names']}")
    print(f"Selection criteria         : {sel['selection_criteria']}")
    print(f"Test data used in selection: {sel.get('test_data_used_for_selection', False)}")
    print(f"Composite scores (val)     : {sel['composite_scores']}")

    for mname, info in out.get("validation_results", {}).items():
        m = info["metrics"]
        print(f"\n[Validation] Model: {mname.upper()}")
        print(f"  Balanced Accuracy : {m['balanced_accuracy']:.4f}")
        print(f"  Macro Recall      : {m['macro_recall']:.4f}")
        print(f"  Macro F1          : {m['macro_f1']:.4f}")
        print("  Rare-class recall (val):")
        for rc in RARE_CLASSES:
            pc = info.get("per_class_metrics", {}).get(rc, {})
            print(f"    {rc:<18}: recall={pc.get('recall', 0):.4f}  f1={pc.get('f1', 0):.4f}")

    print("\n" + "-" * 76)
    print(f"SELECTED MODEL: {best_name.upper()} (selected strictly using validation data)")
    print("-" * 76)

    test_eval = out["final_test_evaluation"]
    tm = test_eval["metrics"]
    print("\n" + "=" * 76)
    print(f"FINAL UNBIASED EVALUATION ON OFFICIAL TEST SET ({test_eval.get('test_attack_records', out.get('test_attack_records', 'N/A'))} records)")
    print(f"Selected Model: {best_name.upper()}")
    print("=" * 76)
    print(f"  Accuracy          : {tm['accuracy']:.4f}")
    print(f"  Balanced Accuracy : {tm['balanced_accuracy']:.4f}")
    print(f"  Macro Precision   : {tm['macro_precision']:.4f}")
    print(f"  Macro Recall      : {tm['macro_recall']:.4f}")
    print(f"  Macro F1          : {tm['macro_f1']:.4f}")
    print(f"  Weighted F1       : {tm['weighted_f1']:.4f}")
    auc = tm.get("roc_auc_ovr_macro")
    ll  = tm.get("log_loss")
    print(f"  ROC-AUC (OvR)     : {auc:.4f}" if auc else "  ROC-AUC (OvR)     : N/A")
    print(f"  Log Loss          : {ll:.4f}"  if ll  else "  Log Loss          : N/A")
    print(f"  Latency           : {test_eval['latency_us']:.2f} µs/sample")
    print("  Rare-class recall (test):")
    for rc in RARE_CLASSES:
        pc = test_eval.get("per_class_metrics", {}).get(rc, {})
        print(f"    {rc:<18}: recall={pc.get('recall', 0):.4f}  f1={pc.get('f1', 0):.4f}")

    print("\n" + "=" * 76)
    print("Artifacts saved:")
    print(f"  Best model       : {AO_BEST_PATH}")
    print(f"  Best model JSON  : {AO_BEST_JSON}")
    print(f"  Results JSON     : {AO_RESULTS_JSON}")
    print(f"  Comparison CSV   : {AO_COMPARISON_CSV}")
    print("=" * 76 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Attack-Only 9-Class Multiclass Experiment")
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training and only evaluate already-saved models.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.eval_only:
        splits = load_splits(PROCESSED_DIR)
        ao_splits = build_attack_only_splits(splits)
        out = evaluate_attack_only_models(ao_splits)
    else:
        out = run_attack_only_experiment()

    _print_summary(out)


if __name__ == "__main__":
    main()
