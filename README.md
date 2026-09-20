# ML-Based Network Intrusion Detection System (Mode 1)

> **Status:** Environment configured — scaffold complete, implementation in progress.

---

## Overview

This project builds a **machine-learning-based Network Intrusion Detection System (NIDS)**
that operates in **Mode 1: offline replay**.

| Property | Value |
|---|---|
| Dataset | UNSW-NB15 |
| Task 1 | Binary classification — Normal vs Attack |
| Task 2 | Multiclass classification — Attack category |
| Inference mode | Offline replay of held-out test records |
| Real traffic | ❌ Never — no live packet capture |
| External scanning | ❌ Never — no active network probing |

---

## Planned Architecture

```
data/raw/           ← UNSW-NB15 CSV files (not committed)
      │
      ▼
 nids.dataset       ← load & validate raw files
      │
      ▼
 nids.cleaning      ← remove duplicates, handle nulls
      │
      ▼
 nids.preprocessing ← encode, scale, train/test split
      │
      ▼
 nids.features      ← feature selection / engineering
      │
      ├──▶ nids.train (binary)      → models/binary_clf.joblib
      └──▶ nids.train (multiclass)  → models/multi_clf.joblib
                │
                ▼
          nids.evaluate   → reports/metrics/
          nids.explain    → reports/explanations/  (SHAP)
                │
                ▼
          nids.replay     ← feed scenario CSV row-by-row
                │
                ▼
          dashboard/app.py  ← Streamlit alert dashboard
```

---

## UNSW-NB15 Attack Categories

| Label | Category |
|---|---|
| 0 | Normal |
| 1 | Fuzzers |
| 2 | Analysis |
| 3 | Backdoors |
| 4 | DoS |
| 5 | Exploits |
| 6 | Generic |
| 7 | Reconnaissance |
| 8 | Shellcode |
| 9 | Worms |

---

## Environment Setup

### Requirements

| Requirement | Version |
|---|---|
| Python | ≥ 3.10 (tested on 3.13) |
| pip | ≥ 23 |

### 1 — Create a virtual environment

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

> **Tip:** Your prompt will show `(.venv)` when the environment is active.

### 2 — Install the `nids` package in editable mode

**Windows (PowerShell)**
```powershell
# Upgrade pip first
python -m pip install --upgrade pip

# Install the nids package in editable mode (makes `import nids` work)
pip install -e .

# Install all dependencies (including SHAP prebuilt wheel)
pip install -r requirements.txt
```

**macOS / Linux**
```bash
pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
```

### 3 — Explainability Backend (SHAP with Permutation Importance Fallback)

The project leverages **SHAP TreeExplainer** for model interpretability when available. Prebuilt wheels are used for Python 3.13 on Windows (`shap>=0.48.0`). If SHAP is unavailable in any environment, the system automatically falls back to **scikit-learn permutation importance** without errors or crashing.

You can verify which explainability backend is active:
```powershell
python -c "from nids.explain import _SHAP_AVAILABLE; print('SHAP available:', _SHAP_AVAILABLE)"
```

### 4 — (Optional) Install Jupyter notebook support

```powershell
pip install -e ".[notebooks]"
```

### 5 — (Optional) Install developer tools (linting + testing)

```powershell
pip install -e ".[dev]"
```

### 6 — Verify the installation

```powershell
# Check the nids package imports correctly
python -c "import nids; print(nids.__version__)"

# Run the smoke-test suite (no dataset required)
python -m pytest tests/ -v

# Confirm core dependencies (SHAP omitted — it is optional)
python -c "
import numpy, pandas, sklearn, xgboost, streamlit, plotly
print('Core dependencies OK')
"
```

Expected output:
```
0.1.0
...passed
All dependencies OK
```

### Dependency overview

| Package | Purpose |
|---|---|
| `numpy` | Numerical arrays |
| `pandas` | Tabular data, CSV I/O |
| `pyarrow` | Fast Parquet / Arrow backend |
| `scikit-learn` | ML algorithms, preprocessing, metrics |
| `imbalanced-learn` | SMOTE over-sampling for class imbalance |
| `xgboost` | Gradient-boosted tree classifiers |
| `shap` | SHAP explainability for model decisions |
| `joblib` | Model serialisation (`.joblib` files) |
| `matplotlib` | Static plots and confusion matrices |
| `seaborn` | Statistical visualisations |
| `plotly` | Interactive charts in the dashboard |
| `streamlit` | Alert dashboard web app |
| `pydantic` | Input/output schema validation |
| `python-dotenv` | Environment variable management |
| `jupyter` | Exploratory notebooks |
| `pytest` + `pytest-cov` | Test suite and coverage reporting |

---

## Quick Start

```bash
# After completing Environment Setup above:

# 1. Download UNSW-NB15 — see data/README.md for instructions

# 2. Run the full pipeline
make data-prep
make preprocess
make train
make evaluate
make explain

# 3. Launch the Streamlit dashboard
make dashboard
```

---

## Project Structure

```
ML-Based-NIDS-and-Attack-Classification-System/
├── data/               ← datasets (not in git — see data/README.md)
├── models/             ← serialised model files (not in git)
├── notebooks/          ← exploratory analysis
├── reports/            ← figures, metrics, SHAP outputs
├── src/nids/           ← importable Python package
├── dashboard/app.py    ← Streamlit UI
├── tests/              ← pytest suite
├── requirements.txt
├── pyproject.toml
└── Makefile
```

---

## Safety Guarantees

- **No live packet capture.** The system only reads pre-recorded CSV files.
- **No active scanning.** No port scanners, probes, or exploit tools are used.
- **No external network calls.** All inference is fully local and offline.

---

## License

MIT
