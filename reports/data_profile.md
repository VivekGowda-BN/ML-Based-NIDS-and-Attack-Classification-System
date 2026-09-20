# UNSW-NB15 Dataset Verification & Profiling Report (Mode 1)

**Generated:** 2026-09-20T21:27:02.356784  
**Target Columns:** Binary: `label` | Multiclass: `attack_cat`

---

## 1. File Integrity & Physical Metadata

| Metric | Training Set | Testing Set |
| :--- | :--- | :--- |
| **Filename** | `UNSW_NB15_training-set.csv` | `UNSW_NB15_testing-set.csv` |
| **File Size** | 30.8 MB (32,293,018 bytes) | 14.67 MB (15,380,800 bytes) |
| **Rows** | 175,341 | 82,332 |
| **Columns** | 45 | 45 |
| **SHA-256** | `bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa` | `734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559` |

---

## 2. Schema Consistency Check

- **Schemas Identical:** `YES`
- **Column Count Train:** 45
- **Column Count Test:** 45
- **Column Order Aligned:** `YES`
- **Data Type Mismatches:** 0

---

## 3. Data Quality (Missing, Infinite, and Duplicate Values)

| Quality Check | Training Set | Testing Set | Status |
| :--- | :--- | :--- | :--- |
| **Missing / NaN Values** | 0 | 0 | Clean |
| **Infinite Values (+/- inf)** | 0 | 0 | Clean |
| **Exact Duplicate Rows (all cols)** | 0 | 0 | Clean |
| **Duplicates excluding `id`** | 67,601 | 26,387 | Expected flow duplicates |

---

## 4. Target Distributions

### 4.1 Binary Target (`label`)

| Label | Meaning | Train Count | Train % | Test Count | Test % |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `0` | Normal | 56,000 | 31.938% | 37,000 | 44.94% |
| `1` | Attack | 119,341 | 68.062% | 45,332 | 55.06% |

### 4.2 Multiclass Target (`attack_cat`)

| Attack Category | Train Count | Train % | Test Count | Test % |
| :--- | :--- | :--- | :--- | :--- |
| **Analysis** | 2,000 | 1.141% | 677 | 0.822% |
| **Backdoor** | 1,746 | 0.996% | 583 | 0.708% |
| **DoS** | 12,264 | 6.994% | 4,089 | 4.966% |
| **Exploits** | 33,393 | 19.045% | 11,132 | 13.521% |
| **Fuzzers** | 18,184 | 10.371% | 6,062 | 7.363% |
| **Generic** | 40,000 | 22.813% | 18,871 | 22.921% |
| **Normal** | 56,000 | 31.938% | 37,000 | 44.94% |
| **Reconnaissance** | 10,491 | 5.983% | 3,496 | 4.246% |
| **Shellcode** | 1,133 | 0.646% | 378 | 0.459% |
| **Worms** | 130 | 0.074% | 44 | 0.053% |

---

## 5. Train / Test Record Overlap Analysis

- **Unique overlapping feature records (excluding `id`):** 940
- **Observation:** Detected 940 identical flow feature patterns appearing in both train and test splits (common in network flow captures due to repetitive background protocols or repeated scans). Models should avoid relying on record memorization.

---

## 6. Generated Figures

- `reports/figures/binary_label_distribution.png`: Comparison of normal vs attack traffic.
- `reports/figures/attack_category_distribution.png`: Breakdown of attack categories.
