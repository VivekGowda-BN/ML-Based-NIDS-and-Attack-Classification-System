# UNSW-NB15 Exploratory Data Analysis (EDA) Report (Mode 1)

**Generated:** 2026-09-20T21:38:15.130166  
**Data Sources:**  
- Cleaned Training Split: `data/interim/UNSW_NB15_training_clean.parquet` (175,341 rows)  
- Cleaned Testing Split: `data/interim/UNSW_NB15_testing_clean.parquet` (82,332 rows)  
**Model Input Features:** 42 features (39 numeric, 3 categorical)  
**Excluded Non-Feature Columns:** `id, attack_cat, label`

---

## 1. Executive Summary & Key Findings

1. **Target Distribution Inversion**:
   - In the **Training set**, Attacks account for **68.06%** (119,341 flows) and Normal traffic accounts for **31.94%** (56,000 flows).
   - In the **Testing set**, Attacks account for **55.06%** (45,332 flows) and Normal traffic accounts for **44.94%** (37,000 flows).
   - The higher proportion of normal records in the test set reflects realistic deployment testing where benign flows dominate.
2. **Extreme Multiclass Imbalance**:
   - Four rare attack categories (`Analysis`, `Backdoor`, `Shellcode`, `Worms`) comprise **< 3%** of training flows combined.
   - Specifically, `Worms` contains only **130 records in training** (0.074%) and **44 records in testing** (0.053%).
3. **Severe Multicollinearity**:
   - Detected **16 feature pairs** with correlation $|r| \ge 0.90$.
   - Perfect correlation ($r = 1.0000$) between `is_ftp_login` and `ct_ftp_cmd`.
   - Very high correlation ($r > 0.99$) between byte counts and packet loss (`sbytes` $\leftrightarrow$ `sloss`, `dbytes` $\leftrightarrow$ `dloss`).
   - `swin` and `dwin` have $r = 0.9901$.
4. **Categorical Domain Shift**:
   - `proto`: 133 unique in Train, 131 in Test. Zero test-only protocols.
   - `service`: 13 unique in Train, 13 in Test (all align).
   - `state`: 9 unique in Train, 7 in Test. **Critical finding:** Test contains 2 states (`ACC` with 4 records, `CLO` with 1 record) that never appear in Train! Categorical encoders must use `handle_unknown="ignore"`.
5. **Data Quality Status**:
   - 0 missing / NaN values and 0 infinite values across all 42 features in both splits.

---

## 2. Target Distributions

### 2.1 Binary Classification Target (`label`)

| Split | Normal (`0`) | Normal % | Attack (`1`) | Attack % | Total Records |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Training** | 56,000 | 31.94% | 119,341 | 68.06% | 175,341 |
| **Testing** | 37,000 | 44.94% | 45,332 | 55.06% | 82,332 |

### 2.2 Multiclass Attack Categories (`attack_cat`)

| Category | Train Count | Train % | Test Count | Test % | Preprocessing Priority |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Analysis** | 2,000 | 1.141% | 677 | 0.822% | Rare (High Risk of Misclassification) |
| **Backdoor** | 1,746 | 0.996% | 583 | 0.708% | Rare (High Risk of Misclassification) |
| **DoS** | 12,264 | 6.994% | 4,089 | 4.966% | Standard |
| **Exploits** | 33,393 | 19.045% | 11,132 | 13.521% | Standard |
| **Fuzzers** | 18,184 | 10.371% | 6,062 | 7.363% | Standard |
| **Generic** | 40,000 | 22.813% | 18,871 | 22.921% | Standard |
| **Normal** | 56,000 | 31.938% | 37,000 | 44.94% | Standard |
| **Reconnaissance** | 10,491 | 5.983% | 3,496 | 4.246% | Standard |
| **Shellcode** | 1,133 | 0.646% | 378 | 0.459% | Rare (High Risk of Misclassification) |
| **Worms** | 130 | 0.074% | 44 | 0.053% | Rare (High Risk of Misclassification) |

---

## 3. Categorical Feature Distributions & Domain Shift

| Feature | Unique (Train) | Unique (Test) | Test-Only Unseen Categories | Train-Only Categories | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `proto` | 133 | 131 | None (`[]`) | `icmp, rtp` | Dominated by `tcp` and `udp`. |
| `service` | 13 | 13 | None (`[]`) | None (`[]`) | `-` denotes unclassified application service. |
| `state` | 9 | 7 | **`CLO, ACC`** | `ECO, URN, PAR, no` | **Warning:** `ACC` and `CLO` occur in test only! |

---

## 4. Numeric Feature Variance & Collinearity

### 4.1 Lowest Variance Features (Training)
- `ackdat`: Variance = `0.001641`
- `synack`: Variance = `0.001884`
- `tcprtt`: Variance = `0.006297`
- `is_sm_ips_ports`: Variance = `0.015504`
- `is_ftp_login`: Variance = `0.015888`
- `ct_ftp_cmd`: Variance = `0.015888`
- `ct_flw_http_mthd`: Variance = `0.491692`
- `trans_depth`: Variance = `0.60359`

### 4.2 Top Multicollinear Feature Pairs ($|r| \ge 0.90$)

| Feature 1 | Feature 2 | Pearson Correlation ($r$) | Implication |
| :--- | :--- | :--- | :--- |
| `is_ftp_login` | `ct_ftp_cmd` | **1.0** | Redundant information / collinearity |
| `dbytes` | `dloss` | **0.9965** | Redundant information / collinearity |
| `sbytes` | `sloss` | **0.9961** | Redundant information / collinearity |
| `swin` | `dwin` | **0.9901** | Redundant information / collinearity |
| `ct_srv_src` | `ct_srv_dst` | **0.9803** | Redundant information / collinearity |
| `dpkts` | `dloss` | **0.9786** | Redundant information / collinearity |
| `ct_dst_src_ltm` | `ct_srv_dst` | **0.9724** | Redundant information / collinearity |
| `dpkts` | `dbytes` | **0.9719** | Redundant information / collinearity |
| `spkts` | `sloss` | **0.9711** | Redundant information / collinearity |
| `ct_srv_src` | `ct_dst_src_ltm` | **0.9671** | Redundant information / collinearity |
| `spkts` | `sbytes` | **0.9638** | Redundant information / collinearity |
| `ct_dst_ltm` | `ct_src_dport_ltm` | **0.9621** | Redundant information / collinearity |
| `tcprtt` | `synack` | **0.9495** | Redundant information / collinearity |
| `tcprtt` | `ackdat` | **0.9418** | Redundant information / collinearity |
| `sinpkt` | `is_sm_ips_ports` | **0.9413** | Redundant information / collinearity |
| `ct_src_dport_ltm` | `ct_dst_sport_ltm` | **0.9068** | Redundant information / collinearity |

---

## 5. Rare Attack Categories Profile

| Rare Category | Train Records | Test Records | Train/Test Ratio | Observation |
| :--- | :--- | :--- | :--- | :--- |
| **Analysis** | 2,000 (1.141%) | 677 (0.822%) | 2.95 | Extreme scarcity; requires class weighting or macro-averaging. |
| **Backdoor** | 1,746 (0.996%) | 583 (0.708%) | 2.99 | Extreme scarcity; requires class weighting or macro-averaging. |
| **Shellcode** | 1,133 (0.646%) | 378 (0.459%) | 3.0 | Extreme scarcity; requires class weighting or macro-averaging. |
| **Worms** | 130 (0.074%) | 44 (0.053%) | 2.95 | Extreme scarcity; requires class weighting or macro-averaging. |

---

## 6. Generated Visualizations

- `reports/figures/eda_binary_distribution.png`: Binary label distributions.
- `reports/figures/eda_attack_categories.png`: Log-scale comparison across all 10 attack classes.
- `reports/figures/eda_numeric_correlations.png`: Heatmap of core traffic volume and temporal features.
- `reports/figures/eda_top_protocols.png`: Top protocols breakdown.
- `reports/figures/eda_top_services.png`: Service distribution breakdown.
- `reports/figures/eda_rare_attacks_breakdown.png`: Scarcity inspection for low-volume attacks.
