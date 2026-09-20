# UNSW-NB15 Leakage-Safe Preprocessing Report (Mode 1)

**Generated:** 2026-09-20T21:57:32.119347  
**Random State:** `42`  
**Training Split:** `C:\Users\vivek\OneDrive\Desktop\College\Year 2\DS\NIDS - Mini Project\data\interim\UNSW_NB15_training_clean.parquet`  
**Testing Split:** `C:\Users\vivek\OneDrive\Desktop\College\Year 2\DS\NIDS - Mini Project\data\interim\UNSW_NB15_testing_clean.parquet`  

---

## 1. Feature Space & Column Allocation

- **Raw Columns Count:** 45
- **Excluded Non-Feature Columns (3):** `id, attack_cat, label`
  - `id`: Row identifier (prevents index memorization)
  - `label`: Binary classification target (0 = Normal, 1 = Attack)
  - `attack_cat`: Multiclass attack category target (10 distinct classes)
- **Model Input Features Detected (Training set only):** **42** columns
  - **Numeric Features (39):** `dur, spkts, dpkts, sbytes, dbytes, rate, sttl, dttl, sload, dload, sloss, dloss, sinpkt, dinpkt, sjit, djit, swin, stcpb, dtcpb, dwin, tcprtt, synack, ackdat, smean, dmean, trans_depth, response_body_len, ct_srv_src, ct_state_ttl, ct_dst_ltm, ct_src_dport_ltm, ct_dst_sport_ltm, ct_dst_src_ltm, is_ftp_login, ct_ftp_cmd, ct_flw_http_mthd, ct_src_ltm, ct_srv_dst, is_sm_ips_ports`
  - **Categorical Features (3):** `proto, service, state`

---

## 2. Transformation Pipeline Architecture

| Channel | Transformations Applied | Rationale |
| :--- | :--- | :--- |
| **Numeric** | `SimpleImputer(strategy="median")` $\rightarrow$ `StandardScaler()` *(scaled config)* | Median handles right-skewed network volumes; scaling normalizes variance for linear/distance models. |
| **Numeric (Tree)** | `SimpleImputer(strategy="median")` *(unscaled config)* | Trees are invariant to monotonic scaling; preserves raw interpretable units. |
| **Categorical** | `SimpleImputer(strategy="most_frequent")` $\rightarrow$ `OneHotEncoder(handle_unknown="ignore")` | `handle_unknown="ignore"` prevents failures on test-only states (`ACC`, `CLO`). |

---

## 3. Preprocessing Dimensions & Matrix Shapes

| Dataset Split | Cleaned Records | Transformed Matrix Shape (Scaled) | Transformed Matrix Shape (Unscaled) | Column Consistency |
| :--- | :--- | :--- | :--- | :--- |
| **Training Set** | 175,341 | `175,341 x 194` | `175,341 x 194` | **Aligned** |
| **Testing Set** | 82,332 | `82,332 x 194` | `82,332 x 194` | **Aligned** |

- **Total Transformed Features:** **194** columns (39 numeric + 155 one-hot columns derived from 133 protocols, 13 services, 9 training states).

---

## 4. Multiclass Label Encoding (`attack_cat`)

The `LabelEncoder` was fitted **exclusively on the training split**. Classes mapped as follows:

| Class Index | Attack Category |
| :--- | :--- |
| `0` | **Analysis** |
| `1` | **Backdoor** |
| `2` | **DoS** |
| `3` | **Exploits** |
| `4` | **Fuzzers** |
| `5` | **Generic** |
| `6` | **Normal** |
| `7` | **Reconnaissance** |
| `8` | **Shellcode** |
| `9` | **Worms** |

---

## 5. Persisted Artifacts

### 5.1 Models Directory (`models/`)
- `preprocessor_scaled.joblib`: Fitted ColumnTransformer with StandardScaler.
- `preprocessor_unscaled.joblib`: Fitted ColumnTransformer with unscaled numeric features.
- `scaler.joblib`: Scaler artifact for inference.
- `label_encoder.joblib`: Fitted LabelEncoder for multiclass attack categories.

### 5.2 Processed Data Directory (`data/processed/`)
- `X_train_scaled.npy` (175,341 x 194)
- `X_test_scaled.npy` (82,332 x 194)
- `X_train_unscaled.npy` (175,341 x 194)
- `X_test_unscaled.npy` (82,332 x 194)
- `y_bin_train.npy`, `y_bin_test.npy` (binary targets)
- `y_multi_train.npy`, `y_multi_test.npy` (multiclass targets)
- `feature_names.json`: Complete record of input, transformed, and excluded feature names.
