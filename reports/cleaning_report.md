# UNSW-NB15 Data Cleaning Report (Mode 1)

**Generated:** 2026-09-20T21:34:11.212775  
**Target Columns:** Binary: `label` | Multiclass: `attack_cat`  
**Non-Feature Columns Excluded:** `id, attack_cat, label` (3 columns)  
**Model Feature Columns Defined:** 42 columns

---

## 1. Before vs After Summary

| Metric | Training (Before) | Training (After) | Testing (Before) | Testing (After) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Row Count** | 175,341 | 175,341 | 82,332 | 82,332 | Exact Preserved |
| **Column Count** | 45 | 45 | 45 | 45 | Preserved (45 cols) |
| **Missing / NaN Values** | 0 | 0 | 0 | 0 | Clean |
| **Infinite Values (+/- inf)** | 0 | 0 | 0 | 0 | Replaced |
| **Exact Duplicate Rows** | 0 | 0 | 0 | 0 | Preserved |

---

## 2. Target Distributions (Cleaned)

### 2.1 Binary Target (`label`)

| Label | Meaning | Training Count | Training % | Testing Count | Testing % |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `0` | Normal | 56,000 | 31.938% | 37,000 | 44.94% |
| `1` | Attack | 119,341 | 68.062% | 45,332 | 55.06% |

### 2.2 Multiclass Target (`attack_cat`)

| Attack Category | Training Count | Training % | Testing Count | Testing % |
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

## 3. Model Feature Column Specification

The following **42** columns are designated as input features for subsequent encoding, scaling, and classification:

`dur, proto, service, state, spkts, dpkts, sbytes, dbytes, rate, sttl, dttl, sload, dload, sloss, dloss, sinpkt, dinpkt, sjit, djit, swin, stcpb, dtcpb, dwin, tcprtt, synack, ackdat, smean, dmean, trans_depth, response_body_len, ct_srv_src, ct_state_ttl, ct_dst_ltm, ct_src_dport_ltm, ct_dst_sport_ltm, ct_dst_src_ltm, is_ftp_login, ct_ftp_cmd, ct_flw_http_mthd, ct_src_ltm, ct_srv_dst, is_sm_ips_ports`

- Excluded `id` to prevent spurious correlation with record index.
- Excluded `label` and `attack_cat` from input space to prevent target leakage.

---

## 4. Persisted Interim Artifacts

- Training Clean Parquet: `C:\Users\vivek\OneDrive\Desktop\College\Year 2\DS\NIDS - Mini Project\data\interim\UNSW_NB15_training_clean.parquet`
- Testing Clean Parquet: `C:\Users\vivek\OneDrive\Desktop\College\Year 2\DS\NIDS - Mini Project\data\interim\UNSW_NB15_testing_clean.parquet`
- Summary Audit JSON: `C:\Users\vivek\OneDrive\Desktop\College\Year 2\DS\NIDS - Mini Project\reports\cleaning_summary.json`
- Audit Report Markdown: `C:\Users\vivek\OneDrive\Desktop\College\Year 2\DS\NIDS - Mini Project\reports\cleaning_report.md`
