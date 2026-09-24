"""
dashboard/app.py — Mode 1 Streamlit Alert Dashboard for ML-Based NIDS.

Mode 1 — Offline UNSW-NB15 Dataset Replay

Safety
------
* Uses only local files (models, parquet scenarios, saved metrics).
* Never sends packets, opens sockets, scans hosts, or calls external URLs.
* Ground-truth fields (label, attack_cat, …) are never displayed in live alerts.

Run
---
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "reports" / "metrics"
FIGURES_DIR = ROOT / "reports" / "figures"
EXPLANATIONS_DIR = ROOT / "reports" / "explanations"
SCENARIOS_DIR = ROOT / "data" / "scenarios"
MODELS_DIR = ROOT / "models"

# Metrics files
BINARY_RESULTS_JSON = METRICS_DIR / "binary_results.json"
BINARY_COMPARISON_CSV = METRICS_DIR / "binary_model_comparison.csv"
BINARY_THRESHOLD_META_JSON = METRICS_DIR / "binary_threshold_selection_metadata.json"
BINARY_THRESHOLD_TEST_JSON = METRICS_DIR / "binary_threshold_selection_test.json"
BINARY_THRESHOLD_COMPARISON_CSV = METRICS_DIR / "binary_threshold_selection_comparison.csv"
MULTICLASS_RESULTS_JSON = METRICS_DIR / "multiclass_results.json"
MULTICLASS_COMPARISON_CSV = METRICS_DIR / "multiclass_model_comparison.csv"
ATTACK_ONLY_RESULTS_JSON = METRICS_DIR / "attack_only_multiclass_results.json"

# ─── Ground-truth fields — NEVER shown in live alert output ──────────────────
FORBIDDEN_FIELDS: frozenset = frozenset({
    "label", "attack_cat", "ground_truth_label",
    "ground_truth_attack_cat", "true_label", "true_attack_category",
})

# ─── Output fields shown in the alert table ───────────────────────────────────
ALERT_DISPLAY_FIELDS: List[str] = [
    "event_id", "timestamp", "prediction", "attack_type",
    "binary_probability", "binary_threshold", "binary_confidence",
    "multiclass_probability", "confidence", "severity", "recommended_action",
    "model_version", "record_index",
]

# ─── Severity colour palette ──────────────────────────────────────────────────
SEVERITY_COLORS: Dict[str, str] = {
    "None": "#2ecc71",
    "Medium": "#f39c12",
    "High": "#e67e22",
    "Critical": "#e74c3c",
}

SCENARIO_DISPLAY_NAMES: Dict[str, str] = {
    "mixed_demo.parquet": "Mixed Demo (all classes)",
    "normal_demo.parquet": "Normal Traffic Only",
    "reconnaissance_demo.parquet": "Reconnaissance Attacks",
    "dos_demo.parquet": "DoS Attacks",
}

EXPECTED_SCENARIOS: List[str] = [
    "mixed_demo.parquet",
    "normal_demo.parquet",
    "reconnaissance_demo.parquet",
    "dos_demo.parquet",
]


# ─────────────────────────────────────────────────────────────────────────────
#  Pure helper utilities (importable without Streamlit runtime)
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file safely; return empty dict if missing or corrupt."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_csv_safe(path: Path) -> pd.DataFrame:
    """Load a CSV file safely; return empty DataFrame if missing."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def get_scenario_files() -> List[str]:
    """Return sorted list of available scenario parquet filenames."""
    if not SCENARIOS_DIR.exists():
        return []
    return [f.name for f in sorted(SCENARIOS_DIR.glob("*.parquet"))]


def events_to_display_df(events: List[Dict]) -> pd.DataFrame:
    """
    Convert prediction-event dicts to a display DataFrame.
    Strips all FORBIDDEN_FIELDS; keeps only ALERT_DISPLAY_FIELDS.
    """
    if not events:
        return pd.DataFrame()
    df = pd.DataFrame(events)
    drop_cols = [c for c in df.columns if c in FORBIDDEN_FIELDS]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    keep = [c for c in ALERT_DISPLAY_FIELDS if c in df.columns]
    return df[keep]


def compute_summary(events: List[Dict]) -> Dict[str, Any]:
    """Compute summary card values from a list of prediction event dicts."""
    total = len(events)
    if total == 0:
        return {
            "total": 0, "normal": 0, "attacks": 0,
            "critical": 0, "avg_confidence": 0.0, "risk_level": "—",
        }
    attacks = sum(1 for e in events if e.get("prediction") == "Attack")
    normal = total - attacks
    critical = sum(1 for e in events if e.get("severity") == "Critical")
    high = sum(1 for e in events if e.get("severity") == "High")
    confs = [e["confidence"] for e in events if "confidence" in e]
    avg_conf = sum(confs) / len(confs) if confs else 0.0

    if critical > 0:
        risk = "🔴 Critical"
    elif high > 0:
        risk = "🟠 High"
    elif attacks > 0:
        risk = "🟡 Medium"
    else:
        risk = "🟢 Normal"

    return {
        "total": total, "normal": normal, "attacks": attacks,
        "critical": critical, "avg_confidence": avg_conf, "risk_level": risk,
    }


def get_predictor_cached(session_state):
    """Load and cache NIDSPredictor into session_state."""
    if "predictor" not in session_state:
        try:
            import sys as _sys
            _sys.path.insert(0, str(ROOT / "src"))
            from nids.predict import NIDSPredictor
            session_state.predictor = NIDSPredictor()
            session_state.predictor_error = None
        except Exception as e:
            session_state.predictor = None
            session_state.predictor_error = str(e)
    return session_state.predictor, session_state.get("predictor_error")


# ─────────────────────────────────────────────────────────────────────────────
#  Streamlit UI — only executed when run via `streamlit run`
# ─────────────────────────────────────────────────────────────────────────────

def _run_dashboard() -> None:
    """Entry point for the Streamlit dashboard (called only at runtime)."""
    import plotly.express as px
    import streamlit as st

    # ── Page config ───────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="NIDS Dashboard — Mode 1",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── CSS ───────────────────────────────────────────────────────────────────
    st.markdown("""
<style>
    .metric-card {
        background: #1e2130; border-radius: 10px;
        padding: 16px; text-align: center;
        border: 1px solid #2d3250;
    }
    .metric-value { font-size: 2rem; font-weight: bold; color: #e0e0ff; }
    .metric-label { font-size: 0.8rem; color: #9999bb; margin-top: 4px; }
    .section-header {
        font-size: 1.2rem; font-weight: 600; color: #c0c8ff;
        margin: 1rem 0 0.5rem; border-left: 4px solid #6c7fff;
        padding-left: 0.5rem;
    }
    .safety-badge {
        background: #1a3a1a; color: #5dde5d;
        border: 1px solid #3a7a3a; border-radius: 6px;
        padding: 8px 12px; font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

    # ── Session state ─────────────────────────────────────────────────────────
    if "replay_events" not in st.session_state:
        st.session_state.replay_events = []
    if "replay_running" not in st.session_state:
        st.session_state.replay_running = False

    # ── Predictor ─────────────────────────────────────────────────────────────
    predictor, pred_err = get_predictor_cached(st.session_state)

    # ─────────────────────────────────────────────────────────────────────────
    #  SIDEBAR
    # ─────────────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🛡️ NIDS Control Panel")
        st.caption("Mode 1 — Offline Dataset Replay")
        st.divider()

        if predictor:
            meta = predictor.get_metadata()
            st.success(f"✅ Models loaded  |  Threshold: **{meta['binary_threshold']:.2f}**")
        else:
            st.error(f"❌ Model load failed: {pred_err}")

        st.divider()
        st.markdown("### 📂 Replay Controls")

        avail_scenarios = get_scenario_files() or EXPECTED_SCENARIOS
        scenario_labels = {SCENARIO_DISPLAY_NAMES.get(s, s): s for s in avail_scenarios}
        selected_label = st.selectbox("Scenario", list(scenario_labels.keys()), index=0)
        selected_scenario_file = scenario_labels[selected_label]

        n_records = st.number_input("Number of records", 1, 500, 20, 1)
        interval_s = st.number_input("Replay interval (s)", 0.0, 2.0, 0.25, 0.05, format="%.2f")
        output_fmt = st.selectbox("Output format", ["JSONL", "CSV", "SQLite"])

        col_start, col_stop = st.columns(2)
        btn_start = col_start.button("▶ Start", type="primary", use_container_width=True)
        btn_stop  = col_stop.button("⏹ Stop", use_container_width=True)
        btn_clear = st.button("🗑 Clear Events", use_container_width=True)

        if btn_clear:
            st.session_state.replay_events = []
            st.session_state.replay_running = False
            st.rerun()

        if btn_stop:
            st.session_state.replay_running = False

        st.divider()
        st.markdown("### 🧭 Navigation")
        nav = st.radio(
            "Section",
            ["Overview", "Live Alert Feed", "Analytics",
             "Model Evaluation", "Explainability", "Export"],
            label_visibility="collapsed",
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  REPLAY
    # ─────────────────────────────────────────────────────────────────────────
    if btn_start and predictor:
        scenario_path = SCENARIOS_DIR / selected_scenario_file
        if not scenario_path.exists():
            st.error(f"Scenario file not found: {scenario_path}")
        else:
            try:
                df = pd.read_parquet(scenario_path).head(int(n_records))
            except Exception as e:
                st.error(f"Failed to load scenario: {e}")
                df = pd.DataFrame()

            if not df.empty:
                st.session_state.replay_running = True
                prog = st.progress(0, text="Starting replay…")
                total = len(df)
                for i, (idx, row) in enumerate(df.iterrows()):
                    if not st.session_state.replay_running:
                        st.warning("⏹ Replay stopped.")
                        break
                    try:
                        event = predictor.predict_record(row.to_dict(), record_index=int(idx))
                        ev_dict = {
                            k: v for k, v in event.model_dump().items()
                            if k in set(ALERT_DISPLAY_FIELDS)
                        }
                        st.session_state.replay_events.append(ev_dict)
                    except Exception as e:
                        st.warning(f"Skipped row {idx}: {e}")
                    prog.progress((i + 1) / total, text=f"Record {i+1}/{total}…")
                    if interval_s > 0:
                        time.sleep(interval_s)
                prog.progress(1.0, text="✅ Replay complete.")
                st.session_state.replay_running = False
                st.rerun()

    elif btn_start and not predictor:
        st.error("Cannot start replay: models failed to load.")

    # ─────────────────────────────────────────────────────────────────────────
    #  CURRENT STATE
    # ─────────────────────────────────────────────────────────────────────────
    current_events: List[Dict] = st.session_state.replay_events
    summary = compute_summary(current_events)

    # ─────────────────────────────────────────────────────────────────────────
    #  HEADER
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(
        "# 🛡️ Machine Learning-Based Network Intrusion Detection "
        "and Attack Classification System"
    )
    st.markdown("**Mode 1 — Offline UNSW-NB15 Dataset Replay**")
    st.markdown("""
<div class="safety-badge">
⚠️ <strong>Safety Disclaimer:</strong>
This dashboard replays held-out benchmark records.
It does not send packets, scan hosts, or monitor a live network.
All inference is performed on local files only.
</div>
""", unsafe_allow_html=True)
    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    #  SUMMARY CARDS
    # ─────────────────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    cards = [
        (c1, "Total Events",    str(summary["total"]),                  "#6c7fff"),
        (c2, "Normal",          str(summary["normal"]),                  "#2ecc71"),
        (c3, "Attacks",         str(summary["attacks"]),                 "#e67e22"),
        (c4, "Critical Alerts", str(summary["critical"]),                "#e74c3c"),
        (c5, "Avg Confidence",  f"{summary['avg_confidence']:.1%}",      "#9b59b6"),
        (c6, "Risk Level",      summary["risk_level"],                   "#3498db"),
    ]
    for col, label, value, color in cards:
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value" style="color:{color}">{value}</div>'
                f'<div class="metric-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    st.divider()

    # ═════════════════════════════════════════════════════════════════════════
    #  SECTION: OVERVIEW
    # ═════════════════════════════════════════════════════════════════════════
    if nav == "Overview":
        st.markdown('<div class="section-header">📌 System Overview</div>', unsafe_allow_html=True)
        col_l, col_r = st.columns([3, 2])

        with col_l:
            st.markdown("""
**System Description**

This system implements a two-stage ML-based network intrusion detection pipeline
trained on the UNSW-NB15 benchmark dataset.

* **Stage 1** (Binary XGBoost): classifies each network flow as *Normal* or *Attack*.
* **Stage 2** (Multiclass XGBoost): classifies confirmed attacks into one of nine
  categories — Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic, Reconnaissance,
  Shellcode, or Worms.

The binary decision threshold **0.80** was selected using a **leakage-safe validation
protocol**: threshold search used only a held-out training-validation split; the
official test set was evaluated exactly once after selection.
            """)

        with col_r:
            thresh_meta = load_json(BINARY_THRESHOLD_META_JSON)
            if predictor:
                m = predictor.get_metadata()
                st.info(
                    f"🔵 Binary Model: `{Path(m['binary_model_artifact']).name}`\n\n"
                    f"🟣 Multiclass Model: `{Path(m['multiclass_model_artifact']).name}`\n\n"
                    f"⚙️ Preprocessor: `{Path(m['preprocessor_artifact']).name}`\n\n"
                    f"🎚️ Decision Threshold: **{m['binary_threshold']:.2f}** (metadata-sourced)\n\n"
                    f"📐 Features: **{m['required_features_count']}** transformed\n\n"
                    f"🏷️ Classes: **{len(m['classes'])}** ({', '.join(m['classes'])})"
                )
            else:
                st.error("Models not loaded — check artifact paths.")

            if thresh_meta:
                st.markdown("**Dataset Split**")
                st.info(
                    f"🚂 Training: **{thresh_meta.get('training_total_records',0):,}** records\n\n"
                    f"🧪 Test: **{thresh_meta.get('test_total_records',0):,}** records\n\n"
                    f"🔍 Validation: **{thresh_meta.get('validation_records',0):,}** records"
                )

        st.markdown('<div class="section-header">⚙️ System Status</div>', unsafe_allow_html=True)
        status_cols = st.columns(4)
        checks = [
            ("Prediction Models", predictor is not None),
            ("Scenario Files", bool(get_scenario_files())),
            ("Metrics Reports", BINARY_RESULTS_JSON.exists()),
            ("Figure Artifacts", FIGURES_DIR.exists() and any(FIGURES_DIR.glob("*.png"))),
        ]
        for (label, ok), col in zip(checks, status_cols):
            col.metric(label, "✅" if ok else "❌")

        st.markdown('<div class="section-header">🔒 Safety & Methodology</div>', unsafe_allow_html=True)
        st.markdown("""
| Property | Value |
|---|---|
| Data source | UNSW-NB15 official held-out test split |
| Live traffic | ❌ Not used — purely offline replay |
| Sockets / packets | ❌ Never created |
| External URLs | ❌ Never called |
| Ground-truth in output | ❌ Strictly excluded |
| Threshold selection | Validation-only (test set used once) |
| Scenario seed | 42 (fixed, reproducible) |
        """)

    # ═════════════════════════════════════════════════════════════════════════
    #  SECTION: LIVE ALERT FEED
    # ═════════════════════════════════════════════════════════════════════════
    elif nav == "Live Alert Feed":
        st.markdown('<div class="section-header">🚨 Live Alert Feed</div>', unsafe_allow_html=True)

        if not current_events:
            st.info(
                "No events yet. Use the **Replay Controls** in the sidebar.\n\n"
                "**Quick start:** Select a scenario → set limit → click ▶ Start."
            )
        else:
            display_df = events_to_display_df(current_events)
            leaked = set(display_df.columns) & FORBIDDEN_FIELDS
            if leaked:
                st.error(f"⚠️ SAFETY: forbidden ground-truth columns detected: {leaked}")
            else:
                def colour_severity(row):
                    colour = SEVERITY_COLORS.get(row.get("severity", "None"), "#95a5a6")
                    return [f"background-color:{colour}18" for _ in row]

                st.dataframe(
                    display_df.style.apply(colour_severity, axis=1),
                    use_container_width=True, hide_index=True,
                )
            st.caption(
                f"{len(current_events)} events — prediction-only fields, no ground-truth labels."
            )

            attack_events = [e for e in current_events if e.get("prediction") == "Attack"]
            if attack_events:
                st.markdown('<div class="section-header">🔍 Latest Attack Alerts</div>', unsafe_allow_html=True)
                for ev in reversed(attack_events[-5:]):
                    sev = ev.get("severity", "None")
                    colour = SEVERITY_COLORS.get(sev, "#95a5a6")
                    with st.expander(
                        f"[{sev}] {ev.get('attack_type','?')}  —  "
                        f"conf={ev.get('confidence',0):.1%}  |  {str(ev.get('timestamp',''))[:19]}",
                        expanded=False,
                    ):
                        ca, cb = st.columns(2)
                        with ca:
                            st.markdown(f"**Event ID:** `{str(ev.get('event_id',''))[:18]}…`")
                            st.markdown(f"**Prediction:** {ev.get('prediction','')}")
                            st.markdown(f"**Attack Type:** {ev.get('attack_type','')}")
                            st.markdown(
                                f"**Severity:** "
                                f'<span style="color:{colour};font-weight:bold">{sev}</span>',
                                unsafe_allow_html=True,
                            )
                        with cb:
                            st.markdown(f"**Binary Prob:** {ev.get('binary_probability',0):.4f}")
                            st.markdown(f"**Threshold:** {ev.get('binary_threshold',0.8):.2f}")
                            mp = ev.get("multiclass_probability")
                            st.markdown(f"**Multi Prob:** {f'{mp:.4f}' if mp else 'N/A'}")
                            st.markdown(f"**Confidence:** {ev.get('confidence',0):.4f}")
                        st.markdown(f"**Recommended Action:** {ev.get('recommended_action','')}")
            else:
                st.success("✅ No attacks detected in the current replay session.")

    # ═════════════════════════════════════════════════════════════════════════
    #  SECTION: ANALYTICS
    # ═════════════════════════════════════════════════════════════════════════
    elif nav == "Analytics":
        st.markdown('<div class="section-header">📊 Analytics</div>', unsafe_allow_html=True)
        if not current_events:
            st.info("Run a replay to see analytics.")
        else:
            df = pd.DataFrame(current_events)

            r1a, r1b = st.columns(2)
            with r1a:
                st.markdown("**Normal vs Attack**")
                pred_c = df["prediction"].value_counts().reset_index()
                pred_c.columns = ["Prediction", "Count"]
                fig = px.pie(pred_c, names="Prediction", values="Count", hole=0.4,
                             color="Prediction",
                             color_discrete_map={"Normal": "#2ecc71", "Attack": "#e74c3c"})
                fig.update_layout(height=280, margin=dict(t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

            with r1b:
                st.markdown("**Attack Category Distribution**")
                atk = df[df["prediction"] == "Attack"]
                if not atk.empty:
                    cat_c = atk["attack_type"].value_counts().reset_index()
                    cat_c.columns = ["Category", "Count"]
                    fig2 = px.bar(cat_c, x="Count", y="Category", orientation="h",
                                  color="Count", color_continuous_scale="Reds")
                    fig2.update_layout(height=280, margin=dict(t=10, b=10), yaxis_title="")
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.success("All traffic normal.")

            r2a, r2b = st.columns(2)
            with r2a:
                st.markdown("**Alert Timeline**")
                if "timestamp" in df.columns:
                    tl = df.copy()
                    tl["ts"] = pd.to_datetime(tl["timestamp"], errors="coerce")
                    tl = tl.dropna(subset=["ts"])
                    if not tl.empty:
                        fig3 = px.scatter(
                            tl, x="ts", y="binary_probability",
                            color="prediction",
                            color_discrete_map={"Normal": "#2ecc71", "Attack": "#e74c3c"},
                            height=270,
                        )
                        thresh_val = float(df["binary_threshold"].iloc[0]) if "binary_threshold" in df.columns else 0.8
                        fig3.add_hline(y=thresh_val, line_dash="dash", line_color="orange",
                                       annotation_text="Threshold")
                        fig3.update_layout(margin=dict(t=10, b=10))
                        st.plotly_chart(fig3, use_container_width=True)

            with r2b:
                st.markdown("**Confidence Distribution**")
                if "confidence" in df.columns:
                    fig4 = px.histogram(df, x="confidence", color="prediction", nbins=20,
                                        color_discrete_map={"Normal": "#2ecc71", "Attack": "#e74c3c"},
                                        barmode="overlay", opacity=0.7, height=270)
                    fig4.update_layout(margin=dict(t=10, b=10))
                    st.plotly_chart(fig4, use_container_width=True)

            r3a, r3b = st.columns(2)
            with r3a:
                st.markdown("**Severity Distribution**")
                if "severity" in df.columns:
                    sev_c = df["severity"].value_counts().reset_index()
                    sev_c.columns = ["Severity", "Count"]
                    fig5 = px.bar(sev_c, x="Severity", y="Count", color="Severity",
                                  color_discrete_map=SEVERITY_COLORS, height=270)
                    fig5.update_layout(margin=dict(t=10, b=10))
                    st.plotly_chart(fig5, use_container_width=True)

            with r3b:
                st.markdown("**Binary Probability Distribution**")
                if "binary_probability" in df.columns:
                    fig6 = px.histogram(df, x="binary_probability", nbins=25,
                                        color="prediction",
                                        color_discrete_map={"Normal": "#2ecc71", "Attack": "#e74c3c"},
                                        barmode="overlay", opacity=0.75, height=270)
                    fig6.update_layout(margin=dict(t=10, b=10))
                    st.plotly_chart(fig6, use_container_width=True)

            st.markdown("**Top Recommended Actions**")
            if "recommended_action" in df.columns:
                act_c = df["recommended_action"].value_counts().head(8).reset_index()
                act_c.columns = ["Action", "Count"]
                act_c["Action"] = act_c["Action"].str[:55] + "…"
                fig7 = px.bar(act_c, x="Count", y="Action", orientation="h",
                              color="Count", color_continuous_scale="Blues", height=320)
                fig7.update_layout(margin=dict(t=10, b=10), yaxis_title="")
                st.plotly_chart(fig7, use_container_width=True)

    # ═════════════════════════════════════════════════════════════════════════
    #  SECTION: MODEL EVALUATION
    # ═════════════════════════════════════════════════════════════════════════
    elif nav == "Model Evaluation":
        st.markdown('<div class="section-header">📈 Model Evaluation Results</div>', unsafe_allow_html=True)
        eval_tab1, eval_tab2, eval_tab3 = st.tabs([
            "Binary Classification", "Multiclass Classification", "Attack-Only Experiment"
        ])

        with eval_tab1:
            st.markdown("### Binary Model Comparison (Baseline — default threshold 0.50)")
            bin_csv = load_csv_safe(BINARY_COMPARISON_CSV)
            if not bin_csv.empty:
                cols_show = [c for c in ["model","accuracy","balanced_accuracy","precision",
                                          "recall","f1_score","roc_auc","false_positive_rate",
                                          "training_time_sec"] if c in bin_csv.columns]
                st.dataframe(bin_csv[cols_show], use_container_width=True, hide_index=True)
            else:
                st.warning("binary_model_comparison.csv not found.")

            st.markdown("### Confusion Matrices")
            c_lr, c_rf, c_xgb = st.columns(3)
            for col, name, key in [
                (c_lr,"Logistic Regression","logistic_regression"),
                (c_rf,"Random Forest","random_forest"),
                (c_xgb,"XGBoost","xgboost"),
            ]:
                img = FIGURES_DIR / f"binary_confusion_matrix_{key}.png"
                with col:
                    st.caption(name)
                    if img.exists():
                        st.image(str(img), use_container_width=True)
                    else:
                        st.info("Image not found.")

            st.markdown("### Leakage-Safe Threshold Selection — XGBoost")
            thresh_meta = load_json(BINARY_THRESHOLD_META_JSON)
            if thresh_meta:
                st.info(
                    f"**Selected threshold:** {thresh_meta.get('selected_threshold',0.8):.2f}  "
                    f"(validation-based; test set evaluated exactly once)"
                )
                st.caption(thresh_meta.get("selection_rule",""))

                col_v, col_t = st.columns(2)
                for col, title, m in [
                    (col_v, "✅ Validation Metrics (used for selection)",
                     thresh_meta.get("validation_metrics_at_selected",{})),
                    (col_t, "🧪 Final Test Metrics (unbiased)",
                     thresh_meta.get("test_metrics_at_selected",{})),
                ]:
                    with col:
                        st.markdown(f"**{title}**")
                        rows = [
                            ("Precision",           f"{m.get('precision',0):.4f}"),
                            ("Recall (Attack)",     f"{m.get('recall',0):.4f}"),
                            ("F1-score",            f"{m.get('f1_score',0):.4f}"),
                            ("Specificity",         f"{m.get('specificity',0):.4f}"),
                            ("FPR",                 f"{m.get('false_positive_rate',0):.4f}"),
                            ("Balanced Accuracy",   f"{m.get('balanced_accuracy',0):.4f}"),
                            ("False Positives",     str(m.get("number_of_false_positives","—"))),
                            ("False Negatives",     str(m.get("number_of_false_negatives","—"))),
                        ]
                        st.dataframe(pd.DataFrame(rows, columns=["Metric","Value"]),
                                     hide_index=True, use_container_width=True)

            img_tradeoff = FIGURES_DIR / "binary_threshold_selection_tradeoff.png"
            if img_tradeoff.exists():
                st.image(str(img_tradeoff), caption="Threshold Selection Tradeoff", use_container_width=True)

        with eval_tab2:
            st.markdown("### Baseline 10-Class Multiclass Model Comparison")
            multi_csv = load_csv_safe(MULTICLASS_COMPARISON_CSV)
            if not multi_csv.empty:
                m_cols = [c for c in ["model","accuracy","balanced_accuracy","macro_f1",
                                       "weighted_f1","roc_auc_ovr_macro","training_time_sec"]
                          if c in multi_csv.columns]
                st.dataframe(multi_csv[m_cols], use_container_width=True, hide_index=True)

            st.markdown("### Multiclass Confusion Matrices")
            mc_lr, mc_rf, mc_xgb = st.columns(3)
            for col, name, key in [
                (mc_lr,"Logistic Regression","logistic_regression"),
                (mc_rf,"Random Forest","random_forest"),
                (mc_xgb,"XGBoost (Selected)","xgboost"),
            ]:
                img = FIGURES_DIR / f"multiclass_confusion_matrix_{key}.png"
                with col:
                    st.caption(name)
                    if img.exists():
                        st.image(str(img), use_container_width=True)
                    else:
                        st.info("Image not found.")

            multi_results = load_json(MULTICLASS_RESULTS_JSON)
            per_class = multi_results.get("models",{}).get("xgboost",{}).get("per_class_metrics",{})
            if per_class:
                st.markdown("### Per-Class Metrics — XGBoost (Baseline)")
                pc_rows = [
                    {"Class": cls, "Precision": f"{m.get('precision',0):.4f}",
                     "Recall": f"{m.get('recall',0):.4f}", "F1": f"{m.get('f1',0):.4f}",
                     "Support": m.get("support","—")}
                    for cls, m in per_class.items()
                ]
                st.dataframe(pd.DataFrame(pc_rows), use_container_width=True, hide_index=True)

        with eval_tab3:
            st.markdown("### Attack-Only Multiclass Experiment")
            if not ATTACK_ONLY_RESULTS_JSON.exists():
                st.info("attack_only_multiclass_results.json not found.")
            else:
                ao = load_json(ATTACK_ONLY_RESULTS_JSON)
                st.json(ao.get("summary", ao), expanded=False)
            for img_name, cap in [
                ("attack_only_multiclass_confusion_matrix_xgboost.png", "Attack-Only XGBoost CM"),
                ("attack_only_multiclass_per_class_f1.png", "Per-Class F1 (Attack-Only)"),
                ("attack_only_multiclass_rare_class_recall.png", "Rare-Class Recall"),
            ]:
                img = FIGURES_DIR / img_name
                if img.exists():
                    st.image(str(img), caption=cap, use_container_width=True)

    # ═════════════════════════════════════════════════════════════════════════
    #  SECTION: EXPLAINABILITY
    # ═════════════════════════════════════════════════════════════════════════
    elif nav == "Explainability":
        st.markdown('<div class="section-header">🔍 Model Explainability</div>', unsafe_allow_html=True)
        st.caption("Feature importance indicates which inputs most influence decisions. **Not causal.**")

        shap_json = EXPLANATIONS_DIR / "binary_shap_importance.json"
        fi_json   = EXPLANATIONS_DIR / "binary_feature_importance.json"

        if shap_json.exists():
            importance = load_json(shap_json)
            if importance:
                shap_df = pd.DataFrame(list(importance.items()), columns=["Feature","Mean |SHAP|"]
                    ).sort_values("Mean |SHAP|", ascending=False).head(20)
                fig_s = px.bar(shap_df, x="Mean |SHAP|", y="Feature", orientation="h",
                               color="Mean |SHAP|", color_continuous_scale="Blues",
                               title="Top 20 Features — Binary XGBoost (SHAP)")
                fig_s.update_layout(yaxis={"categoryorder":"total ascending"}, height=500)
                st.plotly_chart(fig_s, use_container_width=True)
        elif fi_json.exists():
            importance = load_json(fi_json)
            if importance:
                fi_df = pd.DataFrame(list(importance.items()), columns=["Feature","Importance"]
                    ).sort_values("Importance", ascending=False).head(20)
                fig_f = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                               color="Importance", color_continuous_scale="Viridis")
                fig_f.update_layout(yaxis={"categoryorder":"total ascending"}, height=500)
                st.plotly_chart(fig_f, use_container_width=True)
        else:
            st.info("No SHAP or feature-importance JSON found in reports/explanations/.")

        bmc_img = FIGURES_DIR / "binary_model_comparison.png"
        if bmc_img.exists():
            st.markdown("### Binary Model Comparison Chart")
            st.image(str(bmc_img), use_container_width=True)

        st.markdown("### Selected Alert Detail")
        attack_events = [e for e in current_events if e.get("prediction") == "Attack"]
        if not attack_events:
            st.info("Run a replay — then pick an attack event to inspect.")
        else:
            opts = {
                f"[{e.get('severity','')}] {e.get('attack_type','')} — "
                f"conf {e.get('confidence',0):.2%} (rec {e.get('record_index','')})": i
                for i, e in enumerate(attack_events)
            }
            sel = st.selectbox("Pick an attack event", list(opts.keys()))
            ev = attack_events[opts[sel]]
            sev = ev.get("severity","None")
            ia, ib = st.columns(2)
            with ia:
                st.metric("Prediction", ev.get("prediction",""))
                st.metric("Attack Type", ev.get("attack_type",""))
                st.metric("Severity", sev)
            with ib:
                st.metric("Confidence", f"{ev.get('confidence',0):.4f}")
                st.metric("Binary Probability", f"{ev.get('binary_probability',0):.4f}")
                mp = ev.get("multiclass_probability")
                st.metric("Multiclass Probability", f"{mp:.4f}" if mp else "N/A")
            st.markdown(f"**Recommended Action:** `{ev.get('recommended_action','')}`")

    # ═════════════════════════════════════════════════════════════════════════
    #  SECTION: EXPORT
    # ═════════════════════════════════════════════════════════════════════════
    elif nav == "Export":
        st.markdown('<div class="section-header">💾 Export</div>', unsafe_allow_html=True)
        if not current_events:
            st.info("No events to export. Run a replay first.")
        else:
            st.markdown(f"**{len(current_events)} events** ready — prediction-only fields.")
            export_df = events_to_display_df(current_events)

            csv_buf = io.StringIO()
            export_df.to_csv(csv_buf, index=False)
            st.download_button("📥 Download as CSV", csv_buf.getvalue().encode("utf-8"),
                               "nids_replay_events.csv", "text/csv")

            jsonl_lines = "\n".join(json.dumps(row) for row in current_events)
            st.download_button("📥 Download as JSONL", jsonl_lines.encode("utf-8"),
                               "nids_replay_events.jsonl", "application/x-ndjson")

            try:
                import sys as _sys
                _sys.path.insert(0, str(ROOT / "src"))
                from nids.replay import write_sqlite
                from nids.schemas import PredictionResult
                with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                events_pr = []
                for e in current_events:
                    try:
                        events_pr.append(PredictionResult(**e))
                    except Exception:
                        pass
                if events_pr:
                    write_sqlite(events_pr, tmp_path)
                    st.download_button("📥 Download as SQLite", tmp_path.read_bytes(),
                                       "nids_replay_events.sqlite", "application/x-sqlite3")
            except Exception as e:
                st.warning(f"SQLite export unavailable: {e}")

            thresh_meta = load_json(BINARY_THRESHOLD_META_JSON)
            if thresh_meta:
                st.download_button("📥 Download Threshold Metadata",
                                   json.dumps(thresh_meta, indent=2).encode("utf-8"),
                                   "threshold_metadata.json", "application/json")

            summary = compute_summary(current_events)
            summary_csv = pd.DataFrame(
                [{"metric": k, "value": str(v)} for k, v in summary.items()]
            ).to_csv(index=False)
            st.download_button("📥 Download Replay Summary CSV",
                               summary_csv.encode("utf-8"),
                               "replay_summary.csv", "text/csv")

    # ─────────────────────────────────────────────────────────────────────────
    #  FOOTER
    # ─────────────────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "🛡️ ML-Based NIDS — Mode 1 Offline Dashboard  |  "
        "Data: UNSW-NB15  |  Safety: no live traffic  |  "
        "Threshold: 0.80 (metadata-sourced)"
    )


# ─── Streamlit entrypoint ─────────────────────────────────────────────────────
# Only execute the UI when Streamlit is actually running this file.
# When imported by tests, the helper functions above are available directly.
try:
    import streamlit as _st_check
    _st_check.session_state  # will raise if not in a Streamlit context
    _run_dashboard()
except Exception:
    # Not running under Streamlit — this is fine (e.g. during test collection)
    pass
