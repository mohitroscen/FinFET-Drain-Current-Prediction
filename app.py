"""
app.py
------
FinFET Drain Current Prediction — Streamlit Dashboard
Academic demonstration prototype.

Run with:
    streamlit run app.py
"""

import json
import os
import subprocess
import sys
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="FinFET Drain Current Prediction",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Background ── */
.stApp { background: #0d1117; }

/* ── Hide Streamlit default chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
.sidebar-label { font-size: 0.68rem; font-weight: 600; color: #6e7681 !important;
    letter-spacing: 0.08em; text-transform: uppercase; margin-top: 1.2rem; }
.sidebar-value { font-size: 0.84rem; color: #58a6ff !important; margin-bottom: 0.2rem; }
.sidebar-disclaimer {
    font-size: 0.68rem; color: #6e7681 !important;
    background: #161b22; border: 1px solid #21262d;
    border-radius: 6px; padding: 0.6rem; margin-top: 1rem;
}

/* ── Cards ── */
.card {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 12px; padding: 1.2rem 1.4rem; margin-bottom: 0.8rem;
}
.metric-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #21262d; border-radius: 12px;
    padding: 1.1rem 1rem; text-align: center;
}
.metric-card .metric-label {
    font-size: 0.65rem; font-weight: 600; color: #6e7681;
    letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.35rem;
}
.metric-card .metric-value {
    font-size: 1.7rem; font-weight: 700; color: #58a6ff; line-height: 1;
}
.metric-card .metric-sub {
    font-size: 0.72rem; color: #8b949e; margin-top: 0.3rem;
}
.metric-card.highlight { border-color: #388bfd; }

/* ── Prediction result box ── */
.pred-box {
    background: linear-gradient(135deg, #0d2136 0%, #0a1628 100%);
    border: 2px solid #388bfd; border-radius: 14px;
    padding: 1.8rem 2rem; text-align: center;
}
.pred-label { font-size: 0.72rem; font-weight: 600; color: #6e7681;
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.6rem; }
.pred-value { font-size: 3.2rem; font-weight: 700; color: #58a6ff;
    font-family: 'JetBrains Mono', monospace; line-height: 1; }
.pred-unit  { font-size: 1.3rem; font-weight: 400; color: #8b949e; }
.pred-amps  { font-size: 0.9rem; color: #8b949e; margin-top: 0.5rem;
    font-family: 'JetBrains Mono', monospace; }
.pred-badge {
    display: inline-block; background: #1c2128; border: 1px solid #21262d;
    border-radius: 20px; padding: 0.25rem 0.8rem; font-size: 0.72rem;
    color: #8b949e; margin-top: 0.7rem;
}
.pred-disclaimer {
    font-size: 0.65rem; color: #6e7681; margin-top: 0.6rem;
}

/* ── Section headers ── */
.section-header {
    font-size: 1rem; font-weight: 600; color: #e2e8f0;
    border-left: 3px solid #388bfd; padding-left: 0.75rem;
    margin: 1.2rem 0 0.8rem;
}

/* ── Plotly chart captions ── */
.chart-caption { font-size: 0.71rem; color: #6e7681; margin-top: -0.4rem; }

/* ── About box ── */
.about-box {
    background: #161b22; border: 1px solid #21262d; border-radius: 10px;
    padding: 1rem 1.2rem; font-size: 0.82rem; color: #8b949e; line-height: 1.6;
}

/* ── Streamlit slider label colour ── */
[data-testid="stSlider"] label { color: #c9d1d9 !important; }
[data-testid="stNumberInput"] label { color: #c9d1d9 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

FEATURES = ["Lg_nm", "Wfin_nm", "Hfin_nm", "Tox_nm", "Vth_V",
            "VGS_V", "VDS_V", "Temp_K"]

PLOT_THEME = dict(
    paper_bgcolor="rgba(22,27,34,0)",
    plot_bgcolor="rgba(22,27,34,0)",
    font=dict(family="Inter, sans-serif", color="#c9d1d9"),
    margin=dict(l=50, r=20, t=40, b=50),
)
GRID_STYLE = dict(gridcolor="#21262d", zerolinecolor="#30363d")
LEGEND_STYLE = dict(bgcolor="rgba(22,27,34,0.85)", bordercolor="#30363d",
                    borderwidth=1, font=dict(color="#e2e8f0"))

PLOTLY_CFG = {"displayModeBar": False, "scrollZoom": False}


def plotly_chart(fig, *, key=None):
    """Wrapper — passes config dict to avoid deprecated kwargs warning."""
    st.plotly_chart(fig, config=PLOTLY_CFG, width="stretch", key=key)


# ══════════════════════════════════════════════════════════════════════════════
# Auto-train if model.pkl / dataset.csv missing
# ══════════════════════════════════════════════════════════════════════════════

def auto_train():
    if not os.path.exists("dataset.csv"):
        with st.spinner("Generating synthetic dataset …"):
            subprocess.run([sys.executable, "generate_dataset.py"], check=True)

    # Retrain if model.pkl is missing OR if it was pickled with an
    # incompatible Python / scikit-learn version (e.g. local vs. cloud).
    needs_train = not os.path.exists("model.pkl") or not os.path.exists("model_metrics.json")
    if not needs_train:
        try:
            joblib.load("model.pkl")
        except Exception:
            needs_train = True
    if needs_train:
        with st.spinner("Training ML models (first run — ~30 s) …"):
            subprocess.run([sys.executable, "train_model.py"], check=True)


auto_train()

# ══════════════════════════════════════════════════════════════════════════════
# Load model & metrics
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


@st.cache_data
def load_metrics():
    with open("model_metrics.json") as f:
        return json.load(f)


@st.cache_data
def load_dataset():
    return pd.read_csv("dataset.csv")


model   = load_model()
metrics = load_metrics()
df      = load_dataset()

best_model_name = metrics["best_model"]
best_r2   = metrics["models"][best_model_name]["r2_test"]
best_rmse = metrics["models"][best_model_name]["rmse_test"]
n_samples = metrics["n_samples"]
rf_imp    = metrics["rf_importances"]

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚡ FinFET Predictor")
    st.markdown("<div style='font-size:0.78rem;color:#8b949e;'>ML-Based Device Modeling</div>",
                unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="sidebar-label">Project</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-value">ML-Based Prediction of Drain Current in FinFETs</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Domain</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-value">VLSI / Semiconductor Devices / Machine Learning</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Inputs</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-value">Device + Operating Parameters</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Output</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-value">Predicted Drain Current (Id)</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Models</div>', unsafe_allow_html=True)
    st.markdown("""<div class="sidebar-value">
        Linear Regression<br>Random Forest<br>Gradient Boosting
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-disclaimer">⚠ Prototype for academic demonstration. '
                'Results are based on synthetic physically-inspired data.</div>',
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<h1 style="font-size:2rem;font-weight:700;color:#58a6ff;margin-bottom:0.2rem;">
    ⚡ FinFET Drain Current Prediction
</h1>
<p style="font-size:0.95rem;color:#8b949e;margin-top:0;margin-bottom:1rem;">
    Machine Learning-Based Semiconductor Device Modeling
</p>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background:#1c2128;border:1px solid #21262d;border-radius:8px;
    padding:0.7rem 1rem;font-size:0.82rem;color:#8b949e;margin-bottom:1.2rem;">
    🔬 This prototype demonstrates the application of machine learning for predicting
    FinFET drain current from device and operating parameters. The dataset used in this
    demonstration is <strong style="color:#c9d1d9;">synthetic and physically inspired</strong>
    — it is intended for academic demonstration and does <em>not</em> replace TCAD
    simulation or experimental characterization.
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Project Dashboard — summary cards
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">📊 Project Dashboard</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Dataset Samples</div>
        <div class="metric-value">{n_samples:,}</div>
        <div class="metric-sub">Synthetic, physically-inspired</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Input Features</div>
        <div class="metric-value">{len(FEATURES)}</div>
        <div class="metric-sub">Device + operating params</div>
    </div>""", unsafe_allow_html=True)
with c3:
    short = best_model_name.split()[-1]  # "Boosting" / "Forest" / "Regression"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Best ML Model</div>
        <div class="metric-value" style="font-size:1.3rem;">{short}</div>
        <div class="metric-sub">{best_model_name}</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card highlight">
        <div class="metric-label">Test R²</div>
        <div class="metric-value">{best_r2:.4f}</div>
        <div class="metric-sub">Coefficient of determination</div>
    </div>""", unsafe_allow_html=True)
with c5:
    st.markdown(f"""<div class="metric-card highlight">
        <div class="metric-label">Test RMSE</div>
        <div class="metric-value" style="font-size:1.2rem;">{best_rmse:.2e} A</div>
        <div class="metric-sub">Root mean square error</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Demo / retrain buttons ────────────────────────────────────────────────────
btn_col1, btn_col2, _ = st.columns([1, 1, 3])

DEMO_DEFAULTS = dict(Lg=14.0, Wfin=7.0, Hfin=40.0, Tox=1.2,
                     Vth=0.35, VGS=0.80, VDS=0.60, Temp=300.0)

with btn_col1:
    if st.button("🎯  Demo Example", use_container_width=True):
        for k, v in DEMO_DEFAULTS.items():
            st.session_state[f"inp_{k}"] = v
        st.rerun()

with btn_col2:
    if st.button("🔄  Retrain Models", use_container_width=True):
        for path in ("dataset.csv", "model.pkl", "model_metrics.json"):
            if os.path.exists(path):
                os.remove(path)
        st.cache_resource.clear()
        st.cache_data.clear()
        auto_train()
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# Input panel + Prediction (side by side)
# ══════════════════════════════════════════════════════════════════════════════

left, right = st.columns([1, 1], gap="large")

with left:
    # ── Device Parameters ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔧 Device Parameters</div>',
                unsafe_allow_html=True)
    with st.container():
        Lg   = st.slider("Gate Length — Lg (nm)",    7.0,  50.0,
                         float(st.session_state.get("inp_Lg",   14.0)), 0.5, key="inp_Lg")
        Wfin = st.slider("Fin Width — Wfin (nm)",    4.0,  15.0,
                         float(st.session_state.get("inp_Wfin",  7.0)), 0.5, key="inp_Wfin")
        Hfin = st.slider("Fin Height — Hfin (nm)",  20.0,  60.0,
                         float(st.session_state.get("inp_Hfin", 40.0)), 1.0, key="inp_Hfin")
        Tox  = st.slider("Oxide Thickness — Tox (nm)", 0.8, 3.0,
                         float(st.session_state.get("inp_Tox",  1.2)), 0.1, key="inp_Tox")
        Vth  = st.slider("Threshold Voltage — Vth (V)", 0.20, 0.50,
                         float(st.session_state.get("inp_Vth", 0.35)), 0.01, key="inp_Vth")

    # ── Operating Conditions ───────────────────────────────────────────────
    st.markdown('<div class="section-header">⚡ Operating Conditions</div>',
                unsafe_allow_html=True)
    with st.container():
        VGS  = st.slider("Gate Voltage — VGS (V)",   0.0,  1.2,
                         float(st.session_state.get("inp_VGS", 0.80)), 0.01, key="inp_VGS")
        VDS  = st.slider("Drain Voltage — VDS (V)",  0.0,  1.0,
                         float(st.session_state.get("inp_VDS", 0.60)), 0.01, key="inp_VDS")
        Temp = st.slider("Temperature — T (K)",     200.0, 400.0,
                         float(st.session_state.get("inp_Temp", 300.0)), 5.0, key="inp_Temp")

    # ── Predict button ─────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("⚡  Predict Drain Current",
                                use_container_width=True, type="primary")

with right:
    st.markdown('<div class="section-header">🎯 ML Prediction</div>',
                unsafe_allow_html=True)

    # Always show prediction (live update)
    X_input = np.array([[Lg, Wfin, Hfin, Tox, Vth, VGS, VDS, Temp]])
    Id_pred = float(model.predict(X_input)[0])
    Id_pred = max(Id_pred, 0.0)
    Id_uA   = Id_pred * 1e6

    st.markdown(f"""
    <div class="pred-box">
        <div class="pred-label">Predicted Drain Current</div>
        <div class="pred-value">{Id_uA:.3f} <span class="pred-unit">μA</span></div>
        <div class="pred-amps">{Id_pred:.4e} A</div>
        <div class="pred-badge">
            Model: {best_model_name} &nbsp;|&nbsp;
            R² = {best_r2:.4f} &nbsp;|&nbsp;
            RMSE = {best_rmse:.2e} A
        </div>
        <div class="pred-disclaimer">
            ⚠ Based on synthetic, physically-inspired model — not experimentally validated
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Model Performance Comparison ───────────────────────────────────────
    st.markdown('<div class="section-header">📈 Model Performance Comparison</div>',
                unsafe_allow_html=True)

    perf_rows = []
    for name, m in metrics["models"].items():
        marker = "★ " if name == best_model_name else ""
        perf_rows.append({
            "Model":   marker + name,
            "R²":      round(m["r2_test"], 4),
            "MAE (A)": f"{m['mae_test']:.3e}",
            "RMSE (A)": f"{m['rmse_test']:.3e}",
        })
    st.dataframe(pd.DataFrame(perf_rows), hide_index=True, width=600)

    st.markdown("""
    <div style='font-size:0.7rem;color:#6e7681;margin-top:-0.3rem;'>
        ★ = best model (selected automatically by test R²)
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Device Characteristics + Feature Importance tabs
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">📉 Device Characteristics</div>',
            unsafe_allow_html=True)

tab_vgs, tab_vds, tab_pva, tab_fi = st.tabs([
    "📈 Id vs VGS", "📉 Id vs VDS", "🎯 Predicted vs Actual", "✨ Feature Importance"
])

# ── Tab 1 — Id vs VGS ─────────────────────────────────────────────────────────
with tab_vgs:
    vgs_range = np.linspace(0, 1.2, 120)
    vds_vals  = [0.1, 0.5, 0.9]
    colors    = ["#58a6ff", "#3fb950", "#f78166"]

    fig_vgs = go.Figure()
    for vds_val, col in zip(vds_vals, colors):
        X_sweep = np.column_stack([
            np.full(120, Lg), np.full(120, Wfin), np.full(120, Hfin),
            np.full(120, Tox), np.full(120, Vth), vgs_range,
            np.full(120, vds_val), np.full(120, Temp),
        ])
        Id_sweep = np.maximum(model.predict(X_sweep), 0) * 1e6
        fig_vgs.add_trace(go.Scatter(
            x=vgs_range, y=Id_sweep,
            mode="lines", name=f"VDS = {vds_val} V",
            line=dict(color=col, width=2.5),
        ))

    # Current operating point
    fig_vgs.add_trace(go.Scatter(
        x=[VGS], y=[Id_uA], mode="markers", name="Current Op. Point",
        marker=dict(color="#f0883e", size=10, symbol="star"),
    ))
    # Threshold voltage line
    fig_vgs.add_vline(x=Vth, line=dict(color="#8b949e", dash="dash", width=1.5),
                      annotation_text=f"Vth = {Vth:.2f} V",
                      annotation_font_color="#8b949e")

    fig_vgs.update_layout(
        title="Transfer Characteristics — Id vs VGS",
        xaxis=dict(title="Gate Voltage VGS (V)", **GRID_STYLE),
        yaxis=dict(title="Drain Current Id (μA)", **GRID_STYLE),
        legend=LEGEND_STYLE, height=400, **PLOT_THEME,
    )
    plotly_chart(fig_vgs, key="chart_vgs")
    st.markdown('<div class="chart-caption">Transfer characteristics showing Id vs VGS '
                'for three drain voltages. Dashed line marks Vth.</div>',
                unsafe_allow_html=True)

# ── Tab 2 — Id vs VDS ─────────────────────────────────────────────────────────
with tab_vds:
    vds_range = np.linspace(0, 1.0, 100)
    vgs_vals  = [0.4, 0.6, 0.8, 1.0, 1.2]
    palette   = ["#58a6ff", "#3fb950", "#f0883e", "#bc8cff", "#f78166"]

    fig_vds = go.Figure()
    for vgs_val, col in zip(vgs_vals, palette):
        X_sweep = np.column_stack([
            np.full(100, Lg), np.full(100, Wfin), np.full(100, Hfin),
            np.full(100, Tox), np.full(100, Vth), np.full(100, vgs_val),
            vds_range, np.full(100, Temp),
        ])
        Id_sweep = np.maximum(model.predict(X_sweep), 0) * 1e6
        fig_vds.add_trace(go.Scatter(
            x=vds_range, y=Id_sweep,
            mode="lines", name=f"VGS = {vgs_val} V",
            line=dict(color=col, width=2.5),
        ))

    fig_vds.update_layout(
        title="Output Characteristics — Id vs VDS",
        xaxis=dict(title="Drain Voltage VDS (V)", **GRID_STYLE),
        yaxis=dict(title="Drain Current Id (μA)", **GRID_STYLE),
        legend=LEGEND_STYLE, height=400, **PLOT_THEME,
    )
    plotly_chart(fig_vds, key="chart_vds")
    st.markdown('<div class="chart-caption">Output characteristics showing Id vs VDS '
                'for multiple VGS levels. Current saturates at Vdsat ≈ VGS − Vth.</div>',
                unsafe_allow_html=True)

# ── Tab 3 — Predicted vs Actual ───────────────────────────────────────────────
with tab_pva:
    y_test = np.array(metrics["models"][best_model_name]["y_test"])
    y_pred = np.array(metrics["models"][best_model_name]["y_pred"])

    # Clamp to positive (log scale)
    mask   = (y_test > 0) & (y_pred > 0)
    y_t    = y_test[mask] * 1e6
    y_p    = y_pred[mask] * 1e6
    lo, hi = min(y_t.min(), y_p.min()), max(y_t.max(), y_p.max())

    fig_pva = go.Figure()
    fig_pva.add_trace(go.Scatter(
        x=y_t, y=y_p, mode="markers", name="Test samples",
        marker=dict(color="#58a6ff", size=4, opacity=0.55,
                    line=dict(width=0)),
    ))
    fig_pva.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines", name="Ideal (y = x)",
        line=dict(color="#f0883e", dash="dash", width=2),
    ))
    fig_pva.update_layout(
        title="Predicted vs Actual Id (log scale, test set)",
        xaxis=dict(title="Actual Id (μA)", type="log", **GRID_STYLE),
        yaxis=dict(title="Predicted Id (μA)", type="log", **GRID_STYLE),
        legend=LEGEND_STYLE, height=400, **PLOT_THEME,
    )
    plotly_chart(fig_pva, key="chart_pva")
    st.markdown(f'<div class="chart-caption">Log-scale scatter of predicted vs actual Id '
                f'on the 20 % held-out test set. R² = {best_r2:.4f}.</div>',
                unsafe_allow_html=True)

# ── Tab 4 — Feature Importance ────────────────────────────────────────────────
with tab_fi:
    feat_labels = ["Lg (nm)", "Wfin (nm)", "Hfin (nm)", "Tox (nm)",
                   "Vth (V)", "VGS (V)", "VDS (V)", "Temp (K)"]
    pairs = sorted(zip(rf_imp, feat_labels), reverse=True)
    fi_vals, fi_labels = zip(*pairs)
    bar_colors = ["#388bfd" if v == max(fi_vals) else "#58a6ff" for v in fi_vals]

    fig_fi = go.Figure(go.Bar(
        x=list(fi_vals), y=list(fi_labels),
        orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f"{v:.3f}" for v in fi_vals],
        textposition="outside",
        textfont=dict(color="#c9d1d9", size=11),
    ))
    fig_fi.update_layout(
        title="Feature Importance (Random Forest)",
        xaxis=dict(title="Importance Score", range=[0, max(fi_vals) * 1.25],
                   **GRID_STYLE),
        yaxis=dict(**GRID_STYLE),
        height=380, **PLOT_THEME,
    )
    plotly_chart(fig_fi, key="chart_fi")
    st.markdown('<div class="chart-caption">Feature importance scores from the '
                'Random Forest model — higher = more influential parameter.</div>',
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# About section
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">ℹ About this Prototype</div>',
            unsafe_allow_html=True)
st.markdown("""
<div class="about-box">
    This prototype demonstrates the application of machine learning for predicting
    FinFET drain current from device and operating parameters. The dataset used in this
    demonstration is <strong>synthetic and physically inspired</strong>. The application
    is intended for academic demonstration and does <em>not</em> replace TCAD simulation
    or experimental characterization.
</div>
""", unsafe_allow_html=True)
