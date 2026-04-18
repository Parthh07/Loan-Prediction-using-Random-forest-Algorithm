"""
LoanSense AI – Industry-Grade Refinement
Clean · Professional · Minimal · Light
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib, os

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LoanSense AI – Credit Assessment",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Artefacts
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    return (
        joblib.load("model/loan_rf_model.pkl"),
        joblib.load("model/encoders.pkl"),
        joblib.load("model/scaler.pkl"),
        joblib.load("model/metrics.pkl"),
    )

@st.cache_data
def load_data():
    return pd.read_csv("model/loan_data.csv")

def predict_loan(model, encoders, scaler, inputs):
    df = pd.DataFrame([inputs])
    for col in ["Gender","Married","Dependents","Education","Self_Employed","Property_Area"]:
        df[col] = encoders[col].transform([str(df[col].values[0])])
    df[["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"]] = \
        scaler.transform(df[["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"]])
    feat = ["Gender","Married","Dependents","Education","Self_Employed",
            "ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term",
            "Credit_History","Property_Area"]
    pred  = model.predict(df[feat].values)[0]
    proba = model.predict_proba(df[feat].values)[0]
    label = encoders["Loan_Status"].inverse_transform([pred])[0]
    return label, proba


def predict_batch(model, encoders, scaler, df_raw):
    """Run predictions on a cleaned batch DataFrame; returns df with Decision & Probability columns."""
    df = df_raw.copy()
    required = ["Gender","Married","Dependents","Education","Self_Employed",
                "ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term",
                "Credit_History","Property_Area"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return None, f"Missing columns: {', '.join(missing)}"
    try:
        for col in ["Gender","Married","Dependents","Education","Self_Employed","Property_Area"]:
            df[col] = encoders[col].transform(df[col].astype(str))
        df[["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"]] = \
            scaler.transform(df[["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"]])
        preds  = model.predict(df[required].values)
        probas = model.predict_proba(df[required].values)[:, 1]
        labels = encoders["Loan_Status"].inverse_transform(preds)
        result = df_raw.copy()
        result["Decision"]           = labels
        result["Approval_Probability"] = (probas * 100).round(1)
        return result, None
    except Exception as e:
        return None, str(e)

if not os.path.exists("model/loan_rf_model.pkl"):
    with st.spinner("🔄 First-time setup: training model, please wait ~30 seconds..."):
        import subprocess, sys
        subprocess.run([sys.executable, "train_model.py"], check=True)
    st.rerun()

model, encoders, scaler, metrics = load_artefacts()
df_data = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# CSS — Refined Industry Style
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Root background ── */
.stApp { background: #f5f7fa; }
[data-testid="stAppViewContainer"] { background: #f5f7fa; }
[data-testid="block-container"] { padding-top: 0 !important; max-width: 1200px; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #eef0f4; }
::-webkit-scrollbar-thumb { background: #0076CE; border-radius: 4px; }

/* ════════════ TOP NAVBAR ════════════ */
.topnav {
    position: sticky; top: 0; z-index: 100;
    background: white;
    border-bottom: 1px solid #e4e8ef;
    padding: 0 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 58px;
    margin: 0 -4rem 1.5rem -4rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.nav-brand {
    display: flex; align-items: center; gap: 0.6rem;
    font-size: 1.05rem; font-weight: 800;
    color: #0a1f44;
    letter-spacing: -0.3px;
}
.nav-brand span { font-weight: 400; color: #6b7a99; font-size: 0.88rem; border-left: 1px solid #dde2ec; padding-left: 0.7rem; margin-left: 0.3rem; }
.nav-badge {
    background: #e8f4ff;
    border: 1px solid #b3d7f5;
    color: #0076CE;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 0.28rem 0.75rem;
    border-radius: 4px;
}

/* ════════════ HERO ════════════ */
#px-root {
    position: relative;
    width: 100%;
    height: 400px;
    overflow: hidden;
    border-radius: 12px;
    margin-bottom: 1.8rem;
    box-shadow: 0 4px 24px rgba(0,118,206,0.14);
}
#px-bg {
    position: absolute; inset: 0;
    background: linear-gradient(135deg, #0a1f44 0%, #0a3875 45%, #0076CE 100%);
    will-change: transform; transition: transform 0.07s linear;
}
#px-grid {
    position: absolute; inset: 0;
    background-image: radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px);
    background-size: 40px 40px;
    will-change: transform; transition: transform 0.12s linear;
}
#px-lines {
    position: absolute; inset: 0;
    background-image:
        linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px);
    background-size: 80px 80px;
    will-change: transform; transition: transform 0.16s linear;
}
#px-shapes {
    position: absolute; inset: 0;
    will-change: transform; transition: transform 0.20s linear;
}
.px-shape {
    position: absolute; border-radius: 50%;
    background: rgba(255,255,255,0.05);
    animation: shapeFloat ease-in-out infinite alternate;
}
@keyframes shapeFloat {
    from { transform: translateY(0) scale(1); }
    to   { transform: translateY(-16px) scale(1.05); }
}
#px-content {
    position: absolute; inset: 0;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    text-align: center; padding: 2.5rem;
    z-index: 10;
}
.px-overline {
    font-size: 0.68rem; font-weight: 700;
    letter-spacing: 3px; text-transform: uppercase;
    color: rgba(255,255,255,0.55);
    margin-bottom: 0.9rem;
    animation: fadeUp 0.6s ease both;
}
.px-title {
    font-size: clamp(2.2rem, 4vw, 3.4rem);
    font-weight: 800; line-height: 1.1;
    letter-spacing: -1px; color: #ffffff;
    margin-bottom: 0.8rem;
    animation: fadeUp 0.7s 0.08s ease both;
}
.px-title em { font-style: normal; color: #5bc4f5; }
.px-sub {
    font-size: 0.98rem; color: rgba(255,255,255,0.65);
    max-width: 500px; line-height: 1.8;
    animation: fadeUp 0.8s 0.16s ease both;
    margin-bottom: 1.6rem;
}
.px-tags {
    display: flex; gap: 0.5rem; flex-wrap: wrap;
    justify-content: center;
    animation: fadeUp 0.9s 0.24s ease both;
}
.px-tag {
    background: rgba(255,255,255,0.09);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 4px; padding: 0.3rem 0.85rem;
    font-size: 0.75rem; font-weight: 600;
    color: rgba(255,255,255,0.8);
    letter-spacing: 0.2px;
}
.px-cue {
    position: absolute; bottom: 18px; left: 50%;
    transform: translateX(-50%);
    animation: bounce 2s ease-in-out infinite;
    z-index: 11; color: rgba(255,255,255,0.35);
    font-size: 1.2rem;
}
@keyframes bounce {
    0%,100% { transform: translateX(-50%) translateY(0); }
    50%      { transform: translateX(-50%) translateY(6px); }
}
@keyframes fadeUp {
    from { opacity:0; transform:translateY(16px); }
    to   { opacity:1; transform:translateY(0); }
}

/* ════════════ KPI STRIP ════════════ */
.kpi-strip { display:flex; gap:0.9rem; margin-bottom:1.8rem; flex-wrap:wrap; }
.kpi {
    flex: 1; min-width: 130px;
    background: white;
    border: 1px solid #e4e8ef;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    transition: transform 0.2s, box-shadow 0.2s;
    position: relative; overflow: hidden;
}
.kpi::after {
    content: "";
    position: absolute; top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #0076CE, #00B0DA);
}
.kpi:hover { transform: translateY(-3px); box-shadow: 0 6px 20px rgba(0,118,206,0.1); }
.kpi-val   { font-size: 1.75rem; font-weight: 800; color: #0a1f44; line-height: 1; }
.kpi-lbl   { font-size: 0.7rem; font-weight: 600; color: #8a96b0; text-transform: uppercase; letter-spacing: 1px; margin-top: 0.25rem; }
.kpi-icon  { font-size: 1.2rem; margin-bottom: 0.25rem; opacity: 0.9; }

/* ════════════ SECTION HEADER ════════════ */
.sh { margin-bottom: 0.2rem; }
.sh-title { font-size: 1.35rem; font-weight: 800; color: #0a1f44; letter-spacing: -0.3px; }
.sh-desc  { font-size: 0.83rem; color: #8a96b0; margin-top: 0.2rem; margin-bottom: 1rem; }
.sh-rule  {
    height: 2px;
    background: linear-gradient(90deg, #0076CE 0%, #00B0DA 40%, transparent 100%);
    border: none; border-radius: 2px;
    margin-bottom: 1.6rem;
}

/* ════════════ TABS ════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: white;
    border: 1px solid #e4e8ef;
    border-radius: 8px;
    padding: 4px;
    gap: 3px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    margin-bottom: 1.8rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px !important;
    color: #6b7a99 !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
    padding: 0.45rem 1.1rem !important;
}
.stTabs [aria-selected="true"] {
    background: #0076CE !important;
    color: white !important;
}

/* ════════════ FORM CARD ════════════ */
.fc {
    background: white;
    border: 1px solid #e4e8ef;
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.fc-header {
    display: flex; align-items: center; gap: 0.5rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid #f0f2f7;
    margin-bottom: 1rem;
}
.fc-icon  { font-size: 1rem; }
.fc-title { font-size: 0.88rem; font-weight: 700; color: #0a1f44; }
.fc-desc  { font-size: 0.76rem; color: #8a96b0; margin-left: auto; }

/* ════════════ RESULT PANEL ════════════ */
.rp {
    border-radius: 10px;
    padding: 1.8rem;
    text-align: center;
    border: 1px solid;
}
.rp-approved {
    background: #f0fdf6;
    border-color: #86efac;
}
.rp-rejected {
    background: #fff8f8;
    border-color: #fca5a5;
}
.rp-status {
    display: inline-flex; align-items: center; gap: 0.4rem;
    font-size: 0.68rem; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase; padding: 0.3rem 0.8rem;
    border-radius: 4px; margin-bottom: 1rem;
}
.rp-status-approved { background: #dcfce7; color: #16a34a; }
.rp-status-rejected { background: #fee2e2; color: #dc2626; }
.rp-icon  { font-size: 2.8rem; margin-bottom: 0.5rem; line-height: 1; }
.rp-title { font-size: 1.55rem; font-weight: 800; margin-bottom: 0.3rem; letter-spacing: -0.5px; }
.rp-desc  { font-size: 0.84rem; color: #64748b; line-height: 1.6; }

/* ════════════ RECOMMENDATION ITEM ════════════ */
.rec-item {
    display: flex; align-items: flex-start; gap: 0.6rem;
    background: #f8fafc; border: 1px solid #e4e8ef;
    border-left: 3px solid #0076CE;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.84rem; color: #374151;
    line-height: 1.5;
}

/* ════════════ SPEC TABLE ════════════ */
.spec-table { background: white; border: 1px solid #e4e8ef; border-radius: 10px; overflow:hidden; }
.spec-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.65rem 1.2rem;
    border-bottom: 1px solid #f0f2f7;
    font-size: 0.84rem;
}
.spec-row:last-child { border-bottom: none; }
.spec-row:nth-child(even) { background: #fafbfd; }
.spec-k { color: #6b7a99; font-weight: 500; }
.spec-v { color: #0a1f44; font-weight: 700; font-size: 0.82rem; background: #eff6ff; padding: 0.2rem 0.6rem; border-radius: 4px; }

/* ════════════ CONTENT CARD ════════════ */
.cc {
    background: white;
    border: 1px solid #e4e8ef;
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.cc-title { font-size: 0.9rem; font-weight: 700; color: #0a1f44; margin-bottom: 0.3rem; }
.cc-body  { font-size: 0.84rem; color: #475569; line-height: 1.8; }
.cc-list  { font-size: 0.84rem; color: #475569; line-height: 2; padding-left: 1.1rem; margin-top: 0.4rem; }

/* ════════════ TECH PILL ════════════ */
.pill-wrap { display:flex; flex-wrap:wrap; gap:0.45rem; margin-top:0.7rem; }
.pill {
    background: #f0f7ff; border: 1px solid #c7dff7;
    border-radius: 4px; padding: 0.3rem 0.8rem;
    font-size: 0.78rem; color: #0a4fa6; font-weight: 600;
}

/* ════════════ SIDEBAR ════════════ */
[data-testid="stSidebar"] {
    background: white !important;
    border-right: 1px solid #e4e8ef !important;
}
.sb-section { padding: 0.6rem 0; }
.sb-label { font-size: 0.68rem; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; color: #8a96b0; margin-bottom: 0.5rem; }
.sb-stat { display: flex; justify-content: space-between; align-items: center; padding: 0.45rem 0; border-bottom: 1px solid #f0f2f7; font-size: 0.83rem; }
.sb-stat:last-child { border-bottom: none; }
.sb-stat-k { color: #6b7a99; }
.sb-stat-v { font-weight: 700; color: #0a1f44; }
.sb-info {
    background: #f0f7ff; border: 1px solid #c7dff7;
    border-left: 3px solid #0076CE;
    border-radius: 0 6px 6px 0;
    padding: 0.7rem 0.9rem;
    font-size: 0.8rem; color: #374151; line-height: 1.6;
    margin-top: 0.5rem;
}

/* ════════════ INPUTS ════════════ */
label { color: #374151 !important; font-size: 0.83rem !important; font-weight: 600 !important; }
/* hide default streamlit form border */
[data-testid="stForm"] { border: none !important; padding: 0 !important; background: transparent !important; }

/* ════════════ BUTTON ════════════ */
.stButton > button {
    background: #0076CE !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.65rem 2rem !important;
    font-size: 0.88rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.3px;
    box-shadow: 0 2px 8px rgba(0,118,206,0.28) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #005fa3 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(0,118,206,0.4) !important;
}

/* ════════════ PLACEHOLDER CARD ════════════ */
.placeholder {
    background: white; border: 1.5px dashed #d0d8e8;
    border-radius: 10px; padding: 3.5rem 2rem;
    text-align: center;
}
.ph-icon  { font-size: 2.5rem; margin-bottom: 0.8rem; opacity: 0.6; }
.ph-text  { font-size: 0.88rem; color: #94a3b8; line-height: 1.7; }
.ph-cta   { color: #0076CE; font-weight: 700; }

/* Footer */
.footer {
    text-align: center;
    font-size: 0.75rem;
    color: #b0baca;
    padding: 1.8rem 0 0.5rem;
    border-top: 1px solid #e4e8ef;
    margin-top: 2.5rem;
}
</style>

<!-- ══════════ TOP NAV ══════════ -->
<div class="topnav">
  <div class="nav-brand">
    🏦 LoanSense AI
    <span>Credit Assessment Platform</span>
  </div>
  <div class="nav-badge">v1.0 · Random Forest</div>
</div>

<!-- ══════════ PARALLAX HERO ══════════ -->
<div id="px-root">
  <div id="px-bg"></div>
  <div id="px-grid"></div>
  <div id="px-lines"></div>
  <div id="px-shapes">
    <div class="px-shape" style="width:340px;height:340px;top:-100px;left:-80px;animation-duration:8s;"></div>
    <div class="px-shape" style="width:240px;height:240px;bottom:-60px;right:-40px;animation-duration:10s;animation-delay:-4s;"></div>
    <div class="px-shape" style="width:140px;height:140px;top:35%;left:62%;animation-duration:7s;animation-delay:-2s;"></div>
    <div class="px-shape" style="width:90px;height:90px;top:18%;right:22%;animation-duration:9s;animation-delay:-6s;"></div>
  </div>
  <div id="px-content">
    <div class="px-overline">AI-Powered Credit Decision Engine</div>
    <div class="px-title">Intelligent <em>Loan Assessment</em></div>
    <div class="px-sub">
      Predict loan eligibility instantly using machine learning.<br>
      Accurate · Explainable · Built for financial professionals.
    </div>
    <div class="px-tags">
      <span class="px-tag">Random Forest · 300 Trees</span>
      <span class="px-tag">SMOTE · Class Balancing</span>
      <span class="px-tag">5-Fold Cross Validation</span>
      <span class="px-tag">ROC-AUC Optimised</span>
    </div>
  </div>
  <div class="px-cue">⌄</div>
</div>

<script>
(function(){
  const root = document.getElementById('px-root');
  if(!root) return;
  const layers = [
    {id:'px-bg',     fx:5,  fy:3},
    {id:'px-grid',   fx:12, fy:8},
    {id:'px-lines',  fx:18, fy:12},
    {id:'px-shapes', fx:26, fy:18},
  ];
  root.addEventListener('mousemove', function(e){
    const r = root.getBoundingClientRect();
    const dx = (e.clientX - r.left - r.width/2) / r.width;
    const dy = (e.clientY - r.top - r.height/2) / r.height;
    layers.forEach(({id,fx,fy})=>{
      const el = document.getElementById(id);
      if(el) el.style.transform = `translate(${dx*fx}px,${dy*fy}px)`;
    });
  });
  root.addEventListener('mouseleave', function(){
    layers.forEach(({id})=>{
      const el = document.getElementById(id);
      if(el) el.style.transform = 'translate(0,0)';
    });
  });
})();
</script>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# KPI Strip
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="kpi-strip">
  <div class="kpi"><div class="kpi-icon">🎯</div><div class="kpi-val">{metrics['accuracy']}%</div><div class="kpi-lbl">Test Accuracy</div></div>
  <div class="kpi"><div class="kpi-icon">📐</div><div class="kpi-val">{metrics['auc']}</div><div class="kpi-lbl">ROC-AUC Score</div></div>
  <div class="kpi"><div class="kpi-icon">🔁</div><div class="kpi-val">{metrics['cv_mean']}%</div><div class="kpi-lbl">CV Accuracy</div></div>
  <div class="kpi"><div class="kpi-icon">📁</div><div class="kpi-val">{metrics['n_samples']:,}</div><div class="kpi-lbl">Training Records</div></div>
  <div class="kpi"><div class="kpi-icon">✅</div><div class="kpi-val">{metrics['approval_rate']}%</div><div class="kpi-lbl">Approval Rate</div></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Plotly theme
# ─────────────────────────────────────────────────────────────────────────────
PL = dict(
    paper_bgcolor="white",
    plot_bgcolor="#fafbfd",
    font_color="#374151",
    font_family="Inter, sans-serif",
    title_font_size=13,
    title_font_color="#0a1f44",
    margin=dict(t=44, b=24, l=14, r=14),
    legend=dict(bgcolor="white", bordercolor="#e4e8ef", borderwidth=1, font_size=12),
)
COLORS = {"Y": "#0076CE", "N": "#e2e8f0"}
COLORS_NAMED = {"Y": "#0076CE", "N": "#ef4444"}

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍  Credit Assessment",
    "📊  Data Insights",
    "📈  Model Report",
    "ℹ️  Documentation",
    "📂  Batch Upload",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class="sh">
      <div class="sh-title">Credit Assessment Form</div>
      <div class="sh-desc">Complete all sections to generate an AI-powered eligibility decision</div>
    </div>
    <hr class="sh-rule">
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([1.1, 1], gap="large")

    with col_form:
        with st.form("loan_form", border=False):

            # — Personal Details —
            st.markdown("""
            <div class="fc">
              <div class="fc-header">
                <span class="fc-icon">👤</span>
                <span class="fc-title">Personal Details</span>
                <span class="fc-desc">Demographic information</span>
              </div>
            </div>
            """, unsafe_allow_html=True)
            r1a, r1b = st.columns(2)
            gender  = r1a.selectbox("Gender",         ["Male", "Female"])
            married = r1b.selectbox("Marital Status", ["Yes", "No"])

            r2a, r2b = st.columns(2)
            dependents = r2a.selectbox("Dependents",   ["0", "1", "2", "3+"])
            education  = r2b.selectbox("Education Level", ["Graduate", "Not Graduate"])

            r3a, r3b = st.columns(2)
            self_emp  = r3a.selectbox("Employment Type", ["Salaried", "Self Employed"],
                                       format_func=lambda x: x)
            prop_area = r3b.selectbox("Property Area",   ["Semiurban", "Urban", "Rural"])

            # Map "Salaried" back to "No" for model
            self_emp_model = "No" if self_emp == "Salaried" else "Yes"

            st.markdown("<br>", unsafe_allow_html=True)

            # — Financial Details —
            st.markdown("""
            <div class="fc">
              <div class="fc-header">
                <span class="fc-icon">💰</span>
                <span class="fc-title">Financial Details</span>
                <span class="fc-desc">Income &amp; loan specifics</span>
              </div>
            </div>
            """, unsafe_allow_html=True)
            r4a, r4b = st.columns(2)
            app_income   = r4a.number_input("Applicant Monthly Income (₹)", 0, 1_000_000, 55000, 1000)
            coapp_income = r4b.number_input("Co-applicant Income (₹)",       0, 500_000,   0,     1000)

            r5a, r5b = st.columns(2)
            loan_amt  = r5a.number_input("Requested Loan Amount (₹ K)", 1, 10_000, 150, 10)
            loan_term = r5b.selectbox("Repayment Term (months)", [120, 180, 240, 300, 360, 480], index=4)

            st.markdown("<br>", unsafe_allow_html=True)

            # — Credit Profile —
            st.markdown("""
            <div class="fc">
              <div class="fc-header">
                <span class="fc-icon">🏦</span>
                <span class="fc-title">Credit Profile</span>
                <span class="fc-desc">Most significant approval factor</span>
              </div>
            </div>
            """, unsafe_allow_html=True)
            credit = st.radio(
                "Credit History",
                [1.0, 0.0],
                format_func=lambda x: "Good standing — all prior obligations met"
                                       if x == 1.0 else
                                       "Poor standing — has outstanding defaults",
                horizontal=False,
            )

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Submit for Assessment →", use_container_width=True)

    with col_result:
        if submitted:
            label, proba = predict_loan(model, encoders, scaler, dict(
                Gender=gender, Married=married, Dependents=dependents,
                Education=education, Self_Employed=self_emp_model,
                ApplicantIncome=app_income, CoapplicantIncome=coapp_income,
                LoanAmount=loan_amt, Loan_Amount_Term=loan_term,
                Credit_History=credit, Property_Area=prop_area,
            ))
            approved = label == "Y"

            # Decision box
            if approved:
                st.markdown(f"""
                <div class="rp rp-approved">
                  <div><span class="rp-status rp-status-approved">● APPROVED</span></div>
                  <div class="rp-icon">✅</div>
                  <div class="rp-title" style="color:#15803d;">Loan Eligible</div>
                  <div class="rp-desc">This applicant meets the required credit criteria.<br>The application is recommended for approval.</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="rp rp-rejected">
                  <div><span class="rp-status rp-status-rejected">● DECLINED</span></div>
                  <div class="rp-icon">❌</div>
                  <div class="rp-title" style="color:#dc2626;">Loan Ineligible</div>
                  <div class="rp-desc">This application does not meet the current<br>eligibility criteria for loan approval.</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Probability gauge
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba[1] * 100,
                number={"suffix": "%", "font": {"size": 26, "color": "#0a1f44"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#d0d8e8",
                             "tickfont": {"color": "#8a96b0", "size": 11}},
                    "bar":  {"color": "#16a34a" if approved else "#dc2626", "thickness": 0.6},
                    "bgcolor": "#fafbfd", "bordercolor": "#e4e8ef",
                    "steps": [
                        {"range": [0,  40],  "color": "#fef2f2"},
                        {"range": [40, 70],  "color": "#fefce8"},
                        {"range": [70, 100], "color": "#f0fdf4"},
                    ],
                    "threshold": {"line": {"color": "#94a3b8", "width": 2}, "value": 50},
                },
                title={"text": "Approval Probability Score", "font": {"color": "#6b7a99", "size": 12}},
            ))
            fig_g.update_layout(**PL, height=250)
            st.plotly_chart(fig_g, use_container_width=True)

            c1, c2 = st.columns(2)
            c1.metric("Approval Probability",  f"{proba[1]*100:.1f}%")
            c2.metric("Rejection Probability", f"{proba[0]*100:.1f}%")

            # Recommendations
            if not approved:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Recommendations to Improve Eligibility**")
                if credit == 0.0:
                    st.markdown("""<div class="rec-item">🏆 Settle all outstanding debts to establish a positive credit history — this is the single highest-impact factor.</div>""", unsafe_allow_html=True)
                if app_income < 40000:
                    st.markdown("""<div class="rec-item">💼 Strengthen the application with a higher-earning co-applicant or an increase in declared income.</div>""", unsafe_allow_html=True)
                if loan_amt > 200:
                    st.markdown("""<div class="rec-item">📉 Consider reducing the requested loan amount to lower the debt-to-income ratio.</div>""", unsafe_allow_html=True)
                if credit == 1.0 and app_income >= 40000 and loan_amt <= 200:
                    st.markdown("""<div class="rec-item">📋 Review your complete financial profile and reapply after 3–6 months of stable income.</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="placeholder">
              <div class="ph-icon">📋</div>
              <div class="ph-text">
                Complete the assessment form on the left<br>
                and click <span class="ph-cta">Submit for Assessment</span> to receive a decision.
              </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – DATA INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="sh">
      <div class="sh-title">Dataset Insights</div>
      <div class="sh-desc">Exploratory analysis of the loan training dataset</div>
    </div>
    <hr class="sh-rule">
    """, unsafe_allow_html=True)

    total_n = len(df_data)
    app_n   = (df_data["Loan_Status"] == "Y").sum()
    rej_n   = total_n - app_n
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Records",   f"{total_n:,}")
    m2.metric("Approved",        f"{app_n:,}",  f"{app_n/total_n*100:.1f}%")
    m3.metric("Rejected",        f"{rej_n:,}",  f"{rej_n/total_n*100:.1f}%")
    m4.metric("Feature Count",   "11")
    st.markdown("---")

    # Row 1
    ea, eb = st.columns(2)
    with ea:
        fig = px.pie(df_data, names="Loan_Status", title="Approval Distribution",
                     color="Loan_Status", color_discrete_map={"Y":"#0076CE","N":"#ef4444"}, hole=0.55)
        fig.update_layout(**PL)
        st.plotly_chart(fig, use_container_width=True)
    with eb:
        fig = px.histogram(df_data, x="Education", color="Loan_Status", barmode="group",
                           title="Education Level vs Approval",
                           color_discrete_map={"Y":"#0076CE","N":"#ef4444"})
        fig.update_layout(**PL)
        st.plotly_chart(fig, use_container_width=True)

    # Row 2
    ea2, eb2 = st.columns(2)
    with ea2:
        fig = px.box(df_data, x="Loan_Status", y="ApplicantIncome",
                     title="Applicant Income by Decision",
                     color="Loan_Status", color_discrete_map={"Y":"#0076CE","N":"#ef4444"})
        fig.update_layout(**PL)
        st.plotly_chart(fig, use_container_width=True)
    with eb2:
        cdf = df_data.groupby(["Credit_History","Loan_Status"]).size().reset_index(name="Count")
        cdf["Credit_History"] = cdf["Credit_History"].map({1.0: "Good", 0.0: "Poor"})
        fig = px.bar(cdf, x="Credit_History", y="Count", color="Loan_Status", barmode="group",
                     title="Credit History Impact",
                     color_discrete_map={"Y":"#0076CE","N":"#ef4444"})
        fig.update_layout(**PL)
        st.plotly_chart(fig, use_container_width=True)

    # Row 3
    ea3, eb3 = st.columns(2)
    with ea3:
        fig = px.histogram(df_data, x="Property_Area", color="Loan_Status", barmode="group",
                           title="Property Area vs Approval",
                           color_discrete_map={"Y":"#0076CE","N":"#ef4444"})
        fig.update_layout(**PL)
        st.plotly_chart(fig, use_container_width=True)
    with eb3:
        fig = px.histogram(df_data, x="LoanAmount", color="Loan_Status", nbins=50,
                           title="Loan Amount Distribution",
                           color_discrete_map={"Y":"#0076CE","N":"#ef4444"}, opacity=0.78)
        fig.update_layout(**PL)
        st.plotly_chart(fig, use_container_width=True)

    # Correlation
    st.markdown("**Feature Correlation Matrix**")
    num_df = df_data[["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History"]]
    fig = px.imshow(num_df.corr(), text_auto=".2f", color_continuous_scale="Blues",
                    title="Numerical Feature Correlations")
    fig.update_layout(**PL)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View Raw Dataset — First 100 Rows"):
        st.dataframe(df_data.head(100), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – MODEL REPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="sh">
      <div class="sh-title">Model Evaluation Report</div>
      <div class="sh-desc">Performance metrics for the trained Random Forest classifier</div>
    </div>
    <hr class="sh-rule">
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test Accuracy",  f"{metrics['accuracy']}%")
    m2.metric("ROC-AUC Score",  f"{metrics['auc']}")
    m3.metric("CV Mean",        f"{metrics['cv_mean']}%")
    m4.metric("CV Std Dev",     f"±{metrics['cv_std']}%")

    st.markdown("---")
    pc1, pc2 = st.columns(2)

    with pc1:
        cm = np.array(metrics["confusion_matrix"])
        labels = ["Rejected", "Approved"]
        fig = px.imshow(cm, text_auto=True, x=labels, y=labels,
                        labels={"x":"Predicted","y":"Actual"},
                        title="Confusion Matrix",
                        color_continuous_scale=[[0,"#eff6ff"],[1,"#0076CE"]])
        fig.update_layout(**PL)
        st.plotly_chart(fig, use_container_width=True)

    with pc2:
        feat_df = pd.DataFrame({
            "Feature":    metrics["feature_names"],
            "Importance": metrics["feature_importances"],
        }).sort_values("Importance", ascending=True)
        fig = px.bar(feat_df, x="Importance", y="Feature", orientation="h",
                     title="Feature Importance Ranking",
                     color="Importance", color_continuous_scale=[[0,"#bfdbfe"],[1,"#0076CE"]])
        fig.update_layout(**PL)
        st.plotly_chart(fig, use_container_width=True)

    # Accuracy gauge
    st.markdown("---")
    fig_acc = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=metrics["accuracy"],
        delta={"reference": 80, "suffix": "%", "increasing": {"color": "#16a34a"}},
        number={"suffix": "%", "font": {"size": 36, "color": "#0a1f44"}},
        gauge={
            "axis":  {"range": [50, 100], "tickcolor": "#d0d8e8"},
            "bar":   {"color": "#0076CE", "thickness": 0.6},
            "bgcolor": "#fafbfd", "bordercolor": "#e4e8ef",
            "steps": [
                {"range": [50, 70],  "color": "#fef2f2"},
                {"range": [70, 85],  "color": "#fefce8"},
                {"range": [85, 100], "color": "#f0fdf4"},
            ],
        },
        title={"text": "Test Accuracy vs 80% Industry Baseline", "font": {"color": "#6b7a99", "size": 13}},
    ))
    fig_acc.update_layout(**PL, height=290)
    st.plotly_chart(fig_acc, use_container_width=True)

    # Spec table
    st.markdown("---")
    st.markdown("**Model Configuration**")
    specs = [
        ("Algorithm",        "Random Forest Classifier"),
        ("Estimators",       "300 decision trees"),
        ("Max Depth",        "12 levels"),
        ("Max Features",     "sqrt (auto)"),
        ("Class Weight",     "Balanced"),
        ("Oversampling",     "SMOTE"),
        ("Train / Test",     "80% / 20%"),
        ("Cross-Validation", "5-Fold Stratified CV"),
        ("Training Records", f"{metrics['n_samples']:,}"),
        ("Dataset Approval", f"{metrics['approval_rate']}%"),
    ]
    sc1, sc2 = st.columns(2)
    left_specs  = [f'<div class="spec-row"><span class="spec-k">{k}</span><span class="spec-v">{v}</span></div>' for k,v in specs[:5]]
    right_specs = [f'<div class="spec-row"><span class="spec-k">{k}</span><span class="spec-v">{v}</span></div>' for k,v in specs[5:]]
    sc1.markdown(f'<div class="spec-table">{"".join(left_specs)}</div>', unsafe_allow_html=True)
    sc2.markdown(f'<div class="spec-table">{"".join(right_specs)}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – DOCUMENTATION
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div class="sh">
      <div class="sh-title">Documentation</div>
      <div class="sh-desc">System overview, methodology, and technical reference</div>
    </div>
    <hr class="sh-rule">
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="cc">
      <div class="cc-title">System Overview</div>
      <div class="cc-body">
        LoanSense AI is an end-to-end machine learning system designed for automated loan eligibility assessment.
        It processes structured applicant data — including demographics, income, and credit history — through a
        trained <strong>Random Forest Classifier</strong> to produce a binary decision (approved/rejected) with
        calibrated probability scores. The pipeline incorporates SMOTE oversampling to address class imbalance
        inherent in typical credit datasets.
      </div>
    </div>""", unsafe_allow_html=True)

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("""
        <div class="cc">
          <div class="cc-title">ML Pipeline</div>
          <ul class="cc-list">
            <li>Label Encoding — categorical variables</li>
            <li>Standard Scaling — numerical variables</li>
            <li>SMOTE — synthetic minority oversampling</li>
            <li>Random Forest — ensemble classifier</li>
            <li>5-Fold Stratified Cross-Validation</li>
            <li>ROC-AUC + Accuracy evaluation</li>
          </ul>
        </div>""", unsafe_allow_html=True)
    with d2:
        st.markdown("""
        <div class="cc">
          <div class="cc-title">Input Features (11 Total)</div>
          <ul class="cc-list">
            <li>Gender · Marital Status · Dependents</li>
            <li>Education Level · Employment Type</li>
            <li>Applicant &amp; Co-applicant Monthly Income</li>
            <li>Requested Loan Amount (₹ thousands)</li>
            <li>Repayment Term (months)</li>
            <li>Credit History <em>— most significant predictor</em></li>
            <li>Property Area</li>
          </ul>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="cc">
      <div class="cc-title">Technology Stack</div>
      <div class="pill-wrap">
        <span class="pill">Python 3.10+</span>
        <span class="pill">Streamlit 1.32</span>
        <span class="pill">scikit-learn 1.4</span>
        <span class="pill">pandas 2.2</span>
        <span class="pill">numpy 1.26</span>
        <span class="pill">Plotly 5.20</span>
        <span class="pill">imbalanced-learn 0.12</span>
        <span class="pill">joblib 1.3</span>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="cc" style="border-left: 3px solid #f59e0b;">
      <div class="cc-title" style="color:#b45309;">⚠️ Disclaimer</div>
      <div class="cc-body">
        This system is developed for educational and demonstration purposes. Decisions generated by this
        tool should not be used as the sole basis for real-world lending decisions. All final credit
        assessments must comply with applicable financial regulations and involve qualified professionals.
      </div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 – BATCH CSV UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("""
    <div class="sh">
      <div class="sh-title">Batch CSV Prediction</div>
      <div class="sh-desc">Upload a CSV file with multiple applicants and get instant bulk predictions</div>
    </div>
    <hr class="sh-rule">
    """, unsafe_allow_html=True)

    # ── Template download
    TEMPLATE_COLS = [
        "Gender", "Married", "Dependents", "Education", "Self_Employed",
        "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
        "Loan_Amount_Term", "Credit_History", "Property_Area",
    ]
    SAMPLE_ROWS = [
        ["Male",   "Yes", "0",  "Graduate",     "No",  55000, 0,    150, 360, 1.0, "Semiurban"],
        ["Female", "No",  "1",  "Not Graduate", "Yes", 32000, 5000, 120, 360, 1.0, "Urban"],
        ["Male",   "Yes", "2",  "Graduate",     "No",  48000, 8000, 200, 300, 0.0, "Rural"],
        ["Female", "No",  "3+", "Graduate",     "No",  75000, 0,    250, 360, 1.0, "Urban"],
        ["Male",   "Yes", "0",  "Not Graduate", "Yes", 22000, 3000, 80,  180, 0.0, "Semiurban"],
    ]
    template_df = pd.DataFrame(SAMPLE_ROWS, columns=TEMPLATE_COLS)
    template_csv = template_df.to_csv(index=False).encode("utf-8")

    bl, br = st.columns([2, 1])
    with bl:
        st.markdown("""
        <div class="fc">
          <div class="fc-header">
            <span class="fc-icon">📋</span>
            <span class="fc-title">Step 1 — Download the Template</span>
          </div>
          <div style="font-size:0.84rem;color:#475569;line-height:1.8;">
            Download the CSV template below. Fill in one applicant per row using the accepted values:
            <ul style="margin-top:0.5rem;padding-left:1.1rem;">
              <li><strong>Gender</strong>: Male / Female</li>
              <li><strong>Married</strong>: Yes / No</li>
              <li><strong>Dependents</strong>: 0 / 1 / 2 / 3+</li>
              <li><strong>Education</strong>: Graduate / Not Graduate</li>
              <li><strong>Self_Employed</strong>: Yes / No</li>
              <li><strong>ApplicantIncome / CoapplicantIncome</strong>: Monthly income in ₹</li>
              <li><strong>LoanAmount</strong>: Amount in ₹ thousands</li>
              <li><strong>Loan_Amount_Term</strong>: Months (e.g. 360)</li>
              <li><strong>Credit_History</strong>: 1.0 = Good · 0.0 = Poor</li>
              <li><strong>Property_Area</strong>: Semiurban / Urban / Rural</li>
            </ul>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.download_button(
            label="⬇️  Download CSV Template",
            data=template_csv,
            file_name="loansense_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with br:
        st.markdown("""
        <div class="fc" style="height:100%;">
          <div class="fc-header">
            <span class="fc-icon">💡</span>
            <span class="fc-title">Tips</span>
          </div>
          <div style="font-size:0.82rem;color:#475569;line-height:1.9;">
            ✔ Save as <strong>.csv</strong> (UTF-8)<br>
            ✔ Keep column headers exactly as shown<br>
            ✔ Max <strong>5,000</strong> rows per upload<br>
            ✔ Blanks / NaNs will flag an error<br>
            ✔ Results downloadable as CSV
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── File uploader
    st.markdown("""
    <div class="fc">
      <div class="fc-header">
        <span class="fc-icon">📂</span>
        <span class="fc-title">Step 2 — Upload Your CSV</span>
        <span class="fc-desc">Supports .csv files up to 200 MB</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drag & drop or browse to upload your applicant CSV",
        type=["csv"],
        help="The file must contain exactly the 11 required columns.",
        key="batch_csv_uploader",
    )

    if uploaded is not None:
        try:
            raw_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"❌ Could not parse file: {e}")
            raw_df = None

        if raw_df is not None:
            # Guard: row count
            if len(raw_df) > 5000:
                st.warning("⚠️ File exceeds 5,000 rows — only the first 5,000 will be processed.")
                raw_df = raw_df.head(5000)

            # Guard: NaN check
            null_count = raw_df[TEMPLATE_COLS].isnull().sum().sum() if all(c in raw_df.columns for c in TEMPLATE_COLS) else -1
            if null_count > 0:
                st.warning(f"⚠️ {null_count} missing value(s) detected. Rows with missing data may cause errors.")
                raw_df = raw_df.dropna(subset=[c for c in TEMPLATE_COLS if c in raw_df.columns])

            with st.spinner("⚙️ Running batch predictions…"):
                result_df, err = predict_batch(model, encoders, scaler, raw_df)

            if err:
                st.error(f"❌ Prediction error: {err}")
            else:
                # ── Summary KPIs
                total_r   = len(result_df)
                approved_r = (result_df["Decision"] == "Y").sum()
                rejected_r = total_r - approved_r
                avg_prob  = result_df["Approval_Probability"].mean()

                st.markdown("<br>", unsafe_allow_html=True)
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Total Applicants", f"{total_r:,}")
                k2.metric("✅ Approved",       f"{approved_r:,}",  f"{approved_r/total_r*100:.1f}%")
                k3.metric("❌ Rejected",       f"{rejected_r:,}",  f"{rejected_r/total_r*100:.1f}%")
                k4.metric("Avg Approval Score", f"{avg_prob:.1f}%")

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Donut chart
                fig_donut = px.pie(
                    values=[approved_r, rejected_r],
                    names=["Approved", "Rejected"],
                    hole=0.6,
                    color_discrete_sequence=["#0076CE", "#ef4444"],
                    title="Batch Decision Distribution",
                )
                fig_donut.update_layout(**PL, height=280)

                # ── Probability histogram
                fig_hist = px.histogram(
                    result_df, x="Approval_Probability", nbins=20,
                    title="Approval Probability Distribution",
                    color_discrete_sequence=["#0076CE"],
                    labels={"Approval_Probability": "Approval Probability (%)"},
                )
                fig_hist.update_layout(**PL, height=280)

                ch1, ch2 = st.columns(2)
                with ch1:
                    st.plotly_chart(fig_donut, use_container_width=True)
                with ch2:
                    st.plotly_chart(fig_hist, use_container_width=True)

                # ── Results table with styled Decision column
                st.markdown("**Prediction Results**")
                display_df = result_df.copy()
                display_df["Decision"] = display_df["Decision"].map(
                    {"Y": "✅ Approved", "N": "❌ Rejected"}
                )
                display_df["Approval_Probability"] = display_df["Approval_Probability"].apply(
                    lambda x: f"{x:.1f}%"
                )

                # Colour rows: green = approved, red = rejected
                def row_color(row):
                    if "Approved" in str(row["Decision"]):
                        return ["background-color: #f0fdf4"] * len(row)
                    return ["background-color: #fff8f8"] * len(row)

                styled = display_df.style.apply(row_color, axis=1)
                st.dataframe(styled, use_container_width=True, height=400)

                # ── Download results
                out_csv = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️  Download Full Results as CSV",
                    data=out_csv,
                    file_name="loansense_batch_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
    else:
        st.markdown("""
        <div class="placeholder">
          <div class="ph-icon">📂</div>
          <div class="ph-text">
            Upload a CSV to run bulk predictions<br>
            <span class="ph-cta">Download the template above</span> to get started
          </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  LoanSense AI · Credit Assessment Platform · Built with Streamlit &amp; scikit-learn
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**🏦 LoanSense AI**")
    st.markdown("Credit Assessment Platform")
    st.markdown("---")

    st.markdown('<div class="sb-label">Model Performance</div>', unsafe_allow_html=True)
    sb_stats = [
        ("Test Accuracy",   f"{metrics['accuracy']}%"),
        ("ROC-AUC",         f"{metrics['auc']}"),
        ("CV Accuracy",     f"{metrics['cv_mean']}%"),
        ("CV Std Dev",      f"±{metrics['cv_std']}%"),
        ("Training Size",   f"{metrics['n_samples']:,}"),
        ("Approval Rate",   f"{metrics['approval_rate']}%"),
    ]
    rows = "".join(
        f'<div class="sb-stat"><span class="sb-stat-k">{k}</span><span class="sb-stat-v">{v}</span></div>'
        for k, v in sb_stats
    )
    st.markdown(rows, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sb-label">Algorithm</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.83rem; color:#374151; line-height:1.9;">
      Random Forest Classifier<br>
      300 trees · Max depth 12<br>
      SMOTE · 5-Fold Stratified CV<br>
      Balanced class weighting
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="sb-info">
      <strong>Key Insight:</strong> Credit History alone accounts for <strong>66%</strong> of
      the model's decision weight — the single most impactful feature.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("© 2025 LoanSense AI · v1.0")
