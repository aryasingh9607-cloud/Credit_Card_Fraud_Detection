"""
Credit Card Fraud Detection Dashboard
Simple, clean UI with dynamic Light & Dark Mode toggle.
Pages: Overview & Transaction Analyzer.
"""

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib

# ─────────────────────────────────────────────────────────────────────────────
# 1. PAGE SETUP
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. SESSION STATE FOR THEME (LIGHT / DARK) & SAMPLES
# ─────────────────────────────────────────────────────────────────────────────
if 'dark_mode' not in st.session_state:
    st.session_state['dark_mode'] = False

is_dark = st.session_state['dark_mode']

# Load Presets Directly
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR   = os.path.join(BASE_DIR, 'models')
SAMPLE_CSV   = os.path.join(BASE_DIR, 'dataset', 'sample_test_transactions.csv')
PRESETS_FILE = os.path.join(MODELS_DIR, 'sample_presets.json')
FEATURE_COLS = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']

def load_presets_direct():
    if os.path.exists(PRESETS_FILE):
        with open(PRESETS_FILE) as f:
            return json.load(f)
    return {}

presets = load_presets_direct()

# Verified initial default: Sample 3 (Confirmed Real Fraud row 46918 - $648.00)
init_p = presets.get('fraud_high', {})

if 'input_amount' not in st.session_state:
    st.session_state['input_amount'] = float(init_p.get('Amount', 648.00))
if 'input_time' not in st.session_state:
    st.session_state['input_time'] = float(init_p.get('Time', 42988.0))
if 'active_sample_name' not in st.session_state:
    st.session_state['active_sample_name'] = "Sample 3: High-Value Drain ($648.00) — Real Fraud"

# Default V1 to V28
for i in range(1, 29):
    if f'v_{i}' not in st.session_state:
        st.session_state[f'v_{i}'] = float(init_p.get(f'V{i}', 0.0))

def apply_preset(preset_key, display_name):
    p = presets.get(preset_key, {})
    if p:
        st.session_state['input_amount'] = float(p.get('Amount', 100.0))
        st.session_state['input_time']   = float(p.get('Time', 500.0))
        st.session_state['active_sample_name'] = display_name
        for i in range(1, 29):
            st.session_state[f'v_{i}']   = float(p.get(f'V{i}', 0.0))

# ─────────────────────────────────────────────────────────────────────────────
# 3. DYNAMIC THEME COLOR PALETTES (LIGHT VS DARK)
# ─────────────────────────────────────────────────────────────────────────────
if is_dark:
    # 🌙 DARK MODE PALETTE
    bg_app         = "#0a0e1a"
    bg_card        = "#12192c"
    bg_sidebar     = "#0d1322"
    border_color   = "#1f2b45"
    border_input   = "#2e3e60"
    text_primary   = "#f8fafc"
    text_secondary = "#cbd5e1"
    text_muted     = "#94a3b8"
    input_bg       = "#182239"
    banner_bg      = "#151e33"
    safe_bg        = "#064e3b"
    safe_border    = "#059669"
    safe_text      = "#a7f3d0"
    fraud_bg       = "#450a0a"
    fraud_border   = "#dc2626"
    fraud_text     = "#fecaca"
    tab_list_bg    = "#162035"
    tab_active_bg  = "#1e293b"
else:
    # ☀️ LIGHT MODE PALETTE
    bg_app         = "#f8fafc"
    bg_card        = "#ffffff"
    bg_sidebar     = "#ffffff"
    border_color   = "#e2e8f0"
    border_input   = "#cbd5e1"
    text_primary   = "#0f172a"
    text_secondary = "#334155"
    text_muted     = "#64748b"
    input_bg       = "#ffffff"
    banner_bg      = "#f1f5f9"
    safe_bg        = "#f0fdf4"
    safe_border    = "#86efac"
    safe_text      = "#14532d"
    fraud_bg       = "#fef2f2"
    fraud_border   = "#fca5a5"
    fraud_text     = "#7f1d1d"
    tab_list_bg    = "#e2e8f0"
    tab_active_bg  = "#ffffff"

# Inject Dynamic CSS
st.markdown(f"""
<style>
* {{
    box-sizing: border-box;
}}

body, html, .stApp {{
    background-color: {bg_app} !important;
    color: {text_primary} !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}}

.main .block-container {{
    padding: 1.8rem 2.2rem;
    max-width: 1200px;
}}

#MainMenu, header, footer {{
    visibility: hidden;
    height: 0;
}}

/* Text & Headers */
h1, h2, h3, h4, h5, h6,
p, span, div, a, li,
label, label p, label span,
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span {{
    color: {text_primary} !important;
}}

/* Form Labels */
label[data-testid="stWidgetLabel"] p {{
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    color: {text_primary} !important;
    margin-bottom: 0.35rem !important;
}}

/* Inputs (Text, Number) */
input,
input[type="text"],
input[type="number"],
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {{
    background-color: {input_bg} !important;
    color: {text_primary} !important;
    border: 1.5px solid {border_input} !important;
    border-radius: 8px !important;
    padding: 0.5rem 0.75rem !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
}}

input:focus {{
    border-color: #2563eb !important;
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2) !important;
}}

[data-testid="stNumberInput"] button {{
    background-color: {banner_bg} !important;
    color: {text_primary} !important;
    border-color: {border_input} !important;
}}

/* Sidebar */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child,
[data-testid="stSidebarContent"],
section[data-testid="stSidebar"] {{
    background-color: {bg_sidebar} !important;
    border-right: 1px solid {border_color} !important;
}}

[data-testid="stSidebar"] * {{
    color: {text_primary} !important;
}}

/* Sidebar Radio Buttons */
[data-testid="stSidebar"] [data-baseweb="radio"] label {{
    padding: 0.65rem 0.95rem !important;
    border-radius: 8px !important;
    margin-bottom: 4px !important;
    color: {text_secondary} !important;
    font-weight: 500 !important;
    font-size: 0.92rem !important;
    background: transparent !important;
    border: 1px solid transparent !important;
}}

[data-testid="stSidebar"] [data-baseweb="radio"] label:hover {{
    background-color: {banner_bg} !important;
    color: #2563eb !important;
}}

[data-testid="stSidebar"] [data-baseweb="radio"] [aria-checked="true"] + div label {{
    background-color: {'#1e293b' if is_dark else '#eff6ff'} !important;
    color: {'#60a5fa' if is_dark else '#2563eb'} !important;
    font-weight: 700 !important;
    border-color: {'#3b82f6' if is_dark else '#bfdbfe'} !important;
}}

/* Boxes and Cards */
.clean-box {{
    background: {bg_card};
    border: 1px solid {border_color};
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,{'0.2' if is_dark else '0.03'});
}}

.box-heading {{
    font-size: 1.15rem;
    font-weight: 700;
    color: {text_primary} !important;
    margin-bottom: 0.3rem;
}}

.box-subtext {{
    font-size: 0.85rem;
    color: {text_muted} !important;
    margin-bottom: 1rem;
    line-height: 1.5;
}}

.header-card {{
    background: {bg_card};
    border: 1px solid {border_color};
    border-radius: 12px;
    padding: 1.2rem 1.6rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,{'0.2' if is_dark else '0.03'});
    display: flex;
    align-items: center;
    justify-content: space-between;
}}

.header-title-text {{
    font-size: 1.35rem;
    font-weight: 700;
    color: {text_primary} !important;
    margin: 0;
}}

.header-sub-text {{
    font-size: 0.86rem;
    color: {text_muted} !important;
    margin-top: 0.2rem;
}}

.metric-tile {{
    background: {bg_card};
    border: 1px solid {border_color};
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 1px 3px rgba(0,0,0,{'0.2' if is_dark else '0.03'});
}}

.metric-lbl {{
    font-size: 0.76rem;
    font-weight: 700;
    color: {text_muted} !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.3rem;
}}

.metric-val {{
    font-size: 1.85rem;
    font-weight: 800;
    color: {text_primary} !important;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}}

.metric-desc {{
    font-size: 0.78rem;
    color: {text_muted} !important;
}}

/* Preset Samples Banner */
.preset-banner {{
    background: {banner_bg};
    border: 1px solid {border_color};
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 1.2rem;
}}

.preset-title {{
    font-size: 0.88rem;
    font-weight: 700;
    color: {text_primary} !important;
    margin-bottom: 0.4rem;
}}

/* Result Boxes */
.res-safe {{
    background: {safe_bg};
    border: 1.5px solid {safe_border};
    border-radius: 10px;
    padding: 1.2rem;
    margin: 1.2rem 0;
}}

.res-fraud {{
    background: {fraud_bg};
    border: 1.5px solid {fraud_border};
    border-radius: 10px;
    padding: 1.2rem;
    margin: 1.2rem 0;
}}

/* Buttons */
.stButton > button {{
    background: #2563eb !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.2rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}}

.stButton > button:hover {{
    background: #1d4ed8 !important;
}}

.stDownloadButton > button {{
    background: {bg_card} !important;
    color: {'#60a5fa' if is_dark else '#2563eb'} !important;
    border: 1.5px solid {border_color} !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}}

/* Expander & Tabs */
.streamlit-expanderHeader,
[data-testid="stExpander"] details summary {{
    background-color: {bg_card} !important;
    border: 1px solid {border_color} !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    color: {text_primary} !important;
}}

[data-testid="stExpander"] {{
    background-color: {bg_card} !important;
    border: 1px solid {border_color} !important;
    border-radius: 8px !important;
}}

.stTabs [data-baseweb="tab-list"] {{
    background: {tab_list_bg};
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
}}

.stTabs [data-baseweb="tab"] {{
    border-radius: 6px;
    color: {text_muted} !important;
    font-weight: 600;
    padding: 0.4rem 1rem;
}}

.stTabs [aria-selected="true"] {{
    background: {tab_active_bg} !important;
    color: {'#60a5fa' if is_dark else '#2563eb'} !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}}

[data-testid="stFileUploader"] {{
    background: {bg_card} !important;
    border: 1.5px dashed {border_color} !important;
    border-radius: 10px !important;
}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. DATA & MODEL LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    p = os.path.join(MODELS_DIR, 'xgboost.pkl')
    if not os.path.exists(p):
        p = os.path.join(MODELS_DIR, 'random_forest.pkl')
    return joblib.load(p) if os.path.exists(p) else None

@st.cache_resource(show_spinner=False)
def load_scaler():
    p = os.path.join(MODELS_DIR, 'scaler.pkl')
    return joblib.load(p) if os.path.exists(p) else None

@st.cache_data(show_spinner=False)
def load_threshold():
    p = os.path.join(MODELS_DIR, 'threshold.json')
    if os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
            return d.get('xgboost', d.get('random_forest', 0.5))
    return 0.5

model     = load_model()
scaler    = load_scaler()
threshold = load_threshold()


# ─────────────────────────────────────────────────────────────────────────────
# 5. SIDEBAR NAVIGATION WITH THEME TOGGLE AT THE BOTTOM
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.4rem 0 0.8rem 0;">
        <div style="font-size: 1.15rem; font-weight: 700;">🛡️ FraudShield</div>
        <div style="font-size: 0.78rem; opacity: 0.75;">Credit Card Fraud Detection</div>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.radio(
        "Menu",
        options=["📊 Overview", "🔍 Transaction Analyzer"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.78rem; line-height: 1.6; opacity: 0.85;">
        <strong>Dataset:</strong> European Cardholders<br>
        <strong>Algorithm:</strong> XGBoost Classifier<br>
        <strong>Status:</strong> Ready & Active<br>
        <strong>Imbalance:</strong> 99.83% Genuine / 0.17% Fraud
    </div>
    """, unsafe_allow_html=True)

    # ── THEME TOGGLE AT THE VERY END OF SIDEBAR ──
    st.markdown("---")
    st.markdown("<p style='font-size:0.8rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.4rem;'>Theme Appearance</p>", unsafe_allow_html=True)
    
    toggle_val = st.toggle("🌙 Dark Mode", value=is_dark)
    if toggle_val != is_dark:
        st.session_state['dark_mode'] = toggle_val
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# 6. TOP HEADER CARD
# ─────────────────────────────────────────────────────────────────────────────
theme_badge = "🌙 Dark Mode" if is_dark else "☀️ Light Mode"
st.markdown(f"""
<div class="header-card">
    <div>
        <div class="header-title-text">Credit Card Fraud Detection System</div>
        <div class="header-sub-text">Analyze credit card transactions and detect fraudulent activity using Machine Learning.</div>
    </div>
    <div style="background:{'#1e293b' if is_dark else '#f1f5f9'}; border:1px solid {border_color}; border-radius:20px; padding:0.3rem 0.85rem; font-size:0.78rem; font-weight:600;">
        {theme_badge}
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 7. PAGE 1: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.markdown("""
    <div class="box-heading">Dataset Overview</div>
    <div class="box-subtext">Summary statistics from the European Credit Card Fraud Detection benchmark dataset.</div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-lbl">Total Transactions</div>
            <div class="metric-val">284,807</div>
            <div class="metric-desc">Recorded over 2 days</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-lbl">Genuine Charges</div>
            <div class="metric-val" style="color: {'#34d399' if is_dark else '#16a34a'} !important;">284,315</div>
            <div class="metric-desc">99.828% of all transactions</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-lbl">Fraudulent Cases</div>
            <div class="metric-val" style="color: {'#fb7185' if is_dark else '#dc2626'} !important;">492</div>
            <div class="metric-desc">0.172% of all transactions</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-lbl">Model Accuracy</div>
            <div class="metric-val" style="color: {'#60a5fa' if is_dark else '#2563eb'} !important;">99.92%</div>
            <div class="metric-desc">PR-AUC: 0.8413 (XGBoost)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_chart, col_info = st.columns([1.2, 1])

    with col_chart:
        st.markdown(f"""
        <div class="clean-box">
            <div class="box-heading">Class Distribution</div>
            <div class="box-subtext">Visualizing the extreme imbalance between genuine and fraudulent charges.</div>
        """, unsafe_allow_html=True)

        fig = go.Figure(go.Pie(
            labels=['Genuine Transactions (99.83%)', 'Fraud Transactions (0.17%)'],
            values=[284315, 492],
            hole=0.65,
            marker=dict(colors=['#2563eb', '#dc2626']),
            textinfo='none'
        ))
        fig.update_layout(
            paper_bgcolor=bg_card,
            plot_bgcolor=bg_card,
            font=dict(color=text_primary, family='inherit'),
            margin=dict(l=10, r=10, t=10, b=10),
            height=260,
            showlegend=True,
            legend=dict(orientation='h', y=-0.1, font=dict(color=text_secondary, size=11))
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col_info:
        st.markdown(f"""
        <div class="clean-box">
            <div class="box-heading">Key Project Facts</div>
            <div style="font-size: 0.86rem; color: {text_secondary}; line-height: 1.8;">
                • <strong>Source:</strong> ULB Machine Learning Group & Worldline.<br>
                • <strong>Target Label (Class):</strong> 0 = Genuine, 1 = Fraud.<br>
                • <strong>Input Features:</strong> <code>Time</code>, <code>Amount</code>, and <code>V1 to V28</code>.<br>
                • <strong>Privacy Protection (PCA):</strong> V1 to V28 are mathematical principal components to safeguard card numbers and customer identities.<br>
                • <strong>Class Imbalance Handling:</strong> SMOTE oversampling and class weights are applied to the training data.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 8. PAGE 2: TRANSACTION ANALYZER (MANUAL WITH SAMPLES & BATCH CSV)
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Transaction Analyzer":
    st.markdown("""
    <div class="box-heading">Transaction Analyzer</div>
    <div class="box-subtext">Evaluate single transactions using 1-click sample cases or upload a CSV file to test multiple records at once.</div>
    """, unsafe_allow_html=True)

    tab_single, tab_csv = st.tabs(["⚡ Single Transaction Analyzer", "📂 Batch CSV Upload"])

    # ── TAB 1: SINGLE TRANSACTION & SAMPLES ──
    with tab_single:
        st.markdown("<br>", unsafe_allow_html=True)

        # 1-Click Sample Buttons Banner
        st.markdown(f"""
        <div class="preset-banner">
            <div class="preset-title">⚡ 1-Click Quick Test Samples:</div>
            <div style="font-size: 0.8rem; color: {text_muted}; margin-bottom: 0.6rem;">
                Click any sample button below to instantly load verified data from the dataset:
            </div>
        """, unsafe_allow_html=True)

        b1, b2, b3, b4 = st.columns(4)
        with b1:
            if st.button("☕ Sample 1: Small Purchase ($2.69)\n[Genuine]", use_container_width=True):
                apply_preset('legit_small', "Sample 1: Small Purchase ($2.69) — Genuine")
                st.rerun()
        with b2:
            if st.button("🏨 Sample 2: Travel Hotel ($378.66)\n[Genuine]", use_container_width=True):
                apply_preset('legit_large', "Sample 2: Travel Hotel ($378.66) — Genuine")
                st.rerun()
        with b3:
            if st.button("🚨 Sample 3: High-Value Drain ($648.00)\n[Real Fraud]", use_container_width=True):
                apply_preset('fraud_high', "Sample 3: High-Value Drain ($648.00) — Real Fraud")
                st.rerun()
        with b4:
            if st.button("🚨 Sample 4: Card Testing Probe ($0.00)\n[Real Fraud]", use_container_width=True):
                apply_preset('fraud_probe', "Sample 4: Card Testing Probe ($0.00) — Real Fraud")
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        # Active Sample Indicator
        st.markdown(f"""
        <div style="font-size: 0.84rem; color: {text_primary}; background: {bg_card}; border: 1px solid {border_color}; border-radius: 8px; padding: 0.6rem 0.9rem; margin-bottom: 1rem;">
            <strong>Currently Loaded:</strong> <span style="color: {'#60a5fa' if is_dark else '#2563eb'}; font-weight: 600;">{st.session_state['active_sample_name']}</span>
        </div>
        """, unsafe_allow_html=True)

        # Primary Visible Inputs
        col_amt, col_time = st.columns(2)
        with col_amt:
            amount_val = st.number_input(
                "Transaction Amount ($ USD)",
                min_value=0.0,
                max_value=50000.0,
                value=float(st.session_state['input_amount']),
                step=10.0,
                format="%.2f",
                key="amount_input_field"
            )
            st.session_state['input_amount'] = amount_val

        with col_time:
            time_val = st.number_input(
                "Elapsed Time (Seconds)",
                min_value=0.0,
                max_value=200000.0,
                value=float(st.session_state['input_time']),
                step=100.0,
                format="%.0f",
                key="time_input_field"
            )
            st.session_state['input_time'] = time_val

        # Expander for V1-V28
        with st.expander("Advanced: View or Adjust Latent Features (V1 to V28)", expanded=False):
            st.markdown(f"""
            <div style="font-size: 0.83rem; color: {text_secondary}; margin-bottom: 0.8rem; line-height: 1.5;">
                <strong>What are V1 to V28?</strong> In real banking data, credit card numbers, customer names, and merchant names are confidential. The researchers applied <strong>PCA (Principal Component Analysis)</strong> to project 28 confidential raw variables into mathematical features V1 to V28.<br>
                <em>Selecting a sample above automatically loads authentic values for V1 through V28.</em>
            </div>
            """, unsafe_allow_html=True)

            v_cols = st.columns(4)
            for i in range(1, 29):
                c_idx = (i - 1) % 4
                with v_cols[c_idx]:
                    val = st.number_input(
                        f"V{i}",
                        value=float(st.session_state[f'v_{i}']),
                        format="%.3f",
                        key=f"v_widget_{i}"
                    )
                    st.session_state[f'v_{i}'] = val

        # Full Feature Vector
        test_row = {f'V{i}': st.session_state[f'v_{i}'] for i in range(1, 29)}
        test_row['Amount'] = st.session_state['input_amount']
        test_row['Time']   = st.session_state['input_time']
        df_single = pd.DataFrame([test_row])[FEATURE_COLS]

        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🔍  Check Transaction Now", type="primary")

        # Automatically show prediction
        if analyze_btn or True:
            if model is not None and scaler is not None:
                x_scaled = scaler.transform(df_single.values)
                prob = float(model.predict_proba(x_scaled)[0, 1])
                is_fraud = prob >= threshold
                risk_score = int(np.clip(prob ** 0.7 * 100, 0, 100))

                if is_fraud:
                    st.markdown(f"""
                    <div class="res-fraud">
                        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 0.4rem;">
                            <span style="font-size: 26px;">🚨</span>
                            <span style="font-size: 1.25rem; font-weight: 800; color: {fraud_text};">Fraudulent Transaction Detected!</span>
                        </div>
                        <p style="font-size: 0.88rem; color: {fraud_text}; margin: 0 0 0.8rem 0; line-height: 1.5;">
                            This transaction exhibits abnormal latent feature signatures (notably in V14 & V17) strongly correlated with unauthorized card usage or account takeover.
                        </p>
                        <div style="display: flex; gap: 2.5rem; font-size: 0.9rem; color: {fraud_text}; padding-top: 0.4rem; border-top: 1px solid {fraud_border};">
                            <div><strong>Fraud Probability:</strong> {prob * 100:.2f}%</div>
                            <div><strong>Risk Score:</strong> {risk_score} / 100</div>
                            <div><strong>Recommended Action:</strong> Block Card & Alert Customer</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="res-safe">
                        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 0.4rem;">
                            <span style="font-size: 26px;">✅</span>
                            <span style="font-size: 1.25rem; font-weight: 800; color: {safe_text};">Genuine Transaction (Safe)</span>
                        </div>
                        <p style="font-size: 0.88rem; color: {safe_text}; margin: 0 0 0.8rem 0; line-height: 1.5;">
                            All feature coordinates remain safely within legitimate cardholder variance limits. No anomaly detected.
                        </p>
                        <div style="display: flex; gap: 2.5rem; font-size: 0.9rem; color: {safe_text}; padding-top: 0.4rem; border-top: 1px solid {safe_border};">
                            <div><strong>Fraud Probability:</strong> {prob * 100:.2f}%</div>
                            <div><strong>Risk Score:</strong> {risk_score} / 100</div>
                            <div><strong>Recommended Action:</strong> Approve Transaction</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # ── TAB 2: BATCH CSV UPLOAD ──
    with tab_csv:
        st.markdown("<br>", unsafe_allow_html=True)

        col_up, col_help = st.columns([1.5, 1])

        with col_help:
            st.markdown(f"""
            <div style="background: {bg_card}; border: 1px solid {border_color}; border-radius: 10px; padding: 1.2rem;">
                <div style="font-weight: 700; font-size: 0.95rem; color: {text_primary}; margin-bottom: 0.3rem;">Need a Test File?</div>
                <div style="font-size: 0.82rem; color: {text_muted}; margin-bottom: 1rem; line-height: 1.5;">
                    Download a ready-made test CSV containing <strong>50 authentic transactions</strong> (40 genuine + 10 fraud) to test batch predictions instantly:
                </div>
            """, unsafe_allow_html=True)

            if os.path.exists(SAMPLE_CSV):
                with open(SAMPLE_CSV, "rb") as f:
                    st.download_button(
                        "📥 Download 50-Transaction Sample CSV",
                        data=f,
                        file_name="sample_test_transactions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

            st.markdown("</div>", unsafe_allow_html=True)

        with col_up:
            uploaded_file = st.file_uploader("Upload CSV File for Batch Prediction", type=['csv'])

            if uploaded_file is not None:
                try:
                    df_up = pd.read_csv(uploaded_file)
                    missing_cols = [c for c in FEATURE_COLS if c not in df_up.columns]

                    if missing_cols:
                        st.error(f"Missing required columns in CSV: {', '.join(missing_cols)}")
                    else:
                        if model is not None and scaler is not None:
                            b_scaled = scaler.transform(df_up[FEATURE_COLS].values)
                            b_probs  = model.predict_proba(b_scaled)[:, 1]
                            b_preds  = np.where(b_probs >= threshold, '🚨 Fraud', '✅ Genuine')
                            b_scores = [int(np.clip(p ** 0.7 * 100, 0, 100)) for p in b_probs]

                            df_up['Prediction']     = b_preds
                            df_up['Fraud_Prob_%']   = np.round(b_probs * 100, 2)
                            df_up['Risk_Score_100'] = b_scores

                            total_count = len(df_up)
                            fraud_count = int((b_probs >= threshold).sum())

                            st.success(f"Processed {total_count:,} transactions. Found {fraud_count} fraud cases.")

                            showcase_cols = ['Amount', 'Time', 'Prediction', 'Fraud_Prob_%', 'Risk_Score_100']
                            st.dataframe(
                                df_up[showcase_cols].style.format({'Amount': '${:.2f}', 'Fraud_Prob_%': '{:.2f}%'}),
                                use_container_width=True,
                                height=320
                            )

                            out_csv = df_up.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "⬇ Download Complete Predictions (CSV)",
                                data=out_csv,
                                file_name="fraud_detection_predictions.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                except Exception as ex:
                    st.error(f"Error processing CSV: {ex}")
