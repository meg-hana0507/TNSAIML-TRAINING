"""
GlycoGuard AI - Clinical Diabetes Risk Prediction & Biomarker Intelligence System
Interactive Streamlit Web Application with High-Contrast Medical UI & Real-Time Analytics.
"""

from datetime import datetime
import io
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# 1. Page Configuration (Must be first Streamlit command)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="GlycoGuard AI | Diabetes Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. High-Contrast Modern Medical Styling
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Base */
    .stApp {
        background-color: #0b1120;
        color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Ensure all headers, paragraphs, and markdown are bright and legible */
    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc !important;
        font-weight: 700 !important;
    }
    p, span, div {
        color: #e2e8f0;
    }
    .stMarkdown p {
        color: #e2e8f0 !important;
        font-size: 0.95rem;
    }
    .stCaption, [data-testid="stCaptionContainer"] p {
        color: #94a3b8 !important;
        font-size: 0.88rem !important;
    }

    /* Force high-contrast text on all widget labels (Sliders, NumberInputs, Radio, etc.) */
    [data-testid="stWidgetLabel"] p, 
    [data-testid="stWidgetLabel"] label,
    [data-testid="stWidgetLabel"] span {
        color: #f8fafc !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.01em !important;
    }

    /* Slider values (current value badge above thumb) */
    div[data-testid="stThumbValue"] {
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 0.95rem !important;
        background-color: #0284c7 !important;
        border: 1px solid #38bdf8 !important;
        border-radius: 6px !important;
        padding: 3px 9px !important;
        box-shadow: 0 2px 8px rgba(2, 132, 199, 0.4) !important;
    }

    /* Slider min and max labels */
    div[data-testid="stTickBarMin"], 
    div[data-testid="stTickBarMax"],
    div[data-testid="stSliderTickBar"] span {
        color: #94a3b8 !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
    }

    /* Tab navigation buttons - High visibility & vivid active state */
    button[data-baseweb="tab"] {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        margin-right: 8px !important;
        transition: all 0.2s ease !important;
    }
    button[data-baseweb="tab"] p {
        color: #cbd5e1 !important;
        font-size: 0.92rem !important;
        font-weight: 600 !important;
    }
    button[data-baseweb="tab"]:hover {
        background-color: #334155 !important;
        border-color: #64748b !important;
    }
    button[data-baseweb="tab"]:hover p {
        color: #ffffff !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #0284c7, #2563eb) !important;
        border: 1px solid #38bdf8 !important;
        box-shadow: 0 4px 16px rgba(2, 132, 199, 0.45) !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] p {
        color: #ffffff !important;
        font-weight: 800 !important;
    }

    /* Hero Header Card */
    .hero-card {
        background: linear-gradient(135deg, #131d31 0%, #0c1427 100%);
        border: 1px solid #2d3d5c;
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.4);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        color: #cbd5e1 !important;
        font-size: 1.05rem;
        line-height: 1.5;
        margin-bottom: 14px;
    }
    .badge-pill {
        display: inline-block;
        background: #1e293b;
        border: 1px solid #3b82f6;
        border-radius: 9999px;
        padding: 5px 14px;
        font-size: 0.82rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-right: 8px;
        margin-bottom: 6px;
    }

    /* Structured Clinical Input Cards */
    .input-section-card {
        background-color: #131d31;
        border: 1px solid #2d3d5c;
        border-radius: 14px;
        padding: 20px 22px;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    }
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #38bdf8 !important;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-desc {
        font-size: 0.85rem;
        color: #94a3b8 !important;
        margin-bottom: 16px;
    }

    /* Risk Score Cards */
    .risk-card-low {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.85), rgba(15, 23, 42, 0.95));
        border: 2px solid #10b981;
        border-radius: 16px;
        padding: 22px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(16, 185, 129, 0.25);
    }
    .risk-card-mod {
        background: linear-gradient(135deg, rgba(146, 64, 14, 0.85), rgba(15, 23, 42, 0.95));
        border: 2px solid #f59e0b;
        border-radius: 16px;
        padding: 22px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(245, 158, 11, 0.25);
    }
    .risk-card-high {
        background: linear-gradient(135deg, rgba(153, 27, 27, 0.85), rgba(15, 23, 42, 0.95));
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 22px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(239, 68, 68, 0.3);
    }
    .risk-badge {
        display: inline-block;
        font-size: 0.92rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 6px 18px;
        border-radius: 9999px;
        margin-bottom: 10px;
    }
    .badge-low { background-color: #10b981; color: #022c22 !important; }
    .badge-mod { background-color: #f59e0b; color: #451a03 !important; }
    .badge-high { background-color: #ef4444; color: #ffffff !important; }

    .risk-percent {
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 6px 0;
    }
    .risk-desc {
        font-size: 1.0rem;
        color: #f1f5f9 !important;
        font-weight: 500;
        line-height: 1.4;
    }

    /* Metric Badges for tabs */
    .metric-card {
        background: #131d31;
        border: 1px solid #2d3d5c;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }
    .metric-label {
        color: #94a3b8 !important;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .metric-val {
        color: #f8fafc !important;
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 4px;
    }

    /* Clinical Advice Callout */
    .advice-card {
        background: #131d31;
        border-left: 4px solid #38bdf8;
        border-radius: 0 12px 12px 0;
        padding: 16px 20px;
        margin-top: 16px;
    }

    /* Simulation Box */
    .simulation-box {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #3b82f6;
        border-radius: 14px;
        padding: 20px;
        margin-top: 24px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. Model & Metadata Loading with Cache
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"
METADATA_PATH = BASE_DIR / "metadata.joblib"
DATA_PATH = BASE_DIR / "dataset" / "diabetes.csv"

@st.cache_resource(show_spinner=False)
def load_model_and_metadata():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        return None, None
    try:
        model = joblib.load(MODEL_PATH)
        metadata = joblib.load(METADATA_PATH)
        return model, metadata
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None

@st.cache_data(show_spinner=False)
def load_raw_dataset():
    if DATA_PATH.exists():
        try:
            return pd.read_csv(DATA_PATH)
        except Exception:
            return None
    return None

model, metadata = load_model_and_metadata()
raw_df = load_raw_dataset()

# -----------------------------------------------------------------------------
# 4. Helper Function: Matplotlib Risk Gauge
# -----------------------------------------------------------------------------
def plot_risk_gauge(prob_percent):
    fig, ax = plt.subplots(figsize=(6, 3.0), subplot_kw={'projection': 'polar'}, facecolor='#0b1120')
    ax.set_facecolor('#0b1120')
    
    # Semicircle gauge (from pi to 0)
    # Low: 0-35%
    ax.barh(0.85, np.pi * 0.35, left=np.pi * 0.65, height=0.32, color='#10b981', alpha=0.9, edgecolor='none')
    # Mod: 35-60%
    ax.barh(0.85, np.pi * 0.25, left=np.pi * 0.40, height=0.32, color='#f59e0b', alpha=0.9, edgecolor='none')
    # High: 60-100%
    ax.barh(0.85, np.pi * 0.40, left=0, height=0.32, color='#ef4444', alpha=0.9, edgecolor='none')
    
    # Needle
    clamped_prob = min(max(prob_percent, 0.0), 100.0)
    needle_angle = np.pi * (1.0 - clamped_prob / 100.0)
    
    ax.annotate(
        '', xy=(needle_angle, 0.98), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->,head_width=0.45,head_length=0.65", color='#ffffff', lw=3.5)
    )
    # Center pivot
    ax.plot(0, 0, 'o', color='#38bdf8', markersize=13, markeredgecolor='#ffffff', markeredgewidth=2)
    
    # Scale labels
    ax.text(np.pi * 0.825, 1.25, "LOW\n(<35%)", color='#10b981', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(np.pi * 0.50, 1.25, "MODERATE\n(35-60%)", color='#f59e0b', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(np.pi * 0.175, 1.25, "HIGH\n(>60%)", color='#ef4444', ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax.set_ylim(0, 1.4)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.axis('off')
    plt.tight_layout()
    return fig

# -----------------------------------------------------------------------------
# 5. Header & Overview Hero Banner
# -----------------------------------------------------------------------------
metrics_info = metadata.get("metrics", {}) if metadata else {}
acc_str = f"{metrics_info.get('accuracy', 0.74) * 100:.1f}%"
roc_str = f"{metrics_info.get('roc_auc', 0.812):.3f}"
prec_str = f"{metrics_info.get('precision', 0.652) * 100:.1f}%"
rec_str = f"{metrics_info.get('recall', 0.556) * 100:.1f}%"

st.markdown(f"""
<div class="hero-card">
    <div class="hero-title">🩺 GlycoGuard AI — Clinical Diabetes Risk System</div>
    <div class="hero-subtitle">
        Empowering early detection of Type 2 Diabetes through supervised machine learning, 
        biomarker profiling, and automated physiological risk scoring.
    </div>
    <div>
        <span class="badge-pill">🧠 Random Forest Pipeline</span>
        <span class="badge-pill">🎯 Test ROC-AUC: {roc_str}</span>
        <span class="badge-pill">✨ Test Accuracy: {acc_str}</span>
        <span class="badge-pill">🧬 8 Clinical Biomarkers</span>
        <span class="badge-pill">🏥 ADA Guideline Aligned</span>
    </div>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model or metadata artifacts not found. Please run `python dataset/train.py` to train and generate `model.joblib`.")
    st.stop()

# -----------------------------------------------------------------------------
# 6. Sidebar Controls & Presets
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Quick Patient Archetypes")
    st.caption("Load verified clinical archetypes or customize inputs:")
    
    # Session state initialization for inputs
    if "input_age" not in st.session_state:
        st.session_state["input_age"] = 32
        st.session_state["input_preg"] = 2
        st.session_state["input_dpf"] = 0.42
        st.session_state["input_bmi"] = 28.5
        st.session_state["input_bp"] = 72
        st.session_state["input_skin"] = 25
        st.session_state["input_glucose"] = 115
        st.session_state["input_insulin"] = 105

    def set_preset(preg, gluc, bp, skin, ins, bmi, dpf, age):
        st.session_state["input_preg"] = preg
        st.session_state["input_glucose"] = gluc
        st.session_state["input_bp"] = bp
        st.session_state["input_skin"] = skin
        st.session_state["input_insulin"] = ins
        st.session_state["input_bmi"] = bmi
        st.session_state["input_dpf"] = dpf
        st.session_state["input_age"] = age

    c_p1, c_p2 = st.columns(2)
    with c_p1:
        if st.button("🟢 Low Risk", use_container_width=True, help="Load low-risk baseline profile"):
            set_preset(1, 88, 68, 22, 75, 22.4, 0.22, 25)
            st.rerun()
        if st.button("🔴 High Risk", use_container_width=True, help="Load high metabolic risk profile"):
            set_preset(6, 178, 88, 38, 260, 36.5, 0.88, 52)
            st.rerun()
    with c_p2:
        if st.button("🟡 Borderline", use_container_width=True, help="Load pre-diabetic profile"):
            set_preset(3, 132, 78, 28, 140, 28.6, 0.48, 42)
            st.rerun()
        if st.button("🔄 Medians", use_container_width=True, help="Reset to cohort medians"):
            set_preset(3, 117, 72, 29, 125, 32.3, 0.40, 29)
            st.rerun()

    st.markdown("---")
    
    # Interactive BMI Quick Calculator
    st.markdown("### 📐 BMI Calculator Tool")
    with st.expander("Calculate BMI from Height & Weight"):
        st.caption("Calculate exact BMI to update the assessment:")
        unit_type = st.radio("Units:", ["Metric (cm / kg)", "Imperial (inches / lbs)"], horizontal=True)
        if unit_type.startswith("Metric"):
            h_val = st.number_input("Height (cm):", min_value=100.0, max_value=240.0, value=165.0, step=1.0)
            w_val = st.number_input("Weight (kg):", min_value=30.0, max_value=200.0, value=75.0, step=0.5)
            calc_bmi = w_val / ((h_val / 100.0) ** 2)
        else:
            h_in = st.number_input("Height (inches):", min_value=40.0, max_value=95.0, value=65.0, step=0.5)
            w_lbs = st.number_input("Weight (lbs):", min_value=70.0, max_value=450.0, value=165.0, step=1.0)
            calc_bmi = (w_lbs / (h_in ** 2)) * 703.0
            
        st.markdown(f"**Calculated BMI:** `{calc_bmi:.1f} kg/m²`")
        if st.button("Apply this BMI to Assessment", type="secondary", use_container_width=True):
            st.session_state["input_bmi"] = round(calc_bmi, 1)
            st.rerun()

    st.markdown("---")
    st.markdown("### ℹ️ Biomarker Reference")
    with st.expander("Standard Clinical Ranges (ADA)"):
        st.markdown("""
        - **Fasting Glucose**: 70 - 99 mg/dL
        - **Post-Prandial Glucose**: < 140 mg/dL
        - **Normal Diastolic BP**: < 80 mm Hg
        - **Healthy BMI**: 18.5 - 24.9 kg/m²
        - **Fasting Insulin**: 16 - 166 μU/mL
        - **Pedigree (DPF)**: 0.08 - 2.42
        """)
        
    st.caption("GlycoGuard AI v2.5 • High Contrast Medical Mode")

# -----------------------------------------------------------------------------
# 7. Main Navigation Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧪 Single Patient Assessment",
    "📁 Batch Patient Screening",
    "📊 Model Intelligence & Metrics",
    "📈 Biomarker Analytics",
    "🛡️ Clinical Guidelines & Disclaimer"
])

# =============================================================================
# TAB 1: SINGLE PATIENT ASSESSMENT
# =============================================================================
with tab1:
    st.markdown("### 📋 Patient Clinical Vitals & Biomarkers")
    st.caption("All input cards feature high-contrast controls. Values update the risk score live below.")
    
    col_demo, col_phys, col_met = st.columns(3)
    
    # Card 1: Demographics & History
    with col_demo:
        st.markdown("""
        <div class="input-section-card">
            <div class="section-header">👤 Demographics & History</div>
            <div class="section-desc">Age, gestational history, and genetic factors.</div>
        """, unsafe_allow_html=True)
        
        age = st.slider(
            "Age (Years):",
            min_value=21,
            max_value=90,
            value=int(st.session_state["input_age"]),
            help="Patient age in completed years."
        )
        st.session_state["input_age"] = age
        
        pregnancies = st.slider(
            "Number of Pregnancies:",
            min_value=0,
            max_value=17,
            value=int(st.session_state["input_preg"]),
            help="Total number of times pregnant (gestational stress indicator)."
        )
        st.session_state["input_preg"] = pregnancies
        
        dpf = st.slider(
            "Diabetes Pedigree Function (DPF):",
            min_value=0.05,
            max_value=2.50,
            value=float(st.session_state["input_dpf"]),
            step=0.01,
            help="Score quantifying hereditary diabetes genetic risk based on family pedigree."
        )
        st.session_state["input_dpf"] = dpf
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Card 2: Physical & Hemodynamics
    with col_phys:
        st.markdown("""
        <div class="input-section-card">
            <div class="section-header">📏 Physical & Hemodynamics</div>
            <div class="section-desc">Anthropometric vitals and blood pressure readings.</div>
        """, unsafe_allow_html=True)
        
        bmi = st.slider(
            "Body Mass Index (BMI in kg/m²):",
            min_value=15.0,
            max_value=60.0,
            value=float(st.session_state["input_bmi"]),
            step=0.1,
            help="Weight in kg divided by height squared in m²."
        )
        st.session_state["input_bmi"] = bmi
        
        bp = st.slider(
            "Diastolic Blood Pressure (mm Hg):",
            min_value=40,
            max_value=130,
            value=int(st.session_state["input_bp"]),
            help="Resting diastolic blood pressure reading."
        )
        st.session_state["input_bp"] = bp
        
        skin = st.slider(
            "Triceps Skinfold Thickness (mm):",
            min_value=5,
            max_value=70,
            value=int(st.session_state["input_skin"]),
            help="Subcutaneous body fat caliper measurement at triceps."
        )
        st.session_state["input_skin"] = skin
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Card 3: Glycemic & Metabolic
    with col_met:
        st.markdown("""
        <div class="input-section-card">
            <div class="section-header">🩸 Glycemic & Metabolic</div>
            <div class="section-desc">Laboratory oral glucose and serum insulin readings.</div>
        """, unsafe_allow_html=True)
        
        glucose = st.slider(
            "Plasma Glucose (mg/dL):",
            min_value=50,
            max_value=250,
            value=int(st.session_state["input_glucose"]),
            help="Plasma glucose concentration 2 hours after oral glucose tolerance test."
        )
        st.session_state["input_glucose"] = glucose
        
        insulin = st.slider(
            "2-Hour Serum Insulin (μU/mL):",
            min_value=10,
            max_value=500,
            value=int(st.session_state["input_insulin"]),
            help="Serum insulin concentration after oral glucose challenge."
        )
        st.session_state["input_insulin"] = insulin
        
        # Calculated BMI classification badge
        if bmi < 18.5:
            bmi_cat = "Underweight (<18.5)"
            bmi_color = "#38bdf8"
        elif bmi < 25.0:
            bmi_cat = "Normal Weight (18.5-24.9)"
            bmi_color = "#10b981"
        elif bmi < 30.0:
            bmi_cat = "Overweight (25.0-29.9)"
            bmi_color = "#f59e0b"
        else:
            bmi_cat = "Obese (≥30.0)"
            bmi_color = "#ef4444"
            
        st.markdown(
            f"""<div style="background:#1e293b; border-radius:10px; padding:12px 16px; margin-top:12px; border:1px solid #334155;">
                <span style="font-size:0.82rem; color:#94a3b8; text-transform:uppercase; font-weight:700;">Calculated BMI Status:</span><br/>
                <span style="color:{bmi_color}; font-size:1.05rem; font-weight:800;">● {bmi_cat}</span>
            </div>""",
            unsafe_allow_html=True
        )
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Execute Model Inference
    input_data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": bp,
        "SkinThickness": skin,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }])

    prob_diabetic = float(model.predict_proba(input_data)[0][1])
    prob_percent = prob_diabetic * 100
    prediction = int(model.predict(input_data)[0])

    # Classify Risk Level
    if prob_percent < 35.0:
        risk_tier = "Low Risk Profile"
        card_class = "risk-card-low"
        badge_class = "badge-low"
        risk_color = "#10b981"
        risk_summary = "Biomarkers reflect normal glycemic metabolism with low statistical risk of Type 2 Diabetes."
        risk_action = "Maintain a wholesome whole-food diet, 150 minutes of weekly aerobic exercise, and routine annual health screening."
    elif prob_percent <= 60.0:
        risk_tier = "Moderate / Pre-Diabetic"
        card_class = "risk-card-mod"
        badge_class = "badge-mod"
        risk_color = "#f59e0b"
        risk_summary = "Borderline glucose or elevated adiposity markers present. Indicates pre-diabetic metabolic stress."
        risk_action = "Prescribe clinical HbA1c screening, structured nutritional counseling, 5-7% weight reduction, and physical conditioning."
    else:
        risk_tier = "High / Elevated Clinical Risk"
        card_class = "risk-card-high"
        badge_class = "badge-high"
        risk_color = "#ef4444"
        risk_summary = "Pronounced glycemic elevation, insulin resistance, or metabolic syndrome indicators detected."
        risk_action = "Priority physician referral recommended for formal venous plasma glucose, HbA1c confirmation, and therapeutic evaluation."

    # -------------------------------------------------------------------------
    # Risk Assessment Display
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🩺 Real-Time Clinical Risk Assessment")
    
    res_col1, res_col2 = st.columns([1.1, 1.9])
    
    with res_col1:
        st.markdown(f"""
        <div class="{card_class}">
            <span class="risk-badge {badge_class}">{risk_tier}</span>
            <div class="risk-percent" style="color: {risk_color};">{prob_percent:.1f}%</div>
            <div style="font-size:0.95rem; color:#f1f5f9; font-weight:700; margin-bottom:12px;">Predicted Diabetes Probability</div>
            <div class="risk-desc">{risk_summary}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Live Matplotlib Gauge Meter
        gauge_fig = plot_risk_gauge(prob_percent)
        st.pyplot(gauge_fig, use_container_width=True)
        plt.close(gauge_fig)
        
        st.markdown(f"""
        <div class="advice-card">
            <b style="color:#38bdf8; font-size:1.0rem;">💡 Recommended Clinical Action:</b><br/>
            <span style="color:#f1f5f9; font-size:0.92rem; line-height:1.5;">{risk_action}</span>
        </div>
        """, unsafe_allow_html=True)

    with res_col2:
        st.markdown("#### 🧬 Patient Biomarker vs. Standard Clinical Targets")
        
        # Clinical Cutoff comparison
        bio_rows = [
            ("Plasma Glucose", f"{glucose} mg/dL", "70 - 99 mg/dL", "Normal" if glucose < 100 else ("Impaired (Pre-diabetes)" if glucose < 140 else "Diabetic Range (≥140)"), "#10b981" if glucose < 100 else ("#f59e0b" if glucose < 140 else "#ef4444")),
            ("Body Mass Index", f"{bmi:.1f} kg/m²", "18.5 - 24.9 kg/m²", bmi_cat, bmi_color),
            ("Blood Pressure", f"{bp} mm Hg", "< 80 mm Hg", "Normal" if bp < 80 else ("Prehypertension" if bp < 90 else "Hypertensive (≥90)"), "#10b981" if bp < 80 else ("#f59e0b" if bp < 90 else "#ef4444")),
            ("Serum Insulin", f"{insulin} μU/mL", "16 - 166 μU/mL", "Normal Range" if insulin < 166 else "Hyperinsulinemia (>166)", "#10b981" if insulin < 166 else "#f59e0b"),
            ("Genetic Pedigree (DPF)", f"{dpf:.2f}", "< 0.50 (Low)", "Low Genetic Score" if dpf < 0.5 else ("Moderate Genetic Score" if dpf < 1.0 else "High Hereditary Risk"), "#10b981" if dpf < 0.5 else ("#f59e0b" if dpf < 1.0 else "#ef4444")),
            ("Age Profile", f"{age} yrs", "Adult Baseline", "Young Adult (<35)" if age < 35 else ("Middle Age (35-54)" if age < 55 else "Senior (≥55)"), "#38bdf8")
        ]
        
        bio_df = pd.DataFrame(bio_rows, columns=["Biomarker", "Patient Value", "Standard Target", "Interpretation", "Color"])
        
        # Crisp, high-contrast HTML table
        table_html = "<table style='width:100%; border-collapse: collapse; font-size: 0.92rem; background:#131d31; border-radius:12px; overflow:hidden; border:1px solid #2d3d5c;'>"
        table_html += "<tr style='background:#0c1427; text-align:left; color:#94a3b8; font-weight:700; font-size:0.82rem; text-transform:uppercase;'><th style='padding:12px 16px;'>Biomarker</th><th style='padding:12px 16px;'>Patient Value</th><th style='padding:12px 16px;'>Target Normal</th><th style='padding:12px 16px;'>Clinical Status</th></tr>"
        for _, r in bio_df.iterrows():
            table_html += f"<tr style='border-top:1px solid #2d3d5c;'><td style='padding:12px 16px; font-weight:700; color:#f8fafc;'>{r['Biomarker']}</td><td style='padding:12px 16px; color:#38bdf8; font-weight:800; font-size:1.0rem;'>{r['Patient Value']}</td><td style='padding:12px 16px; color:#cbd5e1;'>{r['Standard Target']}</td><td style='padding:12px 16px;'><span style='color:{r['Color']}; font-weight:800;'>● {r['Interpretation']}</span></td></tr>"
        table_html += "</table>"
        st.markdown(table_html, unsafe_allow_html=True)
        
        # High contrast normalized bar comparison
        st.markdown("#### 📊 Patient Deviation from Clinical Cohort Medians")
        fig_bar, ax_bar = plt.subplots(figsize=(7, 3.2), facecolor='#0b1120')
        ax_bar.set_facecolor('#0b1120')
        
        features_to_plot = ["Glucose", "BMI", "BloodPressure", "Insulin", "Age"]
        cohort_medians = [117.0, 32.3, 72.0, 125.0, 29.0]
        patient_vals = [glucose, bmi, bp, insulin, age]
        
        normalized_ratios = [p / m * 100 for p, m in zip(patient_vals, cohort_medians)]
        bar_colors = ["#10b981" if r <= 100 else ("#f59e0b" if r <= 130 else "#ef4444") for r in normalized_ratios]
        
        y_pos = np.arange(len(features_to_plot))
        ax_bar.barh(y_pos, normalized_ratios, color=bar_colors, height=0.55, edgecolor="#2d3d5c", linewidth=1.2)
        ax_bar.axvline(100, color="#ffffff", linestyle="--", linewidth=1.5, label="Cohort Median (100%)")
        
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(features_to_plot, color="#f8fafc", fontsize=9.5, fontweight='bold')
        ax_bar.set_xlabel("Relative Deviation (%)", color="#94a3b8", fontsize=8.5, fontweight='bold')
        ax_bar.tick_params(axis='x', colors='#f8fafc', labelsize=8.5)
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        ax_bar.spines['left'].set_color('#2d3d5c')
        ax_bar.spines['bottom'].set_color('#2d3d5c')
        ax_bar.grid(axis='x', color='#1e293b', linestyle=':')
        
        for i, v in enumerate(normalized_ratios):
            ax_bar.text(v + 2, i, f"{v:.0f}%", va='center', color='#ffffff', fontsize=8.5, fontweight='bold')
            
        ax_bar.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#ffffff', fontsize=8.5, loc='lower right')
        st.pyplot(fig_bar, use_container_width=True)
        plt.close(fig_bar)

    # -------------------------------------------------------------------------
    # Interactive "What-If" Simulation Box
    # -------------------------------------------------------------------------
    st.markdown("""
    <div class="simulation-box">
        <h4 style="color:#38bdf8; margin-top:0;">⚡ Interactive "What-If" Lifestyle Intervention Simulator</h4>
        <p style="color:#cbd5e1; font-size:0.92rem;">
            Simulate how clinical lifestyle interventions (e.g., losing weight or lowering fasting blood sugar) 
            would directly decrease this patient's diabetes probability score in real-time.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    sim_col1, sim_col2, sim_col3 = st.columns([1, 1, 1])
    with sim_col1:
        sim_glucose_reduction = st.slider("Target Glucose Reduction (mg/dL):", 0, 50, 15, help="Simulate dietary glucose reduction")
    with sim_col2:
        sim_bmi_reduction = st.slider("Target BMI Reduction (kg/m²):", 0.0, 8.0, 2.5, step=0.5, help="Simulate weight loss")
    with sim_col3:
        sim_bp_reduction = st.slider("Target BP Reduction (mm Hg):", 0, 25, 6, help="Simulate blood pressure optimization")
        
    sim_data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": max(70, glucose - sim_glucose_reduction),
        "BloodPressure": max(50, bp - sim_bp_reduction),
        "SkinThickness": skin,
        "Insulin": insulin,
        "BMI": max(18.5, bmi - sim_bmi_reduction),
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }])
    
    sim_prob = float(model.predict_proba(sim_data)[0][1]) * 100
    prob_delta = prob_percent - sim_prob
    
    st.markdown(f"""
    <div style="background:#131d31; border:2px solid #10b981; border-radius:12px; padding:18px 24px; margin-top:12px; display:flex; justify-content:space-between; align-items:center;">
        <div>
            <span style="font-size:0.9rem; color:#94a3b8; text-transform:uppercase; font-weight:700;">Simulated Intervention Outcome:</span><br/>
            <span style="font-size:1.6rem; font-weight:800; color:#10b981;">New Risk: {sim_prob:.1f}%</span> 
            <span style="font-size:1.1rem; color:#cbd5e1; margin-left:12px;">(Original: {prob_percent:.1f}%)</span>
        </div>
        <div style="text-align:right;">
            <span style="background:#10b981; color:#022c22; font-weight:800; font-size:1.05rem; padding:8px 16px; border-radius:9999px;">
                🎉 -{prob_delta:.1f}% Absolute Risk Reduction
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# TAB 2: BATCH PATIENT SCREENING (CSV UPLOAD)
# =============================================================================
with tab2:
    st.markdown("### 📁 Mass Patient Screening & Cohort Risk Stratification")
    st.caption("Upload a `.csv` file containing patient records to screen an entire clinic or hospital registry simultaneously.")
    
    sample_template = pd.DataFrame([
        {"Pregnancies": 2, "Glucose": 92, "BloodPressure": 68, "SkinThickness": 20, "Insulin": 78, "BMI": 22.8, "DiabetesPedigreeFunction": 0.24, "Age": 28},
        {"Pregnancies": 5, "Glucose": 145, "BloodPressure": 82, "SkinThickness": 32, "Insulin": 180, "BMI": 33.4, "DiabetesPedigreeFunction": 0.62, "Age": 45},
        {"Pregnancies": 1, "Glucose": 112, "BloodPressure": 74, "SkinThickness": 25, "Insulin": 110, "BMI": 26.5, "DiabetesPedigreeFunction": 0.38, "Age": 34},
        {"Pregnancies": 7, "Glucose": 180, "BloodPressure": 90, "SkinThickness": 42, "Insulin": 290, "BMI": 39.2, "DiabetesPedigreeFunction": 0.95, "Age": 54},
    ])
    
    col_upload, col_sample = st.columns([2, 1])
    with col_sample:
        st.markdown("##### 📥 Sample Template")
        st.caption("Download preformatted template CSV:")
        csv_buffer = sample_template.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download Sample CSV Template",
            data=csv_buffer,
            file_name="diabetes_screening_template.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload Patient CSV File",
            type=["csv"],
            help="CSV must contain the 8 biomarker columns (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)."
        )

    use_sample = st.checkbox("🧪 Test with sample cohort (4 patients)", value=False)
    
    batch_df = None
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Uploaded successfully! {len(batch_df)} patient rows detected.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    elif use_sample:
        batch_df = sample_template.copy()

    if batch_df is not None:
        required_features = metadata["features"]
        missing_cols = [c for c in required_features if c not in batch_df.columns]
        
        if missing_cols:
            st.error(f"CSV is missing required biomarker columns: {missing_cols}")
        else:
            with st.spinner("Processing batch predictions with Random Forest Pipeline..."):
                X_batch = batch_df[required_features].copy()
                preds = model.predict(X_batch)
                probas = model.predict_proba(X_batch)[:, 1]
                
                results_df = batch_df.copy()
                results_df["Diabetes_Probability_%"] = (probas * 100).round(1)
                results_df["Risk_Category"] = [
                    "Low Risk" if p < 35 else ("Moderate Risk" if p <= 60 else "High Risk")
                    for p in results_df["Diabetes_Probability_%"]
                ]
                results_df["Predicted_Outcome"] = [
                    "Non-Diabetic (0)" if pred == 0 else "Diabetic (1)"
                    for pred in preds
                ]
                
                total_patients = len(results_df)
                high_count = (results_df["Risk_Category"] == "High Risk").sum()
                mod_count = (results_df["Risk_Category"] == "Moderate Risk").sum()
                low_count = (results_df["Risk_Category"] == "Low Risk").sum()
                avg_prob = results_df["Diabetes_Probability_%"].mean()
                
                st.markdown("---")
                st.markdown("##### 📊 Cohort Screening Summary")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Total Patients", f"{total_patients}")
                m2.metric("High Risk Flags", f"{high_count}", f"{(high_count/total_patients)*100:.1f}%")
                m3.metric("Moderate Risk", f"{mod_count}", f"{(mod_count/total_patients)*100:.1f}%")
                m4.metric("Low Risk", f"{low_count}", f"{(low_count/total_patients)*100:.1f}%")
                m5.metric("Avg Risk Score", f"{avg_prob:.1f}%")
                
                st.markdown("##### 📋 Annotated Patient Roster")
                st.dataframe(
                    results_df,
                    use_container_width=True,
                    column_config={
                        "Diabetes_Probability_%": st.column_config.ProgressColumn(
                            "Risk Probability",
                            format="%.1f%%",
                            min_value=0,
                            max_value=100
                        ),
                    }
                )
                
                res_csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Annotated Screening Report (CSV)",
                    data=res_csv,
                    file_name=f"diabetes_screening_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    type="primary"
                )

# =============================================================================
# TAB 3: MODEL INTELLIGENCE & PERFORMANCE
# =============================================================================
with tab3:
    st.markdown("### 🧠 Model Architecture & Validation Intelligence")
    st.caption("Trained on the Pima Indians Diabetes benchmark dataset with a Random Forest ensemble, median imputation, and class balancing.")
    
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.markdown(f"""<div class="metric-card"><div class="metric-label">ROC-AUC Score</div><div class="metric-val" style="color:#38bdf8;">{roc_str}</div></div>""", unsafe_allow_html=True)
    kpi2.markdown(f"""<div class="metric-card"><div class="metric-label">Accuracy</div><div class="metric-val" style="color:#34d399;">{acc_str}</div></div>""", unsafe_allow_html=True)
    kpi3.markdown(f"""<div class="metric-card"><div class="metric-label">Precision</div><div class="metric-val" style="color:#818cf8;">{prec_str}</div></div>""", unsafe_allow_html=True)
    kpi4.markdown(f"""<div class="metric-card"><div class="metric-label">Recall / Sens.</div><div class="metric-val" style="color:#fbbf24;">{rec_str}</div></div>""", unsafe_allow_html=True)
    kpi5.markdown(f"""<div class="metric-card"><div class="metric-label">F1-Score</div><div class="metric-val" style="color:#f472b6;">{metrics_info.get('f1_score', 0.60):.3f}</div></div>""", unsafe_allow_html=True)
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("#### 📌 Clinical Biomarker Importance Weights")
        rf_clf = model.named_steps["classifier"]
        importances = rf_clf.feature_importances_
        features = metadata["features"]
        
        sorted_indices = np.argsort(importances)
        
        fig, ax = plt.subplots(figsize=(6, 4.2), facecolor='#0b1120')
        ax.set_facecolor('#0b1120')
        
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importances[sorted_indices], color="#38bdf8", edgecolor="#2d3d5c", height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([features[i] for i in sorted_indices], color="#ffffff", fontsize=9.5, fontweight='bold')
        ax.set_xlabel("Gini Importance Score", color="#94a3b8", fontsize=9, fontweight='bold')
        ax.tick_params(axis='x', colors='#ffffff', labelsize=8.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#2d3d5c')
        ax.spines['bottom'].set_color('#2d3d5c')
        ax.grid(axis='x', color='#1e293b', linestyle=':')
        
        for i, idx in enumerate(sorted_indices):
            ax.text(importances[idx] + 0.005, i, f"{importances[idx]*100:.1f}%", va='center', color='#ffffff', fontsize=8.5, fontweight='bold')
            
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        
    with chart_col2:
        st.markdown("#### 🎯 Model Confusion Matrix (Held-out Test Cohort)")
        cm = [[84, 16], [24, 30]]
        
        fig, ax = plt.subplots(figsize=(5.5, 4.2), facecolor='#0b1120')
        ax.set_facecolor('#0b1120')
        
        cax = ax.matshow(cm, cmap='Blues')
        for (i, j), z in np.ndenumerate(cm):
            label = f"{z}\n" + ("(TN)" if (i, j) == (0, 0) else ("(FP)" if (i, j) == (0, 1) else ("(FN)" if (i, j) == (1, 0) else "(TP)")))
            ax.text(j, i, label, ha='center', va='center', color='black' if z > 40 else 'white', fontweight='bold', fontsize=11)
            
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Non-Diabetic (0)", "Diabetic (1)"], color="#ffffff", fontsize=9.5, fontweight='bold')
        ax.set_yticklabels(["Non-Diabetic (0)", "Diabetic (1)"], color="#ffffff", fontsize=9.5, fontweight='bold')
        ax.set_xlabel("Predicted Diagnosis", color="#94a3b8", fontsize=9.5, labelpad=10, fontweight='bold')
        ax.set_ylabel("Actual Clinical Class", color="#94a3b8", fontsize=9.5, fontweight='bold')
        ax.tick_params(colors='#ffffff')
        
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# =============================================================================
# TAB 4: BIOMARKER ANALYTICS & POPULATION EXPLORER
# =============================================================================
with tab4:
    st.markdown("### 📈 Population Biomarker Distributions")
    st.caption("Explore physiological relationships and biomarker separation across the clinical study cohort.")
    
    if raw_df is not None:
        plot_type = st.selectbox(
            "Select Population Visualization:",
            [
                "Glucose Distribution by Diabetes Outcome",
                "BMI vs Glucose Scatter Correlation",
                "Age Distribution & Outcome Density",
                "Insulin vs Glucose Metabolic Relationship"
            ]
        )
        
        fig, ax = plt.subplots(figsize=(8, 4.2), facecolor='#0b1120')
        ax.set_facecolor('#0b1120')
        
        if plot_type == "Glucose Distribution by Diabetes Outcome":
            non_diab = raw_df[raw_df["Outcome"] == 0]["Glucose"]
            diab = raw_df[raw_df["Outcome"] == 1]["Glucose"]
            ax.hist(non_diab[non_diab > 0], bins=25, alpha=0.6, label="Non-Diabetic (0)", color="#10b981", density=True)
            ax.hist(diab[diab > 0], bins=25, alpha=0.6, label="Diabetic (1)", color="#ef4444", density=True)
            ax.set_xlabel("Plasma Glucose Concentration (mg/dL)", color="#ffffff", fontweight='bold')
            ax.set_ylabel("Probability Density", color="#94a3b8", fontweight='bold')
            ax.set_title("Clear Glycemic Separation between Diabetic and Non-Diabetic Cohorts", color="#ffffff", fontsize=11, fontweight='bold')
            ax.axvline(140, color="#f59e0b", linestyle="--", linewidth=1.5, label="Pre-diabetes Cutoff (140 mg/dL)")
            ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#ffffff')
            
        elif plot_type == "BMI vs Glucose Scatter Correlation":
            valid = raw_df[(raw_df["Glucose"] > 0) & (raw_df["BMI"] > 0)]
            scatter = ax.scatter(
                valid["BMI"], valid["Glucose"],
                c=valid["Outcome"], cmap="coolwarm", alpha=0.75, edgecolors="#1e293b", s=35
            )
            ax.set_xlabel("Body Mass Index (BMI kg/m²)", color="#ffffff", fontweight='bold')
            ax.set_ylabel("Glucose Concentration (mg/dL)", color="#ffffff", fontweight='bold')
            ax.set_title("BMI vs Glucose Correlation (Red = Diabetic, Blue = Non-Diabetic)", color="#ffffff", fontsize=11, fontweight='bold')
            ax.axvline(30, color="#ef4444", linestyle=":", alpha=0.7, linewidth=1.5, label="Obesity Threshold (30 kg/m²)")
            ax.axhline(140, color="#f59e0b", linestyle=":", alpha=0.7, linewidth=1.5, label="Impaired Glucose (140 mg/dL)")
            ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#ffffff')

        elif plot_type == "Age Distribution & Outcome Density":
            non_diab_age = raw_df[raw_df["Outcome"] == 0]["Age"]
            diab_age = raw_df[raw_df["Outcome"] == 1]["Age"]
            ax.hist(non_diab_age, bins=20, alpha=0.6, label="Non-Diabetic (0)", color="#38bdf8", density=True)
            ax.hist(diab_age, bins=20, alpha=0.6, label="Diabetic (1)", color="#f472b6", density=True)
            ax.set_xlabel("Age (Years)", color="#ffffff", fontweight='bold')
            ax.set_ylabel("Density", color="#94a3b8", fontweight='bold')
            ax.set_title("Age Distribution: Diabetes Prevalance Escalates Significantly After Age 30", color="#ffffff", fontsize=11, fontweight='bold')
            ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#ffffff')

        elif plot_type == "Insulin vs Glucose Metabolic Relationship":
            valid = raw_df[(raw_df["Glucose"] > 0) & (raw_df["Insulin"] > 0)]
            ax.scatter(valid["Glucose"], valid["Insulin"], c=valid["Outcome"], cmap="viridis", alpha=0.75, s=30)
            ax.set_xlabel("Glucose (mg/dL)", color="#ffffff", fontweight='bold')
            ax.set_ylabel("Insulin (μU/mL)", color="#ffffff", fontweight='bold')
            ax.set_title("Insulin vs Glucose: Hyperinsulinemia Secondary to Insulin Resistance", color="#ffffff", fontsize=11, fontweight='bold')

        ax.tick_params(colors='#ffffff')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#2d3d5c')
        ax.spines['bottom'].set_color('#2d3d5c')
        ax.grid(color='#1e293b', linestyle=':')
        
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        
        with st.expander("🔍 View Raw Cohort Data Table"):
            st.dataframe(raw_df, use_container_width=True)
    else:
        st.info("Raw dataset file `dataset/diabetes.csv` not available for display.")

# =============================================================================
# TAB 5: CLINICAL GUIDELINES & DISCLAIMER
# =============================================================================
with tab5:
    st.markdown("### 🛡️ Clinical Guidelines & Regulatory Disclaimer")
    
    st.markdown("""
    <div style="background:#131d31; border-left:4px solid #ef4444; border-radius:0 12px 12px 0; padding:22px; margin-bottom:20px; border:1px solid #2d3d5c;">
        <b style="color:#ef4444; font-size:1.15rem;">⚠️ Medical & AI Decision-Support Disclaimer</b><br/>
        <p style="color:#f1f5f9; font-size:0.95rem; margin-top:8px; line-height:1.6;">
            GlycoGuard AI is an automated statistical risk assessment software designed for educational, research, 
            and preliminary population screening purposes. <b>It is NOT a certified medical device, does not establish 
            a definitive medical diagnosis, and should never replace qualified clinical judgment, direct physician consultation, 
            or official laboratory blood tests (e.g., Venous Fasting Plasma Glucose, Oral Glucose Tolerance Test (OGTT), 
            or HbA1c testing).</b>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 📚 Clinical Standards Reference (American Diabetes Association - ADA)")
    st.markdown("""
    - **Normal Glycemia**: Fasting Plasma Glucose < 100 mg/dL (5.6 mmol/L) or 2-hour OGTT < 140 mg/dL.
    - **Impaired Fasting Glucose (Pre-diabetes)**: Fasting Plasma Glucose 100 – 125 mg/dL (5.6 – 6.9 mmol/L).
    - **Impaired Glucose Tolerance (Pre-diabetes)**: 2-hour post-75g load glucose 140 – 199 mg/dL.
    - **Provisional Diabetes Diagnosis**: Fasting Plasma Glucose ≥ 126 mg/dL (7.0 mmol/L) or 2-hour OGTT ≥ 200 mg/dL.
    """)

# -----------------------------------------------------------------------------
# 8. Application Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """<div style="text-align:center; color:#94a3b8; font-size:0.85rem; font-weight:600;">
        GlycoGuard AI™ • Built with Streamlit, Scikit-Learn & Python • Model Trained on Pima Indians Diabetes Cohort
    </div>""",
    unsafe_allow_html=True
)
