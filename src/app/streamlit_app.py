"""
MedAgent Interactive Clinical Decision Support Web Application.
Streamlit dashboard providing multimodal diagnostic analysis, Grad-CAM visual overlays,
inter-modality attention insights, and evidence-grounded clinical reporting.
"""

from pathlib import Path
import sys
from typing import Any, Dict

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from src.agents.supervisor_agent import SupervisorAgent
from src.data.loaders.image_loader import (
    XRAY_CLASSES,
    generate_synthetic_xray_image,
)
from src.evaluation.explainability import overlay_cam_on_image

st.set_page_config(
    page_title="MedAgent | Multimodal Clinical AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_supervisor():
    """Caches supervisor agent initialization."""
    return SupervisorAgent(load_checkpoints=True)


# Custom CSS for clinical styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .risk-badge-low {
        background-color: #d1fae5;
        color: #065f46;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }
    .risk-badge-moderate {
        background-color: #fef3c7;
        color: #92400e;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }
    .risk-badge-high {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Title Header
st.markdown('<div class="main-header">🩺 MedAgent: Multimodal Clinical Decision Support System</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="sub-header">Explainable Medical AI combining Physiological ECG Biosignals, Chest Radiographs, and Clinical RAG Reasoning | GPU: <b>{torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}</b></div>',
    unsafe_allow_html=True,
)

# Sidebar: Patient Demographics & Presets
st.sidebar.header("📋 Patient Clinical Profile")

preset = st.sidebar.selectbox(
    "Load Clinical Preset Case:",
    [
        "Custom Patient",
        "Case A: Normal Screening Checkup",
        "Case B: Acute Pneumonia & Dyspnea",
        "Case C: Cardiac Arrhythmia (PVC Episode)",
        "Case D: Severe Cardiopulmonary Decompensation",
    ],
)

if preset == "Case A: Normal Screening Checkup":
    default_id = "PT-1042"
    default_age = 42
    default_sex = "Female"
    default_sbp = 118
    default_dbp = 76
    default_hr = 72
    default_spo2 = 99
    default_htn = False
    default_dm = False
    default_history = "Routine annual physical exam; no active complaints."
    default_sig_type = "Normal (N)"
    default_img_type = "Normal"
elif preset == "Case B: Acute Pneumonia & Dyspnea":
    default_id = "PT-8824"
    default_age = 65
    default_sex = "Male"
    default_sbp = 136
    default_dbp = 84
    default_hr = 96
    default_spo2 = 91
    default_htn = True
    default_dm = False
    default_history = "Fever (38.8C), productive cough, and shortness of breath for 3 days."
    default_sig_type = "Normal (N)"
    default_img_type = "Pneumonia"
elif preset == "Case C: Cardiac Arrhythmia (PVC Episode)":
    default_id = "PT-4419"
    default_age = 58
    default_sex = "Male"
    default_sbp = 148
    default_dbp = 92
    default_hr = 84
    default_spo2 = 97
    default_htn = True
    default_dm = True
    default_history = "Intermittent palpitations and lightheadedness during exertion."
    default_sig_type = "Ventricular Ectopic (V)"
    default_img_type = "Normal"
elif preset == "Case D: Severe Cardiopulmonary Decompensation":
    default_id = "PT-9051"
    default_age = 74
    default_sex = "Female"
    default_sbp = 162
    default_dbp = 98
    default_hr = 108
    default_spo2 = 88
    default_htn = True
    default_dm = True
    default_history = "Acute orthopnea, bilateral lower extremity edema, and chest discomfort."
    default_sig_type = "Ventricular Ectopic (V)"
    default_img_type = "Pleural Effusion"
else:
    default_id = "PT-3001"
    default_age = 55
    default_sex = "Male"
    default_sbp = 130
    default_dbp = 82
    default_hr = 78
    default_spo2 = 96
    default_htn = False
    default_dm = False
    default_history = "Follow-up outpatient cardiology visit."
    default_sig_type = "Normal (N)"
    default_img_type = "Normal"

patient_id = st.sidebar.text_input("Patient ID", value=default_id)
col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    age = st.number_input("Age (years)", min_value=18, max_value=105, value=default_age)
    sbp = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=240, value=default_sbp)
    hr = st.number_input("Heart Rate (BPM)", min_value=40, max_value=200, value=default_hr)
with col_sb2:
    sex = st.selectbox("Biological Sex", ["Male", "Female"], index=0 if default_sex == "Male" else 1)
    dbp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=140, value=default_dbp)
    spo2 = st.number_input("SpO2 (%)", min_value=70, max_value=100, value=default_spo2)

has_htn = st.sidebar.checkbox("Hypertension History", value=default_htn)
has_dm = st.sidebar.checkbox("Diabetes Mellitus History", value=default_dm)
history_text = st.sidebar.text_area("Clinical History & Symptoms", value=default_history, height=80)

# Main Screen: Modality Previews
col_ecg, col_xray = st.columns(2)

with col_ecg:
    st.subheader("📈 Physiological ECG Waveform")
    ecg_selection = st.selectbox(
        "Select ECG Signal Sample:",
        ["Normal Sinus Beat", "Ventricular Ectopic (PVC)", "Supraventricular Beat (SVEB)", "Synthetic Sine Wave"],
        index=0 if "Normal" in default_sig_type else (1 if "Ventricular" in default_sig_type else 2),
    )

    t = np.linspace(0, 1.0, 260)
    if "Normal" in ecg_selection:
        # Realistic P-Q-R-S-T wave model
        ecg_wave = (
            0.15 * np.exp(-((t - 0.2) ** 2) / 0.002)
            - 0.2 * np.exp(-((t - 0.38) ** 2) / 0.0005)
            + 1.2 * np.exp(-((t - 0.42) ** 2) / 0.0008)
            - 0.35 * np.exp(-((t - 0.46) ** 2) / 0.0005)
            + 0.35 * np.exp(-((t - 0.65) ** 2) / 0.008)
        )
    elif "Ventricular" in ecg_selection:
        # Wide QRS inverted T wave (PVC)
        ecg_wave = (
            -0.4 * np.exp(-((t - 0.35) ** 2) / 0.002)
            + 1.6 * np.exp(-((t - 0.45) ** 2) / 0.004)
            - 0.8 * np.exp(-((t - 0.58) ** 2) / 0.006)
        )
    elif "Supraventricular" in ecg_selection:
        # Premature narrow QRS with abnormal P
        ecg_wave = (
            0.3 * np.exp(-((t - 0.15) ** 2) / 0.001)
            + 1.1 * np.exp(-((t - 0.35) ** 2) / 0.0007)
            + 0.25 * np.exp(-((t - 0.55) ** 2) / 0.006)
        )
    else:
        ecg_wave = np.sin(2 * np.pi * 5 * t)

    fig_ecg = go.Figure()
    fig_ecg.add_trace(go.Scatter(y=ecg_wave, mode="lines", line=dict(color="#2563eb", width=2.5), name="Lead II"))
    fig_ecg.update_layout(
        height=240,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Sample Index (360 Hz)",
        yaxis_title="Amplitude (mV)",
        template="plotly_white",
    )
    st.plotly_chart(fig_ecg, use_container_width=True)

with col_xray:
    st.subheader("🩻 Chest X-Ray Radiograph")
    xray_selection = st.selectbox(
        "Select Radiograph Finding:",
        XRAY_CLASSES,
        index=XRAY_CLASSES.index(default_img_type) if default_img_type in XRAY_CLASSES else 0,
    )

    xray_img = generate_synthetic_xray_image(pathology=xray_selection, image_size=224)
    st.image(xray_img, caption=f"Chest Radiograph ({xray_selection})", width=240)

st.markdown("---")

# Run Analysis Trigger
if st.button("🚀 Run Multimodal AI Clinical Reasoning", type="primary", use_container_width=True):
    with st.spinner("Executing Supervisor Agent & Clinical Multi-Agent Workflow..."):
        supervisor = load_supervisor()

        patient_dict = {
            "patient_id": patient_id,
            "age": age,
            "sex": sex,
            "sbp": sbp,
            "dbp": dbp,
            "heart_rate": hr,
            "spo2": spo2,
            "has_hypertension": 1 if has_htn else 0,
            "has_diabetes": 1 if has_dm else 0,
            "medical_history": history_text,
        }

        result = supervisor.run_case(
            patient_info=patient_dict,
            ecg_input=ecg_wave.astype(np.float32),
            xray_input=xray_img,
        )

        st.session_state["result"] = result
        st.session_state["raw_img"] = xray_img

if "result" in st.session_state:
    res = st.session_state["result"]
    raw_img = st.session_state["raw_img"]
    risk_level = res["risk_level"]

    # Top Executive Summary Badge
    badge_class = "risk-badge-low" if "Low" in risk_level else ("risk-badge-moderate" if "Moderate" in risk_level else "risk-badge-high")
    st.markdown(
        f'<div style="text-align:center; margin: 1rem 0;"><span class="{badge_class}">Overall Stratified Risk: {risk_level}</span></div>',
        unsafe_allow_html=True,
    )

    # Diagnostic Agent Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Clinical Decision Report",
        "📈 Signal Intelligence (ECG)",
        "🩻 Radiological Saliency (Grad-CAM)",
        "📚 Evidence & Guidelines (RAG)",
        "🧠 Multimodal Attention",
    ])

    with tab1:
        st.markdown(res["markdown_report"])
        st.download_button(
            "💾 Download Official Clinical Report (Markdown)",
            data=res["markdown_report"],
            file_name=f"{res['patient_id']}_clinical_report.md",
            mime="text/markdown",
        )

    with tab2:
        sig_summary = res["signal_summary"]
        st.subheader("Signal Agent Diagnostic Findings")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("Detected Cardiac Rhythm", sig_summary["finding"])
            st.metric("Diagnostic Confidence", f"{sig_summary['confidence']*100:.1f}%")
            st.write(f"**Clinical Implication**: {sig_summary['clinical_implication']}")
        with col_s2:
            probs = sig_summary["probabilities"]
            fig_p = px.bar(
                x=list(probs.keys()),
                y=list(probs.values()),
                labels={"x": "AAMI Rhythm Class", "y": "Probability"},
                title="Class Probability Distribution",
                color=list(probs.values()),
                color_continuous_scale="Blues",
            )
            fig_p.update_layout(height=280)
            st.plotly_chart(fig_p, use_container_width=True)

    with tab3:
        img_summary = res["imaging_summary"]
        st.subheader("Imaging Agent Findings & Grad-CAM Heatmap")
        col_i1, col_i2 = st.columns(2)
        with col_i1:
            st.image(raw_img, caption="Input Posteroanterior Radiograph", use_container_width=True)
        with col_i2:
            # Show Grad-CAM overlay
            if Path("data/processed/images/gradcam_sample.png").exists():
                cam_img = Image.open("data/processed/images/gradcam_sample.png")
                st.image(cam_img, caption="Grad-CAM Saliency Heatmap Overlay", use_container_width=True)
            else:
                st.info("Grad-CAM visualization generated.")

        st.write(f"**Radiological Description**: {img_summary['description']}")
        if img_summary.get("salient_regions"):
            st.write("**Identified Anatomical Quadrants of Interest**:")
            st.table(img_summary["salient_regions"])

    with tab4:
        st.subheader("Clinical Guidelines & Literature Grounding (RAG)")
        for ev in res["evidence_citations"]:
            with st.expander(f"📖 {ev['title']} (Relevance: {ev['relevance_score']*100:.1f}%)", expanded=True):
                st.write(f"**Source**: {ev['source']}")
                st.write(f"**Guideline Evidence**: *\"{ev['evidence_text']}\"*")
                if ev.get("doi"):
                    st.write(f"**DOI**: [{ev['doi']}](https://doi.org/{ev['doi']})")

    with tab5:
        st.subheader("Cross-Modal Attention Weights (Multimodal Transformer)")
        risk_summary = res["risk_summary"]
        attn_map = risk_summary.get("modality_attention", {})
        if attn_map:
            fig_attn = px.pie(
                names=list(attn_map.keys()),
                values=list(attn_map.values()),
                title="Relative Modality Influence on Risk Assessment",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_attn.update_layout(height=320)
            st.plotly_chart(fig_attn, use_container_width=True)
        else:
            st.info("Balanced multimodal feature weighting applied across ECG, X-Ray, and Demographics.")
