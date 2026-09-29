"""
MedAgent Clinical Decision Support Web Console.

A multimodal clinical cockpit empowering physicians and healthcare practitioners with:
1. Interactive 1D Physiological Waveform Intelligence (ECG/PPG) & Arrhythmia Detection
2. Explainable Medical Vision (Chest X-Ray) with Grad-CAM Attention Heatmaps
3. Cross-Attention Multimodal Patient Risk Stratification (Vitals + Signals + Imaging)
4. Multi-Agent Clinical Reasoning, RAG Practice Guideline Citations & EHR Report Generation
"""

import io
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.cm as cm
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from src.agents.evidence_agent import EvidenceAgent
from src.agents.imaging_agent import ImagingAgent
from src.agents.report_agent import ReportAgent
from src.agents.risk_agent import RiskAssessmentAgent
from src.agents.signal_agent import SignalAgent
from src.agents.supervisor_agent import SupervisorAgent
from src.data.loaders.ecg_dataset import AAMI_CLASSES
from src.data.loaders.multimodal_dataset import RISK_CLASSES
from src.data.loaders.xray_dataset import XRAY_CLASSES, XRayDataset
from src.data.preprocessing.image_transforms import denormalize_image, get_xray_transforms
from src.models.fusion.multimodal_transformer import MultimodalTransformerFusion
from src.models.signals.ecg_cnn import ECG1DCNN
from src.models.vision.gradcam import GradCAM
from src.models.vision.xray_densenet import XRayDenseNet121


# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="MedAgent Clinical Console",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .risk-badge-low {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    .risk-badge-mod {
        background-color: #FEF08A;
        color: #713F12;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    .risk-badge-high {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    .safety-banner {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 12px;
        border-radius: 4px;
        font-size: 0.88rem;
        color: #1E40AF;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# Model & Resource Caching
# ==============================================================================
@st.cache_resource(show_spinner="Initializing MedAgent neural models and agent graph...")
def load_medagent_systems():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Neural networks
    ecg_model = ECG1DCNN(in_channels=1, num_classes=5, latent_dim=256).to(device)
    ecg_model.eval()

    xray_model = XRayDenseNet121(num_classes=5, pretrained=False, latent_dim=256).to(device)
    xray_model.eval()

    fusion_model = MultimodalTransformerFusion(
        ecg_latent_dim=256,
        vision_latent_dim=256,
        tabular_dim=10,
        embed_dim=256,
        num_risk_classes=3,
    ).to(device)
    fusion_model.eval()

    gradcam = GradCAM(xray_model)

    # Agents
    signal_agent = SignalAgent()
    imaging_agent = ImagingAgent()
    evidence_agent = EvidenceAgent()
    risk_agent = RiskAssessmentAgent()
    report_agent = ReportAgent()
    supervisor_agent = SupervisorAgent(
        signal_agent=signal_agent,
        imaging_agent=imaging_agent,
        evidence_agent=evidence_agent,
        risk_agent=risk_agent,
        report_agent=report_agent,
    )

    return {
        "device": device,
        "ecg_model": ecg_model,
        "xray_model": xray_model,
        "fusion_model": fusion_model,
        "gradcam": gradcam,
        "signal_agent": signal_agent,
        "imaging_agent": imaging_agent,
        "evidence_agent": evidence_agent,
        "risk_agent": risk_agent,
        "report_agent": report_agent,
        "supervisor_agent": supervisor_agent,
    }


@st.cache_data(show_spinner=False)
def load_sample_datasets():
    ecg_file = Path("data/processed/signals/mitbih_aami_beats.npz")
    ecg_beats, ecg_labels = None, None
    if ecg_file.exists():
        d = np.load(ecg_file)
        ecg_beats, ecg_labels = d["beats"], d["labels"]

    xray_dir = Path("data/raw/imaging/synthetic_xray")
    xray_dataset = None
    if xray_dir.exists():
        xray_dataset = XRayDataset(samples=xray_dir)

    return ecg_beats, ecg_labels, xray_dataset


systems = load_medagent_systems()
ecg_beats, ecg_labels, xray_dataset = load_sample_datasets()


# ==============================================================================
# Clinical Demo Presets
# ==============================================================================
PRESET_CASES = {
    "Case 1: Normal Routine Health Check": {
        "ecg_class_idx": 0,
        "xray_class_name": "Normal",
        "age": 42,
        "sex": "Female",
        "hr": 68.0,
        "bp": "118/76",
        "spo2": 99.0,
        "htn": False,
        "dm": False,
        "smk": False,
        "cardiac": False,
        "notes": "Annual physical checkup. Asymptomatic, baseline vitals within normal parameters.",
    },
    "Case 2: Supraventricular Ectopy & Mild Atelectasis": {
        "ecg_class_idx": 1,
        "xray_class_name": "Atelectasis",
        "age": 59,
        "sex": "Male",
        "hr": 84.0,
        "bp": "134/86",
        "spo2": 96.0,
        "htn": True,
        "dm": False,
        "smk": True,
        "cardiac": False,
        "notes": "Post-operative patient with mild exertional dyspnea and occasional palpitations.",
    },
    "Case 3: Ventricular Arrhythmia & Cardiomegaly (Critical)": {
        "ecg_class_idx": 2,
        "xray_class_name": "Cardiomegaly",
        "age": 73,
        "sex": "Male",
        "hr": 110.0,
        "bp": "158/98",
        "spo2": 93.0,
        "htn": True,
        "dm": True,
        "smk": True,
        "cardiac": True,
        "notes": "Emergency department presentation with orthopnea, bilateral leg edema, and chest pressure.",
    },
    "Case 4: Severe Pneumonia with Hypoxia": {
        "ecg_class_idx": 0,
        "xray_class_name": "Pneumonia",
        "age": 66,
        "sex": "Female",
        "hr": 98.0,
        "bp": "128/82",
        "spo2": 89.0,
        "htn": False,
        "dm": True,
        "smk": False,
        "cardiac": False,
        "notes": "Productive cough, high fever (39.1°C), and marked tachypnea.",
    },
}


# ==============================================================================
# Sidebar - Patient Profile & Navigation
# ==============================================================================
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/google/material-design-icons/master/png/action/favorite/materialicons/48dp/2x/baseline_favorite_black_48dp.png",
        width=48,
    )
    st.title("MedAgent Console")
    st.caption("Multimodal Medical AI Clinical Decision Support")

    st.markdown("---")
    st.subheader("👤 Patient Case Selector")
    case_choice = st.selectbox("Select Patient Profile", list(PRESET_CASES.keys()) + ["Custom Patient Case"])

    if case_choice != "Custom Patient Case":
        case_data = PRESET_CASES[case_choice]
        default_age = case_data["age"]
        default_sex = case_data["sex"]
        default_hr = case_data["hr"]
        default_bp = case_data["bp"]
        default_spo2 = case_data["spo2"]
        default_htn = case_data["htn"]
        default_dm = case_data["dm"]
        default_smk = case_data["smk"]
        default_cardiac = case_data["cardiac"]
        default_notes = case_data["notes"]
    else:
        default_age, default_sex, default_hr, default_bp, default_spo2 = 65, "Male", 75.0, "130/80", 97.0
        default_htn, default_dm, default_smk, default_cardiac = False, False, False, False
        default_notes = "Custom patient presentation."

    patient_id = st.text_input("Patient ID", value="PT-2026-0842")
    age = st.slider("Age", 18, 100, default_age)
    sex = st.selectbox("Biological Sex", ["Male", "Female", "Other"], index=0 if default_sex == "Male" else 1)
    hr_bpm = st.number_input("Heart Rate (BPM)", 30.0, 220.0, default_hr, step=1.0)
    bp = st.text_input("Blood Pressure (mmHg)", value=default_bp)
    spo2 = st.slider("SpO2 Oxygen Saturation (%)", 70.0, 100.0, default_spo2, step=0.5)

    st.markdown("##### Clinical Comorbidities")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        hypertension = st.checkbox("Hypertension", value=default_htn)
        diabetes = st.checkbox("Diabetes Mellitus", value=default_dm)
    with col_c2:
        smoking = st.checkbox("Smoking History", value=default_smk)
        prior_cardiac = st.checkbox("Prior Cardiac Event", value=default_cardiac)

    medical_notes = st.text_area("Physician Notes", value=default_notes, height=70)

    st.markdown("---")
    device_name = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    st.info(f"**Hardware Engine**: {device_name}\n\n**Safety Layer**: Active (Human-in-the-Loop)")


# Compile Patient Information
patient_dict = {
    "patient_id": patient_id,
    "age": age,
    "sex": sex,
    "heart_rate_bpm": hr_bpm,
    "blood_pressure": bp,
    "spo2_percent": spo2,
    "hypertension": hypertension,
    "diabetes": diabetes,
    "smoking_history": smoking,
    "prior_cardiac_event": prior_cardiac,
    "medical_history": medical_notes,
}

# Encode Tabular Tensor
sbp_val, dbp_val = 120.0, 80.0
if "/" in bp:
    try:
        parts = bp.split("/")
        sbp_val = float(parts[0])
        dbp_val = float(parts[1])
    except Exception:
        pass

tabular_vector = torch.tensor(
    [[
        age / 100.0,
        1.0 if sex == "Male" else 0.0,
        hr_bpm / 200.0,
        sbp_val / 200.0,
        dbp_val / 120.0,
        spo2 / 100.0,
        1.0 if hypertension else 0.0,
        1.0 if diabetes else 0.0,
        1.0 if smoking else 0.0,
        1.0 if prior_cardiac else 0.0,
    ]],
    dtype=torch.float32,
    device=systems["device"],
)


# ==============================================================================
# Fetch Sample Modality Data based on Selection
# ==============================================================================
selected_ecg_signal = None
if ecg_beats is not None:
    if case_choice != "Custom Patient Case":
        target_ecg_cls = PRESET_CASES[case_choice]["ecg_class_idx"]
        matching_indices = np.where(ecg_labels == target_ecg_cls)[0]
        chosen_idx = matching_indices[0] if len(matching_indices) > 0 else 0
        selected_ecg_signal = ecg_beats[chosen_idx]
    else:
        selected_ecg_signal = ecg_beats[0]
else:
    # Synthetic default sine waveform
    selected_ecg_signal = np.sin(np.linspace(0, 4 * np.pi, 260)).astype(np.float32)

selected_xray_img = None
if xray_dataset is not None:
    if case_choice != "Custom Patient Case":
        target_xray_cls_name = PRESET_CASES[case_choice]["xray_class_name"]
        target_xray_cls_id = [k for k, v in XRAY_CLASSES.items() if v == target_xray_cls_name][0]
        # find matching sample
        matching_samples = [s for s in xray_dataset.samples if s[1] == target_xray_cls_id]
        if matching_samples:
            selected_xray_img = Image.open(matching_samples[0][0]).convert("RGB")
    if selected_xray_img is None and len(xray_dataset.samples) > 0:
        selected_xray_img = Image.open(xray_dataset.samples[0][0]).convert("RGB")
if selected_xray_img is None:
    selected_xray_img = Image.new("RGB", (224, 224), color=(128, 128, 128))


# ==============================================================================
# Header & Safety Banner
# ==============================================================================
st.markdown('<div class="main-header">🏥 MedAgent Clinical Decision Support Console</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Multimodal AI Platform for Cardiothoracic Signal Analysis, Medical Imaging Explainability, and Clinical Practice Guideline Retrieval</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="safety-banner">
        <strong>⚠️ CLINICAL SAFETY ASSURANCE:</strong> MedAgent operates as an assistive diagnostic copilot under strict physician supervision.
        All algorithmic risk scores and guideline citations require independent clinical validation before patient management decisions.
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# Main Interactive Tabs
# ==============================================================================
tab_ecg, tab_vision, tab_multimodal, tab_agent = st.tabs([
    "🫀 Physiological Signal Intelligence",
    "🩻 Medical Vision & Explainability",
    "🌐 Multimodal Patient Risk Stratification",
    "🤖 Multi-Agent Clinical Summary & RAG",
])


# ==============================================================================
# TAB 1: Physiological Signal Intelligence (ECG)
# ==============================================================================
with tab_ecg:
    st.subheader("1D Single-Lead ECG Analysis & Arrhythmia Classification")
    col_sig1, col_sig2 = st.columns([3, 2])

    with col_sig1:
        st.markdown("##### Interactive ECG Beat Waveform (260 Samples, Lead MLII)")
        # Plotly ECG Chart
        fig = go.Figure()
        time_axis = np.arange(len(selected_ecg_signal)) / 360.0 * 1000.0  # ms
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=selected_ecg_signal,
                mode="lines",
                name="ECG Signal",
                line=dict(color="#2563EB", width=2.5),
            )
        )

        # Mark peak
        peak_idx = int(np.argmax(selected_ecg_signal))
        fig.add_trace(
            go.Scatter(
                x=[time_axis[peak_idx]],
                y=[selected_ecg_signal[peak_idx]],
                mode="markers+text",
                name="R-Peak",
                text=["R-Peak"],
                textposition="top center",
                marker=dict(color="#DC2626", size=10, symbol="triangle-up"),
            )
        )

        fig.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Time (ms)",
            yaxis_title="Normalized Amplitude (mV)",
            template="plotly_white",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_sig2:
        # Run ECG Model Inference
        ecg_tensor = torch.from_numpy(selected_ecg_signal).unsqueeze(0).unsqueeze(0).to(systems["device"])
        with torch.no_grad():
            ecg_logits, ecg_latent = systems["ecg_model"](ecg_tensor)
            ecg_probs = F.softmax(ecg_logits, dim=-1).cpu().numpy()[0]

        signal_analysis = systems["signal_agent"].analyze(ecg_logits.cpu().numpy(), heart_rate_bpm=hr_bpm)

        st.markdown("##### Arrhythmia Classification Distribution (AAMI EC57)")
        df_ecg = pd.DataFrame({
            "Arrhythmia Category": [AAMI_CLASSES[i] for i in range(5)],
            "Probability": ecg_probs,
        })
        fig_bar = px.bar(
            df_ecg,
            x="Probability",
            y="Arrhythmia Category",
            orientation="h",
            color="Probability",
            color_continuous_scale="Blues",
            range_x=[0, 1],
        )
        fig_bar.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Signal Agent Clinical Findings Card
    st.markdown("---")
    st.markdown("##### 🩺 SignalAgent Clinical Findings")
    col_f1, col_f2, col_f3 = st.columns([1.5, 2.5, 1])
    with col_f1:
        st.metric(
            label="Classified Condition",
            value=signal_analysis.class_name,
            delta=f"Confidence: {signal_analysis.confidence * 100:.1f}%",
        )
    with col_f2:
        st.info(f"**Clinical Interpretation**: {signal_analysis.clinical_interpretation}")
    with col_f3:
        sev_color = "#DC2626" if signal_analysis.severity == "High" else "#D97706" if signal_analysis.severity == "Moderate" else "#16A34A"
        st.markdown(
            f"<div style='background-color:{sev_color}22; border-left:4px solid {sev_color}; padding:10px; border-radius:4px;'>"
            f"<strong>Severity:</strong> {signal_analysis.severity}<br>"
            f"<strong>BPM:</strong> {hr_bpm:.0f}</div>",
            unsafe_allow_html=True,
        )


# ==============================================================================
# TAB 2: Medical Vision & Explainability (Chest X-Ray)
# ==============================================================================
with tab_vision:
    st.subheader("Thoracic Radiograph Deep Learning & Grad-CAM Spatial Localization")

    col_v_left, col_v_mid, col_v_right = st.columns([1.5, 1.5, 1.5])

    # Transform image for model
    xray_trans = get_xray_transforms(image_size=224, is_training=False)
    xray_tensor = xray_trans(selected_xray_img).unsqueeze(0).to(systems["device"])

    # Grad-CAM Options
    with col_v_left:
        st.markdown("##### 1. Original Radiograph")
        st.image(selected_xray_img, use_container_width=True, caption="Anterior-Posterior (AP/PA) Radiograph")

    with col_v_mid:
        st.markdown("##### 2. Grad-CAM Attention Overlay")
        alpha_slider = st.slider("Heatmap Opacity (α)", 0.0, 1.0, 0.55, step=0.05)
        cmap_choice = st.selectbox("Color Mapping", ["jet", "plasma", "turbo", "magma"], index=0)

        # Generate overlay
        overlay_rgb, heatmap = systems["gradcam"].generate_overlay(
            xray_tensor,
            target_class=None,
            alpha=alpha_slider,
            colormap=cmap_choice,
        )
        st.image(overlay_rgb, use_container_width=True, caption=f"Penultimate Layer Activation (α={alpha_slider})")

    with col_v_right:
        st.markdown("##### 3. Diagnostic Probabilities")
        with torch.no_grad():
            xray_logits, vision_latent = systems["xray_model"](xray_tensor)
            xray_probs = F.softmax(xray_logits, dim=-1).cpu().numpy()[0]

        df_xray = pd.DataFrame({
            "Pathology": [XRAY_CLASSES[i] for i in range(5)],
            "Confidence": xray_probs,
        })
        fig_xray = px.bar(
            df_xray,
            x="Confidence",
            y="Pathology",
            orientation="h",
            color="Confidence",
            color_continuous_scale="Reds",
            range_x=[0, 1],
        )
        fig_xray.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig_xray, use_container_width=True)

    # Imaging Agent Findings Card
    st.markdown("---")
    st.markdown("##### 🩻 ImagingAgent Radiological Report")
    imaging_analysis = systems["imaging_agent"].analyze(xray_logits.cpu().numpy(), gradcam_heatmap=heatmap)

    col_ir1, col_ir2 = st.columns([1, 2])
    with col_ir1:
        st.metric(
            label="Radiological Finding",
            value=imaging_analysis.class_name,
            delta=f"Confidence: {imaging_analysis.confidence * 100:.1f}%",
        )
    with col_ir2:
        st.success(
            f"**Radiological Assessment**: {imaging_analysis.radiological_finding}\n\n"
            f"**Spatial Anatomical Focus**: {imaging_analysis.gradcam_localization}"
        )


# ==============================================================================
# TAB 3: Multimodal Patient Risk Stratification
# ==============================================================================
with tab_multimodal:
    st.subheader("Cross-Attention Multimodal Patient Risk Stratification")

    # Forward pass multimodal fusion transformer
    with torch.no_grad():
        risk_logits, patient_rep = systems["fusion_model"](ecg_latent, vision_latent, tabular_vector)
        risk_probs = F.softmax(risk_logits, dim=-1).cpu().numpy()[0]

    risk_analysis = systems["risk_agent"].evaluate_risk(
        signal_res=signal_analysis,
        imaging_res=imaging_analysis,
        patient_info=patient_dict,
        multimodal_logits_or_probs=risk_probs,
    )

    col_m1, col_m2, col_m3 = st.columns([1.5, 1.5, 2])

    with col_m1:
        st.markdown("##### Composite Risk Tier")
        tier_color = "#DC2626" if risk_analysis.risk_tier_id == 2 else "#D97706" if risk_analysis.risk_tier_id == 1 else "#16A34A"
        badge_cls = "risk-badge-high" if risk_analysis.risk_tier_id == 2 else "risk-badge-mod" if risk_analysis.risk_tier_id == 1 else "risk-badge-low"

        st.markdown(
            f"""
            <div class="metric-card" style="text-align:center; border-top:4px solid {tier_color};">
                <span class="{badge_cls}" style="font-size:1.1rem;">{risk_analysis.risk_tier}</span>
                <h1 style="font-size:3rem; margin:10px 0; color:{tier_color};">{risk_analysis.risk_score * 100:.1f}<span style="font-size:1.2rem;">/100</span></h1>
                <p style="color:#6B7280; font-size:0.9rem;">Calibrated Composite Patient Risk Index</p>
                <strong>Triage Urgency:</strong> {risk_analysis.urgency_level}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m2:
        st.markdown("##### Multimodal Modality Alignment")
        fig_radar = go.Figure(
            data=go.Scatterpolar(
                r=[
                    signal_analysis.confidence * (0.9 if signal_analysis.predicted_class != 0 else 0.2),
                    imaging_analysis.confidence * (0.9 if imaging_analysis.predicted_class != 0 else 0.2),
                    min(1.0, (age / 100.0) * (1.3 if hypertension or diabetes else 0.9)),
                    1.0 if (hr_bpm > 100 or hr_bpm < 55) else 0.3,
                    1.0 if spo2 < 93 else 0.2,
                ],
                theta=["ECG Arrhythmia", "Imaging Pathology", "Age & Demographics", "Heart Rate Stress", "Oxygen Desaturation"],
                fill="toself",
                fillcolor="rgba(37, 99, 235, 0.2)",
                line=dict(color="#2563EB"),
            )
        )
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            height=280,
            margin=dict(l=30, r=30, t=20, b=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_m3:
        st.markdown("##### Primary Clinical Risk Drivers")
        for driver in risk_analysis.primary_risk_drivers:
            st.markdown(f"- 🔴 **{driver}**")

        st.markdown("##### Clinical Rationale")
        st.write(risk_analysis.clinical_rationale)


# ==============================================================================
# TAB 4: Multi-Agent Clinical Summary & RAG
# ==============================================================================
with tab_agent:
    st.subheader("Multi-Agent Clinical Decision Support & Practice Guidelines (RAG)")

    # Execute Evidence Agent retrieval
    evidence_res = systems["evidence_agent"].retrieve_evidence(
        signal_findings=signal_analysis.clinical_interpretation,
        imaging_findings=imaging_analysis.radiological_finding,
        patient_context=f"Age {age}, Sex {sex}, Vitals: HR {hr_bpm}, BP {bp}, SpO2 {spo2}%. Comorbidities: {medical_notes}",
        top_k=2,
    )

    # Generate Report
    report = systems["report_agent"].generate_report(
        patient_info=patient_dict,
        signal_res=signal_analysis,
        imaging_res=imaging_analysis,
        risk_res=risk_analysis,
        evidence_res=evidence_res,
    )

    # Safety Validation via Supervisor
    is_safe, val_notes = systems["supervisor_agent"]._validate_safety(
        signal_res=signal_analysis,
        imaging_res=imaging_analysis,
        risk_res=risk_analysis,
        report=report,
    )

    col_ag1, col_ag2 = st.columns([2, 1])

    with col_ag1:
        st.markdown("##### 📄 Structured EHR Clinical Report")
        st.markdown(report.markdown_content)

        # Download Buttons
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                label="📥 Download Clinical Summary (Markdown)",
                data=report.markdown_content,
                file_name=f"MedAgent_Report_{patient_id}.md",
                mime="text/markdown",
            )
        with col_d2:
            import json

            st.download_button(
                label="📥 Export Structured JSON Data",
                data=json.dumps(report.structured_data, indent=2),
                file_name=f"MedAgent_Data_{patient_id}.json",
                mime="application/json",
            )

    with col_ag2:
        st.markdown("##### 📚 Evidence-Based Practice Guidelines (RAG)")
        for idx, item in enumerate(evidence_res.retrieved_guidelines, 1):
            with st.expander(f"📖 [{item.id}] {item.topic}", expanded=(idx == 1)):
                st.markdown(f"**{item.title}**")
                st.markdown(f"**Guideline Recommendation**: {item.content}")
                st.markdown(f"**Strength / Evidence**: `{item.strength_of_recommendation}`")
                st.caption(f"Source: {item.citation}")

        st.markdown("---")
        st.markdown("##### 🛡️ Supervisor Agent Safety Audit")
        if is_safe:
            st.success(f"**Safety Verification**: Passed\n\n{val_notes}")
        else:
            st.error(f"**Safety Flag Triggered**: {val_notes}")
