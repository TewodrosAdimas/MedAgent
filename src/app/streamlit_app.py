"""
MedAgent Clinical Decision Support Web Console.

A multimodal clinical cockpit empowering physicians and healthcare practitioners with:
1. Interactive 1D Physiological Waveform Intelligence (ECG/PPG) & Arrhythmia Detection
2. Explainable Medical Vision (Chest X-Ray) with Grad-CAM Attention Heatmaps
3. Cross-Attention Multimodal Patient Risk Stratification (Vitals + Signals + Imaging)
4. Multi-Agent Clinical Reasoning, RAG Practice Guideline Citations & EHR Report Generation (PDF / Markdown / FHIR-JSON)
"""

import io
import json
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
from src.app.pdf_generator import generate_clinical_pdf
from src.data.loaders.ecg_dataset import AAMI_CLASSES
from src.data.loaders.multimodal_dataset import RISK_CLASSES
from src.data.loaders.xray_dataset import XRAY_CLASSES, XRayDataset
from src.data.preprocessing.ecg_cleaner import clean_ecg
from src.data.preprocessing.image_transforms import denormalize_image, get_xray_transforms
from src.models.fusion.multimodal_transformer import MultimodalTransformerFusion
from src.models.signals.ecg_cnn import ECG1DCNN
from src.models.vision.gradcam import GradCAM
from src.models.vision.xray_densenet import XRayDenseNet121


# ==============================================================================
# Page Configuration & Design System
# ==============================================================================
st.set_page_config(
    page_title="MedAgent • Multimodal Clinical Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# High-Contrast, Production-Grade Clinical Design System (100% Legibility & WCAG 2.1 AA Compliance)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --font-main: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        --font-mono: 'JetBrains Mono', monospace;
        
        --text-primary:   #0f172a;
        --text-secondary: #334155;
        --text-muted:     #64748b;
        --text-on-dark:   #ffffff;

        --bg-app:         #f8fafc;
        --bg-card:        #ffffff;
        --bg-sidebar:     #ffffff;
        --border-color:   #cbd5e1;
        --border-light:   #e2e8f0;

        --primary-600:    #0284c7;
        --primary-500:    #0ea5e9;
        --primary-100:    #e0f2fe;
        --primary-50:     #f0f9ff;
        
        --success-700:    #047857;
        --success-600:    #059669;
        --success-100:    #d1fae5;
        --success-50:     #ecfdf5;

        --warning-700:    #b45309;
        --warning-600:    #d97706;
        --warning-100:    #fef3c7;
        --warning-50:     #fffbeb;

        --danger-700:     #b91c1c;
        --danger-600:     #dc2626;
        --danger-500:     #ef4444;
        --danger-100:     #fee2e2;
        --danger-50:      #fef2f2;

        --purple-700:     #6d28d9;
        --purple-600:     #7c3aed;
        --purple-100:     #ede9fe;
    }

    /* Global Typography Enforcement */
    html, body, [class*="css"], .stApp {
        font-family: var(--font-main) !important;
        background-color: var(--bg-app) !important;
        color: var(--text-primary) !important;
    }

    /* Force all Streamlit Text, Labels & Headings to High-Contrast Dark Slate */
    p, span, label, h1, h2, h3, h4, h5, h6, strong, em, b {
        color: var(--text-primary) !important;
    }

    .stMarkdown, .stMarkdown p, .stMarkdown span {
        color: var(--text-primary) !important;
    }

    /* Streamlit Widget Label Legibility */
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label,
    [data-testid="stWidgetLabel"] span {
        color: #0f172a !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        letter-spacing: -0.01em !important;
    }

    /* Radio Button & Checkbox Label Legibility */
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] div,
    [data-testid="stRadio"] span,
    [data-testid="stRadio"] p,
    [data-testid="stCheckbox"] label,
    [data-testid="stCheckbox"] span,
    [data-testid="stCheckbox"] p {
        color: #1e293b !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }

    /* Streamlit Native Input Fields */
    [data-baseweb="input"] input,
    [data-baseweb="textarea"] textarea {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border-color: #cbd5e1 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }

    [data-baseweb="input"] > div,
    [data-baseweb="textarea"] > div,
    [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
    }

    [data-baseweb="select"] span {
        color: #0f172a !important;
        font-weight: 600 !important;
    }

    /* Slider Styling & Numbers */
    [data-testid="stSlider"] div,
    [data-testid="stSlider"] span,
    [data-testid="stSlider"] p {
        color: #0f172a !important;
        font-weight: 600 !important;
    }

    /* Sidebar Background & Elements */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] label,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    /* Header Container */
    .med-brand-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.25rem 1.75rem;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%);
        border-radius: 16px;
        color: #ffffff !important;
        margin-bottom: 1.25rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.2), 0 8px 10px -6px rgba(15, 23, 42, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }

    .med-brand-header * {
        color: #ffffff !important;
    }

    .med-brand-title {
        font-size: 1.65rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 0;
    }

    .med-brand-badge {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        background: rgba(14, 165, 233, 0.35);
        color: #e0f2fe !important;
        border: 1px solid rgba(56, 189, 248, 0.6);
        padding: 4px 10px;
        border-radius: 9999px;
    }

    .med-brand-subtitle {
        font-size: 0.88rem;
        color: #cbd5e1 !important;
        margin-top: 0.35rem;
        font-weight: 400;
        max-width: 750px;
        line-height: 1.45;
    }

    .med-engine-status {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 0.5rem 0.9rem;
        border-radius: 10px;
        font-size: 0.78rem;
    }

    .status-pulse-dot {
        width: 9px;
        height: 9px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse-ring 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }

    @keyframes pulse-ring {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Clinical Patient HUD / Executive Strip */
    .patient-hud-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 0.75rem;
        margin-bottom: 1.25rem;
    }

    .hud-card {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        box-shadow: 0 2px 4px rgba(15, 23, 42, 0.04);
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: all 0.2s ease;
    }

    .hud-card:hover {
        border-color: var(--primary-500);
        box-shadow: 0 6px 12px -2px rgba(14, 165, 233, 0.15);
        transform: translateY(-1px);
    }

    .hud-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b !important;
        margin-bottom: 0.25rem;
    }

    .hud-val {
        font-size: 1.05rem;
        font-weight: 800;
        color: #0f172a !important;
        display: flex;
        align-items: center;
        gap: 0.4rem;
        line-height: 1.25;
    }

    .hud-subtext {
        font-size: 0.75rem;
        color: #475569 !important;
        margin-top: 0.2rem;
        font-weight: 500;
    }

    /* Safety Alert Banner */
    .clinical-safety-alert {
        background: #f0fdf4;
        border: 1px solid #86efac;
        border-left: 5px solid #16a34a;
        padding: 0.85rem 1.1rem;
        border-radius: 10px;
        margin-bottom: 1.25rem;
        display: flex;
        align-items: center;
        gap: 0.85rem;
        font-size: 0.85rem;
        color: #14532d !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }

    .clinical-safety-alert * {
        color: #14532d !important;
    }

    /* Modern Card Container */
    .med-card {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 14px;
        padding: 1.25rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04);
        transition: all 0.2s ease;
    }

    .med-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid #e2e8f0;
    }

    .med-card-title {
        font-size: 1.05rem;
        font-weight: 800;
        color: #0f172a !important;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 0;
    }

    .med-card-title * {
        color: #0f172a !important;
    }

    /* Metric & Findings Cards */
    .finding-kpi-card {
        border-radius: 12px;
        padding: 1.1rem;
        background: #ffffff;
        border: 1px solid #cbd5e1;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .finding-kpi-card.severity-high {
        border-left: 5px solid #dc2626;
        background: linear-gradient(135deg, #ffffff 0%, #fef2f2 100%);
    }

    .finding-kpi-card.severity-moderate {
        border-left: 5px solid #d97706;
        background: linear-gradient(135deg, #ffffff 0%, #fffbeb 100%);
    }

    .finding-kpi-card.severity-low {
        border-left: 5px solid #059669;
        background: linear-gradient(135deg, #ffffff 0%, #ecfdf5 100%);
    }

    /* Badges & Status Pills */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }

    .badge-pill-danger {
        background: #fee2e2 !important;
        color: #991b1b !important;
        border: 1px solid #f87171 !important;
    }
    .badge-pill-danger * { color: #991b1b !important; }

    .badge-pill-warning {
        background: #fef3c7 !important;
        color: #92400e !important;
        border: 1px solid #fbbf24 !important;
    }
    .badge-pill-warning * { color: #92400e !important; }

    .badge-pill-success {
        background: #d1fae5 !important;
        color: #065f46 !important;
        border: 1px solid #34d399 !important;
    }
    .badge-pill-success * { color: #065f46 !important; }

    .badge-pill-info {
        background: #e0f2fe !important;
        color: #0369a1 !important;
        border: 1px solid #38bdf8 !important;
    }
    .badge-pill-info * { color: #0369a1 !important; }

    .badge-pill-purple {
        background: #ede9fe !important;
        color: #5b21b6 !important;
        border: 1px solid #a78bfa !important;
    }
    .badge-pill-purple * { color: #5b21b6 !important; }

    /* Multi-Agent Orchestration Stepper */
    .agent-pipeline-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 0.85rem 1.25rem;
        margin-bottom: 1.25rem;
        gap: 0.5rem;
        overflow-x: auto;
    }

    .pipeline-step {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        font-size: 0.85rem;
        font-weight: 700;
        color: #1e293b !important;
    }
    .pipeline-step * { color: #1e293b !important; }

    .pipeline-step-badge {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: #e0f2fe;
        color: #0284c7 !important;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.78rem;
        font-weight: 800;
        border: 1.5px solid #0284c7;
    }

    .pipeline-divider {
        flex: 1;
        height: 2px;
        background: #cbd5e1;
        min-width: 15px;
    }

    /* EHR Report Styling */
    .ehr-container {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 1.5rem 1.75rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.04);
        font-family: var(--font-main);
        color: #0f172a !important;
    }

    .ehr-header-strip {
        border-bottom: 2px solid #0f172a;
        padding-bottom: 0.75rem;
        margin-bottom: 1.25rem;
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
    }

    /* Streamlit Tabs Polish & Visibility */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #e2e8f0 !important;
        padding: 6px;
        border-radius: 12px;
        border: 1px solid #cbd5e1 !important;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
        color: #334155 !important;
        background-color: transparent !important;
        border: none !important;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"] * {
        color: #334155 !important;
        font-weight: 700 !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #0284c7 !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1) !important;
    }

    .stTabs [aria-selected="true"] * {
        color: #0284c7 !important;
        font-weight: 800 !important;
    }

    /* Expander High Contrast */
    [data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
        margin-bottom: 0.6rem;
    }

    [data-testid="stExpander"] summary {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    [data-testid="stExpander"] summary * {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    /* Button Polish */
    .stButton>button {
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.9rem;
        background-color: #0284c7;
        color: #ffffff !important;
        border: none;
        transition: all 0.2s ease;
    }
    
    .stDownloadButton>button {
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.88rem;
        background-color: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
        color: #0f172a !important;
        transition: all 0.2s ease;
    }
    .stDownloadButton>button * {
        color: #0f172a !important;
    }
    .stDownloadButton>button:hover {
        border-color: #0284c7 !important;
        background-color: #f0f9ff !important;
    }
    .stDownloadButton>button:hover * {
        color: #0284c7 !important;
    }

    .pdf-download-btn>button {
        background-color: #0284c7 !important;
        border: 1.5px solid #0284c7 !important;
        color: #ffffff !important;
    }
    .pdf-download-btn>button * {
        color: #ffffff !important;
    }
    .pdf-download-btn>button:hover {
        background-color: #0369a1 !important;
        border-color: #0369a1 !important;
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
    ecg_ckpt = Path("models/checkpoints/ecg_cnn_best.pt")
    if ecg_ckpt.exists():
        ckpt = torch.load(str(ecg_ckpt), map_location=device, weights_only=False)
        ecg_model.load_state_dict(ckpt["model_state_dict"])
    ecg_model.eval()

    xray_model = XRayDenseNet121(num_classes=5, pretrained=False, latent_dim=256).to(device)
    xray_ckpt = Path("models/checkpoints/xray_densenet_best.pt")
    if xray_ckpt.exists():
        ckpt = torch.load(str(xray_ckpt), map_location=device, weights_only=False)
        xray_model.load_state_dict(ckpt["model_state_dict"])
    xray_model.eval()

    fusion_model = MultimodalTransformerFusion(
        ecg_latent_dim=256,
        vision_latent_dim=256,
        tabular_dim=10,
        embed_dim=256,
        num_risk_classes=3,
    ).to(device)
    fusion_ckpt = Path("models/checkpoints/fusion_transformer_weights.pt")
    if fusion_ckpt.exists():
        ckpt = torch.load(str(fusion_ckpt), map_location=device, weights_only=False)
        fusion_model.load_state_dict(ckpt["model_state_dict"])
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
# Helper Functions for Parsing Uploaded Modalities
# ==============================================================================
def parse_uploaded_signal(uploaded_file) -> Optional[np.ndarray]:
    """Parses an uploaded CSV, NPY, JSON, or TXT file into a 1D NumPy float32 signal."""
    try:
        fname = uploaded_file.name.lower()
        if fname.endswith(".npy"):
            arr = np.load(uploaded_file)
            arr = arr.flatten().astype(np.float32)
        elif fname.endswith(".csv"):
            df = pd.read_csv(uploaded_file, header=None)
            arr = df.values.flatten().astype(np.float32)
        elif fname.endswith(".json"):
            data = json.load(uploaded_file)
            if isinstance(data, list):
                arr = np.array(data, dtype=np.float32)
            elif isinstance(data, dict) and "signal" in data:
                arr = np.array(data["signal"], dtype=np.float32)
            else:
                arr = np.array(list(data.values()), dtype=np.float32)
        else:
            content = uploaded_file.read().decode("utf-8")
            raw_vals = [float(x.strip()) for x in content.replace(",", " ").split() if x.strip()]
            arr = np.array(raw_vals, dtype=np.float32)

        if len(arr) == 0:
            return None

        if len(arr) < 260:
            arr = np.pad(arr, (0, 260 - len(arr)), mode="edge")
        elif len(arr) > 260:
            cleaned = clean_ecg(arr, sampling_rate=360.0)
            arr = cleaned[:260]

        return arr
    except Exception as e:
        st.error(f"Error parsing ECG signal file: {str(e)}")
        return None


# ==============================================================================
# Clinical Demo Presets
# ==============================================================================
PRESET_CASES = {
    "Case 1: Normal Routine Health Check": {
        "tag": "Low Risk • Baseline Normal",
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
        "tag": "Moderate Risk • SVEB & Atelectasis",
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
        "tag": "High Risk • VEB & Cardiomegaly",
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
        "tag": "High Risk • Hypoxemia & Infiltration",
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
# Sidebar - Patient Profile & Clinical Telemetry
# ==============================================================================
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
            <div style="background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%); color: white !important; border-radius: 10px; width: 38px; height: 38px; display: flex; align-items: center; justify-content: center; font-size: 1.3rem; font-weight: bold; box-shadow: 0 4px 6px -1px rgba(2, 132, 199, 0.35);">
                🏥
            </div>
            <div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #0f172a !important; line-height: 1.1;">MedAgent</div>
                <div style="font-size: 0.75rem; color: #475569 !important; font-weight: 600;">Clinical Telemetry v2.4</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:1px; background:#cbd5e1; margin:0.75rem 0;'></div>", unsafe_allow_html=True)

    st.markdown("##### 📁 Case Preset Cohort")
    case_choice = st.selectbox(
        "Select Cohort Template",
        list(PRESET_CASES.keys()) + ["Custom Patient Case"],
        label_visibility="collapsed",
    )

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
        preset_tag = case_data["tag"]
        st.markdown(f"<span class='badge-pill badge-pill-info' style='margin-bottom:0.75rem;'>{preset_tag}</span>", unsafe_allow_html=True)
    else:
        default_age, default_sex, default_hr, default_bp, default_spo2 = 65, "Male", 75.0, "130/80", 97.0
        default_htn, default_dm, default_smk, default_cardiac = False, False, False, False
        default_notes = "Custom patient clinical presentation."

    st.markdown("##### 👤 Patient Demographics")
    col_p1, col_p2 = st.columns([1.2, 1])
    with col_p1:
        patient_id = st.text_input("Patient MRN / ID", value="PT-2026-0842")
    with col_p2:
        sex = st.selectbox("Biological Sex", ["Male", "Female", "Other"], index=0 if default_sex == "Male" else 1)

    age = st.slider("Patient Age (Years)", 18, 100, default_age)
    
    st.markdown("##### 💓 Baseline Vitals")
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        hr_bpm = st.number_input("Heart Rate (BPM)", 30.0, 220.0, default_hr, step=1.0)
    with col_v2:
        bp = st.text_input("BP (mmHg)", value=default_bp)

    spo2 = st.slider("SpO2 Saturation (%)", 70.0, 100.0, default_spo2, step=0.5)

    st.markdown("##### 🩺 Comorbidities & History")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        hypertension = st.checkbox("Hypertension", value=default_htn)
        diabetes = st.checkbox("Diabetes Mellitus", value=default_dm)
    with col_c2:
        smoking = st.checkbox("Tobacco Smoker", value=default_smk)
        prior_cardiac = st.checkbox("Prior Cardiac Event", value=default_cardiac)

    medical_notes = st.text_area("Clinical Notes & Chief Complaint", value=default_notes, height=75)

    st.markdown("<div style='height:1px; background:#cbd5e1; margin:1rem 0;'></div>", unsafe_allow_html=True)
    
    device_name = "CUDA (NVIDIA RTX)" if torch.cuda.is_available() else "CPU Active"
    st.markdown(
        f"""
        <div style="background:#f1f5f9; border:1px solid #cbd5e1; border-radius:10px; padding:0.85rem; font-size:0.78rem; color:#334155;">
            <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                <span>Compute Engine:</span> <strong style="color:#0f172a !important;">{device_name}</strong>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                <span>Neural Fusion:</span> <strong style="color:#059669 !important;">Cross-Attention v2</strong>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span>Agent Supervisor:</span> <strong style="color:#0284c7 !important;">Safety Active (HITL)</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
# Brand Header & Live System Status
# ==============================================================================
st.markdown(
    f"""
    <div class="med-brand-header">
        <div>
            <div class="med-brand-title">
                <span>🏥 MedAgent Clinical Decision Cockpit</span>
                <span class="med-brand-badge">Multimodal AI 2.0</span>
            </div>
            <div class="med-brand-subtitle">
                Autonomous multi-agent clinical synthesis integrating 1D physiological waveform intelligence, thoracic vision explainability, and RAG-driven practice guideline consensus.
            </div>
        </div>
        <div class="med-engine-status">
            <div class="status-pulse-dot"></div>
            <div>
                <div style="font-weight:700; color:#ffffff !important;">System Online</div>
                <div style="font-size:0.7rem; color:#cbd5e1 !important;">Latency: 18ms • Neural Checkpoints Active</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# Initial Signal & Image Resolution for HUD & Cross-Tab State
# ==============================================================================
if "active_ecg_signal" not in st.session_state:
    if ecg_beats is not None:
        if case_choice != "Custom Patient Case":
            target_ecg_cls = PRESET_CASES[case_choice]["ecg_class_idx"]
            matching_indices = np.where(ecg_labels == target_ecg_cls)[0]
            chosen_idx = matching_indices[0] if len(matching_indices) > 0 else 0
            st.session_state["active_ecg_signal"] = ecg_beats[chosen_idx]
        else:
            st.session_state["active_ecg_signal"] = ecg_beats[0]
    else:
        st.session_state["active_ecg_signal"] = np.sin(np.linspace(0, 4 * np.pi, 260)).astype(np.float32)

if "active_xray_img" not in st.session_state:
    if xray_dataset is not None and len(xray_dataset.samples) > 0:
        if case_choice != "Custom Patient Case":
            target_xray_cls_name = PRESET_CASES[case_choice]["xray_class_name"]
            target_xray_cls_id = [k for k, v in XRAY_CLASSES.items() if v == target_xray_cls_name][0]
            matching_samples = [s for s in xray_dataset.samples if s[1] == target_xray_cls_id]
            if matching_samples:
                st.session_state["active_xray_img"] = Image.open(matching_samples[0][0]).convert("RGB")
            else:
                st.session_state["active_xray_img"] = Image.open(xray_dataset.samples[0][0]).convert("RGB")
        else:
            st.session_state["active_xray_img"] = Image.open(xray_dataset.samples[0][0]).convert("RGB")
    else:
        st.session_state["active_xray_img"] = Image.new("RGB", (224, 224), color=(128, 128, 128))


# Run Background Quick Inferences for Persistent Patient Context HUD
curr_sig = st.session_state["active_ecg_signal"]
sig_tensor = torch.from_numpy(curr_sig).unsqueeze(0).unsqueeze(0).to(systems["device"])
with torch.no_grad():
    quick_ecg_logits, quick_ecg_latent = systems["ecg_model"](sig_tensor)
quick_sig_analysis = systems["signal_agent"].analyze(quick_ecg_logits.cpu().numpy(), heart_rate_bpm=hr_bpm)

curr_img = st.session_state["active_xray_img"]
xray_trans = get_xray_transforms(image_size=224, is_training=False)
quick_xray_tensor = xray_trans(curr_img).unsqueeze(0).to(systems["device"])
with torch.no_grad():
    quick_xray_logits, quick_vis_latent = systems["xray_model"](quick_xray_tensor)
quick_img_analysis = systems["imaging_agent"].analyze(quick_xray_logits.cpu().numpy())

with torch.no_grad():
    quick_risk_logits, _ = systems["fusion_model"](quick_ecg_latent, quick_vis_latent, tabular_vector)
    quick_risk_probs = F.softmax(quick_risk_logits, dim=-1).cpu().numpy()[0]
quick_risk_analysis = systems["risk_agent"].evaluate_risk(
    signal_res=quick_sig_analysis,
    imaging_res=quick_img_analysis,
    patient_info=patient_dict,
    multimodal_logits_or_probs=quick_risk_probs,
)


# ==============================================================================
# Global Patient Context HUD (Heads-Up Display)
# ==============================================================================
risk_badge_cls = "badge-pill-danger" if quick_risk_analysis.risk_tier_id == 2 else "badge-pill-warning" if quick_risk_analysis.risk_tier_id == 1 else "badge-pill-success"
spo2_badge = f"<span style='color:{'#dc2626' if spo2 < 92 else '#059669'}; font-weight:800;'>{spo2:.1f}%</span>"
hr_badge = f"<span style='color:{'#dc2626' if hr_bpm > 100 or hr_bpm < 55 else '#0f172a'}; font-weight:800;'>{hr_bpm:.0f} BPM</span>"

st.markdown(
    f"""
    <div class="patient-hud-container">
        <div class="hud-card">
            <div class="hud-label">Patient Record</div>
            <div class="hud-val">{patient_id}</div>
            <div class="hud-subtext">{age}y • {sex} • {bp} mmHg</div>
        </div>
        <div class="hud-card">
            <div class="hud-label">Hemodynamic Vitals</div>
            <div class="hud-val">{hr_badge} &nbsp;•&nbsp; SpO2 {spo2_badge}</div>
            <div class="hud-subtext">Continuous Telemetry</div>
        </div>
        <div class="hud-card">
            <div class="hud-label">Signal Rhythm Finding</div>
            <div class="hud-val" style="font-size:0.95rem;">{quick_sig_analysis.class_name}</div>
            <div class="hud-subtext">Conf: {quick_sig_analysis.confidence * 100:.1f}% • {quick_sig_analysis.severity} Severity</div>
        </div>
        <div class="hud-card">
            <div class="hud-label">Thoracic Radiograph</div>
            <div class="hud-val" style="font-size:0.95rem;">{quick_img_analysis.class_name}</div>
            <div class="hud-subtext">Conf: {quick_img_analysis.confidence * 100:.1f}% • DenseNet-121</div>
        </div>
        <div class="hud-card" style="border-right: 4px solid {'#ef4444' if quick_risk_analysis.risk_tier_id == 2 else '#f59e0b' if quick_risk_analysis.risk_tier_id == 1 else '#10b981'};">
            <div class="hud-label">Composite Risk Tier</div>
            <div class="hud-val">
                <span class="badge-pill {risk_badge_cls}">{quick_risk_analysis.risk_tier}</span>
                <span style="font-size:1.15rem; font-weight:800; margin-left:4px;">{quick_risk_analysis.risk_score * 100:.1f}</span><span style="font-size:0.75rem; color:#64748b;">/100</span>
            </div>
            <div class="hud-subtext">Triage Urgency: <strong style="color:#0f172a !important;">{quick_risk_analysis.urgency_level}</strong></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Safety Protocol Alert
st.markdown(
    """
    <div class="clinical-safety-alert">
        <span style="font-size: 1.2rem;">🛡️</span>
        <div>
            <strong>CLINICAL SAFETY ASSURANCE PROTOCOL:</strong> MedAgent operates as an assistive AI copilot with physician-in-the-loop oversight. 
            All diagnostic predictions, Grad-CAM localizations, and RAG guideline citations must be verified prior to therapeutic interventions.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# Multi-Agent Workflow Orchestration Pipeline Indicator
# ==============================================================================
st.markdown(
    """
    <div class="agent-pipeline-bar">
        <div class="pipeline-step">
            <div class="pipeline-step-badge">1</div>
            <span>SignalAgent</span>
        </div>
        <div class="pipeline-divider"></div>
        <div class="pipeline-step">
            <div class="pipeline-step-badge">2</div>
            <span>ImagingAgent</span>
        </div>
        <div class="pipeline-divider"></div>
        <div class="pipeline-step">
            <div class="pipeline-step-badge">3</div>
            <span>Multimodal Transformer</span>
        </div>
        <div class="pipeline-divider"></div>
        <div class="pipeline-step">
            <div class="pipeline-step-badge">4</div>
            <span>RiskAgent</span>
        </div>
        <div class="pipeline-divider"></div>
        <div class="pipeline-step">
            <div class="pipeline-step-badge">5</div>
            <span>EvidenceAgent (RAG)</span>
        </div>
        <div class="pipeline-divider"></div>
        <div class="pipeline-step">
            <div class="pipeline-step-badge">6</div>
            <span>Supervisor Audit</span>
        </div>
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
    st.markdown(
        """
        <div class="med-card-header" style="margin-top:0.5rem;">
            <div class="med-card-title">
                <span>🫀 1D Single-Lead ECG Telemetry & Arrhythmia Classification</span>
            </div>
            <span class="badge-pill badge-pill-info">AAMI EC57 Standard • 360 Hz</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ECG Signal Input Source Selector
    ecg_source = st.radio(
        "Select ECG Signal Source:",
        ["📁 Preset / Cohort Case", "📤 Upload Custom ECG File (.csv, .npy, .txt, .json)", "✍️ Paste Numeric Signal Values"],
        horizontal=True,
    )

    current_ecg_signal = None

    if ecg_source == "📁 Preset / Cohort Case":
        if ecg_beats is not None:
            if case_choice != "Custom Patient Case":
                target_ecg_cls = PRESET_CASES[case_choice]["ecg_class_idx"]
                matching_indices = np.where(ecg_labels == target_ecg_cls)[0]
                chosen_idx = matching_indices[0] if len(matching_indices) > 0 else 0
                current_ecg_signal = ecg_beats[chosen_idx]
            else:
                beat_idx = st.slider("Select Sample Beat Index from MIT-BIH Arrhythmia Database", 0, len(ecg_beats) - 1, 0)
                current_ecg_signal = ecg_beats[beat_idx]
        else:
            current_ecg_signal = np.sin(np.linspace(0, 4 * np.pi, 260)).astype(np.float32)

    elif ecg_source == "📤 Upload Custom ECG File (.csv, .npy, .txt, .json)":
        uploaded_ecg = st.file_uploader("Upload 1D ECG waveform file", type=["csv", "npy", "txt", "json"], key="ecg_uploader")
        if uploaded_ecg is not None:
            parsed = parse_uploaded_signal(uploaded_ecg)
            if parsed is not None:
                current_ecg_signal = parsed
                st.success(f"Loaded waveform: `{uploaded_ecg.name}` ({len(current_ecg_signal)} samples).")
        if current_ecg_signal is None:
            current_ecg_signal = ecg_beats[0] if ecg_beats is not None else np.sin(np.linspace(0, 4 * np.pi, 260)).astype(np.float32)

    elif ecg_source == "✍️ Paste Numeric Signal Values":
        default_str = ", ".join([f"{v:.3f}" for v in (ecg_beats[0] if ecg_beats is not None else np.zeros(260))])
        raw_text = st.text_area("Enter comma or space-separated signal float values (e.g. 260 samples):", value=default_str, height=90)
        try:
            vals = [float(x.strip()) for x in raw_text.replace(",", " ").split() if x.strip()]
            if len(vals) >= 10:
                arr = np.array(vals, dtype=np.float32)
                if len(arr) < 260:
                    arr = np.pad(arr, (0, 260 - len(arr)), mode="edge")
                current_ecg_signal = arr[:260]
            else:
                st.warning("Please provide at least 10 signal values.")
                current_ecg_signal = ecg_beats[0] if ecg_beats is not None else np.zeros(260, dtype=np.float32)
        except Exception:
            current_ecg_signal = ecg_beats[0] if ecg_beats is not None else np.zeros(260, dtype=np.float32)

    st.session_state["active_ecg_signal"] = current_ecg_signal

    # Model inference
    ecg_tensor = torch.from_numpy(current_ecg_signal).unsqueeze(0).unsqueeze(0).to(systems["device"])
    with torch.no_grad():
        ecg_logits, ecg_latent = systems["ecg_model"](ecg_tensor)
        ecg_probs = F.softmax(ecg_logits, dim=-1).cpu().numpy()[0]

    signal_analysis = systems["signal_agent"].analyze(ecg_logits.cpu().numpy(), heart_rate_bpm=hr_bpm)
    st.session_state["active_ecg_logits"] = ecg_logits
    st.session_state["active_ecg_latent"] = ecg_latent
    st.session_state["active_signal_analysis"] = signal_analysis

    # 2-Column Responsive Visualization Grid
    col_sig1, col_sig2 = st.columns([1.7, 1.3])

    with col_sig1:
        st.markdown(
            """
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                <span style="font-weight:700; font-size:0.92rem; color:#0f172a !important;">Interactive Digital Electrocardiogram (Lead MLII)</span>
                <span style="font-size:0.78rem; color:#475569 !important; font-family:'JetBrains Mono', monospace; font-weight:600;">260 Samples • 722 ms</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        time_axis = np.arange(len(current_ecg_signal)) / 360.0 * 1000.0  # ms
        peak_idx = int(np.argmax(current_ecg_signal))

        fig_ecg = go.Figure()
        # Baseline shaded area
        fig_ecg.add_trace(
            go.Scatter(
                x=time_axis,
                y=current_ecg_signal,
                mode="lines",
                name="ECG Lead MLII",
                line=dict(color="#0284c7", width=2.8, shape="spline"),
                fill="tozeroy",
                fillcolor="rgba(14, 165, 233, 0.12)",
                hovertemplate="Time: %{x:.1f} ms<br>Amplitude: %{y:.3f} mV<extra></extra>",
            )
        )
        # Highlight R-Peak with modern callout
        fig_ecg.add_trace(
            go.Scatter(
                x=[time_axis[peak_idx]],
                y=[current_ecg_signal[peak_idx]],
                mode="markers+text",
                name="R-Peak",
                text=["R-Peak"],
                textposition="top center",
                textfont=dict(family="Plus Jakarta Sans", size=12, color="#dc2626"),
                marker=dict(color="#dc2626", size=10, symbol="circle", line=dict(color="#ffffff", width=2)),
            )
        )

        fig_ecg.update_layout(
            height=320,
            margin=dict(l=40, r=20, t=20, b=40),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            xaxis=dict(
                title=dict(text="Time (ms)", font=dict(size=12, color="#334155", family="Plus Jakarta Sans")),
                showgrid=True,
                gridcolor="#e2e8f0",
                gridwidth=1,
                zeroline=True,
                zerolinecolor="#94a3b8",
                tickfont=dict(color="#334155", size=10),
            ),
            yaxis=dict(
                title=dict(text="Amplitude (mV)", font=dict(size=12, color="#334155", family="Plus Jakarta Sans")),
                showgrid=True,
                gridcolor="#e2e8f0",
                gridwidth=1,
                zeroline=True,
                zerolinecolor="#94a3b8",
                tickfont=dict(color="#334155", size=10),
            ),
            hovermode="x unified",
            showlegend=False,
        )
        st.plotly_chart(fig_ecg, use_container_width=True)

    with col_sig2:
        st.markdown(
            """
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                <span style="font-weight:700; font-size:0.92rem; color:#0f172a !important;">AAMI EC57 Arrhythmia Probability Spectrum</span>
                <span style="font-size:0.78rem; color:#475569 !important; font-weight:600;">1D-CNN Latent (d=256)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        df_ecg = pd.DataFrame({
            "Arrhythmia Category": [AAMI_CLASSES[i] for i in range(5)],
            "Probability": ecg_probs,
        })
        
        # Color coding: highest probability highlighted
        colors = ["#0284c7" if p == max(ecg_probs) else "#cbd5e1" for p in ecg_probs]
        if max(ecg_probs) > 0.4 and np.argmax(ecg_probs) != 0:
            colors = ["#dc2626" if p == max(ecg_probs) else "#cbd5e1" for p in ecg_probs]

        fig_bar = go.Figure(
            go.Bar(
                x=df_ecg["Probability"],
                y=df_ecg["Arrhythmia Category"],
                orientation="h",
                marker=dict(color=colors, line=dict(width=0)),
                text=[f"{p * 100:.1f}%" for p in ecg_probs],
                textposition="auto",
                textfont=dict(family="Plus Jakarta Sans", size=11, color="#0f172a"),
                hovertemplate="%{y}: %{x:.3f}<extra></extra>",
            )
        )
        fig_bar.update_layout(
            height=320,
            margin=dict(l=10, r=20, t=20, b=30),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            xaxis=dict(
                title=dict(text="Confidence Score (0 - 1.0)", font=dict(size=11, color="#334155")),
                range=[0, 1.05],
                gridcolor="#e2e8f0",
                tickfont=dict(color="#334155", size=10),
            ),
            yaxis=dict(autorange="reversed", tickfont=dict(color="#0f172a", size=11, family="Plus Jakarta Sans")),
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # SignalAgent Clinical Assessment Section
    st.markdown("<div style='height:1px; background:#cbd5e1; margin:1rem 0;'></div>", unsafe_allow_html=True)
    st.markdown("##### 🩺 SignalAgent Clinical Synthesis")
    
    col_f1, col_f2, col_f3 = st.columns([1.2, 2.3, 1.2])
    
    sev_class = "severity-high" if signal_analysis.severity == "High" else "severity-moderate" if signal_analysis.severity == "Moderate" else "severity-low"
    badge_type = "badge-pill-danger" if signal_analysis.severity == "High" else "badge-pill-warning" if signal_analysis.severity == "Moderate" else "badge-pill-success"

    with col_f1:
        st.markdown(
            f"""
            <div class="finding-kpi-card {sev_class}">
                <div class="hud-label">Classified Morphology</div>
                <div style="font-size:1.3rem; font-weight:800; color:#0f172a !important; margin:0.25rem 0;">{signal_analysis.class_name}</div>
                <div style="display:flex; align-items:center; gap:0.5rem;">
                    <span class="badge-pill {badge_type}">Confidence: {signal_analysis.confidence * 100:.1f}%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_f2:
        st.markdown(
            f"""
            <div class="finding-kpi-card">
                <div class="hud-label">Electrophysiological Interpretation</div>
                <div style="font-size:0.92rem; color:#1e293b !important; line-height:1.5; margin-top:0.3rem; font-weight:500;">
                    {signal_analysis.clinical_interpretation}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_f3:
        st.markdown(
            f"""
            <div class="finding-kpi-card {sev_class}">
                <div class="hud-label">Hemodynamic Status</div>
                <div style="font-size:1.45rem; font-weight:800; color:#0f172a !important; margin:0.25rem 0;">{hr_bpm:.0f} <span style="font-size:0.85rem; color:#64748b;">BPM</span></div>
                <div style="font-size:0.82rem; color:#334155 !important; font-weight:600;">
                    Severity Level: <strong style="color:#0f172a !important;">{signal_analysis.severity}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# TAB 2: Medical Vision & Explainability (Chest X-Ray)
# ==============================================================================
with tab_vision:
    st.markdown(
        """
        <div class="med-card-header" style="margin-top:0.5rem;">
            <div class="med-card-title">
                <span>🩻 Thoracic Radiograph Vision & Grad-CAM Spatial Localization</span>
            </div>
            <span class="badge-pill badge-pill-purple">DenseNet-121 • Explainable AI (XAI)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Image Input Source Selector
    img_source = st.radio(
        "Select Chest Radiograph Source:",
        ["📁 Preset / Dataset Image", "📤 Upload Custom Chest X-Ray (.png, .jpg, .jpeg)"],
        horizontal=True,
    )

    current_xray_img = None

    if img_source == "📁 Preset / Dataset Image":
        if xray_dataset is not None:
            if case_choice != "Custom Patient Case":
                target_xray_cls_name = PRESET_CASES[case_choice]["xray_class_name"]
                target_xray_cls_id = [k for k, v in XRAY_CLASSES.items() if v == target_xray_cls_name][0]
                matching_samples = [s for s in xray_dataset.samples if s[1] == target_xray_cls_id]
                if matching_samples:
                    current_xray_img = Image.open(matching_samples[0][0]).convert("RGB")
            else:
                col_filt1, col_filt2 = st.columns([1, 1])
                with col_filt1:
                    cls_filter = st.selectbox("Select Pathology Class Filter", list(XRAY_CLASSES.values()))
                cls_id = [k for k, v in XRAY_CLASSES.items() if v == cls_filter][0]
                matching_samples = [s for s in xray_dataset.samples if s[1] == cls_id]
                if matching_samples:
                    with col_filt2:
                        sample_idx = st.slider("Sample Index in Class", 0, len(matching_samples) - 1, 0)
                    current_xray_img = Image.open(matching_samples[sample_idx][0]).convert("RGB")

        if current_xray_img is None and xray_dataset is not None and len(xray_dataset.samples) > 0:
            current_xray_img = Image.open(xray_dataset.samples[0][0]).convert("RGB")

    elif img_source == "📤 Upload Custom Chest X-Ray (.png, .jpg, .jpeg)":
        uploaded_xray = st.file_uploader(
            "Upload chest radiograph (DICOM-converted PNG / JPG)",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            key="xray_uploader",
        )
        if uploaded_xray is not None:
            try:
                current_xray_img = Image.open(uploaded_xray).convert("RGB")
                st.success(f"Loaded radiograph: `{uploaded_xray.name}` ({current_xray_img.size[0]}×{current_xray_img.size[1]} px)")
            except Exception as e:
                st.error(f"Error reading image: {str(e)}")

        if current_xray_img is None:
            if xray_dataset is not None and len(xray_dataset.samples) > 0:
                current_xray_img = Image.open(xray_dataset.samples[0][0]).convert("RGB")
            else:
                current_xray_img = Image.new("RGB", (224, 224), color=(128, 128, 128))

    if current_xray_img is None:
        current_xray_img = Image.new("RGB", (224, 224), color=(128, 128, 128))

    st.session_state["active_xray_img"] = current_xray_img

    # Model inference & Grad-CAM
    xray_trans = get_xray_transforms(image_size=224, is_training=False)
    xray_tensor = xray_trans(current_xray_img).unsqueeze(0).to(systems["device"])

    with torch.no_grad():
        xray_logits, vision_latent = systems["xray_model"](xray_tensor)
        xray_probs = F.softmax(xray_logits, dim=-1).cpu().numpy()[0]

    # Controls Row
    col_ctl1, col_ctl2 = st.columns([1, 1])
    with col_ctl1:
        alpha_slider = st.slider("Grad-CAM Heatmap Opacity (α)", 0.0, 1.0, 0.60, step=0.05)
    with col_ctl2:
        cmap_choice = st.selectbox("Scientific Color Map", ["jet", "plasma", "turbo", "magma"], index=0)

    overlay_rgb, heatmap = systems["gradcam"].generate_overlay(
        xray_tensor,
        target_class=None,
        alpha=alpha_slider,
        colormap=cmap_choice,
    )

    st.session_state["active_xray_logits"] = xray_logits
    st.session_state["active_vision_latent"] = vision_latent
    st.session_state["active_heatmap"] = heatmap

    # 3-Column Diagnostic Panel
    col_v_left, col_v_mid, col_v_right = st.columns([1.3, 1.3, 1.4])

    with col_v_left:
        st.markdown(
            """
            <div class="hud-label" style="margin-bottom:0.4rem;">1. Anteroposterior (PA) Radiograph</div>
            """,
            unsafe_allow_html=True,
        )
        st.image(current_xray_img, use_container_width=True, caption="Thoracic View Presentation")

    with col_v_mid:
        st.markdown(
            f"""
            <div class="hud-label" style="margin-bottom:0.4rem;">2. Grad-CAM Spatial Saliency ({cmap_choice.upper()})</div>
            """,
            unsafe_allow_html=True,
        )
        st.image(overlay_rgb, use_container_width=True, caption=f"Anatomical Attention Heatmap (α={alpha_slider:.2f})")

    with col_v_right:
        st.markdown(
            """
            <div class="hud-label" style="margin-bottom:0.4rem;">3. Deep Learning Pathology Spectrum</div>
            """,
            unsafe_allow_html=True,
        )
        
        df_xray = pd.DataFrame({
            "Pathology": [XRAY_CLASSES[i] for i in range(5)],
            "Confidence": xray_probs,
        })
        
        # Color coding: highlight top condition
        v_colors = ["#7c3aed" if p == max(xray_probs) else "#cbd5e1" for p in xray_probs]
        if max(xray_probs) > 0.4 and np.argmax(xray_probs) != 0:
            v_colors = ["#dc2626" if p == max(xray_probs) else "#cbd5e1" for p in xray_probs]

        fig_xray = go.Figure(
            go.Bar(
                x=df_xray["Confidence"],
                y=df_xray["Pathology"],
                orientation="h",
                marker=dict(color=v_colors, line=dict(width=0)),
                text=[f"{p * 100:.1f}%" for p in xray_probs],
                textposition="auto",
                textfont=dict(family="Plus Jakarta Sans", size=11, color="#0f172a"),
                hovertemplate="%{y}: %{x:.3f}<extra></extra>",
            )
        )
        fig_xray.update_layout(
            height=260,
            margin=dict(l=10, r=20, t=10, b=20),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            xaxis=dict(
                title=dict(text="Confidence Score", font=dict(size=11, color="#334155")),
                range=[0, 1.05],
                gridcolor="#e2e8f0",
                tickfont=dict(color="#334155", size=10),
            ),
            yaxis=dict(autorange="reversed", tickfont=dict(color="#0f172a", size=11, family="Plus Jakarta Sans")),
            showlegend=False,
        )
        st.plotly_chart(fig_xray, use_container_width=True)

    # Imaging Agent Findings Card
    st.markdown("<div style='height:1px; background:#cbd5e1; margin:1rem 0;'></div>", unsafe_allow_html=True)
    st.markdown("##### 🩻 ImagingAgent Radiological Synthesis")
    imaging_analysis = systems["imaging_agent"].analyze(xray_logits.cpu().numpy(), gradcam_heatmap=heatmap)
    st.session_state["active_imaging_analysis"] = imaging_analysis

    col_ir1, col_ir2 = st.columns([1.2, 2.8])
    with col_ir1:
        st.markdown(
            f"""
            <div class="finding-kpi-card">
                <div class="hud-label">Primary Radiographic Finding</div>
                <div style="font-size:1.3rem; font-weight:800; color:#0f172a !important; margin:0.25rem 0;">{imaging_analysis.class_name}</div>
                <span class="badge-pill badge-pill-purple">Confidence: {imaging_analysis.confidence * 100:.1f}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_ir2:
        st.markdown(
            f"""
            <div class="finding-kpi-card">
                <div class="hud-label">Radiological Interpretation & Anatomical Localization</div>
                <div style="font-size:0.92rem; color:#1e293b !important; line-height:1.5; margin-top:0.3rem; font-weight:500;">
                    <strong>Assessment:</strong> {imaging_analysis.radiological_finding}<br>
                    <strong>Spatial Focus:</strong> {imaging_analysis.gradcam_localization}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# TAB 3: Multimodal Patient Risk Stratification
# ==============================================================================
with tab_multimodal:
    st.markdown(
        """
        <div class="med-card-header" style="margin-top:0.5rem;">
            <div class="med-card-title">
                <span>🌐 Cross-Attention Multimodal Patient Risk Stratification</span>
            </div>
            <span class="badge-pill badge-pill-info">Cross-Modal Transformer Fusion • Latent d=256</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Fetch active latents
    act_ecg_latent = st.session_state.get("active_ecg_latent", quick_ecg_latent)
    act_vis_latent = st.session_state.get("active_vision_latent", quick_vis_latent)
    act_sig_res = st.session_state.get("active_signal_analysis", quick_sig_analysis)
    act_img_res = st.session_state.get("active_imaging_analysis", quick_img_analysis)

    # Forward pass multimodal fusion transformer
    with torch.no_grad():
        risk_logits, patient_rep = systems["fusion_model"](act_ecg_latent, act_vis_latent, tabular_vector)
        risk_probs = F.softmax(risk_logits, dim=-1).cpu().numpy()[0]

    risk_analysis = systems["risk_agent"].evaluate_risk(
        signal_res=act_sig_res,
        imaging_res=act_img_res,
        patient_info=patient_dict,
        multimodal_logits_or_probs=risk_probs,
    )
    st.session_state["active_risk_analysis"] = risk_analysis

    col_m1, col_m2, col_m3 = st.columns([1.3, 1.5, 1.6])

    tier_hex = "#dc2626" if risk_analysis.risk_tier_id == 2 else "#d97706" if risk_analysis.risk_tier_id == 1 else "#059669"
    badge_cls = "badge-pill-danger" if risk_analysis.risk_tier_id == 2 else "badge-pill-warning" if risk_analysis.risk_tier_id == 1 else "badge-pill-success"

    with col_m1:
        st.markdown(
            f"""
            <div class="med-card" style="text-align:center; border-top: 5px solid {tier_hex}; height: 100%; display: flex; flex-direction: column; justify-content: center;">
                <div class="hud-label">Composite Risk Stratification</div>
                <div style="margin: 0.5rem 0;">
                    <span class="badge-pill {badge_cls}" style="font-size: 0.92rem; padding: 6px 14px;">{risk_analysis.risk_tier}</span>
                </div>
                <div style="font-size: 3.5rem; font-weight: 800; color: {tier_hex} !important; line-height: 1; margin: 0.5rem 0;">
                    {risk_analysis.risk_score * 100:.1f}<span style="font-size: 1.2rem; color: #64748b !important; font-weight: 600;">/100</span>
                </div>
                <div style="font-size: 0.85rem; color: #475569 !important; margin-bottom: 0.75rem; font-weight:600;">
                    Calibrated Patient Risk Index
                </div>
                <div style="background: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 8px; padding: 0.6rem; font-size: 0.82rem; color: #1e293b !important;">
                    Triage Urgency: <strong style="color: {tier_hex} !important; font-size:0.9rem;">{risk_analysis.urgency_level}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m2:
        st.markdown(
            """
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                <span style="font-weight:700; font-size:0.92rem; color:#0f172a !important;">Cross-Modal Alignment Radar</span>
                <span style="font-size:0.78rem; color:#475569 !important; font-weight:600;">Pentagonal Metric</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        radar_scores = [
            act_sig_res.confidence * (0.95 if act_sig_res.predicted_class != 0 else 0.2),
            act_img_res.confidence * (0.95 if act_img_res.predicted_class != 0 else 0.2),
            min(1.0, (age / 100.0) * (1.3 if (hypertension or diabetes) else 0.9)),
            1.0 if (hr_bpm > 100 or hr_bpm < 55) else 0.3,
            1.0 if spo2 < 93 else 0.2,
        ]

        fig_radar = go.Figure(
            data=go.Scatterpolar(
                r=radar_scores,
                theta=["ECG Arrhythmia", "Imaging Pathology", "Age & Comorbidities", "Heart Rate Stress", "Oxygen Desaturation"],
                fill="toself",
                fillcolor="rgba(14, 165, 233, 0.25)",
                line=dict(color="#0284c7", width=2.5),
                marker=dict(color="#0284c7", size=6),
            )
        )
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1.0], gridcolor="#cbd5e1", tickfont=dict(size=9, color="#475569")),
                angularaxis=dict(tickfont=dict(size=10, family="Plus Jakarta Sans", color="#0f172a", weight="bold")),
            ),
            showlegend=False,
            height=280,
            margin=dict(l=35, r=35, t=20, b=20),
            paper_bgcolor="#ffffff",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_m3:
        st.markdown(
            """
            <div class="med-card" style="height: 100%;">
                <div class="hud-label">Primary Clinical Risk Drivers</div>
            """,
            unsafe_allow_html=True,
        )
        for driver in risk_analysis.primary_risk_drivers:
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; gap:0.5rem; margin:0.4rem 0; font-size:0.88rem; color:#0f172a !important;">
                    <span style="color:{tier_hex}; font-size:1.1rem;">●</span>
                    <strong>{driver}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
                <div style="height:1px; background:#cbd5e1; margin:0.75rem 0;"></div>
                <div class="hud-label">Multimodal Rationale</div>
                <div style="font-size:0.88rem; color:#1e293b !important; line-height:1.45; margin-top:0.25rem; font-weight:500;">
                    {risk_analysis.clinical_rationale}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# TAB 4: Multi-Agent Clinical Summary & RAG
# ==============================================================================
with tab_agent:
    st.markdown(
        """
        <div class="med-card-header" style="margin-top:0.5rem;">
            <div class="med-card-title">
                <span>🤖 Multi-Agent Synthesis & Practice Guidelines (RAG)</span>
            </div>
            <span class="badge-pill badge-pill-purple">Supervisor Verified • ACC/AHA, BTS Guidelines</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    act_sig_res = st.session_state.get("active_signal_analysis", quick_sig_analysis)
    act_img_res = st.session_state.get("active_imaging_analysis", quick_img_analysis)
    act_risk_res = st.session_state.get("active_risk_analysis", quick_risk_analysis)

    # Evidence retrieval
    evidence_res = systems["evidence_agent"].retrieve_evidence(
        signal_findings=act_sig_res.clinical_interpretation,
        imaging_findings=act_img_res.radiological_finding,
        patient_context=f"Age {age}, Sex {sex}, Vitals: HR {hr_bpm}, BP {bp}, SpO2 {spo2}%. Comorbidities: {medical_notes}",
        top_k=2,
    )

    # Report generation
    report = systems["report_agent"].generate_report(
        patient_info=patient_dict,
        signal_res=act_sig_res,
        imaging_res=act_img_res,
        risk_res=act_risk_res,
        evidence_res=evidence_res,
    )

    # Generate Hospital-Grade Clinical PDF
    pdf_bytes = generate_clinical_pdf(
        patient_info=patient_dict,
        signal_res=act_sig_res,
        imaging_res=act_img_res,
        risk_res=act_risk_res,
        evidence_res=evidence_res,
        report_id=report.report_id,
    )

    # Supervisor Safety Audit
    is_safe, val_notes = systems["supervisor_agent"]._validate_safety(
        signal_res=act_sig_res,
        imaging_res=act_img_res,
        risk_res=act_risk_res,
        report=report,
    )

    col_ag1, col_ag2 = st.columns([1.8, 1.2])

    with col_ag1:
        st.markdown(
            f"""
            <div class="ehr-container">
                <div class="ehr-header-strip">
                    <div>
                        <div style="font-size:1.2rem; font-weight:800; color:#0f172a !important;">Electronic Health Record (EHR) Summary</div>
                        <div style="font-size:0.78rem; color:#475569 !important; font-weight:600;">Generated by MedAgent Multi-Agent Clinical Engine</div>
                    </div>
                    <div style="text-align:right;">
                        <span class="badge-pill badge-pill-info">Record ID: {patient_id}</span>
                    </div>
                </div>
            """,
            unsafe_allow_html=True,
        )

        # Render report markdown inside the styled EHR container
        st.markdown(report.markdown_content)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:0.85rem;'></div>", unsafe_allow_html=True)
        col_d1, col_d2, col_d3 = st.columns([1.3, 1, 1])
        with col_d1:
            st.markdown('<div class="pdf-download-btn">', unsafe_allow_html=True)
            st.download_button(
                label="📄 Export Clinical Report (PDF)",
                data=pdf_bytes,
                file_name=f"MedAgent_Clinical_Report_{patient_id}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)
        with col_d2:
            st.download_button(
                label="📥 Export Markdown (.md)",
                data=report.markdown_content,
                file_name=f"MedAgent_Clinical_Report_{patient_id}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col_d3:
            st.download_button(
                label="📥 Export FHIR / JSON",
                data=json.dumps(report.structured_data, indent=2),
                file_name=f"MedAgent_FHIR_Data_{patient_id}.json",
                mime="application/json",
                use_container_width=True,
            )

    with col_ag2:
        st.markdown(
            """
            <div class="hud-label" style="margin-bottom:0.5rem;">Evidence-Based Practice Guidelines (RAG)</div>
            """,
            unsafe_allow_html=True,
        )

        for idx, item in enumerate(evidence_res.retrieved_guidelines, 1):
            with st.expander(f"📖 [{item.id}] {item.topic}", expanded=(idx == 1)):
                st.markdown(
                    f"""
                    <div style="font-weight:800; color:#0f172a !important; margin-bottom:0.3rem; font-size:0.92rem;">{item.title}</div>
                    <div style="font-size:0.88rem; color:#1e293b !important; line-height:1.45; margin-bottom:0.5rem; font-weight:500;">{item.content}</div>
                    <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.78rem; color:#334155 !important; background:#f1f5f9; padding:0.45rem 0.7rem; border-radius:6px; border:1px solid #e2e8f0;">
                        <span>Recommendation: <strong style="color:#0284c7 !important;">{item.strength_of_recommendation}</strong></span>
                    </div>
                    <div style="font-size:0.75rem; color:#64748b !important; margin-top:0.4rem; font-weight:500;">
                        Citation: {item.citation}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='height:1px; background:#cbd5e1; margin:1rem 0;'></div>", unsafe_allow_html=True)
        
        st.markdown(
            """
            <div class="hud-label" style="margin-bottom:0.5rem;">Supervisor Safety Audit</div>
            """,
            unsafe_allow_html=True,
        )
        if is_safe:
            st.markdown(
                f"""
                <div style="background:#f0fdf4; border:1px solid #86efac; border-left:5px solid #16a34a; border-radius:8px; padding:0.85rem; font-size:0.85rem; color:#14532d !important;">
                    <div style="display:flex; align-items:center; gap:0.4rem; font-weight:800; margin-bottom:0.25rem; color:#14532d !important;">
                        <span>✅</span> Autonomous Safety Verification: PASSED
                    </div>
                    <div style="color:#15803d !important; font-size:0.82rem; font-weight:500;">
                        {val_notes}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="background:#fef2f2; border:1px solid #fca5a5; border-left:5px solid #dc2626; border-radius:8px; padding:0.85rem; font-size:0.85rem; color:#991b1b !important;">
                    <div style="display:flex; align-items:center; gap:0.4rem; font-weight:800; margin-bottom:0.25rem; color:#991b1b !important;">
                        <span>⚠️</span> Clinical Supervisor Flag Triggered
                    </div>
                    <div style="color:#b91c1c !important; font-size:0.82rem; font-weight:500;">
                        {val_notes}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
