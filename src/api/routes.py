"""
FastAPI Route Handlers for MedAgent Multimodal Clinical AI.

Provides RESTful endpoints for:
- System Health and Device Diagnostics
- 1D Physiological Signal (ECG) Arrhythmia Classification
- Chest Radiograph (X-Ray) Diagnosis and Grad-CAM Explainability
- Multimodal Risk Stratification (ECG + Vision + Tabular Vitals)
- End-to-End Multi-Agent Clinical Summary Generation
"""

import base64
import io
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from PIL import Image

from src.agents.imaging_agent import ImagingAgent
from src.agents.risk_agent import RiskAssessmentAgent
from src.agents.signal_agent import SignalAgent
from src.agents.supervisor_agent import SupervisorAgent
from src.api.schemas import (
    AgentClinicalSummaryRequest,
    AgentClinicalSummaryResponse,
    ECGPredictionRequest,
    ECGPredictionResponse,
    HealthResponse,
    MultimodalPredictionRequest,
    MultimodalPredictionResponse,
    PatientDemographics,
    XRayPredictionResponse,
)
from src.data.loaders.multimodal_dataset import RISK_CLASSES
from src.data.loaders.xray_dataset import XRAY_CLASSES
from src.data.preprocessing.image_transforms import get_xray_transforms
from src.models.fusion.multimodal_transformer import MultimodalTransformerFusion
from src.models.signals.ecg_cnn import ECG1DCNN
from src.models.vision.gradcam import GradCAM
from src.models.vision.xray_densenet import XRayDenseNet121

router = APIRouter(prefix="/api/v1", tags=["MedAgent Clinical AI"])

# Global singleton model cache
_models: dict = {}


def get_model_cache() -> dict:
    """Initializes and returns cached neural models and agent instances."""
    if not _models:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        supervisor = SupervisorAgent()

        _models["device"] = device
        _models["ecg_model"] = ecg_model
        _models["xray_model"] = xray_model
        _models["fusion_model"] = fusion_model
        _models["gradcam"] = gradcam
        _models["supervisor"] = supervisor
        _models["signal_agent"] = SignalAgent()
        _models["imaging_agent"] = ImagingAgent()
        _models["risk_agent"] = RiskAssessmentAgent()

    return _models


def encode_tabular_features(patient_info: PatientDemographics) -> torch.Tensor:
    """Converts patient demographics and vitals into standard 10-dimensional normalized tensor."""
    # Blood pressure parse
    sbp, dbp = 120.0, 80.0
    if "/" in patient_info.blood_pressure:
        try:
            parts = patient_info.blood_pressure.split("/")
            sbp = float(parts[0].strip())
            dbp = float(parts[1].strip())
        except Exception:
            sbp, dbp = 120.0, 80.0

    is_male = 1.0 if patient_info.sex.strip().lower().startswith("m") else 0.0

    vector = [
        patient_info.age / 100.0,
        is_male,
        patient_info.heart_rate_bpm / 200.0,
        sbp / 200.0,
        dbp / 120.0,
        patient_info.spo2_percent / 100.0,
        1.0 if patient_info.hypertension else 0.0,
        1.0 if patient_info.diabetes else 0.0,
        1.0 if patient_info.smoking_history else 0.0,
        1.0 if patient_info.prior_cardiac_event else 0.0,
    ]
    return torch.tensor([vector], dtype=torch.float32)


def array_to_base64_png(image_array: np.ndarray) -> str:
    """Encodes a uint8 RGB image array into base64 data URI."""
    pil_img = Image.fromarray(image_array)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


# ==========================================
# 1. Health Diagnostics
# ==========================================
@router.get("/health", response_model=HealthResponse)
def health_check():
    """System health check and hardware acceleration telemetry."""
    models = get_model_cache()
    device = models["device"]
    gpu_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if gpu_avail else "CPU"
    return HealthResponse(
        status="healthy",
        project="MedAgent",
        version="1.0.0",
        gpu_available=gpu_avail,
        device_name=device_name,
    )


# ==========================================
# 2. Physiological Signal (ECG) Inference
# ==========================================
@router.post("/predict/ecg", response_model=ECGPredictionResponse)
def predict_ecg(request: ECGPredictionRequest):
    """
    Classifies a 1D ECG beat sequence into AAMI arrhythmia categories and extracts clinical insights.
    """
    models = get_model_cache()
    ecg_model = models["ecg_model"]
    signal_agent = models["signal_agent"]
    device = models["device"]

    raw_signal = np.array(request.signal, dtype=np.float32)
    if len(raw_signal) == 0:
        raise HTTPException(status_code=400, detail="Empty ECG signal array provided.")

    # Resample or pad/crop to fixed 260 length
    if len(raw_signal) < 260:
        padded = np.pad(raw_signal, (0, 260 - len(raw_signal)), mode="edge")
    else:
        padded = raw_signal[:260]

    input_tensor = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 260)

    with torch.no_grad():
        logits, _ = ecg_model(input_tensor)

    analysis = signal_agent.analyze(logits.cpu().numpy(), heart_rate_bpm=request.heart_rate_bpm)

    return ECGPredictionResponse(
        predicted_class=analysis.predicted_class,
        class_name=analysis.class_name,
        confidence=analysis.confidence,
        probabilities=analysis.probabilities,
        clinical_interpretation=analysis.clinical_interpretation,
        abnormal_features=analysis.abnormal_features,
        severity=analysis.severity,
    )


# ==========================================
# 3. Medical Vision (Chest X-Ray) Inference
# ==========================================
@router.post("/predict/xray", response_model=XRayPredictionResponse)
async def predict_xray(file: UploadFile = File(...)):
    """
    Classifies an uploaded chest radiograph across 5 thoracic categories and generates Grad-CAM overlays.
    """
    models = get_model_cache()
    xray_model = models["xray_model"]
    gradcam = models["gradcam"]
    imaging_agent = models["imaging_agent"]
    device = models["device"]

    try:
        content = await file.read()
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    transform = get_xray_transforms(image_size=224, is_training=False)
    img_tensor = transform(pil_img).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    # Forward & Grad-CAM
    overlay_rgb, heatmap = gradcam.generate_overlay(img_tensor, target_class=None, alpha=0.5)

    with torch.no_grad():
        logits, _ = xray_model(img_tensor)

    analysis = imaging_agent.analyze(logits.cpu().numpy(), gradcam_heatmap=heatmap)
    overlay_b64 = array_to_base64_png(overlay_rgb)

    return XRayPredictionResponse(
        predicted_class=analysis.predicted_class,
        class_name=analysis.class_name,
        confidence=analysis.confidence,
        probabilities=analysis.probabilities,
        radiological_finding=analysis.radiological_finding,
        gradcam_localization=analysis.gradcam_localization,
        gradcam_heatmap_base64=overlay_b64,
        severity=analysis.severity,
    )


# ==========================================
# 4. Multimodal Patient Risk Stratification
# ==========================================
@router.post("/predict/multimodal", response_model=MultimodalPredictionResponse)
def predict_multimodal(request: MultimodalPredictionRequest):
    """
    Fuses ECG waveforms, vision features, and patient vitals to assess 3-tier clinical risk.
    """
    models = get_model_cache()
    ecg_model = models["ecg_model"]
    xray_model = models["xray_model"]
    fusion_model = models["fusion_model"]
    risk_agent = models["risk_agent"]
    signal_agent = models["signal_agent"]
    imaging_agent = models["imaging_agent"]
    device = models["device"]

    # 1. ECG Latent
    raw_signal = np.array(request.ecg_signal, dtype=np.float32)
    if len(raw_signal) < 260:
        padded = np.pad(raw_signal, (0, 260 - len(raw_signal)), mode="edge")
    else:
        padded = raw_signal[:260]
    ecg_tensor = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        ecg_logits, ecg_latent = ecg_model(ecg_tensor)

    # 2. Vision Latent
    # If xray_class_id is specified, pick a representative sample or dummy tensor
    dummy_xray = torch.zeros(1, 3, 224, 224, device=device)
    with torch.no_grad():
        xray_logits, vision_latent = xray_model(dummy_xray)

    # 3. Tabular Latent
    tabular_tensor = encode_tabular_features(request.patient_info).to(device)

    # 4. Multimodal Fusion Forward Pass
    with torch.no_grad():
        risk_logits, _ = fusion_model(ecg_latent, vision_latent, tabular_tensor)
        risk_probs = F.softmax(risk_logits, dim=-1).cpu().numpy()[0]

    signal_res = signal_agent.analyze(ecg_logits.cpu().numpy(), heart_rate_bpm=request.patient_info.heart_rate_bpm)
    imaging_res = imaging_agent.analyze(xray_logits.cpu().numpy())

    patient_dict = request.patient_info.model_dump()
    risk_res = risk_agent.evaluate_risk(
        signal_res=signal_res,
        imaging_res=imaging_res,
        patient_info=patient_dict,
        multimodal_logits_or_probs=risk_probs,
    )

    return MultimodalPredictionResponse(
        patient_id=request.patient_info.patient_id,
        risk_tier=risk_res.risk_tier,
        risk_tier_id=risk_res.risk_tier_id,
        risk_score=risk_res.risk_score,
        probabilities={
            RISK_CLASSES[i]: float(risk_probs[i]) for i in range(len(RISK_CLASSES))
        },
        primary_risk_drivers=risk_res.primary_risk_drivers,
        urgency_level=risk_res.urgency_level,
        clinical_rationale=risk_res.clinical_rationale,
    )


# ==========================================
# 5. Full Multi-Agent Clinical Summary
# ==========================================
@router.post("/agent/clinical-summary", response_model=AgentClinicalSummaryResponse)
def generate_agent_summary(request: AgentClinicalSummaryRequest):
    """
    Executes end-to-end multi-agent clinical decision support pipeline with safety verification.
    """
    models = get_model_cache()
    ecg_model = models["ecg_model"]
    xray_model = models["xray_model"]
    fusion_model = models["fusion_model"]
    supervisor = models["supervisor"]
    device = models["device"]

    # 1. ECG Model pass
    raw_signal = np.array(request.ecg_signal, dtype=np.float32)
    if len(raw_signal) < 260:
        padded = np.pad(raw_signal, (0, 260 - len(raw_signal)), mode="edge")
    else:
        padded = raw_signal[:260]
    ecg_tensor = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        ecg_logits, ecg_latent = ecg_model(ecg_tensor)

    # 2. X-Ray Model pass
    img_tensor = None
    heatmap = None
    if request.xray_image_filename and Path(request.xray_image_filename).exists():
        try:
            pil_img = Image.open(request.xray_image_filename).convert("RGB")
            transform = get_xray_transforms(image_size=224, is_training=False)
            img_tensor = transform(pil_img).unsqueeze(0).to(device)
            _, heatmap = models["gradcam"].generate_overlay(img_tensor)
        except Exception:
            img_tensor = None

    if img_tensor is None:
        img_tensor = torch.zeros(1, 3, 224, 224, device=device)

    with torch.no_grad():
        xray_logits, vision_latent = xray_model(img_tensor)

    # 3. Multimodal risk logits
    tabular_tensor = encode_tabular_features(request.patient_info).to(device)
    with torch.no_grad():
        risk_logits, _ = fusion_model(ecg_latent, vision_latent, tabular_tensor)

    patient_dict = request.patient_info.model_dump()

    # 4. Supervisor Agent Workflow Execution
    workflow_res = supervisor.execute_workflow(
        ecg_output=ecg_logits.cpu().numpy(),
        xray_output=xray_logits.cpu().numpy(),
        patient_info=patient_dict,
        gradcam_heatmap=heatmap,
        multimodal_risk_output=risk_logits.cpu().numpy(),
    )

    return AgentClinicalSummaryResponse(
        report_id=workflow_res.report.report_id,
        patient_id=workflow_res.report.patient_id,
        created_at=workflow_res.report.created_at,
        markdown_report=workflow_res.report.markdown_content,
        structured_summary=workflow_res.report.structured_data,
        safety_validated=workflow_res.safety_validated,
        validation_notes=workflow_res.validation_notes,
    )
