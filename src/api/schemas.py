"""
Pydantic API Schemas for MedAgent REST Service.

Defines input request models and output response structures for:
- Single-lead ECG Inference
- Chest X-Ray Inference & Grad-CAM Heatmaps
- Multimodal Patient Risk Prediction
- Multi-Agent Clinical Decision Support Summaries
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ==========================================
# Health & Diagnostics
# ==========================================
class HealthResponse(BaseModel):
    status: str
    project: str
    version: str
    gpu_available: bool
    device_name: str


# ==========================================
# ECG Predictions
# ==========================================
class ECGPredictionRequest(BaseModel):
    signal: List[float] = Field(..., description="Raw or preprocessed ECG signal array (e.g. 260 samples)")
    sampling_rate: float = Field(360.0, description="Sampling rate in Hz")
    heart_rate_bpm: Optional[float] = Field(None, description="Patient resting heart rate in BPM")


class ECGPredictionResponse(BaseModel):
    predicted_class: int
    class_name: str
    confidence: float
    probabilities: Dict[str, float]
    clinical_interpretation: str
    abnormal_features: List[str]
    severity: str


# ==========================================
# Chest X-Ray Predictions
# ==========================================
class XRayPredictionResponse(BaseModel):
    predicted_class: int
    class_name: str
    confidence: float
    probabilities: Dict[str, float]
    radiological_finding: str
    gradcam_localization: str
    gradcam_heatmap_base64: Optional[str] = None
    severity: str


# ==========================================
# Multimodal Predictions & Patient Case
# ==========================================
class PatientDemographics(BaseModel):
    patient_id: str = "PT-001"
    age: int = 65
    sex: str = "Male"
    heart_rate_bpm: float = 78.0
    blood_pressure: str = "130/85"
    spo2_percent: float = 96.0
    hypertension: bool = False
    diabetes: bool = False
    smoking_history: bool = False
    prior_cardiac_event: bool = False


class MultimodalPredictionRequest(BaseModel):
    patient_info: PatientDemographics
    ecg_signal: List[float]
    xray_class_id: Optional[int] = None  # Optional pre-classified X-ray label or features


class MultimodalPredictionResponse(BaseModel):
    patient_id: str
    risk_tier: str
    risk_tier_id: int
    risk_score: float
    probabilities: Dict[str, float]
    primary_risk_drivers: List[str]
    urgency_level: str
    clinical_rationale: str


# ==========================================
# Agentic Clinical Decision Support
# ==========================================
class AgentClinicalSummaryRequest(BaseModel):
    patient_info: PatientDemographics
    ecg_signal: List[float]
    xray_image_filename: Optional[str] = None


class AgentClinicalSummaryResponse(BaseModel):
    report_id: str
    patient_id: str
    created_at: str
    markdown_report: str
    structured_summary: Dict[str, Any]
    safety_validated: bool
    validation_notes: str
