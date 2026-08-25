"""
FastAPI REST Service for MedAgent.
Exposes endpoints for multimodal patient case diagnostic inference,
Grad-CAM explainability, and multi-agent clinical reporting.
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import torch

from src.agents.supervisor_agent import SupervisorAgent
from src.data.loaders.image_loader import generate_synthetic_xray_image

from contextlib import asynccontextmanager

# Global supervisor agent instance
supervisor: Optional[SupervisorAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global supervisor
    supervisor = SupervisorAgent(load_checkpoints=True)
    yield


app = FastAPI(
    title="MedAgent Clinical Decision Support API",
    description="Multimodal Medical AI REST API integrating ECG, Chest X-Rays, and Multi-Agent RAG Reasoning",
    version="1.0.0",
    lifespan=lifespan,
)



class PatientCaseRequest(BaseModel):
    patient_id: str = Field("PT-1001", description="Unique Patient Identifier")
    age: int = Field(55, description="Patient age in years")
    sex: str = Field("Male", description="Biological sex (Male/Female)")
    sbp: float = Field(125.0, description="Systolic Blood Pressure")
    dbp: float = Field(82.0, description="Diastolic Blood Pressure")
    heart_rate: float = Field(78.0, description="Resting Heart Rate in BPM")
    spo2: float = Field(97.0, description="Oxygen Saturation SpO2 (%)")
    has_hypertension: int = Field(0, description="1 if hypertensive, else 0")
    has_diabetes: int = Field(0, description="1 if diabetic, else 0")
    medical_history: str = Field("Outpatient diagnostic evaluation.", description="Patient clinical history")
    ecg_signal: Optional[List[float]] = Field(None, description="260-sample 1D ECG waveform array")
    xray_pathology_preset: Optional[str] = Field("Normal", description="X-Ray finding preset (Normal, Pneumonia, Cardiomegaly, Atelectasis, Pleural Effusion)")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "MedAgent Clinical API",
        "device": str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"),
        "cuda_available": torch.cuda.is_available(),
    }


@app.post("/analyze/case")
def analyze_patient_case(request: PatientCaseRequest) -> Dict[str, Any]:
    global supervisor
    if supervisor is None:
        supervisor = SupervisorAgent(load_checkpoints=True)

    # 1. ECG Input
    if request.ecg_signal is not None and len(request.ecg_signal) == 260:
        ecg_arr = np.array(request.ecg_signal, dtype=np.float32)
    else:
        ecg_arr = np.sin(np.linspace(0, 10, 260)).astype(np.float32)

    # 2. X-Ray Input
    xray_img = generate_synthetic_xray_image(pathology=request.xray_pathology_preset)

    # 3. Patient Info
    patient_info = {
        "patient_id": request.patient_id,
        "age": request.age,
        "sex": request.sex,
        "sbp": request.sbp,
        "dbp": request.dbp,
        "heart_rate": request.heart_rate,
        "spo2": request.spo2,
        "has_hypertension": request.has_hypertension,
        "has_diabetes": request.has_diabetes,
        "medical_history": request.medical_history,
    }

    try:
        report = supervisor.run_case(
            patient_info=patient_info,
            ecg_input=ecg_arr,
            xray_input=xray_img,
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
