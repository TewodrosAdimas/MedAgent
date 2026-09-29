"""
Unit Tests for MedAgent FastAPI REST Service.
"""

import io
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "endpoints" in data


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["project"] == "MedAgent"
    assert "gpu_available" in data
    assert "device_name" in data


def test_predict_ecg_endpoint():
    sample_signal = (np.sin(np.linspace(0, 4 * np.pi, 260)) * 0.8).tolist()
    payload = {
        "signal": sample_signal,
        "sampling_rate": 360.0,
        "heart_rate_bpm": 82.0,
    }
    response = client.post("/api/v1/predict/ecg", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert "class_name" in data
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0
    assert "clinical_interpretation" in data
    assert "probabilities" in data
    assert len(data["probabilities"]) == 5


def test_predict_xray_endpoint():
    # Create test image in memory
    img = Image.new("RGB", (224, 224), color=(120, 120, 120))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    files = {"file": ("test_xray.png", buf, "image/png")}
    response = client.post("/api/v1/predict/xray", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert "class_name" in data
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0
    assert "radiological_finding" in data
    assert "gradcam_heatmap_base64" in data
    assert data["gradcam_heatmap_base64"].startswith("data:image/png;base64,")


def test_predict_multimodal_endpoint():
    sample_signal = (np.cos(np.linspace(0, 4 * np.pi, 260)) * 0.5).tolist()
    payload = {
        "patient_info": {
            "patient_id": "TEST-PT-001",
            "age": 68,
            "sex": "Male",
            "heart_rate_bpm": 88.0,
            "blood_pressure": "142/90",
            "spo2_percent": 94.0,
            "hypertension": True,
            "diabetes": True,
            "smoking_history": False,
            "prior_cardiac_event": True,
        },
        "ecg_signal": sample_signal,
    }
    response = client.post("/api/v1/predict/multimodal", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["patient_id"] == "TEST-PT-001"
    assert "risk_tier" in data
    assert "risk_score" in data
    assert "primary_risk_drivers" in data
    assert len(data["primary_risk_drivers"]) > 0
    assert "clinical_rationale" in data


def test_agent_clinical_summary_endpoint():
    sample_signal = (np.sin(np.linspace(0, 2 * np.pi, 260))).tolist()
    payload = {
        "patient_info": {
            "patient_id": "TEST-PT-002",
            "age": 72,
            "sex": "Female",
            "heart_rate_bpm": 95.0,
            "blood_pressure": "150/95",
            "spo2_percent": 91.0,
            "hypertension": True,
            "diabetes": False,
            "smoking_history": True,
            "prior_cardiac_event": False,
        },
        "ecg_signal": sample_signal,
    }
    response = client.post("/api/v1/agent/clinical-summary", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "report_id" in data
    assert data["patient_id"] == "TEST-PT-002"
    assert "markdown_report" in data
    assert "CLINICAL SAFETY DISCLAIMER" in data["markdown_report"]
    assert data["safety_validated"] is True
    assert "structured_summary" in data
