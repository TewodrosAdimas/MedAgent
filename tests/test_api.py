"""
Unit tests for MedAgent FastAPI REST Clinical Service.
"""

from fastapi.testclient import TestClient
import pytest

from src.api.server import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data


def test_analyze_case_endpoint():
    payload = {
        "patient_id": "PT-API-TEST",
        "age": 60,
        "sex": "Female",
        "sbp": 130.0,
        "dbp": 80.0,
        "heart_rate": 75.0,
        "spo2": 98.0,
        "has_hypertension": 1,
        "has_diabetes": 0,
        "medical_history": "Hypertension routine follow-up.",
        "xray_pathology_preset": "Normal",
    }
    response = client.post("/analyze/case", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["patient_id"] == "PT-API-TEST"
    assert "risk_level" in data
    assert "markdown_report" in data
    assert "evidence_citations" in data
