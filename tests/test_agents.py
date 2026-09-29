"""
Unit Tests for MedAgent Clinical Multi-Agent Framework and Medical RAG.
"""

import numpy as np
import pytest
import torch

from src.llm.rag.knowledge_base import GUIDELINE_CORPUS
from src.llm.rag.retriever import MedicalKnowledgeRetriever
from src.agents.signal_agent import SignalAgent
from src.agents.imaging_agent import ImagingAgent
from src.agents.evidence_agent import EvidenceAgent
from src.agents.risk_agent import RiskAssessmentAgent
from src.agents.report_agent import ReportAgent
from src.agents.supervisor_agent import SupervisorAgent


def test_knowledge_base_and_retriever():
    retriever = MedicalKnowledgeRetriever(corpus=GUIDELINE_CORPUS)
    assert len(retriever.corpus) > 0

    # Query for arrhythmia
    matches = retriever.retrieve("premature ventricular contraction arrhythmia", top_k=2)
    assert len(matches) > 0
    top_guideline, score = matches[0]
    assert score > 0.0
    assert "AHA" in top_guideline.id or "Ventricular" in top_guideline.topic

    # Format text
    formatted = retriever.format_evidence_for_prompt(matches)
    assert "Recommendation Level:" in formatted


def test_signal_agent_analysis():
    agent = SignalAgent()

    # Normal ECG
    normal_probs = torch.tensor([0.95, 0.02, 0.01, 0.01, 0.01])
    res_normal = agent.analyze(normal_probs, heart_rate_bpm=72)
    assert res_normal.predicted_class == 0
    assert res_normal.severity == "Normal"
    assert res_normal.confidence > 0.9

    # Ventricular Ectopic ECG with Tachycardia
    vent_probs = torch.tensor([0.05, 0.05, 0.85, 0.03, 0.02])
    res_vent = agent.analyze(vent_probs, heart_rate_bpm=115)
    assert res_vent.predicted_class == 2
    assert res_vent.severity == "Severe / Urgent"
    assert any("Tachycardia" in feat for feat in res_vent.abnormal_features)


def test_imaging_agent_analysis():
    agent = ImagingAgent()

    # Cardiomegaly with Grad-CAM heatmap
    probs = torch.tensor([0.05, 0.05, 0.82, 0.04, 0.04])
    dummy_heatmap = np.zeros((224, 224), dtype=np.float32)
    dummy_heatmap[100:150, 100:150] = 0.9  # Central activation

    res = agent.analyze(probs, gradcam_heatmap=dummy_heatmap)
    assert res.predicted_class == 2
    assert "Cardiomegaly" in res.class_name
    assert res.severity == "Severe"
    assert "central" in res.gradcam_localization or "thoracic" in res.gradcam_localization


def test_risk_assessment_agent():
    agent = RiskAssessmentAgent()
    signal_agent = SignalAgent()
    imaging_agent = ImagingAgent()

    sig_res = signal_agent.analyze(torch.tensor([0.02, 0.03, 0.90, 0.03, 0.02]))
    img_res = imaging_agent.analyze(torch.tensor([0.05, 0.10, 0.75, 0.05, 0.05]))

    patient_info = {
        "patient_id": "PT-TEST-001",
        "age": 72,
        "hypertension": True,
        "prior_cardiac_event": True,
        "heart_rate_bpm": 105,
        "spo2_percent": 91,
    }

    risk_res = agent.evaluate_risk(sig_res, img_res, patient_info)
    assert risk_res.risk_tier == "High / Critical Risk"
    assert risk_res.risk_score > 0.60
    assert len(risk_res.primary_risk_drivers) >= 3


def test_supervisor_agent_end_to_end():
    supervisor = SupervisorAgent()

    ecg_logits = torch.tensor([[0.05, 0.05, 0.85, 0.03, 0.02]])
    xray_logits = torch.tensor([[0.05, 0.10, 0.80, 0.03, 0.02]])
    patient_info = {
        "patient_id": "PT-E2E-999",
        "age": 66,
        "sex": "Female",
        "heart_rate_bpm": 98,
        "blood_pressure": "145/90",
        "spo2_percent": 94,
        "hypertension": True,
    }

    result = supervisor.execute_workflow(
        ecg_output=ecg_logits,
        xray_output=xray_logits,
        patient_info=patient_info,
    )

    assert result.safety_validated is True
    assert "MEDAGENT CLINICAL DECISION SUPPORT REPORT" in result.report.markdown_content
    assert "CLINICAL SAFETY DISCLAIMER" in result.report.markdown_content
    assert result.report.structured_data["patient_id"] == "PT-E2E-999"
    assert len(result.evidence_result.citations) > 0
