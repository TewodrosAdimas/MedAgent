"""
Unit tests for MedAgent Multi-Agent Clinical Reasoning & Medical RAG Layer.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from src.agents.evidence_agent import EvidenceAgent
from src.agents.imaging_agent import ImagingAgent
from src.agents.report_agent import ReportAgent
from src.agents.risk_agent import RiskAgent
from src.agents.signal_agent import SignalAgent
from src.agents.supervisor_agent import SupervisorAgent
from src.llm.llm_client import LLMClient
from src.llm.rag.embeddings import MedicalEmbeddings
from src.llm.rag.retrieval import MedicalRetriever
from src.llm.rag.vector_store import MedicalVectorStore


class TestMedicalRAG:
    def test_embeddings_and_vector_store(self):
        store = MedicalVectorStore(embedding_dim=32)
        results = store.similarity_search("Ventricular arrhythmia PVC", top_k=2)

        assert len(results) == 2
        top_doc, score = results[0]
        assert "AHA" in top_doc["id"] or "Arrhythmia" in top_doc["title"]
        assert score > 0.0

    def test_medical_retriever(self):
        retriever = MedicalRetriever()
        evidence = retriever.retrieve_evidence(
            signal_finding="Ventricular Ectopic (V)",
            imaging_finding="Pneumonia",
            top_k=2,
        )

        assert len(evidence) == 2
        assert "title" in evidence[0]
        assert "evidence_text" in evidence[0]
        assert "source" in evidence[0]


class TestClinicalAgents:
    def test_signal_agent(self):
        agent = SignalAgent()
        dummy_ecg = np.random.randn(260).astype(np.float32)

        output = agent.analyze(ecg_tensor=dummy_ecg, heart_rate=72.0)

        assert "finding" in output
        assert "confidence" in output
        assert 0.0 <= output["confidence"] <= 1.0
        assert output["heart_rate"] == 72.0

    def test_imaging_agent(self):
        agent = ImagingAgent()
        dummy_img = Image.new("RGB", (224, 224), color=(50, 50, 50))

        output = agent.analyze(image_input=dummy_img)

        assert "finding" in output
        assert "confidence" in output
        assert "description" in output

    def test_evidence_agent(self):
        agent = EvidenceAgent()
        citations = agent.retrieve_guidelines(
            signal_finding="Normal",
            imaging_finding="Pleural Effusion",
        )

        assert len(citations) > 0
        assert any("Effusion" in c["title"] or "Pleural" in c["title"] for c in citations)

    def test_risk_agent(self):
        agent = RiskAgent()
        sig_report = {"finding": "Normal (N)", "class_idx": 0}
        img_report = {"finding": "Normal", "class_idx": 0}
        patient_info = {"age": 45, "has_hypertension": 0}

        risk_out = agent.assess_risk(
            signal_report=sig_report,
            imaging_report=img_report,
            patient_info=patient_info,
        )

        assert risk_out["risk_level"] == "Low Risk"
        assert risk_out["risk_score"] < 0.3

    def test_report_agent(self):
        agent = ReportAgent()
        patient_info = {"patient_id": "PT-100", "age": 60, "sex": "Female", "medical_history": "None"}
        sig_report = {"finding": "Normal (N)", "class_idx": 0, "confidence": 0.95}
        img_report = {"finding": "Normal", "class_idx": 0, "confidence": 0.98}
        risk_report = {"risk_level": "Low Risk", "risk_score": 0.12, "primary_drivers": "None", "suggested_diagnostics": "Routine"}
        evidence = [{"title": "AHA Guidelines", "source": "AHA 2022", "evidence_text": "Normal parameters"}]

        report = agent.generate_report(patient_info, sig_report, img_report, risk_report, evidence)

        assert "markdown_report" in report
        assert "MedAgent Clinical Decision Support Report" in report["markdown_report"]
        assert "PT-100" in report["markdown_report"]

    def test_supervisor_agent_end_to_end(self):
        supervisor = SupervisorAgent(load_checkpoints=False)

        patient_data = {
            "patient_id": "PT-TEST-01",
            "age": 62,
            "sex": "Male",
            "medical_history": "Hypertension",
        }
        dummy_ecg = np.zeros(260, dtype=np.float32)
        dummy_xray = Image.new("RGB", (224, 224), color=(30, 30, 30))

        result = supervisor.run_case(
            patient_info=patient_data,
            ecg_input=dummy_ecg,
            xray_input=dummy_xray,
        )

        assert result["patient_id"] == "PT-TEST-01"
        assert "risk_level" in result
        assert "markdown_report" in result
        assert len(result["evidence_citations"]) > 0
