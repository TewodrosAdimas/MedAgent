"""
Clinical Supervisor Agent for MedAgent.
Master orchestrator that coordinates the multimodal clinical reasoning workflow
across Signal, Imaging, Evidence (RAG), Risk, and Report agents.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
from PIL import Image
import torch

from src.agents.evidence_agent import EvidenceAgent
from src.agents.imaging_agent import ImagingAgent
from src.agents.report_agent import ReportAgent
from src.agents.risk_agent import RiskAgent
from src.agents.signal_agent import SignalAgent
from src.config.paths import CHECKPOINTS_DIR
from src.models.fusion.multimodal_transformer import MultimodalTransformer
from src.models.signals.cnn1d import ECGCNN1D
from src.models.vision.densenet import ChestXRayDenseNet121
from src.utils.logger import get_logger

logger = get_logger("SupervisorAgent")


class SupervisorAgent:
    """
    Master multi-agent clinical supervisor orchestrating end-to-end diagnostic reasoning.
    """

    def __init__(self, load_checkpoints: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing SupervisorAgent on device: {self.device}")

        # 1. Initialize Signal Model
        signal_model = ECGCNN1D(num_classes=5)
        sig_ckpt = CHECKPOINTS_DIR / "ecg_cnn1d_best.pth"
        if load_checkpoints and sig_ckpt.exists():
            try:
                ckpt = torch.load(sig_ckpt, map_location=self.device, weights_only=False)
                signal_model.load_state_dict(ckpt["model_state_dict"])
                logger.info("Loaded trained ECG CNN1D checkpoint.")
            except Exception as e:
                logger.warning(f"Could not load ECG checkpoint: {e}")

        # 2. Initialize Vision Model
        vision_model = ChestXRayDenseNet121(num_classes=5, pretrained=False)
        xray_ckpt = CHECKPOINTS_DIR / "xray_densenet_best.pth"
        if load_checkpoints and xray_ckpt.exists():
            try:
                ckpt = torch.load(xray_ckpt, map_location=self.device, weights_only=False)
                vision_model.load_state_dict(ckpt["model_state_dict"])
                logger.info("Loaded trained DenseNet-121 Chest X-ray checkpoint.")
            except Exception as e:
                logger.warning(f"Could not load Vision checkpoint: {e}")

        # 3. Initialize Multimodal Transformer Model
        fusion_model = MultimodalTransformer(
            num_risk_classes=3,
            num_signal_classes=5,
            num_image_classes=5,
            embed_dim=256,
            num_heads=8,
            num_layers=3,
            pretrained_vision=False,
        )
        fusion_ckpt = CHECKPOINTS_DIR / "multimodal_transformer_best.pth"
        if load_checkpoints and fusion_ckpt.exists():
            try:
                ckpt = torch.load(fusion_ckpt, map_location=self.device, weights_only=False)
                fusion_model.load_state_dict(ckpt["model_state_dict"])
                logger.info("Loaded trained Multimodal Transformer checkpoint.")
            except Exception as e:
                logger.warning(f"Could not load Multimodal Transformer checkpoint: {e}")

        # 4. Instantiate Specialized Clinical Subagents
        self.signal_agent = SignalAgent(model=signal_model, device=self.device)
        self.imaging_agent = ImagingAgent(model=vision_model, device=self.device)
        self.evidence_agent = EvidenceAgent()
        self.risk_agent = RiskAgent(fusion_model=fusion_model, device=self.device)
        self.report_agent = ReportAgent()

        logger.info("SupervisorAgent and all specialized clinical subagents ready.")

    def run_case(
        self,
        patient_info: Dict[str, Any],
        ecg_input: Union[torch.Tensor, np.ndarray],
        xray_input: Union[torch.Tensor, np.ndarray, Image.Image],
        tabular_input: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the full multi-agent diagnostic cycle for a patient case.

        Args:
            patient_info: Demographics dict (id, age, sex, medical_history, etc.)
            ecg_input: Raw or filtered ECG waveform (1, 260)
            xray_input: Chest radiograph image or tensor
            tabular_input: Demographic feature vector (8,)

        Returns:
            Dictionary containing finalized clinical report and subagent findings.
        """
        logger.info(f"=== Starting Multi-Agent Clinical Case: Patient {patient_info.get('patient_id', 'Unknown')} ===")

        # Step 1: Signal Agent Analysis
        signal_findings = self.signal_agent.analyze(
            ecg_tensor=ecg_input,
            heart_rate=patient_info.get("heart_rate", 78.0),
        )

        # Step 2: Imaging Agent Analysis & Grad-CAM ROI Extraction
        imaging_findings = self.imaging_agent.analyze(image_input=xray_input)

        # Step 3: Evidence Agent RAG Guideline Retrieval
        evidence_citations = self.evidence_agent.retrieve_guidelines(
            signal_finding=signal_findings["finding"],
            imaging_finding=imaging_findings["finding"],
            patient_history=patient_info.get("medical_history", ""),
            top_k=2,
        )

        # Step 4: Multimodal Risk Assessment
        if tabular_input is None:
            # Build default normalized tabular vector
            age = float(patient_info.get("age", 55))
            sex = 1.0 if patient_info.get("sex", "Male") == "Male" else 0.0
            sbp = float(patient_info.get("sbp", 130))
            dbp = float(patient_info.get("dbp", 85))
            hr = float(patient_info.get("heart_rate", 78))
            spo2 = float(patient_info.get("spo2", 96))
            htn = float(patient_info.get("has_hypertension", 1))
            dm = float(patient_info.get("has_diabetes", 0))
            tabular_input = np.array([
                (age - 55.0) / 20.0,
                sex,
                (sbp - 120.0) / 20.0,
                (dbp - 80.0) / 10.0,
                (hr - 75.0) / 15.0,
                (spo2 - 95.0) / 5.0,
                htn,
                dm,
            ], dtype=np.float32)

        risk_findings = self.risk_agent.assess_risk(
            signal_report=signal_findings,
            imaging_report=imaging_findings,
            patient_info=patient_info,
            ecg_tensor=ecg_input if isinstance(ecg_input, torch.Tensor) else torch.from_numpy(ecg_input).float(),
            image_tensor=xray_input if isinstance(xray_input, torch.Tensor) else None,
            tabular_tensor=torch.from_numpy(tabular_input).float() if isinstance(tabular_input, np.ndarray) else tabular_input,
        )

        # Step 5: Report Agent Clinical Decision Support Assembly
        final_report = self.report_agent.generate_report(
            patient_info=patient_info,
            signal_report=signal_findings,
            imaging_report=imaging_findings,
            risk_report=risk_findings,
            evidence_list=evidence_citations,
        )

        logger.info(f"=== Multi-Agent Case Analysis Complete: Risk Level = {final_report['risk_level']} ===")
        return final_report


if __name__ == "__main__":
    from src.data.loaders.image_loader import generate_synthetic_xray_image

    supervisor = SupervisorAgent(load_checkpoints=True)

    # Simulated acute patient case
    patient_data = {
        "patient_id": "PT-7821",
        "age": 68,
        "sex": "Male",
        "sbp": 148,
        "dbp": 92,
        "heart_rate": 88.0,
        "spo2": 93.0,
        "has_hypertension": 1,
        "has_diabetes": 1,
        "medical_history": "Chronic hypertension, type 2 diabetes, presenting with acute dyspnea and palpitations.",
    }

    dummy_ecg = np.sin(np.linspace(0, 10, 260)).astype(np.float32)
    dummy_xray = generate_synthetic_xray_image(pathology="Pneumonia")

    result = supervisor.run_case(
        patient_info=patient_data,
        ecg_input=dummy_ecg,
        xray_input=dummy_xray,
    )

    print("\n" + "=" * 70)
    print(result["markdown_report"])
    print("=" * 70)
