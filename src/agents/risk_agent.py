"""
Risk Assessment Agent for MedAgent.
Integrates unimodal diagnostic findings, cross-modal transformer embeddings,
and patient physiological profiles to synthesize stratified clinical risk.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn

from src.data.loaders.multimodal_loader import RISK_LEVELS, determine_clinical_risk
from src.utils.logger import get_logger

logger = get_logger("RiskAgent")


class RiskAgent:
    """
    Agent responsible for multimodal clinical risk stratification and severity evaluation.
    """

    def __init__(self, fusion_model: Optional[nn.Module] = None, device: Optional[torch.device] = None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fusion_model = fusion_model
        if self.fusion_model is not None:
            self.fusion_model.to(self.device)
            self.fusion_model.eval()

    def assess_risk(
        self,
        signal_report: Dict[str, Any],
        imaging_report: Dict[str, Any],
        patient_info: Dict[str, Any],
        ecg_tensor: Optional[torch.Tensor] = None,
        image_tensor: Optional[torch.Tensor] = None,
        tabular_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Assesses overall patient clinical risk by combining model outputs and clinical heuristics.
        """
        sig_idx = signal_report.get("class_idx", 0)
        img_idx = imaging_report.get("class_idx", 0)
        age = patient_info.get("age", 55)
        has_htn = patient_info.get("has_hypertension", 0)

        # 1. Base clinical risk calculation
        risk_idx = determine_clinical_risk(sig_idx, img_idx, age, has_htn)
        risk_level = RISK_LEVELS[risk_idx]

        # 2. If multimodal fusion model is loaded, refine with cross-attention model
        modality_attention = {}
        if self.fusion_model is not None and ecg_tensor is not None and image_tensor is not None and tabular_tensor is not None:
            try:
                with torch.no_grad():
                    ecg = ecg_tensor.to(self.device)
                    img = image_tensor.to(self.device)
                    tab = tabular_tensor.to(self.device)
                    if ecg.ndim == 1:
                        ecg = ecg.unsqueeze(0).unsqueeze(0)
                    elif ecg.ndim == 2:
                        ecg = ecg.unsqueeze(0)
                    if img.ndim == 3:
                        img = img.unsqueeze(0)
                    if tab.ndim == 1:
                        tab = tab.unsqueeze(0)


                    out = self.fusion_model(ecg, img, tab)
                    probs = torch.softmax(out["risk_logits"], dim=1).squeeze().cpu().numpy()
                    risk_idx = int(np.argmax(probs))
                    risk_level = RISK_LEVELS[risk_idx]

                    if "attention_weights" in out and out["attention_weights"] is not None:
                        attn = out["attention_weights"][0].cpu().numpy()
                        modality_attention = {
                            "ecg": float(attn[0, 1]),
                            "xray": float(attn[0, 2]),
                            "demographics": float(attn[0, 3]),
                        }
            except Exception as e:
                logger.warning(f"Multimodal transformer inference failed: {e}")

        # Compute composite severity score
        severity_scores = {0: 0.12, 1: 0.58, 2: 0.89}
        base_score = severity_scores.get(risk_idx, 0.2)

        # Primary drivers
        drivers = []
        if sig_idx != 0:
            drivers.append(f"Cardiac Arrhythmia ({signal_report.get('finding')})")
        if img_idx != 0:
            drivers.append(f"Pulmonary Finding ({imaging_report.get('finding')})")
        if age >= 65 and has_htn:
            drivers.append("Geriatric Cardiovascular Risk Factors")
        if not drivers:
            drivers.append("Standard Baseline Health Profile")

        # Suggested diagnostic follow-ups
        diagnostics = []
        if sig_idx in [1, 2]:
            diagnostics.append("24-48h Holter ECG monitoring and serum electrolyte panel (K+, Mg2+)")
        if img_idx in [1, 4]:
            diagnostics.append("Serial chest radiograph and inflammatory biomarkers (CRP, Procalcitonin)")
        if img_idx == 2 or sig_idx == 2:
            diagnostics.append("Transthoracic Echocardiogram (TTE) and NT-proBNP evaluation")
        if not diagnostics:
            diagnostics.append("Routine annual wellness screening")

        output = {
            "risk_level": risk_level,
            "risk_score": round(base_score, 3),
            "risk_idx": risk_idx,
            "primary_drivers": ", ".join(drivers),
            "suggested_diagnostics": "; ".join(diagnostics),
            "modality_attention": modality_attention,
        }

        logger.info(f"RiskAgent calculated Risk: {risk_level} (Score: {base_score:.2f})")
        return output
