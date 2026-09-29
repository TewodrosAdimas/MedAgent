"""
Multimodal Risk Assessment Agent for MedAgent.

Synthesizes physiological signals, medical imaging, patient medical history,
and vital signs into a calibrated composite clinical risk assessment.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import numpy as np
import torch

from src.data.loaders.multimodal_dataset import RISK_CLASSES
from src.agents.imaging_agent import ImagingAnalysisResult
from src.agents.signal_agent import SignalAnalysisResult


@dataclass
class RiskAssessmentResult:
    risk_tier: str  # 'Low Risk', 'Moderate Risk', 'High / Critical Risk'
    risk_tier_id: int
    risk_score: float  # [0.0, 1.0]
    probabilities: Dict[str, float]
    primary_risk_drivers: List[str]
    urgency_level: str  # 'Routine Review', 'Priority Clinical Review', 'Urgent / Emergent Review'
    clinical_rationale: str


class RiskAssessmentAgent:
    """
    Agent responsible for combining multimodal evidence into holistic patient risk estimates.
    """

    def evaluate_risk(
        self,
        signal_res: SignalAnalysisResult,
        imaging_res: ImagingAnalysisResult,
        patient_info: Dict[str, Union[int, float, str, bool]],
        multimodal_logits_or_probs: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> RiskAssessmentResult:
        """
        Combines AI model outputs, patient context, and vitals.
        """
        drivers = []
        base_score = 0.15

        # 1. Evaluate Signal Contribution
        if signal_res.predicted_class in [2, 3]:  # Ventricular / Fusion
            base_score += 0.35
            drivers.append(f"Significant Cardiac Arrhythmia ({signal_res.class_name})")
        elif signal_res.predicted_class == 1:
            base_score += 0.15
            drivers.append(f"Supraventricular Ectopy ({signal_res.class_name})")

        # 2. Evaluate Imaging Contribution
        if imaging_res.predicted_class in [1, 2, 4]:  # Pneumonia, Cardiomegaly, Effusion
            base_score += 0.30
            drivers.append(f"Thoracic Radiograph Abnormality ({imaging_res.class_name})")
        elif imaging_res.predicted_class == 3:  # Atelectasis
            base_score += 0.10
            drivers.append(f"Mild Pulmonary Finding ({imaging_res.class_name})")

        # 3. Evaluate Patient Demographics & Vitals
        age = patient_info.get("age", 50)
        if age >= 65:
            base_score += 0.05
            drivers.append(f"Advanced Age ({age} years)")

        if patient_info.get("hypertension", False):
            base_score += 0.05
            drivers.append("Hypertension History")

        if patient_info.get("prior_cardiac_event", False):
            base_score += 0.10
            drivers.append("History of Prior Cardiac Event")

        hr = patient_info.get("heart_rate_bpm", 72)
        if hr > 100 or hr < 50:
            base_score += 0.05
            drivers.append(f"Abnormal Resting Heart Rate ({hr} BPM)")

        spo2 = patient_info.get("spo2_percent", 98)
        if spo2 < 93:
            base_score += 0.15
            drivers.append(f"Hypoxia / Low SpO2 ({spo2}%)")

        # Calibrate with Multimodal Transformer if provided
        if multimodal_logits_or_probs is not None:
            if isinstance(multimodal_logits_or_probs, torch.Tensor):
                probs = torch.softmax(multimodal_logits_or_probs.squeeze(), dim=-1).detach().cpu().numpy()
            else:
                arr = np.squeeze(multimodal_logits_or_probs)
                if np.isclose(np.sum(arr), 1.0, atol=1e-3):
                    probs = arr
                else:
                    e_x = np.exp(arr - np.max(arr))
                    probs = e_x / np.sum(e_x)
            # Weighted average
            nn_score = float(probs[1] * 0.5 + probs[2] * 1.0)
            final_score = float(np.clip(0.4 * base_score + 0.6 * nn_score, 0.0, 0.99))
        else:
            final_score = float(np.clip(base_score, 0.0, 0.99))
            probs = np.array([
                max(0.0, 1.0 - 2 * final_score),
                max(0.0, 1.0 - abs(final_score - 0.5) * 2),
                final_score,
            ])
            probs = probs / np.sum(probs)

        # Categorize tier
        if final_score >= 0.65:
            risk_tier_id = 2
            risk_tier = "High / Critical Risk"
            urgency = "Urgent / Emergent Review"
            rationale = (
                "Multiple concurring high-severity physiological and radiologic indicators detected. "
                "Immediate attending physician review and diagnostic stabilization recommended."
            )
        elif final_score >= 0.35:
            risk_tier_id = 1
            risk_tier = "Moderate Risk"
            urgency = "Priority Clinical Review"
            rationale = (
                "Mild-to-moderate abnormalities detected across physiological or imaging modalities. "
                "Secondary clinical workup and outpatient evaluation recommended."
            )
        else:
            risk_tier_id = 0
            risk_tier = "Low Risk"
            urgency = "Routine Review"
            rationale = "No critical single-modality or cross-modal risks identified. Standard monitoring suggested."

        prob_dict = {
            RISK_CLASSES[0]: float(probs[0]),
            RISK_CLASSES[1]: float(probs[1]),
            RISK_CLASSES[2]: float(probs[2]),
        }

        return RiskAssessmentResult(
            risk_tier=risk_tier,
            risk_tier_id=risk_tier_id,
            risk_score=final_score,
            probabilities=prob_dict,
            primary_risk_drivers=drivers,
            urgency_level=urgency,
            clinical_rationale=rationale,
        )
