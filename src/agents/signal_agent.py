"""
Signal Intelligence Agent for MedAgent.

Translates deep learning ECG model outputs, waveform morphology,
and rhythm characteristics into structured clinical cardiology findings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import numpy as np
import torch

from src.data.preprocessing.ecg_cleaner import AAMI_CLASSES


@dataclass
class SignalAnalysisResult:
    predicted_class: int
    class_name: str
    confidence: float
    probabilities: Dict[str, float]
    clinical_interpretation: str
    abnormal_features: List[str]
    severity: str  # 'Normal', 'Mild / Moderate', 'Severe / Urgent'


class SignalAgent:
    """
    Agent responsible for analyzing physiological signal AI predictions.
    """

    def __init__(self):
        self.class_descriptions = {
            0: "Normal sinus rhythm / baseline cardiac morphology.",
            1: "Supraventricular ectopic activity (premature atrial or nodal complex).",
            2: "Ventricular ectopic activity (premature ventricular contraction / ventricular escape).",
            3: "Fusion heartbeat (co-activation of ventricular focus and normal supraventricular conduction).",
            4: "Paced / Unclassifiable cardiac electrical complex.",
        }

    def analyze(
        self,
        ecg_logits_or_probs: Union[torch.Tensor, np.ndarray],
        heart_rate_bpm: Optional[float] = None,
    ) -> SignalAnalysisResult:
        """
        Interprets ECG model outputs and vital signs.
        """
        if isinstance(ecg_logits_or_probs, torch.Tensor):
            arr = ecg_logits_or_probs.squeeze().detach().cpu().numpy()
        else:
            arr = np.squeeze(ecg_logits_or_probs)

        if np.isclose(np.sum(arr), 1.0, atol=1e-2) and np.all(arr >= 0):
            probs = arr
        else:
            e_x = np.exp(arr - np.max(arr))
            probs = e_x / np.sum(e_x)

        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        class_name = AAMI_CLASSES.get(pred_class, f"Class {pred_class}")

        prob_dict = {
            AAMI_CLASSES.get(i, f"Class_{i}"): float(probs[i])
            for i in range(len(probs))
        }

        # Formulate clinical observations
        abnormalities = []
        if pred_class == 0:
            severity = "Normal"
            interpretation = (
                f"ECG waveform demonstrates normal morphology with {confidence*100:.1f}% confidence. "
                "No primary repolarization or conduction abnormalities detected on the monitored lead."
            )
        elif pred_class == 1:
            severity = "Mild / Moderate"
            abnormalities.append("Supraventricular premature beat (PAC)")
            interpretation = (
                f"Ectopic atrial/nodal focus detected ({class_name}, confidence: {confidence*100:.1f}%). "
                "Premature P-wave morphology and altered PR interval observed. Typically benign but warrants monitoring if frequent."
            )
        elif pred_class in [2, 3]:
            severity = "Severe / Urgent"
            abnormalities.append("Ventricular ectopic complex (PVC / Fusion)")
            interpretation = (
                f"Aberrant wide QRS complex detected ({class_name}, confidence: {confidence*100:.1f}%). "
                "Signifies premature ventricular depolarization. In the presence of underlying cardiac symptoms, further structural workup is indicated."
            )
        else:
            severity = "Mild / Moderate"
            abnormalities.append("Paced / Unclassified electrical complex")
            interpretation = (
                f"Artificial pacing artifact or anomalous waveform detected ({class_name}, confidence: {confidence*100:.1f}%)."
            )

        if heart_rate_bpm is not None:
            if heart_rate_bpm > 100.0:
                abnormalities.append(f"Sinus Tachycardia ({heart_rate_bpm:.0f} BPM)")
                interpretation += f" Concurrent elevated heart rate ({heart_rate_bpm:.0f} BPM) noted."
            elif heart_rate_bpm < 60.0:
                abnormalities.append(f"Sinus Bradycardia ({heart_rate_bpm:.0f} BPM)")
                interpretation += f" Concurrent low heart rate ({heart_rate_bpm:.0f} BPM) noted."

        return SignalAnalysisResult(
            predicted_class=pred_class,
            class_name=class_name,
            confidence=confidence,
            probabilities=prob_dict,
            clinical_interpretation=interpretation,
            abnormal_features=abnormalities,
            severity=severity,
        )
