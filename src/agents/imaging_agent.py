"""
Medical Imaging Intelligence Agent for MedAgent.

Translates chest radiograph AI classifications and Grad-CAM spatial activation maps
into structured radiologic observations and anatomic localizations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import numpy as np
import torch

from src.data.loaders.xray_dataset import XRAY_CLASSES


@dataclass
class ImagingAnalysisResult:
    predicted_class: int
    class_name: str
    confidence: float
    probabilities: Dict[str, float]
    radiological_finding: str
    gradcam_localization: str
    severity: str  # 'Normal', 'Moderate', 'Severe'


class ImagingAgent:
    """
    Agent responsible for interpreting chest X-ray deep learning models and Grad-CAM visual heatmaps.
    """

    def __init__(self):
        self.localization_descriptions = {
            0: "Diffuse normal thoracic parenchyma without focal consolidation or cardiothoracic enlargement.",
            1: "Focal alveolar consolidation / opacity predominantly in mid-to-lower pulmonary lung fields.",
            2: "Enlarged transverse cardiac silhouette exceeding standard cardiothoracic ratio (> 0.50).",
            3: "Linear / subsegmental compressive volume loss and opacity in dependent basal regions.",
            4: "Costophrenic angle blunting and dependent fluid accumulation in pleural space.",
        }

    def analyze(
        self,
        xray_logits_or_probs: Union[torch.Tensor, np.ndarray],
        gradcam_heatmap: Optional[np.ndarray] = None,
    ) -> ImagingAnalysisResult:
        """
        Interprets vision model predictions and Grad-CAM heatmaps.
        """
        if isinstance(xray_logits_or_probs, torch.Tensor):
            arr = xray_logits_or_probs.squeeze().detach().cpu().numpy()
        else:
            arr = np.squeeze(xray_logits_or_probs)

        if np.isclose(np.sum(arr), 1.0, atol=1e-2) and np.all(arr >= 0):
            probs = arr
        else:
            e_x = np.exp(arr - np.max(arr))
            probs = e_x / np.sum(e_x)

        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        class_name = XRAY_CLASSES.get(pred_class, f"Class {pred_class}")

        prob_dict = {
            XRAY_CLASSES.get(i, f"Class_{i}"): float(probs[i])
            for i in range(len(probs))
        }

        # Analyze Grad-CAM region
        if gradcam_heatmap is not None and np.max(gradcam_heatmap) > 0.3:
            # Find center of maximum activation
            y_max, x_max = np.unravel_index(np.argmax(gradcam_heatmap), gradcam_heatmap.shape)
            h, w = gradcam_heatmap.shape
            vert = "upper" if y_max < h / 3 else ("middle" if y_max < 2 * h / 3 else "lower")
            horiz = "right" if x_max < w / 3 else ("central / perihilar" if x_max < 2 * w / 3 else "left")
            localization = f"Peak activation localized to {vert} {horiz} thoracic region."
        else:
            localization = self.localization_descriptions.get(pred_class, "Standard thoracic distribution.")

        if pred_class == 0:
            severity = "Normal"
            finding = (
                f"Radiograph demonstrates clear lung fields without consolidation, effusion, or pneumothorax "
                f"({class_name}, confidence: {confidence*100:.1f}%)."
            )
        elif pred_class == 1:
            severity = "Severe"
            finding = (
                f"Abnormal pulmonary infiltrate / opacity identified, highly suggestive of Pneumonia "
                f"({class_name}, confidence: {confidence*100:.1f}%). Correlate clinically with infectious markers."
            )
        elif pred_class == 2:
            severity = "Severe"
            finding = (
                f"Significant cardiomegaly detected with prominent cardiac border "
                f"({class_name}, confidence: {confidence*100:.1f}%). Suggestive of chamber enlargement or pericardial effusion."
            )
        elif pred_class == 3:
            severity = "Moderate"
            finding = (
                f"Basal atelectatic changes observed ({class_name}, confidence: {confidence*100:.1f}%). "
                "May represent dependent airway collapse or hypoventilation."
            )
        else:
            severity = "Severe"
            finding = (
                f"Evidence of pleural fluid collection / pleural effusion noted "
                f"({class_name}, confidence: {confidence*100:.1f}%)."
            )

        return ImagingAnalysisResult(
            predicted_class=pred_class,
            class_name=class_name,
            confidence=confidence,
            probabilities=prob_dict,
            radiological_finding=finding,
            gradcam_localization=localization,
            severity=severity,
        )
