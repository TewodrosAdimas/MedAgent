"""
Late Fusion Model for Multimodal Medical AI.
Combines unimodal model predictions using learnable gating attention
to produce consensus-driven clinical risk assessments.
"""

from typing import Dict
import torch
import torch.nn as nn

from src.models.signals.cnn1d import ECGCNN1D
from src.models.vision.densenet import ChestXRayDenseNet121


class LateFusionModel(nn.Module):
    """
    Late Decision Fusion Model for Multimodal Medical AI.
    """

    def __init__(
        self,
        num_risk_classes: int = 3,
        num_signal_classes: int = 5,
        num_image_classes: int = 5,
        tabular_dim: int = 8,
        pretrained_vision: bool = True,
    ):
        super().__init__()
        # Unimodal Predictors
        self.signal_branch = ECGCNN1D(num_classes=num_signal_classes)
        self.vision_branch = ChestXRayDenseNet121(
            num_classes=num_image_classes, pretrained=pretrained_vision
        )
        self.tabular_branch = nn.Sequential(
            nn.Linear(tabular_dim, 32),
            nn.GELU(),
            nn.Linear(32, num_risk_classes),
        )

        # Gating network computing modality importance weights
        combined_logits_dim = num_signal_classes + num_image_classes + num_risk_classes  # 5 + 5 + 3 = 13
        self.gate = nn.Sequential(
            nn.Linear(combined_logits_dim, 32),
            nn.GELU(),
            nn.Linear(32, 3),  # 3 modality attention weights
            nn.Softmax(dim=1),
        )

        # Risk prediction from fused decision representations
        self.risk_classifier = nn.Sequential(
            nn.Linear(combined_logits_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_risk_classes),
        )

    def forward(
        self,
        ecg: torch.Tensor,
        image: torch.Tensor,
        tabular: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        sig_logits = self.signal_branch(ecg)        # (B, 5)
        img_logits = self.vision_branch(image)      # (B, 5)
        tab_logits = self.tabular_branch(tabular)   # (B, 3)

        concat_logits = torch.cat((sig_logits, img_logits, tab_logits), dim=1)  # (B, 13)
        modality_weights = self.gate(concat_logits)  # (B, 3)

        risk_logits = self.risk_classifier(concat_logits)

        return {
            "risk_logits": risk_logits,
            "sig_logits": sig_logits,
            "img_logits": img_logits,
            "modality_weights": modality_weights,
        }
