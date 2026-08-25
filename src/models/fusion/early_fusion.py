"""
Early Fusion Model for Multimodal Healthcare Patient Representation.
Concatenates latent features from ECG signals, Chest X-Rays, and clinical demographics
prior to passing through a joint multi-layer representation network.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from src.models.signals.cnn1d import ECGCNN1D
from src.models.vision.densenet import ChestXRayDenseNet121


class EarlyFusionModel(nn.Module):
    """
    Early Feature Concatenation Model for Multimodal Medical AI.
    """

    def __init__(
        self,
        num_risk_classes: int = 3,
        num_signal_classes: int = 5,
        num_image_classes: int = 5,
        tabular_dim: int = 8,
        dropout: float = 0.3,
        joint_hidden: int = 256,
        pretrained_vision: bool = True,
    ):
        super().__init__()
        # 1. Biosignal Encoder (ECG) -> 256-d embedding
        self.signal_model = ECGCNN1D(num_classes=num_signal_classes)
        self.sig_dim = self.signal_model.latent_dim  # 256

        # 2. Medical Vision Encoder (Chest X-Ray) -> 1024-d embedding
        self.vision_model = ChestXRayDenseNet121(
            num_classes=num_image_classes, pretrained=pretrained_vision
        )
        self.img_dim = self.vision_model.latent_dim  # 1024

        # 3. Tabular Clinical Encoder -> 64-d embedding
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
        )
        self.tab_dim = 64

        # Total concatenated feature dimension
        total_dim = self.sig_dim + self.img_dim + self.tab_dim  # 1344

        # 4. Joint Multimodal Representation Network
        self.fusion_network = nn.Sequential(
            nn.Linear(total_dim, joint_hidden * 2),
            nn.BatchNorm1d(joint_hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(joint_hidden * 2, joint_hidden),
            nn.BatchNorm1d(joint_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.latent_dim = joint_hidden

        # 5. Multi-task Heads
        self.risk_head = nn.Linear(joint_hidden, num_risk_classes)
        self.sig_head = nn.Linear(joint_hidden, num_signal_classes)
        self.img_head = nn.Linear(joint_hidden, num_image_classes)

    def get_features(
        self,
        ecg: torch.Tensor,
        image: torch.Tensor,
        tabular: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extracts fused multimodal patient embedding.

        Returns:
            Latent patient embedding of shape (batch_size, joint_hidden)
        """
        sig_feat = self.signal_model.get_features(ecg)        # (B, 256)
        img_feat = self.vision_model.get_features(image)      # (B, 1024)
        tab_feat = self.tabular_encoder(tabular)             # (B, 64)

        concat_feat = torch.cat((sig_feat, img_feat, tab_feat), dim=1)  # (B, 1344)
        patient_embedding = self.fusion_network(concat_feat)            # (B, 256)
        return patient_embedding

    def forward(
        self,
        ecg: torch.Tensor,
        image: torch.Tensor,
        tabular: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass yielding joint clinical risk and unimodal pathology predictions.
        """
        patient_emb = self.get_features(ecg, image, tabular)

        risk_logits = self.risk_head(patient_emb)
        sig_logits = self.sig_head(patient_emb)
        img_logits = self.img_head(patient_emb)

        return {
            "risk_logits": risk_logits,
            "sig_logits": sig_logits,
            "img_logits": img_logits,
            "patient_embedding": patient_emb,
        }
