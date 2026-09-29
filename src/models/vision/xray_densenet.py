"""
DenseNet-121 Deep Neural Network for Thoracic Chest X-Ray Classification.

Features:
- Standard clinical benchmark architecture with dense connectivity across feature blocks.
- Pretrained weights integration with fine-tuning head.
- Extracts 1024-dimensional latent representations for multimodal fusion.
- Hooks into penultimate convolutional layers for Grad-CAM explainability.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class XRayDenseNet121(nn.Module):
    """
    DenseNet-121 Classifier for Chest X-Rays with Latent Feature Extraction.
    """

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = False,
        latent_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        # Load DenseNet-121 backbone
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        densenet = models.densenet121(weights=weights)

        # Feature extractor up to final norm layer
        self.features = densenet.features
        in_features = densenet.classifier.in_features  # 1024

        # Latent projection for Multimodal Fusion (1024 -> latent_dim)
        self.latent_proj = nn.Sequential(
            nn.Linear(in_features, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Classification Head
        self.classifier = nn.Linear(latent_dim, num_classes)

    def extract_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts pooled 1024-d features from DenseNet backbone.
        """
        features = self.features(x)
        out = F.relu(features, inplace=False)
        pooled = F.adaptive_avg_pool2d(out, (1, 1)).flatten(1)
        return pooled

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts projected 256-d latent representation for multimodal fusion.
        """
        pooled = self.extract_backbone_features(x)
        latent = self.latent_proj(pooled)
        return latent

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Chest radiograph image tensor of shape (B, 3, 224, 224).

        Returns:
            Tuple of:
                - logits: (B, num_classes)
                - latent: (B, latent_dim)
        """
        latent = self.extract_features(x)
        logits = self.classifier(latent)
        return logits, latent
