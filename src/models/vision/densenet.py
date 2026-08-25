"""
DenseNet-121 Radiographic Classifier for Chest X-Ray Analysis.
Extracts deep spatial representations and provides Grad-CAM hook endpoints.
"""

from typing import Optional
import torch
import torch.nn as nn
import torchvision.models as models


class ChestXRayDenseNet121(nn.Module):
    """
    DenseNet-121 architecture optimized for Chest X-Ray multi-pathology classification.

    Input shape:  (batch_size, 3, 224, 224)
    Output shape: (batch_size, num_classes) e.g. (32, 5)
    """

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.3,
        fc_hidden: int = 256,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Load DenseNet-121 backbone
        try:
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            backbone = models.densenet121(weights=weights)
        except Exception:
            backbone = models.densenet121(weights=None)

        self.features = backbone.features
        self.latent_dim = backbone.classifier.in_features  # 1024

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Medical classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )

    def get_target_layer(self) -> nn.Module:
        """
        Returns the final convolutional layer for Grad-CAM explainability hooks.
        """
        return self.features.denseblock4.denselayer16.conv2

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts 1024-dimensional feature embeddings for multimodal fusion.

        Args:
            x: Image tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Latent embedding tensor of shape (batch_size, 1024)
        """
        features_map = self.features(x)
        pooled = self.global_pool(features_map)
        embeddings = torch.flatten(pooled, 1)
        return embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing pathology class logits.
        """
        features = self.get_features(x)
        logits = self.classifier(features)
        return logits
