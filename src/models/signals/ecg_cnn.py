"""
1D Residual Convolutional Neural Network for ECG Heartbeat Classification.

Features:
- Multi-scale 1D residual blocks for robust morphological waveform feature extraction.
- Zero-phase receptive field expansion across P-QRS-T complexes.
- Dual-head output: 5-class AAMI arrhythmia logits and 256-d latent embedding vector for multimodal fusion.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ECGResidualBlock1D(nn.Module):
    """
    1D Residual block with batch normalization, dropout, and projection shortcut.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        out = self.relu(out)
        return out


class ECG1DCNN(nn.Module):
    """
    1D Deep Residual CNN for ECG Arrhythmia Classification and Latent Embedding.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        latent_dim: int = 256,
        dropout: float = 0.2,
    ):
        """
        Args:
            in_channels: Number of input signal channels (default: 1 for single lead).
            num_classes: Number of target arrhythmia classes (default: 5 for AAMI).
            latent_dim: Dimensionality of latent feature embedding vector (default: 256).
            dropout: Dropout probability.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Initial Feature Stem: captures broad waveform shape
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Residual Stages: progressive feature abstraction
        self.stage1 = ECGResidualBlock1D(32, 64, kernel_size=5, stride=2, dropout=dropout)
        self.stage2 = ECGResidualBlock1D(64, 128, kernel_size=5, stride=2, dropout=dropout)
        self.stage3 = ECGResidualBlock1D(128, 256, kernel_size=3, stride=2, dropout=dropout)

        # Global Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Latent Projection Layer
        self.latent_proj = nn.Sequential(
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Arrhythmia Classifier Head
        self.classifier = nn.Linear(latent_dim, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts the 256-dimensional latent representation for multimodal fusion.

        Args:
            x: Input tensor of shape (B, 1, 260) or (B, 260).

        Returns:
            Latent embedding tensor of shape (B, 256).
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)

        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.global_pool(out).squeeze(-1)
        latent = self.latent_proj(out)
        return latent

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input ECG waveform tensor of shape (B, 1, 260) or (B, 260).

        Returns:
            Tuple of:
                - logits: (B, num_classes)
                - latent: (B, latent_dim)
        """
        latent = self.extract_features(x)
        logits = self.classifier(latent)
        return logits, latent
