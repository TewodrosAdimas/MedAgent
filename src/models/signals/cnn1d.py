"""
1D Convolutional Neural Network (CNN1D) for ECG Arrhythmia Classification.
Serves as the high-speed, robust baseline model for physiological waveform analysis.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn


class ConvBlock1D(nn.Module):
    """
    Standard 1D Convolutional Block with Conv1D, BatchNorm, GELU, and optional Pooling/Dropout.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        padding: Optional[int] = None,
        pool_size: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=pool_size, stride=pool_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ECGCNN1D(nn.Module):
    """
    Deep 1D CNN for ECG arrhythmia classification.

    Input shape:  (batch_size, in_channels, sequence_length) e.g. (64, 1, 260)
    Output shape: (batch_size, num_classes) e.g. (64, 5)
    """

    def __init__(
        self,
        in_channels: int = 1,
        sequence_length: int = 260,
        num_classes: int = 5,
        conv_channels: Optional[List[int]] = None,
        kernel_sizes: Optional[List[int]] = None,
        dropout: float = 0.3,
        fc_hidden: int = 128,
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64, 128, 256]
        if kernel_sizes is None:
            kernel_sizes = [7, 5, 3, 3]

        assert len(conv_channels) == len(kernel_sizes), "Channels and kernels must have same length"

        self.in_channels = in_channels
        self.sequence_length = sequence_length
        self.num_classes = num_classes

        # Construct Convolutional Backbone
        blocks = []
        curr_in = in_channels
        for out_ch, k_size in zip(conv_channels, kernel_sizes):
            blocks.append(
                ConvBlock1D(
                    in_channels=curr_in,
                    out_channels=out_ch,
                    kernel_size=k_size,
                    dropout=dropout,
                )
            )
            curr_in = out_ch

        self.feature_extractor = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.latent_dim = conv_channels[-1]

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts 1D latent feature representations for multimodal fusion.

        Args:
            x: Input tensor of shape (batch_size, in_channels, sequence_length)

        Returns:
            Latent embedding tensor of shape (batch_size, latent_dim) e.g. (64, 256)
        """
        features = self.feature_extractor(x)
        pooled = self.global_pool(features).squeeze(-1)
        return pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing classification logits.

        Args:
            x: Input tensor of shape (batch_size, in_channels, sequence_length)

        Returns:
            Class logits of shape (batch_size, num_classes)
        """
        features = self.get_features(x)
        logits = self.classifier(features)
        return logits
