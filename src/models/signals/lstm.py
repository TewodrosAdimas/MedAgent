"""
Bidirectional LSTM (BiLSTM) for ECG Sequence Classification.
Captures sequential temporal dynamics and rhythm transitions across heartbeat waveforms.
"""

from typing import Optional
import torch
import torch.nn as nn


class ECGBiLSTM(nn.Module):
    """
    Bidirectional LSTM architecture for ECG signal classification.

    Input shape:  (batch_size, in_channels, sequence_length) e.g. (64, 1, 260)
    Output shape: (batch_size, num_classes) e.g. (64, 5)
    """

    def __init__(
        self,
        in_channels: int = 1,
        sequence_length: int = 260,
        num_classes: int = 5,
        hidden_dim: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        fc_hidden: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_classes = num_classes

        # Optional 1D Conv front-end to extract local morphological features before LSTM
        self.conv_in = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
        )

        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        direction_factor = 2 if bidirectional else 1
        self.latent_dim = hidden_dim * direction_factor

        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts temporal summary representation for multimodal fusion.
        """
        # x: (B, 1, L) -> conv: (B, 32, L/2) -> transpose: (B, L/2, 32)
        conv_out = self.conv_in(x)
        lstm_in = conv_out.transpose(1, 2)

        lstm_out, (h_n, _) = self.lstm(lstm_in)

        if self.bidirectional:
            # Concatenate forward and backward hidden states from final layer
            h_fwd = h_n[-2, :, :]
            h_bwd = h_n[-1, :, :]
            features = torch.cat((h_fwd, h_bwd), dim=1)
        else:
            features = h_n[-1, :, :]

        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.get_features(x)
        logits = self.classifier(features)
        return logits
