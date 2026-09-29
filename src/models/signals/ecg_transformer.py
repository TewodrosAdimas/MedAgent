"""
1D Vision-Style Transformer Encoder for ECG Sequence Modeling.

Architecture:
1. 1D Temporal Patch Embedding: Splits continuous 260-sample heartbeat waveforms into token embeddings.
2. Learnable [CLS] classification token and learnable positional encodings.
3. Multi-Head Self-Attention Transformer Encoder blocks capturing long-range morphological dependencies.
4. Dual-head output: 5-class AAMI arrhythmia classification and 256-d latent representation.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ECGPatchEmbedding1D(nn.Module):
    """
    Splits a 1D ECG signal into non-overlapping temporal patches and projects them to embedding dimension.
    """

    def __init__(
        self,
        sequence_length: int = 260,
        patch_size: int = 10,
        in_channels: int = 1,
        embed_dim: int = 256,
    ):
        super().__init__()
        assert sequence_length % patch_size == 0, (
            f"Sequence length ({sequence_length}) must be divisible by patch size ({patch_size})"
        )
        self.num_patches = sequence_length // patch_size
        self.proj = nn.Conv1d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 1, L)
        # proj(x) shape: (B, embed_dim, num_patches)
        # transpose to: (B, num_patches, embed_dim)
        return self.proj(x).transpose(1, 2)


class ECGTransformer(nn.Module):
    """
    Transformer Encoder Architecture for ECG Biosignal Analysis.
    """

    def __init__(
        self,
        sequence_length: int = 260,
        patch_size: int = 10,
        in_channels: int = 1,
        num_classes: int = 5,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # 1. 1D Patch Embedding
        self.patch_embed = ECGPatchEmbedding1D(
            sequence_length=sequence_length,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # 2. [CLS] token and Positional Encodings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),
        )

        # 4. Classification Head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts the 256-d latent representation at the [CLS] token.
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)

        B = x.shape[0]
        # (B, num_patches, embed_dim)
        tokens = self.patch_embed(x)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)

        # Add positional embedding
        tokens = self.pos_drop(tokens + self.pos_embed)

        # Transformer blocks
        encoded = self.transformer_encoder(tokens)

        # Extract [CLS] token output
        cls_out = encoded[:, 0, :]
        return cls_out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.extract_features(x)
        logits = self.head(latent)
        return logits, latent
