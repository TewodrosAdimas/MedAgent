"""
Signal Transformer Architecture for ECG Arrhythmia Classification.
Uses 1D patch embeddings, positional encodings, and multi-head self-attention
to capture global waveform dynamics and long-range morphological dependencies.
"""

from typing import Optional
import torch
import torch.nn as nn


class PatchEmbedding1D(nn.Module):
    """
    Projects contiguous 1D signal slices into a sequence of token embeddings.
    """

    def __init__(
        self,
        in_channels: int = 1,
        patch_size: int = 20,
        embed_dim: int = 128,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) -> proj: (B, embed_dim, num_patches) -> transpose: (B, num_patches, embed_dim)
        x = self.proj(x)
        return x.transpose(1, 2)


class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer Encoder Block.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        mlp_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-Attention residual
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out

        # MLP residual
        x = x + self.mlp(self.norm2(x))
        return x


class ECGSignalTransformer(nn.Module):
    """
    Transformer model for 1D physiological ECG signals.

    Input shape:  (batch_size, in_channels, sequence_length) e.g. (64, 1, 260)
    Output shape: (batch_size, num_classes) e.g. (64, 5)
    """

    def __init__(
        self,
        in_channels: int = 1,
        sequence_length: int = 260,
        num_classes: int = 5,
        patch_size: int = 20,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        mlp_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        assert sequence_length % patch_size == 0, f"Seq length {sequence_length} must be divisible by patch size {patch_size}"

        self.num_patches = sequence_length // patch_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # 1D Patch Embedding
        self.patch_embed = PatchEmbedding1D(
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        # Learnable [CLS] token & Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # Stacked Transformer Encoder Layers
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Classification Head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts global [CLS] token representation for multimodal fusion.

        Args:
            x: Input tensor of shape (batch_size, in_channels, sequence_length)

        Returns:
            Latent embedding tensor of shape (batch_size, embed_dim)
        """
        B = x.shape[0]
        tokens = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)  # (B, num_patches + 1, embed_dim)

        tokens = self.pos_drop(tokens + self.pos_embed)

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)
        cls_output = tokens[:, 0]  # Extract [CLS] token
        return cls_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing classification logits.
        """
        features = self.get_features(x)
        logits = self.head(features)
        return logits
