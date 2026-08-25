"""
Vision Transformer (ViT) Architecture for Medical Radiograph Classification.
Applies patch-based self-attention to capture non-local radiographic patterns.
"""

from typing import Optional
import torch
import torch.nn as nn


class PatchEmbedding2D(nn.Module):
    """
    Splits 2D image into flattened patches and linearly projects into embed_dim.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 384,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> proj: (B, embed_dim, H/P, W/P) -> flatten: (B, embed_dim, N) -> transpose: (B, N, embed_dim)
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class ViTBlock(nn.Module):
    """
    Standard Vision Transformer Encoder Layer with Pre-LayerNorm.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
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
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ChestXRayViT(nn.Module):
    """
    Vision Transformer for Chest X-Ray classification.

    Input shape:  (batch_size, 3, 224, 224)
    Output shape: (batch_size, num_classes)
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 5,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding2D(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        self.blocks = nn.ModuleList([
            ViTBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.latent_dim = embed_dim

        # Medical classification head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts the [CLS] token embedding for multimodal fusion.
        """
        B = x.shape[0]
        tokens = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens = self.pos_drop(tokens + self.pos_embed)

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)
        cls_feature = tokens[:, 0]
        return cls_feature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.get_features(x)
        logits = self.head(features)
        return logits
