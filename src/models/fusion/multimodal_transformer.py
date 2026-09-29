"""
Cross-Attention Multimodal Transformer Fusion Engine for MedAgent.

Integrates:
1. Signal Latent Embeddings (256-d) from ECG 1D CNN / Transformer.
2. Vision Latent Embeddings (256-d) from DenseNet-121.
3. Clinical Tabular Context Embeddings (256-d) from patient demographics, vitals, and comorbidities.
4. Learnable modality type encodings and holistic patient [CLS] classification token.
5. Multimodal Multi-Head Cross-Attention Transformer yielding calibrated 3-tier risk predictions.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.loaders.multimodal_dataset import RISK_CLASSES, TABULAR_FEATURE_NAMES


class MultimodalTransformerFusion(nn.Module):
    """
    Cross-Attention Multimodal Patient Representation & Risk Stratification Transformer.
    """

    def __init__(
        self,
        ecg_latent_dim: int = 256,
        vision_latent_dim: int = 256,
        tabular_dim: int = 10,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        num_risk_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_risk_classes = num_risk_classes

        # 1. Modality-specific linear projections to shared embedding space
        self.ecg_proj = nn.Sequential(
            nn.Linear(ecg_latent_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_latent_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.tabular_proj = nn.Sequential(
            nn.Linear(tabular_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2. Learnable Modality Type Encodings (0: CLS, 1: ECG, 2: Vision, 3: Tabular)
        self.modality_type_embed = nn.Embedding(4, embed_dim)

        # 3. Holistic Patient [CLS] Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 4. Multi-Head Cross-Attention Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),
        )

        # 5. Composite Risk Classification Head
        self.risk_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_risk_classes),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.modality_type_embed.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        ecg_latent: torch.Tensor,
        vision_latent: torch.Tensor,
        tabular_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            ecg_latent: (B, ecg_latent_dim)
            vision_latent: (B, vision_latent_dim)
            tabular_features: (B, tabular_dim)

        Returns:
            Tuple of:
                - risk_logits: (B, num_risk_classes)
                - patient_representation: (B, embed_dim)
        """
        B = ecg_latent.size(0)
        device = ecg_latent.device

        # Project modalities into common latent dimension
        tok_ecg = self.ecg_proj(ecg_latent).unsqueeze(1)       # (B, 1, D)
        tok_vis = self.vision_proj(vision_latent).unsqueeze(1)   # (B, 1, D)
        tok_tab = self.tabular_proj(tabular_features).unsqueeze(1) # (B, 1, D)

        # Expand [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)          # (B, 1, D)

        # Add modality type indicators
        mod_ids = torch.tensor([0, 1, 2, 3], device=device).unsqueeze(0).expand(B, -1)
        mod_embeds = self.modality_type_embed(mod_ids)          # (B, 4, D)

        # Sequence of 4 multimodal tokens
        tokens = torch.cat([cls_tokens, tok_ecg, tok_vis, tok_tab], dim=1) + mod_embeds

        # Transformer cross-attention
        encoded = self.transformer(tokens)

        # Extract patient representation from [CLS] token
        patient_representation = encoded[:, 0, :]

        # Predict clinical risk level
        risk_logits = self.risk_head(patient_representation)
        return risk_logits, patient_representation
