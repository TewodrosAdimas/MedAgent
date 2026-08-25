"""
Cross-Modal Attention Transformer for Unified Patient Representation Learning.
Fuses heterogeneous physiological biosignals, medical imaging, and clinical demographics
via self-attention and cross-modal token interaction.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from src.models.signals.cnn1d import ECGCNN1D
from src.models.vision.densenet import ChestXRayDenseNet121


class MultimodalTransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer Block capturing inter-modality cross-attention.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 512,
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_x = self.norm1(x)
        attn_out, attn_weights = self.attn(norm_x, norm_x, norm_x, need_weights=True)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights


class MultimodalTransformer(nn.Module):
    """
    Multimodal Patient Representation Transformer for MedAgent.

    Fuses:
      Token 0: [PATIENT_SUMMARY] (Learnable CLS token)
      Token 1: ECG Signal Embedding (Projected to embed_dim)
      Token 2: Chest X-Ray Visual Embedding (Projected to embed_dim)
      Token 3: Clinical Demographic Profile (Projected to embed_dim)
    """

    def __init__(
        self,
        num_risk_classes: int = 3,
        num_signal_classes: int = 5,
        num_image_classes: int = 5,
        tabular_dim: int = 8,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        mlp_dim: int = 512,
        dropout: float = 0.2,
        pretrained_vision: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # 1. Unimodal Feature Extractors
        self.signal_encoder = ECGCNN1D(num_classes=num_signal_classes)
        self.vision_encoder = ChestXRayDenseNet121(
            num_classes=num_image_classes, pretrained=pretrained_vision
        )

        # 2. Modality Projection Layers to uniform embed_dim
        self.sig_proj = nn.Sequential(
            nn.Linear(self.signal_encoder.latent_dim, embed_dim),  # 256 -> embed_dim
            nn.LayerNorm(embed_dim),
        )
        self.img_proj = nn.Sequential(
            nn.Linear(self.vision_encoder.latent_dim, embed_dim),  # 1024 -> embed_dim
            nn.LayerNorm(embed_dim),
        )
        self.tab_proj = nn.Sequential(
            nn.Linear(tabular_dim, embed_dim),                    # 8 -> embed_dim
            nn.LayerNorm(embed_dim),
        )

        # 3. Learnable Patient Summary Token & Modality Encodings (4 tokens total)
        self.patient_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.modality_type_embed = nn.Embedding(4, embed_dim)  # 0: Summary, 1: ECG, 2: Xray, 3: Tabular
        self.pos_drop = nn.Dropout(dropout)

        # 4. Multimodal Transformer Encoder Layers
        self.blocks = nn.ModuleList([
            MultimodalTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # 5. Multi-Task Heads
        self.risk_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_risk_classes),
        )
        self.sig_head = nn.Linear(embed_dim, num_signal_classes)
        self.img_head = nn.Linear(embed_dim, num_image_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.patient_token, std=0.02)

    def get_patient_embedding(
        self,
        ecg: torch.Tensor,
        image: torch.Tensor,
        tabular: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts unified patient representation and inter-modality attention map.

        Returns:
            Tuple of:
              - unified_patient_embedding: (B, embed_dim)
              - attention_matrix: (B, 4, 4)
        """
        B = ecg.shape[0]

        # Extract features
        sig_feat = self.signal_encoder.get_features(ecg)    # (B, 256)
        img_feat = self.vision_encoder.get_features(image)  # (B, 1024)

        # Project to modality tokens: (B, 1, embed_dim)
        sig_tok = self.sig_proj(sig_feat).unsqueeze(1)
        img_tok = self.img_proj(img_feat).unsqueeze(1)
        tab_tok = self.tab_proj(tabular).unsqueeze(1)
        pat_tok = self.patient_token.expand(B, -1, -1)

        # Construct multimodal sequence: (B, 4, embed_dim)
        tokens = torch.cat((pat_tok, sig_tok, img_tok, tab_tok), dim=1)

        # Add modality type embeddings
        device = ecg.device
        modality_ids = torch.tensor([0, 1, 2, 3], device=device).unsqueeze(0).expand(B, -1)
        tokens = tokens + self.modality_type_embed(modality_ids)
        tokens = self.pos_drop(tokens)

        # Transformer cross-attention forward pass
        last_attn = None
        for block in self.blocks:
            tokens, last_attn = block(tokens)

        tokens = self.norm(tokens)
        patient_summary = tokens[:, 0]  # Extract unified patient summary token

        return patient_summary, last_attn

    def forward(
        self,
        ecg: torch.Tensor,
        image: torch.Tensor,
        tabular: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        patient_emb, attn_weights = self.get_patient_embedding(ecg, image, tabular)

        risk_logits = self.risk_head(patient_emb)
        sig_logits = self.sig_head(patient_emb)
        img_logits = self.img_head(patient_emb)

        return {
            "risk_logits": risk_logits,
            "sig_logits": sig_logits,
            "img_logits": img_logits,
            "patient_embedding": patient_emb,
            "attention_weights": attn_weights,
        }
