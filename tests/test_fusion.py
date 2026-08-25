"""
Unit tests for MedAgent Multimodal Fusion Architectures & Cross-Modal Attention.
"""

import numpy as np
import pytest
import torch

from src.data.loaders.multimodal_loader import determine_clinical_risk
from src.models.fusion.early_fusion import EarlyFusionModel
from src.models.fusion.late_fusion import LateFusionModel
from src.models.fusion.multimodal_transformer import MultimodalTransformer


class TestMultimodalModels:
    def test_early_fusion_forward(self):
        batch_size = 4
        model = EarlyFusionModel(
            num_risk_classes=3,
            num_signal_classes=5,
            num_image_classes=5,
            tabular_dim=8,
            joint_hidden=128,
            pretrained_vision=False,
        )

        ecg = torch.randn(batch_size, 1, 260)
        img = torch.randn(batch_size, 3, 224, 224)
        tab = torch.randn(batch_size, 8)

        out = model(ecg, img, tab)

        assert out["risk_logits"].shape == (batch_size, 3)
        assert out["sig_logits"].shape == (batch_size, 5)
        assert out["img_logits"].shape == (batch_size, 5)
        assert out["patient_embedding"].shape == (batch_size, 128)
        assert not torch.isnan(out["risk_logits"]).any()

    def test_late_fusion_forward(self):
        batch_size = 4
        model = LateFusionModel(
            num_risk_classes=3,
            num_signal_classes=5,
            num_image_classes=5,
            tabular_dim=8,
            pretrained_vision=False,
        )

        ecg = torch.randn(batch_size, 1, 260)
        img = torch.randn(batch_size, 3, 224, 224)
        tab = torch.randn(batch_size, 8)

        out = model(ecg, img, tab)

        assert out["risk_logits"].shape == (batch_size, 3)
        assert out["modality_weights"].shape == (batch_size, 3)
        # Weights should sum to 1
        sums = torch.sum(out["modality_weights"], dim=1)
        assert torch.allclose(sums, torch.ones(batch_size), atol=1e-4)

    def test_multimodal_transformer_forward(self):
        batch_size = 2
        model = MultimodalTransformer(
            num_risk_classes=3,
            num_signal_classes=5,
            num_image_classes=5,
            tabular_dim=8,
            embed_dim=128,
            num_heads=4,
            num_layers=2,
            pretrained_vision=False,
        )

        ecg = torch.randn(batch_size, 1, 260)
        img = torch.randn(batch_size, 3, 224, 224)
        tab = torch.randn(batch_size, 8)

        out = model(ecg, img, tab)

        assert out["risk_logits"].shape == (batch_size, 3)
        assert out["patient_embedding"].shape == (batch_size, 128)
        assert out["attention_weights"].shape == (batch_size, 4, 4)
        assert not torch.isnan(out["risk_logits"]).any()

    def test_multimodal_gradient_flow(self):
        model = MultimodalTransformer(
            num_risk_classes=3,
            embed_dim=64,
            num_heads=2,
            num_layers=1,
            pretrained_vision=False,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        ecg = torch.randn(2, 1, 260)
        img = torch.randn(2, 3, 224, 224)
        tab = torch.randn(2, 8)
        targets = torch.tensor([0, 2], dtype=torch.long)

        out = model(ecg, img, tab)
        loss = torch.nn.CrossEntropyLoss()(out["risk_logits"], targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0


class TestClinicalRiskLogic:
    def test_risk_rules(self):
        # Normal + Normal = Low Risk (0)
        assert determine_clinical_risk(signal_label=0, image_label=0, age=40, has_hypertension=0) == 0
        # Normal ECG + Pneumonia = Moderate Risk (1)
        assert determine_clinical_risk(signal_label=0, image_label=1, age=40, has_hypertension=0) == 1
        # Ventricular Ectopic Beat + Pneumonia = High Risk (2)
        assert determine_clinical_risk(signal_label=2, image_label=1, age=40, has_hypertension=0) == 2
        # Ventricular Ectopic Beat in Elderly Hypertensive = High Risk (2)
        assert determine_clinical_risk(signal_label=2, image_label=0, age=65, has_hypertension=1) == 2
