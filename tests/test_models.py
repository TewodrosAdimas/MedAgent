"""
Unit Tests for MedAgent Deep Learning Architectures and Explainability (Grad-CAM).
"""

import numpy as np
import pytest
import torch

from src.models.signals.ecg_cnn import ECG1DCNN
from src.models.signals.ecg_transformer import ECGTransformer
from src.models.vision.xray_densenet import XRayDenseNet121
from src.models.vision.gradcam import GradCAM
from src.models.fusion.multimodal_transformer import MultimodalTransformerFusion


def test_ecg_1d_cnn_forward():
    model = ECG1DCNN(in_channels=1, num_classes=5, latent_dim=256)
    model.eval()

    batch_size = 4
    x = torch.randn(batch_size, 1, 260)
    logits, latent = model(x)

    assert logits.shape == (batch_size, 5)
    assert latent.shape == (batch_size, 256)
    assert not torch.isnan(logits).any()
    assert not torch.isnan(latent).any()


def test_ecg_transformer_forward():
    model = ECGTransformer(
        sequence_length=260,
        patch_size=10,
        in_channels=1,
        num_classes=5,
        embed_dim=256,
        num_heads=4,
        num_layers=2,
    )
    model.eval()

    batch_size = 4
    x = torch.randn(batch_size, 1, 260)
    logits, latent = model(x)

    assert logits.shape == (batch_size, 5)
    assert latent.shape == (batch_size, 256)
    assert not torch.isnan(logits).any()
    assert not torch.isnan(latent).any()


def test_xray_densenet_forward():
    model = XRayDenseNet121(num_classes=5, pretrained=False, latent_dim=256)
    model.eval()

    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)
    logits, latent = model(x)

    assert logits.shape == (batch_size, 5)
    assert latent.shape == (batch_size, 256)
    assert not torch.isnan(logits).any()


def test_gradcam_heatmap_and_overlay():
    model = XRayDenseNet121(num_classes=5, pretrained=False, latent_dim=256)
    gradcam = GradCAM(model)

    x = torch.randn(1, 3, 224, 224)
    heatmap = gradcam.generate_heatmap(x, target_class=1)

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (224, 224)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0

    overlay, _ = gradcam.generate_overlay(x, target_class=1, alpha=0.5)
    assert isinstance(overlay, np.ndarray)
    assert overlay.shape == (224, 224, 3)
    assert overlay.dtype == np.uint8

    gradcam.remove_hooks()


def test_multimodal_transformer_fusion_forward():
    fusion_model = MultimodalTransformerFusion(
        ecg_latent_dim=256,
        vision_latent_dim=256,
        tabular_dim=10,
        embed_dim=256,
        num_heads=4,
        num_layers=2,
        num_risk_classes=3,
    )
    fusion_model.eval()

    batch_size = 4
    ecg_lat = torch.randn(batch_size, 256)
    vis_lat = torch.randn(batch_size, 256)
    tab_feat = torch.randn(batch_size, 10)

    risk_logits, patient_rep = fusion_model(ecg_lat, vis_lat, tab_feat)

    assert risk_logits.shape == (batch_size, 3)
    assert patient_rep.shape == (batch_size, 256)
    assert not torch.isnan(risk_logits).any()
    assert not torch.isnan(patient_rep).any()
