"""
Unit tests for MedAgent Medical Imaging (Vision) Pipeline & Grad-CAM Explainability.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.loaders.image_loader import (
    XRAY_CLASSES,
    ChestXRayDataset,
    generate_synthetic_xray_image,
    get_xray_transforms,
)
from src.evaluation.explainability import (
    GradCAM,
    identify_anatomical_quadrants,
    overlay_cam_on_image,
)
from src.models.vision.densenet import ChestXRayDenseNet121
from src.models.vision.vit import ChestXRayViT


class TestVisionPreprocessing:
    def test_synthetic_image_generator(self):
        for pathology in XRAY_CLASSES:
            img = generate_synthetic_xray_image(pathology=pathology, image_size=224)
            assert isinstance(img, Image.Image)
            assert img.size == (224, 224)

    def test_transforms_pipeline(self):
        img = Image.new("RGB", (300, 300), color=(100, 100, 100))
        transform = get_xray_transforms(image_size=224, is_training=True)
        tensor = transform(img)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)
        assert not torch.isnan(tensor).any()


class TestVisionModels:
    def test_densenet_forward_and_features(self):
        batch_size = 4
        num_classes = 5

        model = ChestXRayDenseNet121(num_classes=num_classes, pretrained=False)
        dummy_images = torch.randn(batch_size, 3, 224, 224)

        logits = model(dummy_images)
        features = model.get_features(dummy_images)

        assert logits.shape == (batch_size, num_classes)
        assert features.shape == (batch_size, 1024)
        assert not torch.isnan(logits).any()

    def test_vit_forward_and_features(self):
        batch_size = 2
        num_classes = 5

        model = ChestXRayViT(
            img_size=224,
            patch_size=16,
            num_classes=num_classes,
            embed_dim=128,
            depth=2,
            num_heads=4,
        )
        dummy_images = torch.randn(batch_size, 3, 224, 224)

        logits = model(dummy_images)
        features = model.get_features(dummy_images)

        assert logits.shape == (batch_size, num_classes)
        assert features.shape == (batch_size, 128)
        assert not torch.isnan(logits).any()


class TestGradCAMExplainability:
    def test_gradcam_generation(self):
        model = ChestXRayDenseNet121(num_classes=5, pretrained=False)
        target_layer = model.get_target_layer()

        grad_cam = GradCAM(model=model, target_layer=target_layer)
        dummy_image = torch.randn(1, 3, 224, 224)

        cam = grad_cam.generate_cam(dummy_image, target_class=1)

        assert isinstance(cam, np.ndarray)
        assert cam.shape == (224, 224)
        assert 0.0 <= cam.min()
        assert cam.max() <= 1.0

    def test_overlay_and_quadrants(self):
        dummy_img = np.ones((224, 224, 3), dtype=np.uint8) * 128
        cam = np.zeros((224, 224), dtype=np.float32)
        cam[150:200, 30:80] = 0.9  # Saliency hotspot in right lower lung

        overlay = overlay_cam_on_image(dummy_img, cam, alpha=0.5)
        assert overlay.shape == (224, 224, 3)
        assert overlay.dtype == np.uint8

        regions = identify_anatomical_quadrants(cam, threshold=0.5)
        assert len(regions) > 0
        assert regions[0]["region"] == "Right Lower Lung"
