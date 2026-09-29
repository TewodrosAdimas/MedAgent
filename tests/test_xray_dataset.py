"""
Unit Tests for MedAgent Medical Imaging Preprocessing and X-Ray DataLoaders.
"""

from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

from src.data.preprocessing.image_transforms import (
    DEFAULT_IMAGENET_MEAN,
    DEFAULT_IMAGENET_STD,
    denormalize_image,
    get_xray_transforms,
)
from src.data.loaders.xray_dataset import (
    XRAY_CLASSES,
    XRayDataset,
    get_xray_dataloaders,
)


def test_xray_transforms_train():
    transform = get_xray_transforms(image_size=224, is_training=True)
    img = Image.new("RGB", (300, 300), color=(128, 128, 128))
    tensor = transform(img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_xray_transforms_eval():
    transform = get_xray_transforms(image_size=224, is_training=False)
    img = Image.new("RGB", (400, 300), color=(100, 100, 100))
    tensor = transform(img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_denormalize_image():
    transform = get_xray_transforms(image_size=224, is_training=False)
    img = Image.new("RGB", (224, 224), color=(200, 150, 100))
    tensor = transform(img)

    denorm = denormalize_image(tensor)
    assert isinstance(denorm, np.ndarray)
    assert denorm.shape == (224, 224, 3)
    assert denorm.dtype == np.uint8
    assert denorm.min() >= 0
    assert denorm.max() <= 255


def test_xray_dataset_from_directory():
    xray_dir = Path("data/raw/imaging/synthetic_xray")
    if not xray_dir.exists():
        pytest.skip("Synthetic X-ray directory not found.")

    transform = get_xray_transforms(image_size=224, is_training=False)
    dataset = XRayDataset(samples=xray_dir, transform=transform, return_path=True)

    assert len(dataset) == 250  # 5 classes * 50 images
    img_tensor, label_tensor, path_str = dataset[0]

    assert img_tensor.shape == (3, 224, 224)
    assert label_tensor.dtype == torch.long
    assert 0 <= label_tensor.item() < len(XRAY_CLASSES)
    assert Path(path_str).exists()


def test_get_xray_dataloaders():
    xray_dir = Path("data/raw/imaging/synthetic_xray")
    if not xray_dir.exists():
        pytest.skip("Synthetic X-ray directory not found.")

    train_loader, val_loader, test_loader, weights = get_xray_dataloaders(
        root_dir=xray_dir,
        batch_size=16,
        image_size=224,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
    )

    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0
    assert weights.shape == (len(XRAY_CLASSES),)

    batch_imgs, batch_lbls = next(iter(train_loader))
    assert batch_imgs.shape == (16, 3, 224, 224)
    assert batch_lbls.shape == (16,)
    assert batch_lbls.min() >= 0
    assert batch_lbls.max() < len(XRAY_CLASSES)
