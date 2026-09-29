"""
Medical Image Transformations and Augmentation Pipelines for MedAgent.

Supports:
1. Standardized 224x224x3 ImageNet preprocessing for thoracic radiographs.
2. Clinically realistic data augmentations (mild rotation, affine translation, contrast/brightness jitter).
3. Deterministic evaluation transforms.
4. Denormalization utility for Grad-CAM overlay rendering and visualization.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# Standard ImageNet statistics for pretrained backbones (DenseNet121, ResNet, ViT)
DEFAULT_IMAGENET_MEAN = [0.485, 0.456, 0.406]
DEFAULT_IMAGENET_STD = [0.229, 0.224, 0.225]


def get_xray_transforms(
    image_size: int = 224,
    is_training: bool = True,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    augment_intensity: bool = True,
) -> transforms.Compose:
    """
    Returns torchvision transforms for chest X-ray images.

    Args:
        image_size: Target square resolution (default: 224).
        is_training: If True, includes stochastic augmentations.
        mean: Normalization channel means.
        std: Normalization channel standard deviations.
        augment_intensity: If True, applies mild brightness/contrast jitter.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    mean = mean if mean is not None else DEFAULT_IMAGENET_MEAN
    std = std if std is not None else DEFAULT_IMAGENET_STD

    if is_training:
        transform_list = [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.04, 0.04),
                scale=(0.96, 1.04),
            ),
        ]
        if augment_intensity:
            transform_list.append(
                transforms.ColorJitter(brightness=0.1, contrast=0.1)
            )
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        transform_list = [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]

    return transforms.Compose(transform_list)


def denormalize_image(
    tensor: Union[torch.Tensor, np.ndarray],
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Inverts ImageNet normalization to return a displayable RGB NumPy image in [0, 255] uint8.

    Args:
        tensor: Tensor of shape (3, H, W) or (1, 3, H, W).
        mean: Channel means used during normalization.
        std: Channel std used during normalization.

    Returns:
        NumPy uint8 array of shape (H, W, 3).
    """
    mean = np.array(mean if mean is not None else DEFAULT_IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(std if std is not None else DEFAULT_IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)

    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    if tensor.ndim == 4:
        tensor = tensor[0]

    # Invert: img = (tensor * std) + mean
    denorm = (tensor * std) + mean
    denorm = np.clip(denorm, 0.0, 1.0)

    # Transpose from (C, H, W) to (H, W, C) and scale to [0, 255]
    rgb = (np.transpose(denorm, (1, 2, 0)) * 255.0).astype(np.uint8)
    return rgb
