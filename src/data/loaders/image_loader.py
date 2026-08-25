"""
Medical Imaging Data Loader & Dataset Module for MedAgent.
Handles Chest X-ray loading, radiological transformations, synthetic dataset creation,
and PyTorch DataLoader generation for multi-pathology classification.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config.paths import DATA_PROCESSED_IMAGES_DIR, DATA_RAW_IMAGING_DIR
from src.utils.logger import get_logger

logger = get_logger("ImageLoader")

# Standard Chest X-Ray Pathology Classes
XRAY_CLASSES: List[str] = [
    "Normal",
    "Pneumonia",
    "Cardiomegaly",
    "Atelectasis",
    "Pleural Effusion",
]


def get_xray_transforms(
    image_size: int = 224,
    is_training: bool = True,
) -> transforms.Compose:
    """
    Returns standardized radiological image transformation pipeline.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])


def generate_synthetic_xray_image(
    pathology: str = "Normal",
    image_size: int = 224,
) -> Image.Image:
    """
    Generates a realistic synthetic Chest X-ray image structure for testing and verification.
    Simulates rib cage, lung fields, mediastinum, cardiac silhouette, and pathology opacities.
    """
    # Background: dark tissue
    base = Image.new("L", (image_size, image_size), color=25)
    draw = ImageDraw.Draw(base)

    # Thoracic cavity & rib outline
    draw.ellipse([20, 20, image_size - 20, image_size - 10], fill=60)

    # Left & Right lung fields (radiolucent = darker)
    mid_x = image_size // 2
    # Left lung (patient's anatomical right)
    draw.ellipse([35, 40, mid_x - 10, image_size - 30], fill=15)
    # Right lung (patient's anatomical left)
    draw.ellipse([mid_x + 10, 40, image_size - 35, image_size - 30], fill=15)

    # Spine & Mediastinum (radio-opaque = brighter)
    draw.rectangle([mid_x - 12, 20, mid_x + 12, image_size - 20], fill=180)

    # Cardiac silhouette
    heart_width = 55 if pathology != "Cardiomegaly" else 85
    draw.ellipse(
        [mid_x - 20, image_size // 2 - 10, mid_x + heart_width, image_size - 35],
        fill=190,
    )

    # Add specific pathological opacities
    if pathology == "Pneumonia":
        # Patchy consolidation opacity in right lower lobe
        draw.ellipse([mid_x + 20, image_size - 90, image_size - 45, image_size - 40], fill=140)
    elif pathology == "Pleural Effusion":
        # Blunting of costophrenic angle
        draw.polygon(
            [(35, image_size - 40), (mid_x - 15, image_size - 30), (35, image_size - 70)],
            fill=160,
        )
    elif pathology == "Atelectasis":
        # Linear/band opacity in lung base
        draw.line([(mid_x + 15, image_size - 60), (image_size - 40, image_size - 50)], fill=150, width=5)

    # Apply Gaussian blur for smooth radiographic density gradients
    xray = base.filter(ImageFilter.GaussianBlur(radius=3))
    return xray.convert("RGB")


def create_synthetic_xray_dataset(
    output_dir: Optional[Path] = None,
    samples_per_class: int = 40,
) -> Path:
    """
    Creates and saves a structured synthetic Chest X-ray dataset.
    """
    if output_dir is None:
        output_dir = DATA_RAW_IMAGING_DIR / "synthetic_xray"

    output_dir.mkdir(parents=True, exist_ok=True)

    for c_name in XRAY_CLASSES:
        class_dir = output_dir / c_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for i in range(samples_per_class):
            img_path = class_dir / f"{c_name.lower().replace(' ', '_')}_{i:04d}.png"
            if not img_path.exists():
                img = generate_synthetic_xray_image(pathology=c_name)
                img.save(img_path)

    logger.info(f"Synthetic Chest X-ray dataset verified at {output_dir} ({samples_per_class * len(XRAY_CLASSES)} images)")
    return output_dir


class ChestXRayDataset(Dataset):
    """
    PyTorch Dataset for Chest Radiograph Images.
    Returns (image_tensor, label_int) where image_tensor has shape (3, 224, 224).
    """

    def __init__(
        self,
        image_paths: List[Path],
        labels: List[int],
        transform: Optional[Callable] = None,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform if transform is not None else get_xray_transforms(is_training=False)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return image_tensor, label_tensor


def get_xray_dataloaders(
    data_dir: Optional[Path] = None,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42,
    image_size: int = 224,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Loads Chest X-ray dataset and creates Train, Validation, and Test DataLoaders.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names)
    """
    if data_dir is None:
        data_dir = DATA_RAW_IMAGING_DIR / "synthetic_xray"

    if not data_dir.exists() or not any(data_dir.iterdir()):
        logger.info(f"Imaging directory {data_dir} is empty. Generating synthetic X-ray dataset...")
        create_synthetic_xray_dataset(output_dir=data_dir, samples_per_class=50)

    # Collect image paths and labels
    image_paths = []
    labels = []

    for c_idx, c_name in enumerate(XRAY_CLASSES):
        class_folder = data_dir / c_name
        if class_folder.exists():
            for ext in ["*.png", "*.jpg", "*.jpeg"]:
                for p in class_folder.glob(ext):
                    image_paths.append(p)
                    labels.append(c_idx)

    if len(image_paths) == 0:
        # Check flat directory structure
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            for p in data_dir.glob(ext):
                image_paths.append(p)
                labels.append(0)  # Default normal if unclassified

    labels_arr = np.array(labels, dtype=np.int64)

    from sklearn.model_selection import train_test_split

    total_test_val = val_split + test_split
    val_relative_ratio = val_split / total_test_val

    train_idx, temp_idx = train_test_split(
        np.arange(len(labels_arr)),
        test_size=total_test_val,
        stratify=labels_arr,
        random_state=random_seed,
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - val_relative_ratio),
        stratify=labels_arr[temp_idx],
        random_state=random_seed,
    )

    train_paths = [image_paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]

    val_paths = [image_paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    test_paths = [image_paths[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    train_dataset = ChestXRayDataset(
        train_paths, train_labels, transform=get_xray_transforms(image_size, is_training=True)
    )
    val_dataset = ChestXRayDataset(
        val_paths, val_labels, transform=get_xray_transforms(image_size, is_training=False)
    )
    test_dataset = ChestXRayDataset(
        test_paths, test_labels, transform=get_xray_transforms(image_size, is_training=False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(
        f"X-Ray DataLoaders created | Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader, XRAY_CLASSES
