"""
PyTorch Dataset and DataLoaders for Thoracic Chest X-Ray Images.

Supports:
1. Scanning directory structures with 5 thoracic abnormality classes:
   (0: Normal, 1: Pneumonia, 2: Cardiomegaly, 3: Atelectasis, 4: Pleural Effusion).
2. Direct image list / metadata dataframe loading.
3. Training augmentations and deterministic evaluation transforms.
4. Stratified Train/Val/Test splitting with balanced class weights.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.data.preprocessing.image_transforms import get_xray_transforms

# Standard Class ID to Name mapping
XRAY_CLASSES: Dict[int, str] = {
    0: "Normal",
    1: "Pneumonia",
    2: "Cardiomegaly",
    3: "Atelectasis",
    4: "Pleural Effusion",
}

# Mapping folder names (case-insensitive normalized) to class IDs
FOLDER_TO_CLASS_ID: Dict[str, int] = {
    "normal": 0,
    "pneumonia": 1,
    "cardiomegaly": 2,
    "atelectasis": 3,
    "pleural effusion": 4,
    "pleuraleffusion": 4,
    "effusion": 4,
}


class XRayDataset(Dataset):
    """
    PyTorch Dataset for Chest X-Ray images across 5 diagnostic categories.
    """

    def __init__(
        self,
        samples: Union[str, Path, List[Tuple[Union[str, Path], int]]],
        transform: Optional[Union[transforms.Compose, Callable]] = None,
        return_path: bool = False,
        cache_images: bool = False,
    ):
        """
        Args:
            samples: Either a root directory containing class subfolders,
                     or a list of (image_path, label_id) tuples.
            transform: Optional torchvision transform pipeline.
            return_path: If True, __getitem__ returns (image_tensor, label_tensor, path_str).
            cache_images: If True, caches PIL images in memory.
        """
        self.transform = transform
        self.return_path = return_path
        self.cache_images = cache_images
        self._cache: Dict[int, Image.Image] = {}

        if isinstance(samples, (str, Path)):
            root = Path(samples)
            if not root.exists():
                raise FileNotFoundError(f"X-Ray directory not found: {root}")
            self.samples: List[Tuple[Path, int]] = self._scan_directory(root)
        elif isinstance(samples, list):
            self.samples = [(Path(p), int(l)) for p, l in samples]
        else:
            raise TypeError(f"Unsupported samples type: {type(samples)}")

        if len(self.samples) == 0:
            raise ValueError("XRayDataset initialized with 0 images.")

        self.labels = np.array([lbl for _, lbl in self.samples], dtype=np.int64)

    def _scan_directory(self, root: Path) -> List[Tuple[Path, int]]:
        """Scans subfolders matching XRAY_CLASSES."""
        items: List[Tuple[Path, int]] = []
        for folder in sorted(root.iterdir()):
            if folder.is_dir():
                norm_name = folder.name.strip().lower()
                class_id = FOLDER_TO_CLASS_ID.get(norm_name, None)
                if class_id is None:
                    # Try matching by class name directly
                    for cid, cname in XRAY_CLASSES.items():
                        if cname.lower() == norm_name:
                            class_id = cid
                            break

                if class_id is not None:
                    image_files = list(folder.glob("*.png")) + list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg"))
                    for img_p in image_files:
                        items.append((img_p, class_id))
        return items

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, str]]:
        img_path, label = self.samples[idx]

        if self.cache_images and idx in self._cache:
            img = self._cache[idx]
        else:
            img = Image.open(img_path).convert("RGB")
            if self.cache_images:
                self._cache[idx] = img

        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            # Default minimal ToTensor
            img_tensor = transforms.ToTensor()(img)

        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.return_path:
            return img_tensor, label_tensor, str(img_path)
        return img_tensor, label_tensor


def get_xray_dataloaders(
    root_dir: Union[str, Path],
    batch_size: int = 16,
    image_size: int = 224,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    num_workers: int = 0,
    cache_images: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Creates stratified Train, Validation, and Test PyTorch DataLoaders for chest X-rays.

    Args:
        root_dir: Path to directory containing class subfolders (e.g. data/raw/imaging/synthetic_xray).
        batch_size: Batch size for loaders.
        image_size: Target square pixel dimension (default: 224).
        train_ratio: Proportion for train split.
        val_ratio: Proportion for validation split.
        test_ratio: Proportion for test split.
        random_seed: Reproducibility seed.
        num_workers: Number of workers for DataLoader.
        cache_images: Whether to cache in memory for speed.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_weights_tensor).
    """
    root_path = Path(root_dir)
    full_dataset = XRayDataset(samples=root_path)

    samples = full_dataset.samples
    labels = full_dataset.labels

    temp_ratio = val_ratio + test_ratio
    indices = np.arange(len(samples))

    # Split 1: Train vs Temp (Val + Test)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=temp_ratio,
        random_state=random_seed,
        stratify=labels,
    )

    # Split 2: Val vs Test
    rel_test_ratio = test_ratio / temp_ratio
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=rel_test_ratio,
        random_state=random_seed,
        stratify=labels[temp_idx],
    )

    # Compute balanced class weights on training split
    train_labels = labels[train_idx]
    present_classes = np.unique(train_labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=present_classes,
        y=train_labels,
    )
    full_weights = np.ones(len(XRAY_CLASSES), dtype=np.float32)
    for c, w in zip(present_classes, weights):
        full_weights[c] = w
    class_weights_tensor = torch.tensor(full_weights, dtype=torch.float32)

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]

    train_transform = get_xray_transforms(image_size=image_size, is_training=True)
    eval_transform = get_xray_transforms(image_size=image_size, is_training=False)

    train_dataset = XRayDataset(train_samples, transform=train_transform, cache_images=cache_images)
    val_dataset = XRayDataset(val_samples, transform=eval_transform, cache_images=cache_images)
    test_dataset = XRayDataset(test_samples, transform=eval_transform, cache_images=cache_images)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, class_weights_tensor
