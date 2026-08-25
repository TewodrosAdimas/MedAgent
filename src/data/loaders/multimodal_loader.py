"""
Multimodal Healthcare Data Loader & Dataset Module for MedAgent.
Pairs physiological ECG signals, Chest X-Ray radiographs, and clinical demographic
profiles into synchronized multimodal patient representations with joint risk targets.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.loaders.ecg_loader import extract_beats_from_records
from src.data.loaders.image_loader import (
    XRAY_CLASSES,
    ChestXRayDataset,
    create_synthetic_xray_dataset,
    get_xray_transforms,
)
from src.data.preprocessing.segmentation import AAMI_CLASS_NAMES
from src.utils.logger import get_logger

logger = get_logger("MultimodalLoader")

# Clinical Risk Target Categories
RISK_LEVELS: List[str] = [
    "Low Risk",
    "Moderate Risk",
    "High / Critical Risk",
]


def determine_clinical_risk(signal_label: int, image_label: int, age: float, has_hypertension: float) -> int:
    """
    Deterministic clinical risk stratification rule based on multimodal evidence:
    - Low Risk (0): Normal ECG (0) and Normal X-ray (0)
    - Moderate Risk (1): Single abnormality (e.g. S-beat or mild pulmonary finding)
    - High Risk (2): Multiple abnormalities (e.g. Ventricular ectopic beat + Pneumonia/Effusion)
                     OR abnormal finding in patient with advanced age & hypertension.
    """
    is_ecg_abnormal = (signal_label != 0)
    is_image_abnormal = (image_label != 0)

    # Ventricular ectopic (2) or Fusion (3) with lung opacity is high risk
    if (signal_label in [2, 3] and is_image_abnormal) or (signal_label == 2 and age >= 60 and has_hypertension):
        return 2  # High Risk
    elif is_ecg_abnormal and is_image_abnormal:
        return 2  # High Risk
    elif is_ecg_abnormal or is_image_abnormal:
        return 1  # Moderate Risk
    else:
        return 0  # Low Risk


class MultimodalPatientDataset(Dataset):
    """
    PyTorch Multimodal Dataset combining ECG waveforms, Chest X-Rays, and Clinical Demographics.
    """

    def __init__(
        self,
        ecg_beats: np.ndarray,
        ecg_labels: np.ndarray,
        image_dataset: ChestXRayDataset,
        num_samples: Optional[int] = None,
        random_seed: int = 42,
    ):
        self.rng = np.random.RandomState(random_seed)
        self.ecg_beats = ecg_beats
        self.ecg_labels = ecg_labels
        self.image_dataset = image_dataset

        self.num_samples = num_samples if num_samples is not None else min(len(ecg_beats), len(image_dataset))
        self._generate_aligned_pairs()

    def _generate_aligned_pairs(self):
        """Generates aligned multimodal patient samples with synthetic clinical demographics."""
        self.samples = []
        n_ecg = len(self.ecg_beats)
        n_img = len(self.image_dataset)

        for i in range(self.num_samples):
            ecg_idx = i % n_ecg
            img_idx = i % n_img

            sig_label = int(self.ecg_labels[ecg_idx])
            _, img_label_tensor = self.image_dataset[img_idx]
            img_label = int(img_label_tensor.item())

            # Generate realistic clinical demographic vector:
            # [age (normalized), sex (0/1), sbp (norm), dbp (norm), hr (norm), spo2 (norm), htn (0/1), dm (0/1)]
            age = float(self.rng.randint(25, 85))
            sex = float(self.rng.choice([0.0, 1.0]))  # 0: Female, 1: Male
            sbp = float(self.rng.normal(125, 18))
            dbp = float(self.rng.normal(82, 10))
            hr = float(self.rng.normal(78, 14))
            spo2 = float(np.clip(self.rng.normal(97, 2.5), 88, 100))
            htn = float(1.0 if (age > 55 and self.rng.rand() < 0.6) else 0.0)
            dm = float(1.0 if self.rng.rand() < 0.25 else 0.0)

            tabular_vector = np.array([
                (age - 55.0) / 20.0,
                sex,
                (sbp - 120.0) / 20.0,
                (dbp - 80.0) / 10.0,
                (hr - 75.0) / 15.0,
                (spo2 - 95.0) / 5.0,
                htn,
                dm,
            ], dtype=np.float32)

            risk_label = determine_clinical_risk(sig_label, img_label, age, htn)

            self.samples.append({
                "ecg_idx": ecg_idx,
                "img_idx": img_idx,
                "tabular": tabular_vector,
                "sig_label": sig_label,
                "img_label": img_label,
                "risk_label": risk_label,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_meta = self.samples[idx]

        # ECG Tensor: (1, 260)
        beat = self.ecg_beats[sample_meta["ecg_idx"]]
        ecg_tensor = torch.from_numpy(beat).unsqueeze(0).float()

        # Image Tensor: (3, 224, 224)
        image_tensor, _ = self.image_dataset[sample_meta["img_idx"]]

        # Tabular Tensor: (8,)
        tabular_tensor = torch.from_numpy(sample_meta["tabular"]).float()

        return {
            "ecg": ecg_tensor,
            "image": image_tensor,
            "tabular": tabular_tensor,
            "sig_label": torch.tensor(sample_meta["sig_label"], dtype=torch.long),
            "img_label": torch.tensor(sample_meta["img_label"], dtype=torch.long),
            "risk_label": torch.tensor(sample_meta["risk_label"], dtype=torch.long),
        }


def get_multimodal_dataloaders(
    batch_size: int = 32,
    num_samples: int = 1500,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, List[str]]]:
    """
    Creates multimodal train, validation, and test dataloaders.
    """
    # 1. Load ECG beats
    ecg_beats, ecg_labels, _ = extract_beats_from_records(use_cache=True)

    # 2. Load Chest X-ray images
    from src.config.paths import DATA_RAW_IMAGING_DIR
    img_dir = DATA_RAW_IMAGING_DIR / "synthetic_xray"
    if not img_dir.exists() or not any(img_dir.iterdir()):
        create_synthetic_xray_dataset(output_dir=img_dir, samples_per_class=60)

    image_paths = []
    image_labels = []
    for c_idx, c_name in enumerate(XRAY_CLASSES):
        class_folder = img_dir / c_name
        if class_folder.exists():
            for p in class_folder.glob("*.png"):
                image_paths.append(p)
                image_labels.append(c_idx)

    img_dataset = ChestXRayDataset(
        image_paths, image_labels, transform=get_xray_transforms(image_size=224, is_training=False)
    )

    full_dataset = MultimodalPatientDataset(
        ecg_beats=ecg_beats,
        ecg_labels=ecg_labels,
        image_dataset=img_dataset,
        num_samples=num_samples,
        random_seed=random_seed,
    )

    # Split dataset
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(full_dataset))
    risk_targets = [s["risk_label"] for s in full_dataset.samples]

    total_test_val = val_split + test_split
    val_relative_ratio = val_split / total_test_val

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=total_test_val,
        stratify=risk_targets,
        random_state=random_seed,
    )

    # Check if all classes in temp_idx have at least 2 members for second split stratification
    temp_targets = [risk_targets[i] for i in temp_idx]
    counts = np.bincount(temp_targets)
    can_stratify_temp = bool(np.all(counts[np.unique(temp_targets)] >= 2))

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - val_relative_ratio),
        stratify=temp_targets if can_stratify_temp else None,
        random_state=random_seed,
    )


    from torch.utils.data import Subset
    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)
    test_subset = Subset(full_dataset, test_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    metadata = {
        "risk_classes": RISK_LEVELS,
        "signal_classes": AAMI_CLASS_NAMES,
        "image_classes": XRAY_CLASSES,
    }

    logger.info(
        f"Multimodal DataLoaders created | Train: {len(train_subset)} | Val: {len(val_subset)} | Test: {len(test_subset)}"
    )

    return train_loader, val_loader, test_loader, metadata
