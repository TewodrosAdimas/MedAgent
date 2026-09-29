"""
PyTorch Multimodal Dataset and DataLoaders for MedAgent.

Pairs synchronized:
1. Physiological ECG Waveforms (1, 260)
2. Thoracic Chest Radiographs (3, 224, 224)
3. Tabular Clinical Context (Age, Sex, Vital Signs, Comorbidities)
4. Multi-Task Labels:
   - ECG Arrhythmia Class (0: Normal, 1: S, 2: V, 3: F, 4: Q)
   - X-Ray Diagnostic Class (0: Normal, 1: Pneumonia, 2: Cardiomegaly, 3: Atelectasis, 4: Pleural Effusion)
   - Composite Clinical Risk (0: Low Risk, 1: Moderate Risk, 2: High / Critical Risk)
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.model_selection import train_test_split

from src.data.preprocessing.image_transforms import get_xray_transforms
from src.data.loaders.ecg_dataset import ECGAugmentation
from src.data.loaders.xray_dataset import XRAY_CLASSES, XRayDataset

# Clinical Multimodal Risk Tiers
RISK_CLASSES: Dict[int, str] = {
    0: "Low Risk",
    1: "Moderate Risk",
    2: "High / Critical Risk",
}

# Tabular Feature Indices
TABULAR_FEATURE_NAMES = [
    "normalized_age",           # Age / 100.0
    "sex_is_male",              # 1.0 = Male, 0.0 = Female
    "normalized_heart_rate",    # HR / 200.0
    "normalized_systolic_bp",   # SBP / 200.0
    "normalized_diastolic_bp",  # DBP / 120.0
    "normalized_spo2",          # SpO2 / 100.0
    "hypertension",             # 1.0 = Yes, 0.0 = No
    "diabetes",                 # 1.0 = Yes, 0.0 = No
    "smoking_history",          # 1.0 = Yes, 0.0 = No
    "prior_cardiac_event",      # 1.0 = Yes, 0.0 = No
]


def determine_clinical_risk(
    ecg_label: int,
    xray_label: int,
    has_hypertension: bool = False,
    has_prior_cardiac: bool = False,
    is_hypoxic: bool = False,
) -> int:
    """
    Clinically realistic multi-criteria heuristic for composite patient risk stratification:
    - Low Risk (0): Normal ECG (0) + Normal X-ray (0) + stable vitals.
    - Moderate Risk (1): Isolated supraventricular ectopy (1) OR isolated mild pulmonary finding
      (e.g., atelectasis 3) OR hypertension with otherwise normal findings.
    - High / Critical Risk (2): Ventricular ectopy (2), Fusion (3), Cardiomegaly (2),
      Pneumonia (1), Pleural Effusion (4) with abnormal ECG, or multiorgan comorbidities.
    """
    # High-risk conditions
    is_severe_arrhythmia = ecg_label in [2, 3]  # Ventricular or Fusion
    is_severe_imaging = xray_label in [1, 2, 4]  # Pneumonia, Cardiomegaly, Pleural Effusion
    is_both_abnormal = (ecg_label != 0) and (xray_label != 0)

    if is_severe_arrhythmia or (is_severe_imaging and (ecg_label != 0 or has_prior_cardiac)) or is_both_abnormal or is_hypoxic:
        return 2  # High / Critical Risk
    elif ecg_label != 0 or xray_label != 0 or has_hypertension or has_prior_cardiac:
        return 1  # Moderate Risk
    else:
        return 0  # Low Risk


class MultimodalDataset(Dataset):
    """
    Unified PyTorch Dataset for paired ECG beats, Chest X-rays, and tabular clinical records.
    """

    def __init__(
        self,
        ecg_beats: np.ndarray,
        ecg_labels: np.ndarray,
        xray_paths: List[Union[str, Path]],
        xray_labels: np.ndarray,
        tabular_records: np.ndarray,
        risk_labels: Optional[np.ndarray] = None,
        patient_ids: Optional[List[str]] = None,
        ecg_transform: Optional[Callable] = None,
        xray_transform: Optional[Callable] = None,
        cache_images: bool = False,
    ):
        """
        Args:
            ecg_beats: NumPy array of shape (N, 260) or (N, 1, 260)
            ecg_labels: 1D array of AAMI ECG classes
            xray_paths: List of N file paths to chest X-rays
            xray_labels: 1D array of X-ray diagnostic classes
            tabular_records: NumPy array of shape (N, num_features)
            risk_labels: Optional 1D array of composite risk labels (0: Low, 1: Med, 2: High)
            patient_ids: Optional list of patient identifiers
            ecg_transform: Optional ECG augmentation transform
            xray_transform: Optional image torchvision transform
            cache_images: Whether to cache PIL images in memory
        """
        self.ecg_beats = ecg_beats.astype(np.float32)
        self.ecg_labels = ecg_labels.astype(np.int64)
        self.xray_paths = [Path(p) for p in xray_paths]
        self.xray_labels = xray_labels.astype(np.int64)
        self.tabular_records = tabular_records.astype(np.float32)

        N = len(self.ecg_beats)
        assert len(self.xray_paths) == N, f"Mismatch: {N} ECG beats vs {len(self.xray_paths)} X-rays"
        assert len(self.tabular_records) == N, f"Mismatch: {N} ECG beats vs {len(self.tabular_records)} tabular records"

        if risk_labels is not None:
            self.risk_labels = risk_labels.astype(np.int64)
        else:
            # Auto-calculate risk labels
            calculated = []
            for i in range(N):
                hyp = bool(self.tabular_records[i, 6] > 0.5)
                prior = bool(self.tabular_records[i, 9] > 0.5)
                spo2_norm = self.tabular_records[i, 5]
                hypoxic = bool(spo2_norm < 0.92)  # < 92% SpO2
                r = determine_clinical_risk(
                    ecg_label=self.ecg_labels[i],
                    xray_label=self.xray_labels[i],
                    has_hypertension=hyp,
                    has_prior_cardiac=prior,
                    is_hypoxic=hypoxic,
                )
                calculated.append(r)
            self.risk_labels = np.array(calculated, dtype=np.int64)

        self.patient_ids = (
            patient_ids if patient_ids is not None else [f"PT-{i+1:05d}" for i in range(N)]
        )
        self.ecg_transform = ecg_transform
        self.xray_transform = xray_transform
        self.cache_images = cache_images
        self._image_cache: Dict[int, Image.Image] = {}

    def __len__(self) -> int:
        return len(self.ecg_beats)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        # 1. ECG Signal
        beat = self.ecg_beats[idx]
        if self.ecg_transform is not None:
            beat = self.ecg_transform(beat)
        if beat.ndim == 1:
            ecg_tensor = torch.from_numpy(beat).unsqueeze(0).float()
        else:
            ecg_tensor = torch.from_numpy(beat).float()

        # 2. Chest X-Ray
        if self.cache_images and idx in self._image_cache:
            img = self._image_cache[idx]
        else:
            img_path = self.xray_paths[idx]
            img = Image.open(img_path).convert("RGB")
            if self.cache_images:
                self._image_cache[idx] = img

        if self.xray_transform is not None:
            xray_tensor = self.xray_transform(img)
        else:
            xray_tensor = get_xray_transforms(is_training=False)(img)

        # 3. Tabular Clinical Context
        tabular_tensor = torch.from_numpy(self.tabular_records[idx]).float()

        return {
            "ecg": ecg_tensor,
            "image": xray_tensor,
            "tabular": tabular_tensor,
            "ecg_label": torch.tensor(self.ecg_labels[idx], dtype=torch.long),
            "xray_label": torch.tensor(self.xray_labels[idx], dtype=torch.long),
            "risk_label": torch.tensor(self.risk_labels[idx], dtype=torch.long),
            "patient_id": self.patient_ids[idx],
        }

    @classmethod
    def create_synthetic_cohort(
        cls,
        ecg_npz_path: Union[str, Path],
        xray_root_dir: Union[str, Path],
        num_samples: int = 500,
        random_seed: int = 42,
        is_training: bool = False,
        cache_images: bool = False,
    ) -> "MultimodalDataset":
        """
        Synthesizes a realistic paired multimodal patient cohort by aligning
        ECG signals from MIT-BIH, chest radiographs from synthetic_xray, and
        statistically correlated patient demographic & vital records.
        """
        np.random.seed(random_seed)
        
        # Load ECG data
        ecg_data = np.load(str(ecg_npz_path))
        raw_beats = ecg_data["beats"]
        raw_ecg_labels = ecg_data["labels"]

        # Load X-Ray data
        xray_dataset = XRayDataset(samples=xray_root_dir)
        xray_samples = xray_dataset.samples  # List of (Path, label_id)

        # Sample indices
        ecg_indices = np.random.choice(len(raw_beats), size=num_samples, replace=(num_samples > len(raw_beats)))
        xray_indices = np.random.choice(len(xray_samples), size=num_samples, replace=(num_samples > len(xray_samples)))

        sampled_beats = raw_beats[ecg_indices]
        sampled_ecg_labels = raw_ecg_labels[ecg_indices]

        sampled_xray_paths = [xray_samples[i][0] for i in xray_indices]
        sampled_xray_labels = np.array([xray_samples[i][1] for i in xray_indices], dtype=np.int64)

        # Generate correlated tabular patient records
        tabular_list = []
        patient_ids = []

        for i in range(num_samples):
            e_lbl = sampled_ecg_labels[i]
            x_lbl = sampled_xray_labels[i]

            # Age: older patients have higher probability of comorbidities
            age = np.random.randint(25, 85)
            sex = np.random.choice([0.0, 1.0])  # 0: Female, 1: Male

            # Heart Rate: elevated in arrhythmias / pneumonia
            base_hr = 72.0
            if e_lbl in [1, 2]:
                base_hr += np.random.uniform(15, 35)
            elif x_lbl == 1:  # Pneumonia
                base_hr += np.random.uniform(10, 25)
            hr = np.clip(base_hr + np.random.normal(0, 5), 45.0, 180.0)

            # Blood Pressure
            sbp = 120.0 + (age - 40) * 0.3 + (15.0 if e_lbl != 0 else 0.0) + np.random.normal(0, 8)
            dbp = 80.0 + (age - 40) * 0.1 + np.random.normal(0, 5)

            # SpO2: lower in pneumonia or pleural effusion
            base_spo2 = 98.0
            if x_lbl in [1, 4]:  # Pneumonia or Pleural Effusion
                base_spo2 -= np.random.uniform(4, 10)
            spo2 = np.clip(base_spo2 + np.random.normal(0, 1), 85.0, 100.0)

            # Comorbidities
            htn = 1.0 if (sbp > 135.0 or np.random.rand() < 0.35) else 0.0
            dm = 1.0 if (age > 50 and np.random.rand() < 0.25) else 0.0
            smk = 1.0 if np.random.rand() < 0.30 else 0.0
            cardiac = 1.0 if (e_lbl in [2, 3] or x_lbl == 2 or np.random.rand() < 0.15) else 0.0

            row = [
                age / 100.0,
                sex,
                hr / 200.0,
                sbp / 200.0,
                dbp / 120.0,
                spo2 / 100.0,
                htn,
                dm,
                smk,
                cardiac,
            ]
            tabular_list.append(row)
            patient_ids.append(f"MED-PT-{i+1:04d}")

        tabular_arr = np.array(tabular_list, dtype=np.float32)

        ecg_trans = ECGAugmentation() if is_training else None
        xray_trans = get_xray_transforms(image_size=224, is_training=is_training)

        return cls(
            ecg_beats=sampled_beats,
            ecg_labels=sampled_ecg_labels,
            xray_paths=sampled_xray_paths,
            xray_labels=sampled_xray_labels,
            tabular_records=tabular_arr,
            patient_ids=patient_ids,
            ecg_transform=ecg_trans,
            xray_transform=xray_trans,
            cache_images=cache_images,
        )


def get_multimodal_dataloaders(
    ecg_npz_path: Union[str, Path],
    xray_root_dir: Union[str, Path],
    num_samples: int = 500,
    batch_size: int = 16,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    num_workers: int = 0,
    cache_images: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates stratified Train, Validation, and Test PyTorch DataLoaders for multimodal patient data.

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    full_dataset = MultimodalDataset.create_synthetic_cohort(
        ecg_npz_path=ecg_npz_path,
        xray_root_dir=xray_root_dir,
        num_samples=num_samples,
        random_seed=random_seed,
        is_training=False,
        cache_images=cache_images,
    )

    indices = np.arange(len(full_dataset))
    risk_labels = full_dataset.risk_labels

    temp_ratio = val_ratio + test_ratio

    # Check if stratification is possible (every class has at least 2 samples)
    _, counts_all = np.unique(risk_labels, return_counts=True)
    stratify_1 = risk_labels if (len(counts_all) > 1 and np.min(counts_all) >= 2) else None

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=temp_ratio,
        random_state=random_seed,
        stratify=stratify_1,
    )

    rel_test_ratio = test_ratio / temp_ratio
    _, counts_temp = np.unique(risk_labels[temp_idx], return_counts=True)
    stratify_2 = risk_labels[temp_idx] if (len(counts_temp) > 1 and np.min(counts_temp) >= 2) else None

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=rel_test_ratio,
        random_state=random_seed,
        stratify=stratify_2,
    )

    train_dataset = MultimodalDataset(
        ecg_beats=full_dataset.ecg_beats[train_idx],
        ecg_labels=full_dataset.ecg_labels[train_idx],
        xray_paths=[full_dataset.xray_paths[i] for i in train_idx],
        xray_labels=full_dataset.xray_labels[train_idx],
        tabular_records=full_dataset.tabular_records[train_idx],
        risk_labels=full_dataset.risk_labels[train_idx],
        patient_ids=[full_dataset.patient_ids[i] for i in train_idx],
        ecg_transform=ECGAugmentation(),
        xray_transform=get_xray_transforms(is_training=True),
        cache_images=cache_images,
    )

    val_dataset = MultimodalDataset(
        ecg_beats=full_dataset.ecg_beats[val_idx],
        ecg_labels=full_dataset.ecg_labels[val_idx],
        xray_paths=[full_dataset.xray_paths[i] for i in val_idx],
        xray_labels=full_dataset.xray_labels[val_idx],
        tabular_records=full_dataset.tabular_records[val_idx],
        risk_labels=full_dataset.risk_labels[val_idx],
        patient_ids=[full_dataset.patient_ids[i] for i in val_idx],
        ecg_transform=None,
        xray_transform=get_xray_transforms(is_training=False),
        cache_images=cache_images,
    )

    test_dataset = MultimodalDataset(
        ecg_beats=full_dataset.ecg_beats[test_idx],
        ecg_labels=full_dataset.ecg_labels[test_idx],
        xray_paths=[full_dataset.xray_paths[i] for i in test_idx],
        xray_labels=full_dataset.xray_labels[test_idx],
        tabular_records=full_dataset.tabular_records[test_idx],
        risk_labels=full_dataset.risk_labels[test_idx],
        patient_ids=[full_dataset.patient_ids[i] for i in test_idx],
        ecg_transform=None,
        xray_transform=get_xray_transforms(is_training=False),
        cache_images=cache_images,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
