"""
PyTorch Dataset and DataLoaders for ECG Arrhythmia Signals.

Supports:
1. Loading preprocessed .npz beat files (mitbih_aami_beats.npz).
2. Direct in-memory NumPy array ingestion.
3. On-the-fly extraction from raw PhysioNet/WFDB MIT-BIH recordings.
4. Signal augmentations: Gaussian noise, amplitude scaling, random baseline drift, and time jitter.
5. Stratified train/val/test splitting with balanced class-weight computation.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.data.preprocessing.ecg_cleaner import (
    AAMI_CLASSES,
    extract_beats_from_record,
)


class ECGAugmentation:
    """
    Data augmentation transforms for 1D ECG heartbeats during training.
    """

    def __init__(
        self,
        noise_std: float = 0.03,
        scale_min: float = 0.85,
        scale_max: float = 1.15,
        shift_max_samples: int = 6,
        prob_noise: float = 0.5,
        prob_scale: float = 0.5,
        prob_shift: float = 0.5,
    ):
        self.noise_std = noise_std
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.shift_max_samples = shift_max_samples
        self.prob_noise = prob_noise
        self.prob_scale = prob_scale
        self.prob_shift = prob_shift

    def __call__(self, beat: np.ndarray) -> np.ndarray:
        """
        Applies stochastic transformations to a 1D ECG beat.

        Args:
            beat: 1D NumPy array of shape (L,) or 2D of shape (1, L).

        Returns:
            Augmented ECG beat array.
        """
        aug_beat = beat.copy()

        # 1. Random amplitude scaling
        if np.random.rand() < self.prob_scale:
            factor = np.random.uniform(self.scale_min, self.scale_max)
            aug_beat = aug_beat * factor

        # 2. Additive Gaussian noise
        if np.random.rand() < self.prob_noise:
            noise = np.random.normal(0.0, self.noise_std, size=aug_beat.shape).astype(np.float32)
            aug_beat = aug_beat + noise

        # 3. Small temporal circular or padding shift
        if np.random.rand() < self.prob_shift and self.shift_max_samples > 0:
            shift = np.random.randint(-self.shift_max_samples, self.shift_max_samples + 1)
            if shift != 0:
                aug_beat = np.roll(aug_beat, shift, axis=-1)

        return aug_beat.astype(np.float32)


class ECGDataset(Dataset):
    """
    PyTorch Dataset for ECG heartbeat signals conforming to AAMI 5-class standard.
    """

    def __init__(
        self,
        data_source: Union[str, Path, np.ndarray, Dict[str, np.ndarray]],
        labels: Optional[np.ndarray] = None,
        record_origins: Optional[np.ndarray] = None,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        return_origins: bool = False,
    ):
        """
        Args:
            data_source: Either:
                - Path to a .npz file with 'beats' and 'labels' arrays
                - 2D NumPy array of beats of shape (N, total_samples)
                - Dictionary containing 'beats' and 'labels'
            labels: 1D array of class integer labels (if data_source is an array)
            record_origins: Optional array of record identifiers per beat
            transform: Optional augmentation callable
            return_origins: If True, __getitem__ returns (beat, label, origin)
        """
        self.transform = transform
        self.return_origins = return_origins

        if isinstance(data_source, (str, Path)):
            path = Path(data_source)
            if path.suffix == ".npz":
                loaded = np.load(str(path))
                self.beats = loaded["beats"].astype(np.float32)
                self.labels = loaded["labels"].astype(np.int64)
                self.record_origins = (
                    loaded["record_origins"]
                    if "record_origins" in loaded
                    else np.array(["unknown"] * len(self.labels))
                )
            else:
                raise ValueError(f"Unsupported file format for ECG data: {path.suffix}. Expected .npz.")
        elif isinstance(data_source, dict):
            self.beats = data_source["beats"].astype(np.float32)
            self.labels = data_source["labels"].astype(np.int64)
            self.record_origins = data_source.get("record_origins", np.array(["unknown"] * len(self.labels)))
        elif isinstance(data_source, np.ndarray):
            self.beats = data_source.astype(np.float32)
            if labels is None:
                raise ValueError("Labels must be provided when data_source is a NumPy array.")
            self.labels = labels.astype(np.int64)
            self.record_origins = (
                record_origins if record_origins is not None else np.array(["unknown"] * len(self.labels))
            )
        else:
            raise TypeError(f"Invalid data_source type: {type(data_source)}")

        # Ensure beats have 2D shape (N, sequence_length)
        if self.beats.ndim == 1:
            self.beats = self.beats.reshape(1, -1)

        assert len(self.beats) == len(self.labels), (
            f"Length mismatch: {len(self.beats)} beats vs {len(self.labels)} labels."
        )

    def __len__(self) -> int:
        return len(self.beats)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, str]]:
        beat = self.beats[idx]

        if self.transform is not None:
            beat = self.transform(beat)

        # PyTorch expects shape (channels, sequence_length), e.g. (1, 260)
        if beat.ndim == 1:
            beat_tensor = torch.from_numpy(beat).unsqueeze(0).float()
        else:
            beat_tensor = torch.from_numpy(beat).float()

        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.return_origins:
            return beat_tensor, label_tensor, str(self.record_origins[idx])
        return beat_tensor, label_tensor

    @classmethod
    def from_wfdb_records(
        cls,
        records_dir: Union[str, Path],
        record_ids: List[str],
        lead_name: str = "MLII",
        window_before: int = 90,
        window_after: int = 170,
        transform: Optional[Callable] = None,
    ) -> "ECGDataset":
        """
        Creates an ECGDataset by reading and parsing a list of raw PhysioNet records.
        """
        all_beats = []
        all_labels = []
        all_origins = []
        records_dir = Path(records_dir)

        for rec_id in record_ids:
            rec_path = records_dir / rec_id
            try:
                beats, labels, _ = extract_beats_from_record(
                    record_path_or_id=rec_path,
                    lead_name=lead_name,
                    window_before=window_before,
                    window_after=window_after,
                )
                if len(beats) > 0:
                    all_beats.append(beats)
                    all_labels.append(labels)
                    all_origins.extend([str(rec_id)] * len(labels))
            except Exception as e:
                print(f"Warning: Could not process record {rec_id}: {e}")

        if not all_beats:
            raise RuntimeError(f"No beats extracted from records in {records_dir}")

        merged_beats = np.concatenate(all_beats, axis=0)
        merged_labels = np.concatenate(all_labels, axis=0)
        merged_origins = np.array(all_origins)

        return cls(
            data_source=merged_beats,
            labels=merged_labels,
            record_origins=merged_origins,
            transform=transform,
        )


def get_ecg_dataloaders(
    npz_path: Union[str, Path],
    batch_size: int = 64,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    augment_train: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Creates stratified Train, Validation, and Test PyTorch DataLoaders from an .npz file,
    along with balanced inverse-frequency class weights for loss computation.

    Args:
        npz_path: Path to mitbih_aami_beats.npz
        batch_size: Batch size for loaders
        train_ratio: Proportion for training split (default: 0.70)
        val_ratio: Proportion for validation split (default: 0.15)
        test_ratio: Proportion for test split (default: 0.15)
        random_seed: Reproducibility seed
        augment_train: If True, applies ECGAugmentation to the training loader
        num_workers: Number of workers for DataLoader

    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_weights_tensor)
    """
    data = np.load(str(npz_path))
    beats = data["beats"].astype(np.float32)
    labels = data["labels"].astype(np.int64)
    origins = data["record_origins"] if "record_origins" in data else np.array(["unknown"] * len(labels))

    # First split: Train vs Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    indices = np.arange(len(labels))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=temp_ratio,
        random_state=random_seed,
        stratify=labels,
    )

    # Second split: Val vs Test
    relative_test_ratio = test_ratio / temp_ratio
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test_ratio,
        random_state=random_seed,
        stratify=labels[temp_idx],
    )

    # Compute balanced class weights based on training split
    present_classes = np.unique(labels[train_idx])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=present_classes,
        y=labels[train_idx],
    )
    # Initialize weights tensor for all 5 AAMI classes (default 1.0 for missing classes)
    full_weights = np.ones(len(AAMI_CLASSES), dtype=np.float32)
    for c, w in zip(present_classes, weights):
        full_weights[c] = w
    class_weights_tensor = torch.tensor(full_weights, dtype=torch.float32)

    # Instantiate datasets
    train_transform = ECGAugmentation() if augment_train else None

    train_dataset = ECGDataset(
        data_source=beats[train_idx],
        labels=labels[train_idx],
        record_origins=origins[train_idx],
        transform=train_transform,
    )
    val_dataset = ECGDataset(
        data_source=beats[val_idx],
        labels=labels[val_idx],
        record_origins=origins[val_idx],
        transform=None,
    )
    test_dataset = ECGDataset(
        data_source=beats[test_idx],
        labels=labels[test_idx],
        record_origins=origins[test_idx],
        transform=None,
    )

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
