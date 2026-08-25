"""
ECG Data Loader & PyTorch Dataset for MedAgent.
Parses the MIT-BIH Arrhythmia Database, extracts AAMI-classified heartbeats,
and provides stratified PyTorch DataLoaders with class imbalance weights.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import wfdb

from src.config.paths import DATA_PROCESSED_SIGNALS_DIR, DATA_RAW_MITBIH_DIR
from src.data.preprocessing.augmentation import ECGAugmentationPipeline
from src.data.preprocessing.normalization import zscore_normalize
from src.data.preprocessing.segmentation import (
    AAMI_CLASS_NAMES,
    extract_heartbeat_segments,
)
from src.data.preprocessing.signal_filtering import preprocess_ecg_signal
from src.utils.logger import get_logger

logger = get_logger("ECGLoader")

# Standard 48 MIT-BIH patient record IDs
MITBIH_RECORD_IDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]


def load_mitbih_record(
    record_id: str,
    raw_dir: Optional[Path] = None,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], int, List[str]]:
    """
    Loads a single MIT-BIH record and its annotations.

    Args:
        record_id: Record identifier string (e.g. '100')
        raw_dir: Directory containing MIT-BIH record files
        max_samples: Optional maximum samples to read

    Returns:
        Tuple of (p_signal, r_peaks, symbols, sampling_rate, signal_names)
    """
    if raw_dir is None:
        raw_dir = DATA_RAW_MITBIH_DIR

    record_path = raw_dir / record_id
    if not record_path.with_suffix(".hea").exists():
        raise FileNotFoundError(f"Record {record_id} header not found at {record_path}.hea")

    record = wfdb.rdrecord(str(record_path), sampto=max_samples)
    annotation = wfdb.rdann(str(record_path), "atr", sampto=max_samples)

    return (
        record.p_signal.astype(np.float32),
        annotation.sample,
        annotation.symbol,
        record.fs,
        record.sig_name,
    )


def extract_beats_from_records(
    record_ids: Optional[List[str]] = None,
    raw_dir: Optional[Path] = None,
    pre_r: int = 90,
    post_r: int = 170,
    apply_filter: bool = True,
    normalize: bool = True,
    lead_idx: int = 0,
    use_cache: bool = True,
    cache_name: str = "mitbih_aami_beats.npz",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts AAMI heartbeats from multiple MIT-BIH records with caching support.

    Args:
        record_ids: List of record IDs to process (default: all 48 records)
        raw_dir: Directory of raw MIT-BIH files
        pre_r: Pre-R-peak samples
        post_r: Post-R-peak samples
        apply_filter: Whether to apply bandpass + notch filtering
        normalize: Whether to apply Z-score normalization
        lead_idx: Channel index to extract (default 0: MLII)
        use_cache: If True, loads/saves cached .npz file for speed
        cache_name: Cache filename in data/processed/signals/

    Returns:
        Tuple of (beats, labels, record_origins)
            - beats: np.ndarray of shape (N, 260)
            - labels: np.ndarray of shape (N,) with classes 0..4
            - record_origins: np.ndarray of shape (N,) tracking patient record
    """
    if raw_dir is None:
        raw_dir = DATA_RAW_MITBIH_DIR

    if record_ids is None:
        record_ids = MITBIH_RECORD_IDS

    cache_path = DATA_PROCESSED_SIGNALS_DIR / cache_name
    if use_cache and cache_path.exists():
        logger.info(f"Loading preprocessed ECG beats from cache: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data["beats"], data["labels"], data["record_origins"]

    all_beats = []
    all_labels = []
    all_origins = []

    logger.info(f"Extracting heartbeat windows from {len(record_ids)} MIT-BIH records...")

    for r_id in record_ids:
        try:
            signal, r_peaks, symbols, fs, sig_names = load_mitbih_record(r_id, raw_dir=raw_dir)
        except Exception as e:
            logger.warning(f"Skipping record {r_id} due to error: {e}")
            continue

        if apply_filter:
            signal = preprocess_ecg_signal(signal, fs=fs, lowcut=0.5, highcut=45.0, notch_freq=60.0)

        if normalize:
            signal = zscore_normalize(signal)

        beats, labels, _ = extract_heartbeat_segments(
            signal, r_peaks, symbols, pre_r=pre_r, post_r=post_r, lead_idx=lead_idx
        )

        if len(beats) > 0:
            all_beats.append(beats)
            all_labels.append(labels)
            all_origins.extend([r_id] * len(labels))

    if len(all_beats) == 0:
        raise RuntimeError("No heartbeats extracted from any record.")

    beats_arr = np.concatenate(all_beats, axis=0).astype(np.float32)
    labels_arr = np.concatenate(all_labels, axis=0).astype(np.int64)
    origins_arr = np.array(all_origins)

    logger.info(f"Extracted total {len(beats_arr):,} heartbeat segments across {len(record_ids)} records.")
    for c_idx, c_name in enumerate(AAMI_CLASS_NAMES):
        count = np.sum(labels_arr == c_idx)
        pct = (count / len(labels_arr)) * 100
        logger.info(f"  Class {c_idx} [{c_name}]: {count:,} beats ({pct:.2f}%)")

    if use_cache:
        DATA_PROCESSED_SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            beats=beats_arr,
            labels=labels_arr,
            record_origins=origins_arr,
        )
        logger.info(f"Saved processed beats cache to {cache_path}")

    return beats_arr, labels_arr, origins_arr


class MITBIHECGDataset(Dataset):
    """
    PyTorch Dataset for ECG Heartbeat Signals.
    Yields (signal_tensor, label) where signal_tensor has shape (1, sequence_length).
    """

    def __init__(
        self,
        beats: np.ndarray,
        labels: np.ndarray,
        transform: Optional[ECGAugmentationPipeline] = None,
    ):
        self.beats = np.asarray(beats, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.beats)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        beat = self.beats[idx]

        if self.transform is not None:
            beat = self.transform(beat)

        # Reshape to (1, sequence_length) for 1D CNN / Transformer compatibility
        beat_tensor = torch.from_numpy(beat).unsqueeze(0).float()
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        return beat_tensor, label_tensor


def compute_class_weights(labels: np.ndarray, num_classes: int = 5) -> torch.Tensor:
    """
    Computes balanced class weights for CrossEntropyLoss:
    weight[c] = total_samples / (num_classes * class_count[c])
    """
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    weights = np.zeros(num_classes, dtype=np.float32)
    for i in range(num_classes):
        if counts[i] > 0:
            weights[i] = total / (num_classes * counts[i])
        else:
            weights[i] = 1.0
    return torch.tensor(weights, dtype=torch.float32)


def get_ecg_dataloaders(
    batch_size: int = 64,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42,
    use_augmentation: bool = True,
    use_cache: bool = True,
    record_ids: Optional[List[str]] = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Splits the extracted ECG beats into Stratified Train, Validation, and Test sets.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_weights_tensor)
    """
    beats, labels, origins = extract_beats_from_records(
        record_ids=record_ids, use_cache=use_cache
    )

    from sklearn.model_selection import train_test_split

    total_test_val = val_split + test_split
    val_relative_ratio = val_split / total_test_val

    # Stratified split to ensure rare classes (e.g. Fusion) are proportionally present in all splits
    train_idx, temp_idx = train_test_split(
        np.arange(len(labels)),
        test_size=total_test_val,
        stratify=labels,
        random_state=random_seed,
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - val_relative_ratio),
        stratify=labels[temp_idx],
        random_state=random_seed,
    )

    # Compute loss weights from training set distribution
    class_weights = compute_class_weights(labels[train_idx])

    train_transform = ECGAugmentationPipeline() if use_augmentation else None

    train_dataset = MITBIHECGDataset(beats[train_idx], labels[train_idx], transform=train_transform)
    val_dataset = MITBIHECGDataset(beats[val_idx], labels[val_idx], transform=None)
    test_dataset = MITBIHECGDataset(beats[test_idx], labels[test_idx], transform=None)

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
        f"DataLoaders created | Train: {len(train_dataset):,} beats | "
        f"Val: {len(val_dataset):,} beats | Test: {len(test_dataset):,} beats"
    )

    return train_loader, val_loader, test_loader, class_weights
