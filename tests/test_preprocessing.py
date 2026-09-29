"""
Unit Tests for MedAgent Signal Preprocessing and ECG Data Loaders.
"""

from pathlib import Path
import numpy as np
import pytest
import torch

from src.data.preprocessing.ecg_cleaner import (
    AAMI_CLASSES,
    AAMI_SYMBOL_MAP,
    apply_bandpass_filter,
    apply_notch_filter,
    clean_ecg,
    extract_beats_from_record,
    map_aami_symbol,
    normalize_signal,
    segment_beats_from_r_peaks,
)
from src.data.loaders.ecg_dataset import (
    ECGAugmentation,
    ECGDataset,
    get_ecg_dataloaders,
)


def test_bandpass_filter():
    fs = 360.0
    t = np.linspace(0, 5, int(5 * fs), endpoint=False)

    # 1 Hz (in-band), 10 Hz (in-band), 0.1 Hz (baseline wander), 120 Hz (high freq noise)
    sig = (
        np.sin(2 * np.pi * 10 * t)
        + 2.0 * np.sin(2 * np.pi * 0.1 * t)
        + 0.5 * np.sin(2 * np.pi * 120 * t)
    )

    filtered = apply_bandpass_filter(sig, sampling_rate=fs, lowcut=0.5, highcut=45.0, order=4)

    assert filtered.shape == sig.shape
    assert not np.isnan(filtered).any()
    # High frequency and low frequency components should be attenuated
    assert np.std(filtered) < np.std(sig)


def test_notch_filter():
    fs = 360.0
    t = np.linspace(0, 2, int(2 * fs), endpoint=False)
    sig_clean = np.sin(2 * np.pi * 5 * t)
    sig_powerline = sig_clean + 2.0 * np.sin(2 * np.pi * 60 * t)

    filtered = apply_notch_filter(sig_powerline, sampling_rate=fs, notch_freq=60.0)

    assert filtered.shape == sig_powerline.shape
    assert not np.isnan(filtered).any()
    # 60 Hz oscillation should be significantly reduced
    assert np.std(filtered - sig_clean) < np.std(sig_powerline - sig_clean)


def test_normalize_signal():
    data = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)

    # Z-score test
    z_norm = normalize_signal(data, method="z_score")
    assert np.isclose(np.mean(z_norm), 0.0, atol=1e-5)
    assert np.isclose(np.std(z_norm), 1.0, atol=1e-5)

    # Min-max test
    mm_norm = normalize_signal(data, method="min_max")
    assert np.isclose(np.min(mm_norm), -1.0, atol=1e-5)
    assert np.isclose(np.max(mm_norm), 1.0, atol=1e-5)


def test_clean_ecg():
    fs = 360.0
    raw = np.random.randn(1000).astype(np.float32)
    cleaned = clean_ecg(raw, sampling_rate=fs, lowcut=0.5, highcut=45.0, notch_freq=60.0)

    assert cleaned.shape == (1000,)
    assert cleaned.dtype == np.float32
    assert not np.isnan(cleaned).any()
    assert np.isclose(np.mean(cleaned), 0.0, atol=1e-5)


def test_map_aami_symbol():
    assert map_aami_symbol("N") == 0
    assert map_aami_symbol("L") == 0
    assert map_aami_symbol("R") == 0
    assert map_aami_symbol("A") == 1
    assert map_aami_symbol("a") == 1
    assert map_aami_symbol("V") == 2
    assert map_aami_symbol("E") == 2
    assert map_aami_symbol("F") == 3
    assert map_aami_symbol("/") == 4
    assert map_aami_symbol("Q") == 4
    assert map_aami_symbol("+") is None  # Non-beat rhythm annotation
    assert map_aami_symbol("~") is None


def test_segment_beats_from_r_peaks():
    signal_length = 2000
    ecg_signal = np.random.randn(signal_length).astype(np.float32)

    r_peaks = [50, 300, 700, 1200, 1950]  # Note: 50 is too close to start, 1950 is too close to end
    symbols = ["N", "N", "V", "A", "N"]

    beats, labels, valid_peaks = segment_beats_from_r_peaks(
        signal_data=ecg_signal,
        r_peaks=r_peaks,
        symbols=symbols,
        window_before=90,
        window_after=170,
    )

    # R-peaks at 300, 700, 1200 should be valid (3 beats extracted)
    assert len(beats) == 3
    assert beats.shape == (3, 260)
    assert len(labels) == 3
    assert labels.tolist() == [0, 2, 1]  # N -> 0, V -> 2, A -> 1
    assert valid_peaks.tolist() == [300, 700, 1200]


def test_ecg_dataset_from_npz():
    npz_path = Path("data/processed/signals/mitbih_aami_beats.npz")
    if not npz_path.exists():
        pytest.skip("Preprocessed mitbih_aami_beats.npz not found.")

    dataset = ECGDataset(data_source=npz_path)
    assert len(dataset) > 0

    beat_tensor, label_tensor = dataset[0]
    assert isinstance(beat_tensor, torch.Tensor)
    assert isinstance(label_tensor, torch.Tensor)
    assert beat_tensor.shape == (1, 260)
    assert beat_tensor.dtype == torch.float32
    assert label_tensor.dtype == torch.long
    assert 0 <= label_tensor.item() < len(AAMI_CLASSES)


def test_ecg_augmentation():
    beat = np.sin(np.linspace(0, np.pi, 260)).astype(np.float32)
    aug = ECGAugmentation(prob_noise=1.0, prob_scale=1.0, prob_shift=1.0)
    aug_beat = aug(beat)

    assert aug_beat.shape == beat.shape
    assert not np.array_equal(aug_beat, beat)
    assert not np.isnan(aug_beat).any()


def test_get_ecg_dataloaders():
    npz_path = Path("data/processed/signals/mitbih_aami_beats.npz")
    if not npz_path.exists():
        pytest.skip("mitbih_aami_beats.npz not found.")

    train_loader, val_loader, test_loader, weights = get_ecg_dataloaders(
        npz_path=npz_path,
        batch_size=32,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        augment_train=True,
    )

    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0
    assert weights.shape == (len(AAMI_CLASSES),)
    assert (weights > 0).all()

    # Test batch structure
    batch_beats, batch_labels = next(iter(train_loader))
    assert batch_beats.shape == (32, 1, 260)
    assert batch_labels.shape == (32,)


def test_extract_beats_from_raw_wfdb():
    record_path = Path("data/raw/ecg/mit-bih-arrhythmia-database-1.0.0/100")
    if not Path("data/raw/ecg/mit-bih-arrhythmia-database-1.0.0/100.dat").exists():
        pytest.skip("MIT-BIH record 100 not found.")

    beats, labels, valid_peaks = extract_beats_from_record(
        record_path_or_id=record_path,
        lead_name="MLII",
        window_before=90,
        window_after=170,
    )

    assert len(beats) > 1000
    assert beats.shape[1] == 260
    assert len(labels) == len(beats)
    assert len(valid_peaks) == len(beats)
    assert set(labels).issubset({0, 1, 2, 3, 4})
