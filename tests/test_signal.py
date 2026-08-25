"""
Unit tests for MedAgent Medical Signal Preprocessing Modules.
"""

import numpy as np
import pytest
import torch

from src.data.preprocessing.signal_filtering import (
    bandpass_filter,
    notch_filter,
    preprocess_ecg_signal,
)
from src.data.preprocessing.normalization import (
    zscore_normalize,
    minmax_normalize,
    robust_normalize,
)
from src.data.preprocessing.segmentation import (
    AAMI_MAPPING,
    AAMI_CLASS_NAMES,
    map_symbol_to_aami,
    extract_heartbeat_segments,
    extract_fixed_windows,
)
from src.data.preprocessing.augmentation import (
    add_gaussian_noise,
    random_amplitude_scale,
    add_baseline_wander,
    random_time_shift,
    random_cutout_mask,
    ECGAugmentationPipeline,
)


class TestSignalFiltering:
    def test_bandpass_filter_1d(self):
        fs = 360.0
        t = np.linspace(0, 2, int(2 * fs))
        # 1 Hz signal + 10 Hz signal + 100 Hz high frequency noise
        sig = np.sin(2 * np.pi * 1.0 * t) + np.sin(2 * np.pi * 10.0 * t) + np.sin(2 * np.pi * 100.0 * t)

        filtered = bandpass_filter(sig, lowcut=0.5, highcut=45.0, fs=fs, order=4)
        assert isinstance(filtered, np.ndarray)
        assert filtered.shape == sig.shape
        assert filtered.dtype == np.float32

    def test_bandpass_filter_torch_tensor(self):
        tensor_sig = torch.randn(360)
        filtered = bandpass_filter(tensor_sig, lowcut=0.5, highcut=45.0, fs=360.0)
        assert isinstance(filtered, np.ndarray)
        assert filtered.shape == (360,)

    def test_notch_filter(self):
        fs = 360.0
        t = np.linspace(0, 1, int(fs))
        # 60 Hz powerline interference
        sig = np.sin(2 * np.pi * 60.0 * t)
        filtered = notch_filter(sig, freq=60.0, fs=fs, q=30.0)
        # Power of 60 Hz should be significantly attenuated
        assert np.var(filtered) < np.var(sig) * 0.1

    def test_preprocess_pipeline(self):
        sig = np.random.randn(720)
        cleaned = preprocess_ecg_signal(sig, fs=360.0, lowcut=0.5, highcut=45.0, notch_freq=60.0)
        assert cleaned.shape == (720,)
        assert not np.isnan(cleaned).any()


class TestSignalNormalization:
    def test_zscore_normalize_numpy(self):
        arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
        norm = zscore_normalize(arr)
        assert np.isclose(np.mean(norm), 0.0, atol=1e-5)
        assert np.isclose(np.std(norm), 1.0, atol=1e-5)

    def test_zscore_normalize_torch(self):
        tensor = torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0]], dtype=torch.float32)
        norm = zscore_normalize(tensor)
        assert isinstance(norm, torch.Tensor)
        assert torch.isclose(torch.mean(norm), torch.tensor(0.0), atol=1e-4)
        assert torch.isclose(torch.std(norm), torch.tensor(1.0), atol=1e-4)

    def test_minmax_normalize(self):
        arr = np.array([2.0, 4.0, 6.0, 8.0, 10.0], dtype=np.float32)
        norm = minmax_normalize(arr, feature_range=(0.0, 1.0))
        assert np.isclose(norm.min(), 0.0, atol=1e-5)
        assert np.isclose(norm.max(), 1.0, atol=1e-5)

    def test_robust_normalize(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 100.0], dtype=np.float32)
        norm = robust_normalize(arr)
        assert not np.isnan(norm).any()


class TestSignalSegmentation:
    def test_aami_mapping(self):
        assert map_symbol_to_aami("N") == 0
        assert map_symbol_to_aami("L") == 0
        assert map_symbol_to_aami("A") == 1
        assert map_symbol_to_aami("V") == 2
        assert map_symbol_to_aami("F") == 3
        assert map_symbol_to_aami("/") == 4
        assert map_symbol_to_aami("+") is None  # Non-beat rhythm change

    def test_extract_heartbeat_segments(self):
        # Create a mock 10-second ECG signal at 360 Hz (3600 samples)
        sig = np.sin(np.linspace(0, 20 * np.pi, 3600))
        # Place 3 R-peaks at sample 500, 1000, 1500
        r_peaks = np.array([500, 1000, 1500])
        symbols = ["N", "V", "A"]

        beats, labels, peaks = extract_heartbeat_segments(
            sig, r_peaks, symbols, pre_r=90, post_r=170
        )

        assert beats.shape == (3, 260)
        assert list(labels) == [0, 2, 1]
        assert list(peaks) == [500, 1000, 1500]

    def test_extract_heartbeat_boundary_protection(self):
        sig = np.random.randn(500)
        # Peak 10 is too close to start (needs 90 samples before), Peak 480 is too close to end
        r_peaks = np.array([10, 200, 480])
        symbols = ["N", "N", "N"]

        beats, labels, peaks = extract_heartbeat_segments(
            sig, r_peaks, symbols, pre_r=90, post_r=170
        )
        # Only middle peak at index 200 should be extracted
        assert beats.shape == (1, 260)
        assert list(peaks) == [200]

    def test_extract_fixed_windows(self):
        sig = np.random.randn(5000)
        windows = extract_fixed_windows(sig, window_size=2500, stride=2500)
        assert windows.shape == (2, 2500)


class TestAugmentation:
    def test_gaussian_noise(self):
        sig = np.zeros(260, dtype=np.float32)
        noisy = add_gaussian_noise(sig, std=0.05)
        assert np.var(noisy) > 0.0
        assert noisy.shape == (260,)

    def test_amplitude_scale(self):
        sig = np.ones(260, dtype=np.float32)
        scaled = random_amplitude_scale(sig, scale_range=(0.8, 1.2))
        assert 0.79 <= scaled[0] <= 1.21

    def test_pipeline_callable(self):
        pipeline = ECGAugmentationPipeline(
            gaussian_noise_prob=1.0,
            amplitude_scale_prob=1.0,
            baseline_wander_prob=1.0,
            time_shift_prob=1.0,
            cutout_prob=1.0,
        )
        sig = np.random.randn(260).astype(np.float32)
        augmented = pipeline(sig)
        assert augmented.shape == (260,)
        assert not np.isnan(augmented).any()
