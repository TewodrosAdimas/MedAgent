"""
ECG Data Augmentation Module for MedAgent.
Provides physiological and numerical augmentations to improve model generalization.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import torch


def add_gaussian_noise(
    signal: np.ndarray,
    std: float = 0.02,
) -> np.ndarray:
    """Adds zero-mean Gaussian noise (sensor jitter)."""
    noise = np.random.normal(0.0, std, size=signal.shape).astype(np.float32)
    return signal + noise


def random_amplitude_scale(
    signal: np.ndarray,
    scale_range: Tuple[float, float] = (0.85, 1.15),
) -> np.ndarray:
    """Scales signal amplitude by a random factor within scale_range."""
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return (signal * scale).astype(np.float32)


def add_baseline_wander(
    signal: np.ndarray,
    fs: float = 360.0,
    max_amplitude: float = 0.15,
    freq_range: Tuple[float, float] = (0.1, 0.4),
) -> np.ndarray:
    """Simulates respiratory / movement baseline wander via low-frequency sinusoid."""
    t = np.arange(signal.shape[-1]) / fs
    freq = np.random.uniform(freq_range[0], freq_range[1])
    phase = np.random.uniform(0, 2 * np.pi)
    amp = np.random.uniform(0.02, max_amplitude)
    wander = amp * np.sin(2 * np.pi * freq * t + phase).astype(np.float32)
    return signal + wander


def random_time_shift(
    signal: np.ndarray,
    max_shift: int = 15,
) -> np.ndarray:
    """Applies a random circular time shift to the waveform."""
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(signal, shift, axis=-1)


def random_cutout_mask(
    signal: np.ndarray,
    mask_length: int = 20,
) -> np.ndarray:
    """Zeroes out a random contiguous portion of the signal to simulate transient dropouts."""
    sig = signal.copy()
    seq_len = sig.shape[-1]
    if seq_len > mask_length:
        start = np.random.randint(0, seq_len - mask_length)
        sig[..., start : start + mask_length] = 0.0
    return sig


class ECGAugmentationPipeline:
    """
    Composable augmentation pipeline for ECG waveforms during training.
    """

    def __init__(
        self,
        gaussian_noise_prob: float = 0.5,
        gaussian_noise_std: float = 0.02,
        amplitude_scale_prob: float = 0.5,
        amplitude_scale_range: Tuple[float, float] = (0.85, 1.15),
        baseline_wander_prob: float = 0.3,
        time_shift_prob: float = 0.3,
        max_shift: int = 10,
        cutout_prob: float = 0.2,
        cutout_length: int = 15,
        fs: float = 360.0,
    ):
        self.gaussian_noise_prob = gaussian_noise_prob
        self.gaussian_noise_std = gaussian_noise_std
        self.amplitude_scale_prob = amplitude_scale_prob
        self.amplitude_scale_range = amplitude_scale_range
        self.baseline_wander_prob = baseline_wander_prob
        self.time_shift_prob = time_shift_prob
        self.max_shift = max_shift
        self.cutout_prob = cutout_prob
        self.cutout_length = cutout_length
        self.fs = fs

    def __call__(self, signal: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        is_torch = isinstance(signal, torch.Tensor)
        if is_torch:
            arr = signal.detach().cpu().numpy().copy()
        else:
            arr = np.array(signal, copy=True, dtype=np.float32)

        if np.random.rand() < self.gaussian_noise_prob:
            arr = add_gaussian_noise(arr, std=self.gaussian_noise_std)

        if np.random.rand() < self.amplitude_scale_prob:
            arr = random_amplitude_scale(arr, scale_range=self.amplitude_scale_range)

        if np.random.rand() < self.baseline_wander_prob:
            arr = add_baseline_wander(arr, fs=self.fs)

        if np.random.rand() < self.time_shift_prob:
            arr = random_time_shift(arr, max_shift=self.max_shift)

        if np.random.rand() < self.cutout_prob:
            arr = random_cutout_mask(arr, mask_length=self.cutout_length)

        if is_torch:
            return torch.from_numpy(arr).type_as(signal)
        return arr.astype(np.float32)
