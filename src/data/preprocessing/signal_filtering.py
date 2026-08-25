"""
Signal Filtering Module for MedAgent.
Provides zero-phase digital filters for ECG and physiological biosignals.
"""

from typing import Optional, Tuple, Union
import numpy as np
import torch
from scipy.signal import butter, filtfilt, iirnotch


def bandpass_filter(
    signal: Union[np.ndarray, torch.Tensor],
    lowcut: float = 0.5,
    highcut: float = 45.0,
    fs: float = 360.0,
    order: int = 4,
) -> np.ndarray:
    """
    Applies a zero-phase Butterworth bandpass filter to remove baseline wander
    and high-frequency noise.

    Args:
        signal: 1D or 2D biosignal array (channels, time) or (time,)
        lowcut: Lower cutoff frequency in Hz (default 0.5 Hz)
        highcut: Upper cutoff frequency in Hz (default 45.0 Hz)
        fs: Sampling frequency in Hz (default 360.0 Hz)
        order: Butterworth filter order (default 4)

    Returns:
        Filtered signal as np.ndarray (same shape as input)
    """
    is_torch = isinstance(signal, torch.Tensor)
    if is_torch:
        arr = signal.detach().cpu().numpy().copy()
    else:
        arr = np.array(signal, dtype=np.float64)

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    # Ensure normalized frequencies are in valid range (0, 1)
    low = max(0.0001, min(low, 0.9999))
    high = max(low + 0.0001, min(high, 0.9999))

    b, a = butter(order, [low, high], btype="bandpass")

    if arr.ndim == 1:
        # Pad if signal is shorter than default filtfilt padlen
        padlen = 3 * max(len(a), len(b))
        if len(arr) <= padlen:
            padlen = len(arr) - 1
        filtered = filtfilt(b, a, arr, padlen=padlen)
    elif arr.ndim == 2:
        axis = 1 if arr.shape[1] > arr.shape[0] else 0
        padlen = 3 * max(len(a), len(b))
        if arr.shape[axis] <= padlen:
            padlen = arr.shape[axis] - 1
        filtered = filtfilt(b, a, arr, axis=axis, padlen=padlen)
    else:
        raise ValueError(f"Signal must be 1D or 2D, got shape {arr.shape}")

    return filtered.astype(np.float32)


def notch_filter(
    signal: Union[np.ndarray, torch.Tensor],
    freq: float = 60.0,
    fs: float = 360.0,
    q: float = 30.0,
) -> np.ndarray:
    """
    Applies an IIR notch filter to remove powerline interference (50 Hz or 60 Hz).

    Args:
        signal: 1D or 2D biosignal array
        freq: Frequency to reject in Hz (e.g. 50.0 or 60.0 Hz)
        fs: Sampling frequency in Hz (default 360.0 Hz)
        q: Quality factor (default 30.0)

    Returns:
        Filtered signal as np.ndarray
    """
    is_torch = isinstance(signal, torch.Tensor)
    if is_torch:
        arr = signal.detach().cpu().numpy().copy()
    else:
        arr = np.array(signal, dtype=np.float64)

    nyq = 0.5 * fs
    w0 = freq / nyq

    if w0 >= 1.0 or w0 <= 0.0:
        return arr.astype(np.float32)

    b, a = iirnotch(w0, q)

    if arr.ndim == 1:
        padlen = 3 * max(len(a), len(b))
        if len(arr) <= padlen:
            padlen = len(arr) - 1
        filtered = filtfilt(b, a, arr, padlen=padlen)
    elif arr.ndim == 2:
        axis = 1 if arr.shape[1] > arr.shape[0] else 0
        padlen = 3 * max(len(a), len(b))
        if arr.shape[axis] <= padlen:
            padlen = arr.shape[axis] - 1
        filtered = filtfilt(b, a, arr, axis=axis, padlen=padlen)
    else:
        raise ValueError(f"Signal must be 1D or 2D, got shape {arr.shape}")

    return filtered.astype(np.float32)


def preprocess_ecg_signal(
    signal: Union[np.ndarray, torch.Tensor],
    fs: float = 360.0,
    lowcut: float = 0.5,
    highcut: float = 45.0,
    notch_freq: Optional[float] = 60.0,
    notch_q: float = 30.0,
) -> np.ndarray:
    """
    Complete signal filtering pipeline: Bandpass + Optional Powerline Notch filter.

    Args:
        signal: Raw ECG signal
        fs: Sampling rate in Hz
        lowcut: Bandpass lower cutoff
        highcut: Bandpass upper cutoff
        notch_freq: Notch filter rejection frequency (or None to disable)
        notch_q: Notch filter Q-factor

    Returns:
        Cleaned, filtered ECG signal
    """
    cleaned = bandpass_filter(signal, lowcut=lowcut, highcut=highcut, fs=fs)
    if notch_freq is not None:
        cleaned = notch_filter(cleaned, freq=notch_freq, fs=fs, q=notch_q)
    return cleaned
