"""
ECG Signal Segmentation & AAMI Beat Mapping Module for MedAgent.
Provides R-peak centered heartbeat window extraction and continuous windowing.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

# Standard AAMI 5-Class Categorization Map
AAMI_MAPPING: Dict[str, int] = {
    # Class 0: Normal / Bundle Branch Block (N)
    "N": 0,
    "L": 0,
    "R": 0,
    "B": 0,
    "e": 0,
    "j": 0,
    # Class 1: Supraventricular Ectopic Beat (S)
    "A": 1,
    "a": 1,
    "J": 1,
    "S": 1,
    # Class 2: Ventricular Ectopic Beat (V)
    "V": 2,
    "E": 2,
    # Class 3: Fusion Beat (F)
    "F": 3,
    # Class 4: Unknown / Paced / Unclassifiable (Q)
    "/": 4,
    "f": 4,
    "Q": 4,
    "?": 4,
}

AAMI_CLASS_NAMES: List[str] = [
    "Normal (N)",
    "Supraventricular Ectopic (S)",
    "Ventricular Ectopic (V)",
    "Fusion (F)",
    "Unknown / Paced (Q)",
]


def map_symbol_to_aami(symbol: str) -> Optional[int]:
    """
    Maps an MIT-BIH annotation symbol to an AAMI 5-class integer (0..4).
    Returns None if symbol is a non-beat annotation (e.g. rhythm change '+', '~', '|').
    """
    return AAMI_MAPPING.get(symbol, None)


def extract_heartbeat_segments(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    symbols: List[str],
    pre_r: int = 90,
    post_r: int = 170,
    lead_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts individual beat segments centered on R-peaks.

    Args:
        signal: 2D (time, channels) or 1D (time,) array
        r_peaks: Array of sample indices corresponding to R-peaks
        symbols: List of annotation symbols matching r_peaks
        pre_r: Number of samples before R-peak (default 90, ~250ms at 360Hz)
        post_r: Number of samples after R-peak (default 170, ~472ms at 360Hz)
        lead_idx: Channel index to extract if signal is multi-lead (default 0: MLII)

    Returns:
        Tuple of:
            - beats: 2D array of shape (num_valid_beats, window_length)
            - labels: 1D array of integer class labels (0..4)
            - peak_indices: 1D array of original R-peak sample indices
    """
    if signal.ndim == 2:
        sig_channel = signal[:, lead_idx]
    else:
        sig_channel = signal

    sig_len = len(sig_channel)
    window_len = pre_r + post_r

    beats = []
    labels = []
    valid_peaks = []

    for peak, symbol in zip(r_peaks, symbols):
        aami_class = map_symbol_to_aami(symbol)
        if aami_class is None:
            continue  # Skip non-beat annotations

        start_idx = peak - pre_r
        end_idx = peak + post_r

        # Ensure window is fully within recording bounds
        if start_idx < 0 or end_idx > sig_len:
            continue

        beat_window = sig_channel[start_idx:end_idx]
        if len(beat_window) == window_len:
            beats.append(beat_window)
            labels.append(aami_class)
            valid_peaks.append(peak)

    if len(beats) == 0:
        return (
            np.empty((0, window_len), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    return (
        np.array(beats, dtype=np.float32),
        np.array(labels, dtype=np.int64),
        np.array(valid_peaks, dtype=np.int64),
    )


def extract_fixed_windows(
    signal: np.ndarray,
    window_size: int = 2500,
    stride: Optional[int] = None,
    lead_idx: Optional[int] = 0,
) -> np.ndarray:
    """
    Slices a continuous signal into fixed-length windows.

    Args:
        signal: 1D or 2D biosignal array
        window_size: Number of samples per window (default 2500)
        stride: Step size between window starts (default = window_size, no overlap)
        lead_idx: Channel index to extract (if None and signal is 2D, returns multi-channel windows)

    Returns:
        Array of sliced windows:
            Shape (num_windows, window_size) for single lead, or
            Shape (num_windows, channels, window_size) for multi-lead
    """
    if stride is None:
        stride = window_size

    if signal.ndim == 2:
        if lead_idx is not None:
            sig = signal[:, lead_idx]
            sig_len = len(sig)
            num_windows = (sig_len - window_size) // stride + 1
            windows = [sig[i * stride : i * stride + window_size] for i in range(num_windows)]
            return np.array(windows, dtype=np.float32)
        else:
            sig_len = signal.shape[0]
            num_windows = (sig_len - window_size) // stride + 1
            windows = [signal[i * stride : i * stride + window_size, :].T for i in range(num_windows)]
            return np.array(windows, dtype=np.float32)
    else:
        sig_len = len(signal)
        num_windows = (sig_len - window_size) // stride + 1
        windows = [signal[i * stride : i * stride + window_size] for i in range(num_windows)]
        return np.array(windows, dtype=np.float32)
