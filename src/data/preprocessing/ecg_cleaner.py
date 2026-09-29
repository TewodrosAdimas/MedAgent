"""
ECG Signal Cleaner and Beat Segmentation Module.

Implements standard biomedical signal processing:
1. Butterworth Bandpass Filtering (0.5 - 45 Hz) to remove baseline wander and EMG noise.
2. IIR Notch Filtering (60 Hz / 50 Hz) to eliminate AC powerline interference.
3. Amplitude Normalization (Z-score or Min-Max).
4. R-Peak centered window segmentation (260 samples: 90 pre-peak, 170 post-peak).
5. AAMI EC57 annotation mapping to 5 standard clinical arrhythmia categories.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import signal
import wfdb


# AAMI EC57 Class definitions
AAMI_CLASSES: Dict[int, str] = {
    0: "Normal (N)",
    1: "Supraventricular Ectopic (S)",
    2: "Ventricular Ectopic (V)",
    3: "Fusion (F)",
    4: "Unknown / Paced (Q)",
}

# Mapping from MIT-BIH PhysioNet annotation symbols to AAMI 5-class integer IDs
# N: Normal beat, Left/Right bundle branch block, Atrial/Nodal escape
# S: Atrial premature, Aberrated atrial premature, Nodal premature, Supraventricular premature
# V: Premature ventricular contraction, Ventricular escape
# F: Fusion of ventricular and normal beat
# Q: Paced beat, Fusion of paced and normal beat, Unclassifiable beat
AAMI_SYMBOL_MAP: Dict[str, int] = {
    # Class 0: Normal / Non-ectopic (N)
    "N": 0,
    "L": 0,
    "R": 0,
    "e": 0,
    "j": 0,
    "B": 0,
    # Class 1: Supraventricular Ectopic (S)
    "A": 1,
    "a": 1,
    "J": 1,
    "S": 1,
    # Class 2: Ventricular Ectopic (V)
    "V": 2,
    "E": 2,
    "r": 2,
    # Class 3: Fusion (F)
    "F": 3,
    # Class 4: Unknown / Paced (Q)
    "/": 4,
    "f": 4,
    "Q": 4,
    "?": 4,
}


def apply_bandpass_filter(
    data: np.ndarray,
    sampling_rate: float = 360.0,
    lowcut: float = 0.5,
    highcut: float = 45.0,
    order: int = 4,
    axis: int = -1,
) -> np.ndarray:
    """
    Applies a zero-phase Butterworth bandpass filter to ECG signals.

    Args:
        data: Input ECG signal array (1D or 2D).
        sampling_rate: Sampling frequency in Hz (default: 360.0 Hz for MIT-BIH).
        lowcut: Low cutoff frequency in Hz (default: 0.5 Hz for baseline wander removal).
        highcut: High cutoff frequency in Hz (default: 45.0 Hz for EMG/high freq noise).
        order: Order of the Butterworth filter (default: 4).
        axis: Array axis along which to apply the filter.

    Returns:
        Filtered ECG signal array with zero phase distortion.
    """
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist

    # Ensure filter frequencies are within (0, 1) Nyquist interval
    low = max(0.001, min(low, 0.999))
    high = max(low + 0.001, min(high, 0.999))

    sos = signal.butter(order, [low, high], btype="bandpass", output="sos")
    filtered = signal.sosfiltfilt(sos, data, axis=axis)
    return filtered.astype(np.float32)


def apply_notch_filter(
    data: np.ndarray,
    sampling_rate: float = 360.0,
    notch_freq: float = 60.0,
    quality_factor: float = 30.0,
    axis: int = -1,
) -> np.ndarray:
    """
    Applies a zero-phase IIR notch filter to remove powerline interference.

    Args:
        data: Input ECG signal array.
        sampling_rate: Sampling frequency in Hz (default: 360.0 Hz).
        notch_freq: Notch target frequency (default: 60.0 Hz; 50.0 Hz for EU/UK).
        quality_factor: Filter quality factor Q (default: 30.0).
        axis: Array axis along which to apply the filter.

    Returns:
        Filtered ECG signal array.
    """
    nyquist = 0.5 * sampling_rate
    if notch_freq >= nyquist:
        # If notch frequency is at or above Nyquist, skip notch filter
        return data.astype(np.float32)

    b, a = signal.iirnotch(w0=notch_freq, Q=quality_factor, fs=sampling_rate)
    filtered = signal.filtfilt(b, a, data, axis=axis)
    return filtered.astype(np.float32)


def normalize_signal(
    data: np.ndarray,
    method: str = "z_score",
    axis: int = -1,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Normalizes ECG signal amplitude.

    Args:
        data: Input ECG signal array.
        method: 'z_score' (zero mean, unit variance) or 'min_max' (scaled to [-1, 1]).
        axis: Array axis along which to compute statistics.
        epsilon: Small numerical constant to avoid division by zero.

    Returns:
        Normalized signal array.
    """
    if method == "z_score":
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)
        return ((data - mean) / (std + epsilon)).astype(np.float32)
    elif method == "min_max":
        min_val = np.min(data, axis=axis, keepdims=True)
        max_val = np.max(data, axis=axis, keepdims=True)
        diff = max_val - min_val
        diff[diff == 0] = epsilon
        scaled = 2.0 * ((data - min_val) / diff) - 1.0
        return scaled.astype(np.float32)
    else:
        raise ValueError(f"Unsupported normalization method '{method}'. Use 'z_score' or 'min_max'.")


def clean_ecg(
    raw_signal: np.ndarray,
    sampling_rate: float = 360.0,
    lowcut: float = 0.5,
    highcut: float = 45.0,
    notch_freq: Optional[float] = 60.0,
    order: int = 4,
    normalize: bool = True,
    norm_method: str = "z_score",
) -> np.ndarray:
    """
    Full end-to-end ECG signal cleaning pipeline:
    Raw -> Bandpass (0.5 - 45 Hz) -> Notch (60 Hz) -> Normalization.

    Args:
        raw_signal: 1D or 2D NumPy array of raw ECG measurements.
        sampling_rate: Sampling frequency in Hz.
        lowcut: Low cutoff for bandpass filter.
        highcut: High cutoff for bandpass filter.
        notch_freq: Notch filter frequency (or None to skip).
        order: Filter order.
        normalize: Whether to apply normalization.
        norm_method: 'z_score' or 'min_max'.

    Returns:
        Cleaned and normalized ECG signal array.
    """
    # Handle NaN or Inf values by interpolating or replacing
    clean_data = np.nan_to_num(raw_signal, nan=0.0, posinf=0.0, neginf=0.0)

    # 1. Bandpass filter
    filtered = apply_bandpass_filter(
        clean_data,
        sampling_rate=sampling_rate,
        lowcut=lowcut,
        highcut=highcut,
        order=order,
    )

    # 2. Notch filter
    if notch_freq is not None and notch_freq > 0:
        filtered = apply_notch_filter(
            filtered,
            sampling_rate=sampling_rate,
            notch_freq=notch_freq,
        )

    # 3. Normalization
    if normalize:
        filtered = normalize_signal(filtered, method=norm_method)

    return filtered.astype(np.float32)


def map_aami_symbol(symbol: str) -> Optional[int]:
    """
    Maps a PhysioNet annotation symbol to its AAMI 5-class integer code (0 to 4).
    Returns None for non-beat annotations (e.g. rhythm markers '+', '~', '|').

    Args:
        symbol: PhysioNet annotation character (e.g. 'N', 'V', 'A', 'L', 'R', 'F', '/').

    Returns:
        Integer class (0: N, 1: S, 2: V, 3: F, 4: Q) or None if non-beat.
    """
    return AAMI_SYMBOL_MAP.get(symbol, None)


def segment_beats_from_r_peaks(
    signal_data: np.ndarray,
    r_peaks: Union[List[int], np.ndarray],
    symbols: Optional[Union[List[str], np.ndarray]] = None,
    window_before: int = 90,
    window_after: int = 170,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Segments individual heartbeat waveforms centered around R-peak locations.

    Total beat window length = window_before + window_after (e.g. 90 + 170 = 260 samples).
    At 360 Hz:
      - 90 samples = 250 ms (P-wave and PR interval)
      - 170 samples = 472 ms (QRS complex and ST-T wave)
      - 260 samples = 722 ms total beat cycle

    Args:
        signal_data: 1D cleaned ECG signal array.
        r_peaks: Array or list of R-peak sample indices.
        symbols: Optional array or list of annotation symbols matching r_peaks.
        window_before: Number of samples prior to the R-peak (default: 90).
        window_after: Number of samples after the R-peak (default: 170).

    Returns:
        Tuple of:
            - beats: 2D array of shape (N_beats, window_before + window_after) as float32
            - labels: 1D array of AAMI class integer labels (or empty array if no symbols provided)
            - valid_peak_indices: 1D array of sample positions of accepted beats
    """
    beats_list = []
    labels_list = []
    valid_indices = []

    total_length = len(signal_data)
    expected_beat_len = window_before + window_after

    for idx, r_idx in enumerate(r_peaks):
        start = r_idx - window_before
        end = r_idx + window_after

        # Exclude boundary beats where window exceeds recording boundaries
        if start < 0 or end > total_length:
            continue

        beat = signal_data[start:end]
        if len(beat) != expected_beat_len:
            continue

        # Check label if annotations are provided
        if symbols is not None and idx < len(symbols):
            sym = symbols[idx]
            aami_class = map_aami_symbol(sym)
            if aami_class is None:
                # Skip non-beat annotations (rhythm changes, artifacts)
                continue
            labels_list.append(aami_class)

        beats_list.append(beat)
        valid_indices.append(r_idx)

    if len(beats_list) == 0:
        beats = np.empty((0, expected_beat_len), dtype=np.float32)
        labels = np.empty((0,), dtype=np.int64)
        valid_indices_arr = np.empty((0,), dtype=np.int64)
    else:
        beats = np.array(beats_list, dtype=np.float32)
        labels = np.array(labels_list, dtype=np.int64) if labels_list else np.empty((0,), dtype=np.int64)
        valid_indices_arr = np.array(valid_indices, dtype=np.int64)

    return beats, labels, valid_indices_arr


def extract_beats_from_record(
    record_path_or_id: Union[str, Path],
    lead_name: str = "MLII",
    window_before: int = 90,
    window_after: int = 170,
    sampling_rate: float = 360.0,
    clean: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads a raw PhysioNet / WFDB record, cleans the ECG signal on the requested lead,
    and extracts AAMI-labeled segmented heartbeats.

    Args:
        record_path_or_id: Absolute path or record name (e.g. 'data/raw/ecg/mit-bih.../100').
        lead_name: Desired ECG lead name ('MLII', 'V1', 'V5', etc.).
        window_before: Pre-R-peak window samples (default: 90).
        window_after: Post-R-peak window samples (default: 170).
        sampling_rate: Target sampling rate (default: 360 Hz).
        clean: Whether to apply bandpass, notch, and z-score normalization.

    Returns:
        Tuple of (beats, labels, valid_peak_indices).
    """
    record_str = str(record_path_or_id)
    # Remove file extension if supplied (e.g., .dat or .hea)
    for ext in [".dat", ".hea", ".atr"]:
        if record_str.endswith(ext):
            record_str = record_str[: -len(ext)]

    # Load WFDB record and annotations
    record = wfdb.rdrecord(record_str)
    annotation = wfdb.rdann(record_str, "atr")

    # Select lead
    sig_names = record.sig_name
    if lead_name in sig_names:
        lead_idx = sig_names.index(lead_name)
    else:
        # Default to the first available channel if requested lead is absent
        lead_idx = 0

    raw_lead_signal = record.p_signal[:, lead_idx]
    actual_fs = float(record.fs)

    # Clean signal
    if clean:
        processed_signal = clean_ecg(
            raw_lead_signal,
            sampling_rate=actual_fs,
            lowcut=0.5,
            highcut=45.0,
            notch_freq=60.0,
            normalize=True,
        )
    else:
        processed_signal = raw_lead_signal.astype(np.float32)

    # Segment beats
    r_peaks = annotation.sample
    symbols = annotation.symbol

    beats, labels, valid_peaks = segment_beats_from_r_peaks(
        signal_data=processed_signal,
        r_peaks=r_peaks,
        symbols=symbols,
        window_before=window_before,
        window_after=window_after,
    )

    return beats, labels, valid_peaks
