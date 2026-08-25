"""
Signal Normalization Module for MedAgent.
Provides Z-score, Min-Max, and Robust scaling for biosignals and PyTorch tensors.
"""

from typing import Union
import numpy as np
import torch


def zscore_normalize(
    signal: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-8,
    axis: int = -1,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Applies Z-score normalization: z = (x - mean) / (std + eps)

    Args:
        signal: Input array or tensor (1D, 2D, or 3D)
        eps: Small constant to avoid division by zero
        axis: Axis along which to compute mean and std (default last axis: time)

    Returns:
        Z-score normalized array/tensor of identical shape and type
    """
    if isinstance(signal, torch.Tensor):
        mean = torch.mean(signal, dim=axis, keepdim=True)
        std = torch.std(signal, dim=axis, keepdim=True)
        return (signal - mean) / (std + eps)
    else:
        arr = np.asarray(signal, dtype=np.float32)
        mean = np.mean(arr, axis=axis, keepdims=True)
        std = np.std(arr, axis=axis, keepdims=True)
        return (arr - mean) / (std + eps)


def minmax_normalize(
    signal: Union[np.ndarray, torch.Tensor],
    feature_range: tuple = (0.0, 1.0),
    eps: float = 1e-8,
    axis: int = -1,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Applies Min-Max scaling to map values into feature_range (e.g. [0, 1] or [-1, 1]).

    Args:
        signal: Input array or tensor
        feature_range: Tuple (min_val, max_val)
        eps: Small constant to avoid division by zero
        axis: Axis along which to compute min and max

    Returns:
        Scaled array or tensor
    """
    min_target, max_target = feature_range

    if isinstance(signal, torch.Tensor):
        s_min = torch.amin(signal, dim=axis, keepdim=True)
        s_max = torch.amax(signal, dim=axis, keepdim=True)
        norm = (signal - s_min) / (s_max - s_min + eps)
        return norm * (max_target - min_target) + min_target
    else:
        arr = np.asarray(signal, dtype=np.float32)
        s_min = np.min(arr, axis=axis, keepdims=True)
        s_max = np.max(arr, axis=axis, keepdims=True)
        norm = (arr - s_min) / (s_max - s_min + eps)
        return norm * (max_target - min_target) + min_target


def robust_normalize(
    signal: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-8,
    axis: int = -1,
) -> np.ndarray:
    """
    Applies Robust scaling using median and Interquartile Range (IQR):
    scaled = (x - median) / (IQR + eps)
    Robust against extreme ECG peak outliers.

    Args:
        signal: Input array or tensor
        eps: Small constant to prevent division by zero
        axis: Axis along which to compute percentiles

    Returns:
        Robustly scaled np.ndarray
    """
    if isinstance(signal, torch.Tensor):
        arr = signal.detach().cpu().numpy().copy()
    else:
        arr = np.asarray(signal, dtype=np.float32)

    median = np.median(arr, axis=axis, keepdims=True)
    q75 = np.percentile(arr, 75, axis=axis, keepdims=True)
    q25 = np.percentile(arr, 25, axis=axis, keepdims=True)
    iqr = q75 - q25

    scaled = (arr - median) / (iqr + eps)
    return scaled.astype(np.float32)
