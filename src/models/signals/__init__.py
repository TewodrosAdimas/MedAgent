"""
MedAgent Signal Models Module.
Contains deep learning architectures for physiological biosignals (ECG, PPG).
"""

from src.models.signals.ecg_cnn import ECG1DCNN, ECGResidualBlock1D
from src.models.signals.ecg_transformer import ECGPatchEmbedding1D, ECGTransformer

__all__ = [
    "ECG1DCNN",
    "ECGResidualBlock1D",
    "ECGPatchEmbedding1D",
    "ECGTransformer",
]
