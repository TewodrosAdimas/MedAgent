"""
MedAgent Data Loaders Module.
Contains PyTorch Dataset and DataLoader implementations for ECG signals, X-rays, and multimodal patient records.
"""

from src.data.loaders.ecg_dataset import (
    ECGAugmentation,
    ECGDataset,
    get_ecg_dataloaders,
)
from src.data.loaders.xray_dataset import (
    FOLDER_TO_CLASS_ID,
    XRAY_CLASSES,
    XRayDataset,
    get_xray_dataloaders,
)
from src.data.loaders.multimodal_dataset import (
    RISK_CLASSES,
    TABULAR_FEATURE_NAMES,
    MultimodalDataset,
    determine_clinical_risk,
    get_multimodal_dataloaders,
)

__all__ = [
    "ECGDataset",
    "ECGAugmentation",
    "get_ecg_dataloaders",
    "XRayDataset",
    "get_xray_dataloaders",
    "XRAY_CLASSES",
    "FOLDER_TO_CLASS_ID",
    "MultimodalDataset",
    "get_multimodal_dataloaders",
    "RISK_CLASSES",
    "determine_clinical_risk",
    "TABULAR_FEATURE_NAMES",
]
