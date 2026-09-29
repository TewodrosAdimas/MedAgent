"""
Filesystem path definitions for MedAgent.
Centralizes all paths to data, models, logs, checkpoints, and docs.
"""

from pathlib import Path

# Project Root Directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data Directories
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_EXTERNAL_DIR = DATA_DIR / "external"

DATA_RAW_ECG_DIR = DATA_RAW_DIR / "ecg"
DATA_RAW_IMAGING_DIR = DATA_RAW_DIR / "imaging"
DATA_RAW_PPG_DIR = DATA_RAW_DIR / "ppg"

DATA_PROCESSED_SIGNALS_DIR = DATA_PROCESSED_DIR / "signals"
DATA_PROCESSED_IMAGES_DIR = DATA_PROCESSED_DIR / "images"
DATA_PROCESSED_MULTIMODAL_DIR = DATA_PROCESSED_DIR / "multimodal"

# Models & Checkpoints Directories
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
EXPORTED_MODELS_DIR = MODELS_DIR / "exported"
PRETRAINED_MODELS_DIR = MODELS_DIR / "pretrained"

# Logging Directory
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure runtime directories exist
for path in [
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    DATA_EXTERNAL_DIR,
    DATA_PROCESSED_SIGNALS_DIR,
    DATA_PROCESSED_IMAGES_DIR,
    DATA_PROCESSED_MULTIMODAL_DIR,
    CHECKPOINTS_DIR,
    EXPORTED_MODELS_DIR,
    PRETRAINED_MODELS_DIR,
    LOGS_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)
