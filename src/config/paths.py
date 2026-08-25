"""
Path configuration for MedAgent.
Resolves dynamic absolute paths across Windows/Linux environments.
"""

from pathlib import Path

# Project Root Directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Source Directories
SRC_DIR = PROJECT_ROOT / "src"
CONFIG_DIR = SRC_DIR / "config"
CONFIG_FILE = CONFIG_DIR / "config.yaml"

# Data Directories
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_RAW_ECG_DIR = DATA_RAW_DIR / "ecg"
DATA_RAW_MITBIH_DIR = DATA_RAW_ECG_DIR / "mit-bih-arrhythmia-database-1.0.0"
DATA_RAW_PPG_DIR = DATA_RAW_DIR / "ppg"
DATA_RAW_EDA_DIR = DATA_RAW_DIR / "eda"
DATA_RAW_IMAGING_DIR = DATA_RAW_DIR / "imaging"
DATA_RAW_ACCEL_DIR = DATA_RAW_DIR / "accelerometer"
DATA_EXTERNAL_DIR = DATA_DIR / "external"

# Processed Data Directories
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_PROCESSED_SIGNALS_DIR = DATA_PROCESSED_DIR / "signals"
DATA_PROCESSED_IMAGES_DIR = DATA_PROCESSED_DIR / "images"
DATA_PROCESSED_MULTIMODAL_DIR = DATA_PROCESSED_DIR / "multimodal"

# Models & Checkpoints Directories
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
EXPORTED_MODELS_DIR = MODELS_DIR / "exported"
PRETRAINED_DIR = MODELS_DIR / "pretrained"

# Logs & Experiments Directories
LOGS_DIR = PROJECT_ROOT / "logs"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
DOCS_DIR = PROJECT_ROOT / "docs"

# Ensure essential output directories exist
for directory in [
    DATA_PROCESSED_SIGNALS_DIR,
    DATA_PROCESSED_IMAGES_DIR,
    DATA_PROCESSED_MULTIMODAL_DIR,
    CHECKPOINTS_DIR,
    EXPORTED_MODELS_DIR,
    PRETRAINED_DIR,
    LOGS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
