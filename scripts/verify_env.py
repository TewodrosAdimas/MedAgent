"""
Environment & Hardware Verification Script for MedAgent.
Validates PyTorch GPU/CUDA execution, WFDB data reading, and dependencies.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import DATA_RAW_MITBIH_DIR, PROJECT_ROOT
from src.utils.logger import get_logger

logger = get_logger("EnvVerify")


def verify_gpu():
    logger.info("--- Checking PyTorch & GPU Acceleration ---")
    import torch

    logger.info(f"PyTorch Version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA Available: {cuda_available}")

    if cuda_available:
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA Device Count: {device_count}")
        logger.info(f"Primary GPU: {device_name}")

        # Test simple CUDA tensor operations
        x = torch.randn(1000, 1000, device="cuda")
        y = torch.matmul(x, x)
        logger.info(f"GPU Matmul Test Passed. Allocated VRAM: {torch.cuda.memory_allocated(0) / (1024**2):.2f} MB")
    else:
        logger.warning("CUDA is not available. PyTorch will use CPU.")


def verify_wfdb_and_data():
    logger.info("--- Checking WFDB & MIT-BIH Arrhythmia Dataset ---")
    import wfdb

    record_path = DATA_RAW_MITBIH_DIR / "100"
    if record_path.with_suffix(".hea").exists():
        record = wfdb.rdrecord(str(record_path), sampto=1000)
        annotation = wfdb.rdann(str(record_path), "atr", sampto=1000)
        logger.info(f"Successfully loaded MIT-BIH Record 100! Signal shape: {record.p_signal.shape}, Sampling Rate: {record.fs} Hz")
        logger.info(f"Sample annotations found: {len(annotation.sample)} beats, Symbols: {annotation.symbol[:5]}")
    else:
        logger.error(f"MIT-BIH Record 100 not found at {record_path}")


def verify_signal_libraries():
    logger.info("--- Checking Signal Processing & ML Libraries ---")
    import numpy as np
    import pandas as pd
    import scipy
    from scipy.signal import butter, filtfilt
    import sklearn

    # Test Butterworth filter design
    b, a = butter(4, [0.5 / 180.0, 45.0 / 180.0], btype="bandpass")
    test_sig = np.random.randn(360)
    filtered = filtfilt(b, a, test_sig)
    logger.info(f"Scipy ({scipy.__version__}) Butterworth Filter Test Passed. Output length: {len(filtered)}")
    logger.info(f"Numpy ({np.__version__}), Pandas ({pd.__version__}), Scikit-Learn ({sklearn.__version__}) verified.")


if __name__ == "__main__":
    logger.info("=== MedAgent Environment Verification ===")
    verify_gpu()
    verify_wfdb_and_data()
    verify_signal_libraries()
    logger.info("=== Environment Verification Complete - Ready for Step 2! ===")
