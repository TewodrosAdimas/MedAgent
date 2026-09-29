"""
Unit Tests for Multimodal Dataset and DataLoader Pipelines in MedAgent.
"""

from pathlib import Path
import numpy as np
import pytest
import torch

from src.data.loaders.multimodal_dataset import (
    RISK_CLASSES,
    TABULAR_FEATURE_NAMES,
    MultimodalDataset,
    determine_clinical_risk,
    get_multimodal_dataloaders,
)


def test_determine_clinical_risk():
    # 1. Healthy patient -> Low Risk (0)
    assert determine_clinical_risk(ecg_label=0, xray_label=0, has_hypertension=False) == 0

    # 2. Isolated moderate finding -> Moderate Risk (1)
    assert determine_clinical_risk(ecg_label=1, xray_label=0) == 1
    assert determine_clinical_risk(ecg_label=0, xray_label=3) == 1

    # 3. High risk: severe arrhythmia / severe imaging / multiple findings
    assert determine_clinical_risk(ecg_label=2, xray_label=0) == 2  # Ventricular ectopy
    assert determine_clinical_risk(ecg_label=1, xray_label=1) == 2  # S-ectopy + Pneumonia
    assert determine_clinical_risk(ecg_label=0, xray_label=2, has_prior_cardiac=True) == 2
    assert determine_clinical_risk(ecg_label=0, xray_label=0, is_hypoxic=True) == 2


def test_multimodal_dataset_synthetic_cohort():
    ecg_path = Path("data/processed/signals/mitbih_aami_beats.npz")
    xray_dir = Path("data/raw/imaging/synthetic_xray")

    if not ecg_path.exists() or not xray_dir.exists():
        pytest.skip("Required ECG or X-ray data not found.")

    num_samples = 30
    dataset = MultimodalDataset.create_synthetic_cohort(
        ecg_npz_path=ecg_path,
        xray_root_dir=xray_dir,
        num_samples=num_samples,
        random_seed=42,
    )

    assert len(dataset) == num_samples

    item = dataset[0]
    assert isinstance(item, dict)
    assert "ecg" in item
    assert "image" in item
    assert "tabular" in item
    assert "ecg_label" in item
    assert "xray_label" in item
    assert "risk_label" in item
    assert "patient_id" in item

    assert item["ecg"].shape == (1, 260)
    assert item["image"].shape == (3, 224, 224)
    assert item["tabular"].shape == (len(TABULAR_FEATURE_NAMES),)
    assert item["ecg_label"].dtype == torch.long
    assert item["xray_label"].dtype == torch.long
    assert item["risk_label"].dtype == torch.long
    assert 0 <= item["risk_label"].item() < len(RISK_CLASSES)


def test_get_multimodal_dataloaders():
    ecg_path = Path("data/processed/signals/mitbih_aami_beats.npz")
    xray_dir = Path("data/raw/imaging/synthetic_xray")

    if not ecg_path.exists() or not xray_dir.exists():
        pytest.skip("Required data files not found.")

    train_loader, val_loader, test_loader = get_multimodal_dataloaders(
        ecg_npz_path=ecg_path,
        xray_root_dir=xray_dir,
        num_samples=40,
        batch_size=8,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
    )

    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0

    batch = next(iter(train_loader))
    assert batch["ecg"].shape == (8, 1, 260)
    assert batch["image"].shape == (8, 3, 224, 224)
    assert batch["tabular"].shape == (8, len(TABULAR_FEATURE_NAMES))
    assert batch["ecg_label"].shape == (8,)
    assert batch["xray_label"].shape == (8,)
    assert batch["risk_label"].shape == (8,)
