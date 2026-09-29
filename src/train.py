"""
Master Training Pipeline for MedAgent Multimodal Clinical AI.

Trains and checkpoints:
1. 1D Residual CNN for AAMI Arrhythmia Classification (ECG Lead MLII).
2. DenseNet-121 for 5-Class Thoracic Chest Radiograph Classification.
3. Cross-Attention Multimodal Transformer for 3-Tier Clinical Risk Stratification.

Accelerated on NVIDIA CUDA GPUs with Automatic Mixed Precision (AMP).
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config.paths import CHECKPOINTS_DIR, DATA_PROCESSED_SIGNALS_DIR, DATA_RAW_IMAGING_DIR
from src.data.loaders.ecg_dataset import AAMI_CLASSES, get_ecg_dataloaders
from src.data.loaders.multimodal_dataset import RISK_CLASSES, get_multimodal_dataloaders
from src.data.loaders.xray_dataset import XRAY_CLASSES, get_xray_dataloaders
from src.evaluation.evaluator import ModelEvaluator, Trainer
from src.models.fusion.multimodal_transformer import MultimodalTransformerFusion
from src.models.signals.ecg_cnn import ECG1DCNN
from src.models.vision.xray_densenet import XRayDenseNet121
from src.utils.logger import setup_logger
from src.utils.seed import set_seed

logger = setup_logger("MedAgent.Train")


class EndToEndMultimodalModel(nn.Module):
    """Wraps frozen ECG and Vision encoders with trainable Cross-Attention Fusion Transformer."""

    def __init__(
        self,
        ecg_model: nn.Module,
        xray_model: nn.Module,
        fusion_model: nn.Module,
    ):
        super().__init__()
        self.ecg_model = ecg_model
        self.xray_model = xray_model
        self.fusion_model = fusion_model

    def forward(
        self,
        ecg: torch.Tensor,
        image: torch.Tensor,
        tabular: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            ecg_latent = self.ecg_model.extract_features(ecg)
            vision_latent = self.xray_model.extract_features(image)
        return self.fusion_model(ecg_latent, vision_latent, tabular)


def train_ecg_model(
    data_path: Path,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict]:
    """Trains and evaluates 1D Residual CNN on MIT-BIH AAMI arrhythmia heartbeats."""
    logger.info("=" * 65)
    logger.info("PHASE 1: Training 1D Residual CNN for ECG Arrhythmia Classification")
    logger.info("=" * 65)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, class_weights = get_ecg_dataloaders(
        npz_path=data_path,
        batch_size=batch_size,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
    )

    model = ECG1DCNN(in_channels=1, num_classes=5, latent_dim=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    ckpt_path = CHECKPOINTS_DIR / "ecg_cnn_best.pt"
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        use_amp=True,
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        checkpoint_path=ckpt_path,
        patience=5,
        class_names=AAMI_CLASSES,
    )

    # Load best checkpoint and evaluate on Test Split
    logger.info("Loading best ECG checkpoint for test evaluation...")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    evaluator = ModelEvaluator(device=device)
    test_loss, test_metrics = evaluator.evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        class_names=AAMI_CLASSES,
    )

    logger.info(f"ECG Test Loss: {test_loss:.4f} | Test Accuracy: {test_metrics['accuracy']:.4f} | Test Macro-F1: {test_metrics['macro_f1']:.4f}")
    if test_metrics.get("macro_auroc") is not None:
        logger.info(f"ECG Test Macro-AUROC: {test_metrics['macro_auroc']:.4f}")

    return model, test_metrics


def train_xray_model(
    data_dir: Path,
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 5e-4,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict]:
    """Trains and evaluates DenseNet-121 on Chest Radiographs."""
    logger.info("=" * 65)
    logger.info("PHASE 2: Training DenseNet-121 for Chest Radiograph Classification")
    logger.info("=" * 65)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, class_weights = get_xray_dataloaders(
        root_dir=data_dir,
        batch_size=batch_size,
        image_size=224,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
    )

    model = XRayDenseNet121(num_classes=5, pretrained=False, latent_dim=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    ckpt_path = CHECKPOINTS_DIR / "xray_densenet_best.pt"
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        use_amp=True,
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        checkpoint_path=ckpt_path,
        patience=5,
        class_names=XRAY_CLASSES,
    )

    # Load best checkpoint and evaluate on Test Split
    logger.info("Loading best X-Ray checkpoint for test evaluation...")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    evaluator = ModelEvaluator(device=device)
    test_loss, test_metrics = evaluator.evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        class_names=XRAY_CLASSES,
    )

    logger.info(f"X-Ray Test Loss: {test_loss:.4f} | Test Accuracy: {test_metrics['accuracy']:.4f} | Test Macro-F1: {test_metrics['macro_f1']:.4f}")
    if test_metrics.get("macro_auroc") is not None:
        logger.info(f"X-Ray Test Macro-AUROC: {test_metrics['macro_auroc']:.4f}")

    return model, test_metrics


def train_multimodal_fusion(
    ecg_model: nn.Module,
    xray_model: nn.Module,
    ecg_path: Path,
    xray_dir: Path,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    num_samples: int = 1000,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict]:
    """Trains and evaluates Cross-Attention Multimodal Transformer for Patient Risk Stratification."""
    logger.info("=" * 65)
    logger.info("PHASE 3: Training Cross-Attention Multimodal Transformer Fusion Engine")
    logger.info("=" * 65)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ecg_model.eval().to(device)
    xray_model.eval().to(device)

    train_loader, val_loader, test_loader = get_multimodal_dataloaders(
        ecg_npz_path=ecg_path,
        xray_root_dir=xray_dir,
        num_samples=num_samples,
        batch_size=batch_size,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
    )

    fusion_model = MultimodalTransformerFusion(
        ecg_latent_dim=256,
        vision_latent_dim=256,
        tabular_dim=10,
        embed_dim=256,
        num_heads=8,
        num_layers=3,
        num_risk_classes=3,
    ).to(device)

    e2e_model = EndToEndMultimodalModel(
        ecg_model=ecg_model,
        xray_model=xray_model,
        fusion_model=fusion_model,
    ).to(device)

    optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    ckpt_path = CHECKPOINTS_DIR / "fusion_transformer_best.pt"
    trainer = Trainer(
        model=e2e_model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        use_amp=True,
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        checkpoint_path=ckpt_path,
        patience=5,
        class_names=RISK_CLASSES,
        is_multimodal=True,
    )

    # Save standalone fusion model state dict
    torch.save(
        {
            "model_state_dict": fusion_model.state_dict(),
            "ecg_latent_dim": 256,
            "vision_latent_dim": 256,
            "tabular_dim": 10,
            "embed_dim": 256,
            "num_risk_classes": 3,
        },
        str(CHECKPOINTS_DIR / "fusion_transformer_weights.pt"),
    )

    # Evaluate on Test Split
    evaluator = ModelEvaluator(device=device)
    test_loss, test_metrics = evaluator.evaluate(
        model=e2e_model,
        data_loader=test_loader,
        criterion=criterion,
        class_names=RISK_CLASSES,
        is_multimodal=True,
    )

    logger.info(f"Multimodal Risk Test Loss: {test_loss:.4f} | Test Accuracy: {test_metrics['accuracy']:.4f} | Test Macro-F1: {test_metrics['macro_f1']:.4f}")
    if test_metrics.get("macro_auroc") is not None:
        logger.info(f"Multimodal Risk Test Macro-AUROC: {test_metrics['macro_auroc']:.4f}")

    return fusion_model, test_metrics


def run_full_training_pipeline():
    """Executes full training across all modalities and risk stratification."""
    parser = argparse.ArgumentParser(description="MedAgent Multi-Modal Deep Learning Training")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Target device (cuda or cpu)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    logger.info(f"Starting MedAgent Training Pipeline on {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})...")

    ecg_data_path = DATA_PROCESSED_SIGNALS_DIR / "mitbih_aami_beats.npz"
    xray_data_dir = DATA_RAW_IMAGING_DIR / "synthetic_xray"

    # Step 1: Train ECG
    ecg_model, ecg_metrics = train_ecg_model(
        data_path=ecg_data_path,
        epochs=args.epochs,
        batch_size=64,
        device=device,
    )

    # Step 2: Train X-Ray
    xray_model, xray_metrics = train_xray_model(
        data_dir=xray_data_dir,
        epochs=args.epochs,
        batch_size=16,
        device=device,
    )

    # Step 3: Train Multimodal Fusion
    fusion_model, fusion_metrics = train_multimodal_fusion(
        ecg_model=ecg_model,
        xray_model=xray_model,
        ecg_path=ecg_data_path,
        xray_dir=xray_data_dir,
        epochs=args.epochs,
        batch_size=32,
        num_samples=1000,
        device=device,
    )

    logger.info("=" * 65)
    logger.info("🎉 MEDAGENT FULL MULTIMODAL TRAINING COMPLETED SUCCESSFULLY!")
    logger.info(f"Saved Checkpoints at: {CHECKPOINTS_DIR.resolve()}")
    logger.info(f"ECG Classifier Accuracy: {ecg_metrics['accuracy']*100:.2f}% | Macro-F1: {ecg_metrics['macro_f1']:.4f}")
    logger.info(f"X-Ray Classifier Accuracy: {xray_metrics['accuracy']*100:.2f}% | Macro-F1: {xray_metrics['macro_f1']:.4f}")
    logger.info(f"Multimodal Risk Classifier Accuracy: {fusion_metrics['accuracy']*100:.2f}% | Macro-F1: {fusion_metrics['macro_f1']:.4f}")
    logger.info("=" * 65)


if __name__ == "__main__":
    run_full_training_pipeline()
