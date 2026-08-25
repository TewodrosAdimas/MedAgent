"""
Multimodal Fusion Model Training Pipeline for MedAgent.
Trains Early Fusion, Late Fusion, and Cross-Modal Attention Transformers on synchronized
ECG waveforms, Chest X-Rays, and clinical patient demographics.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config.paths import CHECKPOINTS_DIR
from src.data.loaders.multimodal_loader import RISK_LEVELS, get_multimodal_dataloaders
from src.evaluation.metrics import compute_clinical_metrics, format_metrics_summary
from src.models.fusion.early_fusion import EarlyFusionModel
from src.models.fusion.late_fusion import LateFusionModel
from src.models.fusion.multimodal_transformer import MultimodalTransformer
from src.utils.logger import get_logger

logger = get_logger("TrainMultimodal")


def create_fusion_model(
    model_type: str = "transformer",
    num_risk_classes: int = 3,
    num_signal_classes: int = 5,
    num_image_classes: int = 5,
    tabular_dim: int = 8,
    pretrained_vision: bool = True,
) -> nn.Module:
    """Instantiates the requested multimodal fusion neural network architecture."""
    model_type = model_type.lower()
    if model_type == "transformer":
        return MultimodalTransformer(
            num_risk_classes=num_risk_classes,
            num_signal_classes=num_signal_classes,
            num_image_classes=num_image_classes,
            tabular_dim=tabular_dim,
            embed_dim=256,
            num_heads=8,
            num_layers=3,
            pretrained_vision=pretrained_vision,
        )
    elif model_type == "early":
        return EarlyFusionModel(
            num_risk_classes=num_risk_classes,
            num_signal_classes=num_signal_classes,
            num_image_classes=num_image_classes,
            tabular_dim=tabular_dim,
            joint_hidden=256,
            pretrained_vision=pretrained_vision,
        )
    elif model_type == "late":
        return LateFusionModel(
            num_risk_classes=num_risk_classes,
            num_signal_classes=num_signal_classes,
            num_image_classes=num_image_classes,
            tabular_dim=tabular_dim,
            pretrained_vision=pretrained_vision,
        )
    else:
        raise ValueError(f"Unknown fusion model type '{model_type}'. Choose from 'transformer', 'early', 'late'")


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Runs a single multimodal training epoch with multi-task objective."""
    model.train()
    running_loss = 0.0
    correct_risk = 0
    total = 0

    for batch in loader:
        ecg = batch["ecg"].to(device, non_blocking=True)
        image = batch["image"].to(device, non_blocking=True)
        tabular = batch["tabular"].to(device, non_blocking=True)
        risk_labels = batch["risk_label"].to(device, non_blocking=True)
        sig_labels = batch["sig_label"].to(device, non_blocking=True)
        img_labels = batch["img_label"].to(device, non_blocking=True)

        optimizer.zero_grad()
        out = model(ecg, image, tabular)

        # Multi-task loss
        loss_risk = criterion(out["risk_logits"], risk_labels)
        loss_sig = criterion(out["sig_logits"], sig_labels)
        loss_img = criterion(out["img_logits"], img_labels)

        loss = loss_risk + 0.4 * loss_sig + 0.4 * loss_img
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * len(risk_labels)
        preds = torch.argmax(out["risk_logits"], dim=1)
        correct_risk += (preds == risk_labels).sum().item()
        total += len(risk_labels)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct_risk / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def evaluate_fusion_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict]:
    """Evaluates the multimodal model on validation or test sets."""
    model.eval()
    running_loss = 0.0
    total = 0

    all_risk_preds = []
    all_risk_labels = []
    all_risk_probs = []

    with torch.no_grad():
        for batch in loader:
            ecg = batch["ecg"].to(device, non_blocking=True)
            image = batch["image"].to(device, non_blocking=True)
            tabular = batch["tabular"].to(device, non_blocking=True)
            risk_labels = batch["risk_label"].to(device, non_blocking=True)

            out = model(ecg, image, tabular)
            loss = criterion(out["risk_logits"], risk_labels)
            probs = torch.softmax(out["risk_logits"], dim=1)
            preds = torch.argmax(probs, dim=1)

            running_loss += loss.item() * len(risk_labels)
            total += len(risk_labels)

            all_risk_preds.extend(preds.cpu().numpy())
            all_risk_labels.extend(risk_labels.cpu().numpy())
            all_risk_probs.extend(probs.cpu().numpy())

    val_loss = running_loss / total if total > 0 else 0.0
    metrics = compute_clinical_metrics(
        y_true=np.array(all_risk_labels),
        y_pred=np.array(all_risk_preds),
        y_prob=np.array(all_risk_probs),
        class_names=RISK_LEVELS,
    )
    return val_loss, metrics


def train_multimodal_pipeline(
    model_type: str = "transformer",
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 0.0003,
    weight_decay: float = 1e-4,
    use_cuda: bool = True,
    patience: int = 5,
    num_samples: int = 600,
):
    """Executes the full Multimodal training, validation, and evaluation pipeline."""
    logger.info(f"=== Starting Multimodal Fusion Pipeline Training: {model_type.upper()} ===")

    device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
    logger.info(f"Execution Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # Load dataloaders
    train_loader, val_loader, test_loader, metadata = get_multimodal_dataloaders(
        batch_size=batch_size,
        num_samples=num_samples,
    )

    model = create_fusion_model(model_type=model_type).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model initialized: {model_type.upper()} with {total_params:,} trainable parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = CHECKPOINTS_DIR / f"multimodal_{model_type}_best.pth"

    best_macro_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_metrics = evaluate_fusion_model(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        val_acc = val_metrics["accuracy"]
        val_macro_f1 = val_metrics["macro_f1"]

        scheduler.step()
        curr_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch [{epoch:02d}/{epochs:02d}] | "
            f"Train Loss: {train_loss:.4f} | Train Risk Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Risk Acc: {val_acc*100:.2f}% | "
            f"Val Macro F1: {val_macro_f1*100:.2f}% | LR: {curr_lr:.6f}"
        )

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "model_type": model_type,
                    "risk_classes": RISK_LEVELS,
                },
                best_checkpoint_path,
            )
            logger.info(f"  --> Saved new best checkpoint to {best_checkpoint_path.name} (Risk Macro F1: {best_macro_f1*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

    total_time = time.time() - start_time
    logger.info(f"Multimodal training finished in {total_time/60:.2f} minutes. Best Epoch: {best_epoch}")

    # Evaluate best checkpoint on unseen Test set
    logger.info("\n=== Evaluating Best Multimodal Model on Unseen Test Set ===")
    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_metrics = evaluate_fusion_model(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    summary_table = format_metrics_summary(test_metrics)
    logger.info(f"\nFinal Test Set Clinical Risk Stratification Performance:\n{summary_table}")

    # Sample Cross-Modal Attention Analysis
    if model_type == "transformer":
        sample_batch = next(iter(test_loader))
        ecg = sample_batch["ecg"][:1].to(device)
        img = sample_batch["image"][:1].to(device)
        tab = sample_batch["tabular"][:1].to(device)

        with torch.no_grad():
            out = model(ecg, img, tab)
            attn = out["attention_weights"][0].cpu().numpy()  # (4, 4)
            logger.info("\n--- Cross-Modal Attention Weights [Patient Summary -> Modalities] ---")
            modality_names = ["Patient Summary", "ECG Signal", "Chest X-Ray", "Demographics"]
            for idx, name in enumerate(modality_names):
                logger.info(f"  Attention to {name:<18}: {attn[0, idx]:.4f}")

    return model, test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedAgent Multimodal Fusion Training")
    parser.add_argument("--model", type=str, default="transformer", choices=["transformer", "early", "late"], help="Fusion architecture")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--samples", type=int, default=500, help="Number of multimodal patient samples")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA")

    args = parser.parse_args()

    train_multimodal_pipeline(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_cuda=not args.no_cuda,
        patience=args.patience,
        num_samples=args.samples,
    )
