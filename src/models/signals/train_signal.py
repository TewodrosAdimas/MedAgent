"""
ECG Signal Model Training Pipeline for MedAgent.
Supports training and evaluation for 1D CNN, Signal Transformer, and BiLSTM models
with class-weighted loss, cosine learning rate scheduling, and GPU acceleration.
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
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

from src.config.paths import CHECKPOINTS_DIR
from src.data.loaders.ecg_loader import get_ecg_dataloaders
from src.data.preprocessing.segmentation import AAMI_CLASS_NAMES
from src.evaluation.metrics import compute_clinical_metrics, format_metrics_summary
from src.models.signals.cnn1d import ECGCNN1D
from src.models.signals.lstm import ECGBiLSTM
from src.models.signals.transformer import ECGSignalTransformer
from src.utils.logger import get_logger

logger = get_logger("TrainSignal")


def create_model(
    model_type: str = "cnn1d",
    in_channels: int = 1,
    sequence_length: int = 260,
    num_classes: int = 5,
) -> nn.Module:
    """Instantiates the requested neural network architecture."""
    model_type = model_type.lower()
    if model_type == "cnn1d":
        return ECGCNN1D(
            in_channels=in_channels,
            sequence_length=sequence_length,
            num_classes=num_classes,
            conv_channels=[32, 64, 128, 256],
            kernel_sizes=[7, 5, 3, 3],
            dropout=0.3,
        )
    elif model_type == "transformer":
        return ECGSignalTransformer(
            in_channels=in_channels,
            sequence_length=sequence_length,
            num_classes=num_classes,
            patch_size=20,
            embed_dim=128,
            num_heads=8,
            num_layers=4,
            mlp_dim=256,
            dropout=0.2,
        )
    elif model_type == "lstm":
        return ECGBiLSTM(
            in_channels=in_channels,
            sequence_length=sequence_length,
            num_classes=num_classes,
            hidden_dim=128,
            num_layers=2,
            bidirectional=True,
            dropout=0.3,
        )
    else:
        raise ValueError(f"Unknown model type '{model_type}'. Choose from: 'cnn1d', 'transformer', 'lstm'")


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Runs a single training epoch and returns (loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for signals, labels in loader:
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping to prevent gradient explosion
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * len(labels)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict]:
    """Evaluates the model on validation or test sets."""
    model.eval()
    running_loss = 0.0
    total = 0

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(signals)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            running_loss += loss.item() * len(labels)
            total += len(labels)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    val_loss = running_loss / total if total > 0 else 0.0
    metrics = compute_clinical_metrics(
        y_true=np.array(all_labels),
        y_pred=np.array(all_preds),
        y_prob=np.array(all_probs),
        class_names=AAMI_CLASS_NAMES,
    )
    return val_loss, metrics


def train_ecg_pipeline(
    model_type: str = "cnn1d",
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    use_cuda: bool = True,
    use_cache: bool = True,
    patience: int = 7,
    record_subset: Optional[int] = None,
):
    """Executes the full training, validation, and testing lifecycle."""
    logger.info(f"=== Starting ECG Signal Pipeline Training: {model_type.upper()} ===")

    device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
    logger.info(f"Execution Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # Limit records for quick development runs if specified
    records = None
    if record_subset is not None:
        from src.data.loaders.ecg_loader import MITBIH_RECORD_IDS
        records = MITBIH_RECORD_IDS[:record_subset]
        logger.info(f"Using record subset: {len(records)} records")

    # Load datasets
    train_loader, val_loader, test_loader, class_weights = get_ecg_dataloaders(
        batch_size=batch_size,
        use_cache=use_cache,
        record_ids=records,
    )

    class_weights = class_weights.to(device)
    logger.info(f"Class imbalance loss weights: {class_weights.cpu().numpy().round(3)}")

    # Instantiate model
    model = create_model(model_type=model_type).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model initialized: {model_type.upper()} with {total_params:,} trainable parameters")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = CHECKPOINTS_DIR / f"ecg_{model_type}_best.pth"

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

        val_loss, val_metrics = evaluate_model(
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
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
            f"Val Macro F1: {val_macro_f1*100:.2f}% | LR: {curr_lr:.6f}"
        )

        # Track best model checkpoint
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
                    "class_names": AAMI_CLASS_NAMES,
                },
                best_checkpoint_path,
            )
            logger.info(f"  --> Saved new best checkpoint to {best_checkpoint_path.name} (Macro F1: {best_macro_f1*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch} (No improvement for {patience} epochs)")
                break

    total_time = time.time() - start_time
    logger.info(f"Training finished in {total_time/60:.2f} minutes. Best Epoch: {best_epoch} (Macro F1: {best_macro_f1*100:.2f}%)")

    # Load best checkpoint and evaluate on unseen Test set
    logger.info("\n=== Evaluating Best Checkpoint on Unseen Test Set ===")
    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])


    test_loss, test_metrics = evaluate_model(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    summary_table = format_metrics_summary(test_metrics)
    logger.info(f"\nFinal Test Set Clinical Performance:\n{summary_table}")

    return model, test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedAgent ECG Signal Training")
    parser.add_argument("--model", type=str, default="cnn1d", choices=["cnn1d", "transformer", "lstm"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA execution")
    parser.add_argument("--no-cache", action="store_true", help="Re-process raw records without loading cache")
    parser.add_argument("--subset", type=int, default=None, help="Process only first N records for fast testing")

    args = parser.parse_args()

    train_ecg_pipeline(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_cuda=not args.no_cuda,
        use_cache=not args.no_cache,
        patience=args.patience,
        record_subset=args.subset,
    )
