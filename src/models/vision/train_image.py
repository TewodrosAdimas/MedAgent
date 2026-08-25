"""
Chest X-Ray Vision Model Training Pipeline for MedAgent.
Supports DenseNet-121 and Vision Transformer (ViT) training with GPU acceleration,
evaluation against clinical metrics, and automated Grad-CAM explainability verification.
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

from src.config.paths import CHECKPOINTS_DIR, DATA_PROCESSED_IMAGES_DIR
from src.data.loaders.image_loader import XRAY_CLASSES, get_xray_dataloaders
from src.evaluation.explainability import GradCAM, identify_anatomical_quadrants, overlay_cam_on_image
from src.evaluation.metrics import compute_clinical_metrics, format_metrics_summary
from src.models.vision.densenet import ChestXRayDenseNet121
from src.models.vision.vit import ChestXRayViT
from src.utils.logger import get_logger

logger = get_logger("TrainImage")


def create_vision_model(
    model_type: str = "densenet",
    num_classes: int = 5,
    pretrained: bool = True,
) -> nn.Module:
    """Instantiates the requested vision neural network architecture."""
    model_type = model_type.lower()
    if model_type == "densenet":
        return ChestXRayDenseNet121(num_classes=num_classes, pretrained=pretrained)
    elif model_type == "vit":
        return ChestXRayViT(num_classes=num_classes, embed_dim=384, depth=6, num_heads=6)
    else:
        raise ValueError(f"Unknown vision model type '{model_type}'. Choose from 'densenet', 'vit'")


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Runs a single training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * len(labels)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def evaluate_vision_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict]:
    """Evaluates the vision model on validation or test sets."""
    model.eval()
    running_loss = 0.0
    total = 0

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
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
        class_names=XRAY_CLASSES,
    )
    return val_loss, metrics


def train_xray_pipeline(
    model_type: str = "densenet",
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 0.0005,
    weight_decay: float = 1e-4,
    use_cuda: bool = True,
    patience: int = 5,
):
    """Executes the full Chest X-ray training, evaluation, and explainability lifecycle."""
    logger.info(f"=== Starting Chest X-Ray Vision Pipeline Training: {model_type.upper()} ===")

    device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
    logger.info(f"Execution Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # Load dataloaders
    train_loader, val_loader, test_loader, class_names = get_xray_dataloaders(
        batch_size=batch_size,
    )

    model = create_vision_model(model_type=model_type, num_classes=len(class_names)).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model initialized: {model_type.upper()} with {total_params:,} trainable parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = CHECKPOINTS_DIR / f"xray_{model_type}_best.pth"

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

        val_loss, val_metrics = evaluate_vision_model(
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
                    "class_names": class_names,
                },
                best_checkpoint_path,
            )
            logger.info(f"  --> Saved new best checkpoint to {best_checkpoint_path.name} (Macro F1: {best_macro_f1*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

    total_time = time.time() - start_time
    logger.info(f"Vision training finished in {total_time/60:.2f} minutes. Best Epoch: {best_epoch}")

    # Evaluate best checkpoint on unseen Test set
    logger.info("\n=== Evaluating Best Vision Model on Unseen Test Set ===")
    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_metrics = evaluate_vision_model(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    summary_table = format_metrics_summary(test_metrics)
    logger.info(f"\nFinal Test Set Clinical Performance:\n{summary_table}")

    # Demonstrate Grad-CAM Explainability on DenseNet
    if model_type == "densenet":
        logger.info("\n--- Generating Grad-CAM Radiographic Explainability Sample ---")
        sample_img, sample_label = next(iter(test_loader))
        sample_img = sample_img[:1].to(device)

        grad_cam = GradCAM(model=model, target_layer=model.get_target_layer())
        cam = grad_cam.generate_cam(sample_img)
        salient_regions = identify_anatomical_quadrants(cam)

        DATA_PROCESSED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        overlay = overlay_cam_on_image(sample_img[0], cam)
        from PIL import Image
        Image.fromarray(overlay).save(DATA_PROCESSED_IMAGES_DIR / "gradcam_sample.png")

        logger.info(f"Grad-CAM overlay saved to {DATA_PROCESSED_IMAGES_DIR / 'gradcam_sample.png'}")
        logger.info(f"Identified Salient Anatomical Regions: {salient_regions}")

    return model, test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedAgent Chest X-Ray Vision Training")
    parser.add_argument("--model", type=str, default="densenet", choices=["densenet", "vit"], help="Vision architecture")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA")

    args = parser.parse_args()

    train_xray_pipeline(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_cuda=not args.no_cuda,
        patience=args.patience,
    )
