"""
Model Evaluator and Unified PyTorch Trainer for MedAgent.

Supports:
1. Universal evaluation across Signal, Vision, and Multimodal PyTorch models.
2. Mixed-precision (FP16/AMP) hardware-accelerated training loops.
3. Automated best-model checkpointing and early stopping.
4. Class-weighted CrossEntropyLoss for severe class imbalance.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_classification_metrics


class ModelEvaluator:
    """
    Inference and Metric Evaluation Engine.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
        class_names: Optional[Dict[int, str]] = None,
        is_multimodal: bool = False,
    ) -> Tuple[float, Dict[str, Union[float, Dict]]]:
        """
        Runs validation/test evaluation loop and returns loss and metrics.
        """
        model.eval()
        model.to(self.device)

        total_loss = 0.0
        num_batches = 0
        all_preds = []
        all_labels = []
        all_probs = []

        for batch in data_loader:
            if is_multimodal or isinstance(batch, dict):
                ecg = batch["ecg"].to(self.device)
                image = batch["image"].to(self.device)
                tabular = batch["tabular"].to(self.device)
                target = batch["risk_label"].to(self.device)

                if hasattr(model, "ecg_model"):
                    logits, _ = model(ecg, image, tabular)
                elif hasattr(model, "ecg_proj"):
                    logits, _ = model(ecg.squeeze(1)[:, :256], image[:, 0, :16, :16].flatten(1)[:, :256], tabular)
                else:
                    logits, _ = model(ecg, image, tabular)
            else:
                inputs, target = batch[0].to(self.device), batch[1].to(self.device)
                out = model(inputs)
                logits = out[0] if isinstance(out, (tuple, list)) else out

            if criterion is not None:
                loss = criterion(logits, target)
                total_loss += loss.item()
                num_batches += 1

            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(target.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

        avg_loss = total_loss / max(1, num_batches)
        metrics = compute_classification_metrics(
            y_true=np.array(all_labels),
            y_pred=np.array(all_preds),
            y_prob=np.array(all_probs),
            class_names=class_names,
        )
        return avg_loss, metrics


class Trainer:
    """
    Standardized Trainer for MedAgent neural networks with early stopping and checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: Optional[torch.device] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        use_amp: bool = True,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion.to(self.device)
        self.scheduler = scheduler
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.evaluator = ModelEvaluator(device=self.device)

    def train_epoch(self, train_loader: DataLoader, is_multimodal: bool = False) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            self.optimizer.zero_grad()

            if is_multimodal or isinstance(batch, dict):
                ecg = batch["ecg"].to(self.device)
                image = batch["image"].to(self.device)
                tabular = batch["tabular"].to(self.device)
                target = batch["risk_label"].to(self.device)

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    if hasattr(self.model, "ecg_model"):
                        out = self.model(ecg, image, tabular)
                    elif hasattr(self.model, "ecg_proj"):
                        out = self.model(
                            ecg.squeeze(1)[:, :256],
                            image[:, 0, :16, :16].flatten(1)[:, :256],
                            tabular,
                        )
                    else:
                        out = self.model(ecg, image, tabular)
                    logits = out[0] if isinstance(out, (tuple, list)) else out
                    loss = self.criterion(logits, target)
            else:
                inputs, target = batch[0].to(self.device), batch[1].to(self.device)
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    out = self.model(inputs)
                    logits = out[0] if isinstance(out, (tuple, list)) else out
                    loss = self.criterion(logits, target)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1

        if self.scheduler is not None:
            self.scheduler.step()

        return total_loss / max(1, num_batches)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20,
        checkpoint_path: Optional[Union[str, Path]] = None,
        patience: int = 5,
        class_names: Optional[Dict[int, str]] = None,
        is_multimodal: bool = False,
    ) -> Dict[str, List[float]]:
        """
        Runs full training loop with early stopping.
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_macro_f1": [],
        }

        best_val_f1 = -1.0
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader, is_multimodal=is_multimodal)
            val_loss, val_metrics = self.evaluator.evaluate(
                self.model,
                val_loader,
                criterion=self.criterion,
                class_names=class_names,
                is_multimodal=is_multimodal,
            )

            val_acc = val_metrics["accuracy"]
            val_f1 = val_metrics["macro_f1"]

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)
            history["val_macro_f1"].append(val_f1)

            # Checkpoint best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                if checkpoint_path is not None:
                    ckpt_p = Path(checkpoint_path)
                    ckpt_p.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "val_macro_f1": val_f1,
                            "val_metrics": val_metrics,
                        },
                        str(ckpt_p),
                    )
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        return history
