"""
Unit Tests for MedAgent Evaluation Metrics and Model Training Engine.
"""

from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.evaluator import ModelEvaluator, Trainer


def test_compute_classification_metrics():
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 2, 0, 1, 1])
    y_prob = np.eye(3)[y_pred]  # One-hot simulated probabilities

    class_names = {0: "Normal", 1: "Arrhythmia", 2: "Ischemia"}
    metrics = compute_classification_metrics(y_true, y_pred, y_prob=y_prob, class_names=class_names)

    assert "accuracy" in metrics
    assert "balanced_accuracy" in metrics
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics
    assert "per_class" in metrics
    assert metrics["accuracy"] > 0.5
    assert "Normal" in metrics["per_class"]
    assert metrics["per_class"]["Normal"]["sensitivity"] == 1.0


def test_model_evaluator():
    # Simple linear model
    model = nn.Linear(10, 3)
    x = torch.randn(20, 10)
    y = torch.randint(0, 3, (20,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=5)

    evaluator = ModelEvaluator(device=torch.device("cpu"))
    loss, metrics = evaluator.evaluate(model, loader, criterion=nn.CrossEntropyLoss())

    assert loss >= 0.0
    assert "accuracy" in metrics
    assert "macro_f1" in metrics


def test_trainer_fit(tmp_path):
    model = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    x_train = torch.randn(40, 10)
    y_train = torch.randint(0, 2, (40,))
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=8)

    x_val = torch.randn(16, 10)
    y_val = torch.randint(0, 2, (16,))
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=8)

    ckpt_path = tmp_path / "best_model.pth"
    trainer = Trainer(model, optimizer, criterion, device=torch.device("cpu"), use_amp=False)
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=3,
        checkpoint_path=ckpt_path,
        patience=2,
    )

    assert len(history["train_loss"]) == 3
    assert len(history["val_accuracy"]) == 3
    assert ckpt_path.exists()
