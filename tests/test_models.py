"""
Unit tests for MedAgent Deep Learning Signal Models & Evaluation Metrics.
"""

import numpy as np
import pytest
import torch

from src.evaluation.metrics import compute_clinical_metrics
from src.models.signals.cnn1d import ECGCNN1D
from src.models.signals.lstm import ECGBiLSTM
from src.models.signals.transformer import ECGSignalTransformer


class TestSignalModels:
    def test_cnn1d_shapes(self):
        batch_size = 16
        seq_len = 260
        num_classes = 5

        model = ECGCNN1D(
            in_channels=1,
            sequence_length=seq_len,
            num_classes=num_classes,
        )

        dummy_x = torch.randn(batch_size, 1, seq_len)
        logits = model(dummy_x)
        features = model.get_features(dummy_x)

        assert logits.shape == (batch_size, num_classes)
        assert features.shape == (batch_size, 256)
        assert not torch.isnan(logits).any()

    def test_transformer_shapes(self):
        batch_size = 8
        seq_len = 260
        num_classes = 5

        model = ECGSignalTransformer(
            in_channels=1,
            sequence_length=seq_len,
            num_classes=num_classes,
            patch_size=20,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
        )

        dummy_x = torch.randn(batch_size, 1, seq_len)
        logits = model(dummy_x)
        features = model.get_features(dummy_x)

        assert logits.shape == (batch_size, num_classes)
        assert features.shape == (batch_size, 64)
        assert not torch.isnan(logits).any()

    def test_lstm_shapes(self):
        batch_size = 8
        seq_len = 260
        num_classes = 5

        model = ECGBiLSTM(
            in_channels=1,
            sequence_length=seq_len,
            num_classes=num_classes,
            hidden_dim=64,
            bidirectional=True,
        )

        dummy_x = torch.randn(batch_size, 1, seq_len)
        logits = model(dummy_x)
        features = model.get_features(dummy_x)

        assert logits.shape == (batch_size, num_classes)
        assert features.shape == (batch_size, 128)
        assert not torch.isnan(logits).any()


class TestMetrics:
    def test_clinical_metrics_computation(self):
        y_true = np.array([0, 0, 1, 1, 2, 2, 3, 4])
        y_pred = np.array([0, 0, 1, 0, 2, 2, 3, 4])  # 1 mistake in class 1

        metrics = compute_clinical_metrics(y_true, y_pred)

        assert "accuracy" in metrics
        assert "macro_f1" in metrics
        assert "class_metrics" in metrics
        assert metrics["accuracy"] == 7 / 8  # 87.5%

        # Check normal class 0 sensitivity
        class_0 = metrics["class_metrics"]["Normal (N)"]
        assert class_0["sensitivity"] == 1.0  # Both true 0s detected
