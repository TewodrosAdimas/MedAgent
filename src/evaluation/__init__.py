"""
MedAgent Evaluation Module.
Contains clinical metrics, model evaluators, and training engines.
"""

from src.evaluation.evaluator import ModelEvaluator, Trainer
from src.evaluation.metrics import compute_classification_metrics

__all__ = [
    "compute_classification_metrics",
    "ModelEvaluator",
    "Trainer",
]
