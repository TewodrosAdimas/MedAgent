"""
Clinical & Machine Learning Evaluation Metrics for MedAgent.
Computes Accuracy, Macro F1, Sensitivity, Specificity, AUROC, and Confusion Matrix.
"""

from typing import Dict, List, Optional, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.preprocessing.segmentation import AAMI_CLASS_NAMES


def compute_clinical_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    y_prob: Optional[Union[np.ndarray, list]] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Union[float, np.ndarray, Dict]]:
    """
    Computes comprehensive clinical and statistical performance metrics.

    Args:
        y_true: Ground-truth integer class labels (0..K-1)
        y_pred: Predicted integer class labels (0..K-1)
        y_prob: Optional predicted class probability matrix of shape (N, K)
        class_names: Optional list of class names

    Returns:
        Dictionary containing overall and class-wise clinical metrics.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    if class_names is None:
        class_names = AAMI_CLASS_NAMES

    num_classes = len(class_names)

    # Global Classification Metrics
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # Confusion Matrix: shape (num_classes, num_classes)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # Per-Class Clinical Metrics (Sensitivity, Specificity, Precision, F1)
    class_metrics = {}
    total_samples = len(y_true)

    for i, c_name in enumerate(class_names):
        tp = float(cm[i, i])
        fn = float(np.sum(cm[i, :]) - tp)
        fp = float(np.sum(cm[:, i]) - tp)
        tn = float(total_samples - (tp + fn + fp))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * prec * sensitivity) / (prec + sensitivity) if (prec + sensitivity) > 0 else 0.0

        class_metrics[c_name] = {
            "support": int(np.sum(cm[i, :])),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "precision": float(prec),
            "f1_score": float(f1),
        }

    # Optional Multi-class AUROC
    macro_auroc = None
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=np.float32)
        try:
            # Only compute for classes actually present in y_true
            present_classes = np.unique(y_true)
            if len(present_classes) > 1:
                macro_auroc = float(
                    roc_auc_score(
                        y_true,
                        y_prob,
                        multi_class="ovr",
                        average="macro",
                        labels=list(range(num_classes)),
                    )
                )
        except Exception:
            macro_auroc = None

    metrics_dict = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_auroc": macro_auroc,
        "confusion_matrix": cm,
        "class_metrics": class_metrics,
    }

    return metrics_dict


def format_metrics_summary(metrics: Dict) -> str:
    """
    Formats the clinical metrics dictionary into a readable clinical summary table.
    """
    lines = [
        "=" * 70,
        f"  Accuracy: {metrics['accuracy'] * 100:.2f}% | Macro F1: {metrics['macro_f1'] * 100:.2f}% | "
        + (f"AUROC: {metrics['macro_auroc']:.4f}" if metrics['macro_auroc'] is not None else ""),
        "-" * 70,
        f"{'Class Name':<28} | {'Support':<8} | {'Sensitivity':<11} | {'Specificity':<11} | {'F1-Score':<8}",
        "-" * 70,
    ]

    for c_name, m in metrics["class_metrics"].items():
        lines.append(
            f"{c_name:<28} | {m['support']:<8} | {m['sensitivity'] * 100:>9.2f}% | "
            f"{m['specificity'] * 100:>9.2f}% | {m['f1_score'] * 100:>6.2f}%"
        )
    lines.append("=" * 70)
    return "\n".join(lines)
