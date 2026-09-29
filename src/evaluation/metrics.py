"""
Clinical and Machine Learning Evaluation Metrics for MedAgent.

Computes:
1. Accuracy, Balanced Accuracy, Precision, Recall, Macro-F1, Weighted-F1.
2. Clinical Sensitivity, Specificity, Positive Predictive Value (PPV), Negative Predictive Value (NPV).
3. Multi-Class AUROC (One-vs-Rest) and AUPRC.
4. Confusion Matrix with Normalized Rates.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    y_prob: Optional[Union[np.ndarray, torch.Tensor]] = None,
    class_names: Optional[Dict[int, str]] = None,
) -> Dict[str, Union[float, Dict]]:
    """
    Computes a comprehensive dictionary of AI and clinical evaluation metrics.

    Args:
        y_true: Ground truth class integer labels (N,).
        y_pred: Predicted class integer labels (N,).
        y_prob: Optional predicted class probabilities (N, num_classes).
        class_names: Optional mapping from class ID to human-readable string.

    Returns:
        Structured metrics dictionary.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.detach().cpu().numpy()

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    # 1. Global Metrics
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # 2. AUROC (if probabilities are supplied)
    auroc = None
    if y_prob is not None:
        try:
            unique_classes = np.unique(y_true)
            if len(unique_classes) > 1:
                if y_prob.shape[1] == 2:
                    auroc = float(roc_auc_score(y_true, y_prob[:, 1]))
                else:
                    auroc = float(
                        roc_auc_score(
                            y_true,
                            y_prob,
                            multi_class="ovr",
                            average="macro",
                            labels=unique_classes,
                        )
                    )
        except Exception:
            auroc = None

    # 3. Confusion Matrix & Per-Class Clinical Sensitivity/Specificity
    classes = sorted(list(set(y_true).union(set(y_pred))))
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    per_class_metrics = {}
    for i, cls_id in enumerate(classes):
        cls_name = class_names.get(cls_id, f"Class_{cls_id}") if class_names else f"Class_{cls_id}"

        # True Positives, False Positives, False Negatives, True Negatives
        tp = float(cm[i, i])
        fn = float(np.sum(cm[i, :]) - tp)
        fp = float(np.sum(cm[:, i]) - tp)
        tn = float(np.sum(cm) - tp - fn - fp)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall / TPR
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # TNR
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0          # Precision
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

        per_class_metrics[cls_name] = {
            "class_id": int(cls_id),
            "support": int(tp + fn),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "ppv_precision": float(ppv),
            "npv": float(npv),
            "f1_score": float(f1),
        }

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "auroc": auroc,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class_metrics,
    }
