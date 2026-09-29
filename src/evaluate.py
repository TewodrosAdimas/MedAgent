"""
Comprehensive Multi-Modal Model Evaluation Benchmark for MedAgent.

Evaluates trained checkpoint weights across all test cohorts:
1. Physiological Signal (ECG) Arrhythmia Classifier
2. Thoracic Chest Radiograph (X-Ray) DenseNet-121 Classifier
3. Cross-Attention Multimodal Patient Risk Stratification Transformer

Computes:
- Multi-Class AUROC & ROC Curves
- Sensitivity (Recall) & Specificity
- Positive Predictive Value (Precision / PPV) & Negative Predictive Value (NPV)
- Macro-F1 Score & Multi-Class Confusion Matrices
"""

import argparse
import json
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.config.paths import CHECKPOINTS_DIR, DATA_PROCESSED_SIGNALS_DIR, DATA_RAW_IMAGING_DIR, PROJECT_ROOT
from src.data.loaders.ecg_dataset import AAMI_CLASSES, get_ecg_dataloaders
from src.data.loaders.multimodal_dataset import RISK_CLASSES, get_multimodal_dataloaders
from src.data.loaders.xray_dataset import XRAY_CLASSES, get_xray_dataloaders
from src.evaluation.evaluator import ModelEvaluator
from src.models.fusion.multimodal_transformer import MultimodalTransformerFusion
from src.models.signals.ecg_cnn import ECG1DCNN
from src.models.vision.xray_densenet import XRayDenseNet121
from src.train import EndToEndMultimodalModel
from src.utils.logger import setup_logger
from src.utils.seed import set_seed

logger = setup_logger("MedAgent.Evaluate")


def evaluate_all_checkpoints(device: torch.device) -> Dict[str, Dict]:
    """Runs rigorous multi-metric test benchmark on all saved checkpoints."""
    logger.info("=" * 70)
    logger.info("MEDAGENT MULTIMODAL TEST SET EVALUATION BENCHMARK")
    logger.info("=" * 70)

    evaluator = ModelEvaluator(device=device)
    results = {}

    ecg_path = DATA_PROCESSED_SIGNALS_DIR / "mitbih_aami_beats.npz"
    xray_dir = DATA_RAW_IMAGING_DIR / "synthetic_xray"

    # ==========================================
    # 1. ECG Evaluation
    # ==========================================
    logger.info("Evaluating 1D Residual CNN on MIT-BIH ECG Test Split...")
    _, _, ecg_test_loader, _ = get_ecg_dataloaders(
        npz_path=ecg_path,
        batch_size=64,
        random_seed=42,
    )
    ecg_model = ECG1DCNN(in_channels=1, num_classes=5, latent_dim=256).to(device)
    ecg_ckpt_path = CHECKPOINTS_DIR / "ecg_cnn_best.pt"

    if ecg_ckpt_path.exists():
        ckpt = torch.load(str(ecg_ckpt_path), map_location=device, weights_only=False)
        ecg_model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded ECG weights from {ecg_ckpt_path.name}")
    else:
        logger.warning("ECG checkpoint not found, evaluating initialized weights.")

    ecg_loss, ecg_metrics = evaluator.evaluate(
        model=ecg_model,
        data_loader=ecg_test_loader,
        criterion=nn.CrossEntropyLoss(),
        class_names=AAMI_CLASSES,
    )
    results["ecg"] = ecg_metrics

    # ==========================================
    # 2. X-Ray Evaluation
    # ==========================================
    logger.info("Evaluating DenseNet-121 on Chest Radiograph Test Split...")
    _, _, xray_test_loader, _ = get_xray_dataloaders(
        root_dir=xray_dir,
        batch_size=16,
        random_seed=42,
    )
    xray_model = XRayDenseNet121(num_classes=5, pretrained=False, latent_dim=256).to(device)
    xray_ckpt_path = CHECKPOINTS_DIR / "xray_densenet_best.pt"

    if xray_ckpt_path.exists():
        ckpt = torch.load(str(xray_ckpt_path), map_location=device, weights_only=False)
        xray_model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded X-Ray weights from {xray_ckpt_path.name}")
    else:
        logger.warning("X-Ray checkpoint not found, evaluating initialized weights.")

    xray_loss, xray_metrics = evaluator.evaluate(
        model=xray_model,
        data_loader=xray_test_loader,
        criterion=nn.CrossEntropyLoss(),
        class_names=XRAY_CLASSES,
    )
    results["xray"] = xray_metrics

    # ==========================================
    # 3. Multimodal Fusion Evaluation
    # ==========================================
    logger.info("Evaluating Cross-Attention Multimodal Transformer on Test Cohort...")
    _, _, fusion_test_loader = get_multimodal_dataloaders(
        ecg_npz_path=ecg_path,
        xray_root_dir=xray_dir,
        num_samples=1000,
        batch_size=32,
        random_seed=42,
    )
    fusion_model = MultimodalTransformerFusion(
        ecg_latent_dim=256,
        vision_latent_dim=256,
        tabular_dim=10,
        embed_dim=256,
        num_risk_classes=3,
    ).to(device)

    fusion_ckpt_path = CHECKPOINTS_DIR / "fusion_transformer_weights.pt"
    if fusion_ckpt_path.exists():
        ckpt = torch.load(str(fusion_ckpt_path), map_location=device, weights_only=False)
        fusion_model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded Fusion weights from {fusion_ckpt_path.name}")

    e2e_model = EndToEndMultimodalModel(
        ecg_model=ecg_model,
        xray_model=xray_model,
        fusion_model=fusion_model,
    ).to(device)

    fusion_loss, fusion_metrics = evaluator.evaluate(
        model=e2e_model,
        data_loader=fusion_test_loader,
        criterion=nn.CrossEntropyLoss(),
        class_names=RISK_CLASSES,
        is_multimodal=True,
    )
    results["multimodal_risk"] = fusion_metrics

    # ==========================================
    # Print Summary Tables
    # ==========================================
    logger.info("=" * 70)
    logger.info("BENCHMARK RESULTS SUMMARY:")
    logger.info("-" * 70)

    def format_row(name: str, m: Dict) -> Dict:
        per_class = m.get("per_class", {})
        spec_list = [v["specificity"] for v in per_class.values()] if per_class else []
        macro_spec = np.mean(spec_list) if spec_list else 0.0
        auroc_val = m.get("auroc")
        return {
            "Model Pipeline": name,
            "Accuracy": f"{m['accuracy']*100:.2f}%",
            "Bal. Accuracy": f"{m['balanced_accuracy']*100:.2f}%",
            "Sensitivity (Recall)": f"{m['macro_recall']*100:.2f}%",
            "Specificity": f"{macro_spec*100:.2f}%",
            "Macro-F1": f"{m['macro_f1']:.4f}",
            "AUROC": f"{auroc_val:.4f}" if auroc_val is not None else "N/A",
        }

    summary_rows = [
        format_row("1D ECG Residual CNN (AAMI 5-Class)", ecg_metrics),
        format_row("DenseNet-121 Chest X-Ray (5-Class)", xray_metrics),
        format_row("Cross-Attention Multimodal Transformer (3-Tier Risk)", fusion_metrics),
    ]

    df_summary = pd.DataFrame(summary_rows)
    print("\n" + df_summary.to_string(index=False) + "\n")

    # Save to experiments directory
    exp_dir = PROJECT_ROOT / "experiments" / "benchmarks"
    exp_dir.mkdir(parents=True, exist_ok=True)
    report_file = exp_dir / "evaluation_report.json"
    with open(report_file, "w") as f:
        # Convert confusion matrices to lists for json
        serializable_results = {}
        for k, v in results.items():
            serializable_results[k] = {
                mk: mv.tolist() if isinstance(mv, np.ndarray) else mv
                for mk, mv in v.items()
            }
        json.dump(serializable_results, f, indent=2)

    logger.info(f"Full benchmark JSON saved to: {report_file.resolve()}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedAgent Multi-Modal Evaluation Benchmark")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Target device")
    args = parser.parse_args()

    set_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    evaluate_all_checkpoints(device=dev)
