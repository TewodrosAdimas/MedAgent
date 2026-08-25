"""
Visual Explainability (Grad-CAM) Module for MedAgent.
Generates gradient-weighted class activation maps to provide interpretable visual evidence
and anatomical ROI localization for medical imaging findings.
"""

from typing import Dict, List, Optional, Tuple, Union
import matplotlib.cm as cm
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    Highlights salient radiographic regions supporting a model's clinical prediction.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generates a 2D Grad-CAM heatmap for a given input image tensor.

        Args:
            input_tensor: Image tensor of shape (1, C, H, W)
            target_class: Integer class index to explain (if None, uses predicted class)

        Returns:
            Normalized 2D numpy array heatmap of shape (H, W) in range [0, 1]
        """
        self.model.eval()
        self.model.zero_grad()

        input_tensor = input_tensor.clone().requires_grad_(True)
        logits = self.model(input_tensor)

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations not captured by GradCAM hooks.")

        # Global average pooling of gradients: weights alpha_k
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H_act, W_act)

        # Apply ReLU to retain only positive influences
        cam = F.relu(cam)

        # Upsample to match original image dimensions
        H, W = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)

        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)


def overlay_cam_on_image(
    original_image: Union[np.ndarray, Image.Image, torch.Tensor],
    cam: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    """
    Blends a Grad-CAM heatmap with the original X-ray image.

    Args:
        original_image: Raw image (PIL, ndarray, or tensor)
        cam: 2D heatmap array normalized to [0, 1]
        alpha: Heatmap blend opacity (default 0.5)
        colormap: Matplotlib colormap name

    Returns:
        Blended RGB image array of shape (H, W, 3) with values in [0, 255] uint8
    """
    if isinstance(original_image, torch.Tensor):
        img_np = original_image.squeeze().cpu().numpy()
        if img_np.ndim == 3 and img_np.shape[0] in [1, 3]:
            img_np = np.transpose(img_np, (1, 2, 0))
        # Denormalize ImageNet standard if needed
        if img_np.min() < 0:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)
        img_uint8 = (img_np * 255).astype(np.uint8)
    elif isinstance(original_image, Image.Image):
        img_uint8 = np.array(original_image.convert("RGB"))
    else:
        img_uint8 = np.array(original_image, dtype=np.uint8)

    # Apply colormap to CAM
    import matplotlib.pyplot as plt
    try:
        import matplotlib
        cmap = matplotlib.colormaps[colormap]
    except Exception:
        cmap = cm.get_cmap(colormap)
    heatmap = cmap(cam)[:, :, :3]  # (H, W, 3) float [0, 1]

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)

    # Blend original image and heatmap
    blended = (1.0 - alpha) * img_uint8.astype(np.float32) + alpha * heatmap_uint8.astype(np.float32)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return blended


def identify_anatomical_quadrants(
    cam: np.ndarray,
    threshold: float = 0.6,
) -> List[Dict[str, Union[str, float]]]:
    """
    Analyzes which radiological thoracic regions have the highest saliency activation.
    Provides structured anatomical localization for clinical reasoning agents.

    Regions:
      - Right Upper Lung, Right Lower Lung
      - Left Upper Lung, Left Lower Lung
      - Mediastinum / Cardiac Region
    """
    H, W = cam.shape
    mid_y, mid_x = H // 2, W // 2

    quadrants = {
        "Right Upper Lung": cam[:mid_y, :mid_x],
        "Right Lower Lung": cam[mid_y:, :mid_x],
        "Left Upper Lung": cam[:mid_y, mid_x:],
        "Left Lower Lung": cam[mid_y:, mid_x:],
        "Cardiac / Mediastinal Region": cam[mid_y - H // 4 : mid_y + H // 4, mid_x - W // 4 : mid_x + W // 4],
    }

    findings = []
    for region_name, region_mask in quadrants.items():
        if region_mask.size > 0:
            avg_intensity = float(np.mean(region_mask))
            max_intensity = float(np.max(region_mask))
            if max_intensity >= threshold:
                findings.append({
                    "region": region_name,
                    "max_saliency": round(max_intensity, 3),
                    "mean_saliency": round(avg_intensity, 3),
                })

    findings.sort(key=lambda x: x["max_saliency"], reverse=True)
    return findings
