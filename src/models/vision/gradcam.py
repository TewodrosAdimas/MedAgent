"""
Gradient-Weighted Class Activation Mapping (Grad-CAM) for Medical Image Explainability.

Generates visual explanations highlighting thoracic regions (e.g., lower lung infiltrates,
cardiac silhouette enlargement, pleural fluid accumulations) that influence AI predictions.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.cm as cm
from PIL import Image

from src.data.preprocessing.image_transforms import denormalize_image


class GradCAM:
    """
    Grad-CAM engine for Convolutional and DenseNet architectures.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        """
        Args:
            model: PyTorch vision model.
            target_layer: Target convolutional layer for feature maps.
                          If None, automatically selects model.features.
        """
        self.model = model
        self.model.eval()

        if target_layer is None:
            if hasattr(model, "features"):
                # DenseNet features
                self.target_layer = model.features
            else:
                # Find last Conv2d layer
                conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
                if conv_layers:
                    self.target_layer = conv_layers[-1]
                else:
                    raise ValueError("No target convolutional layer could be inferred from model.")
        else:
            self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handles: List = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self._handles.extend([h1, h2])

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generates 2D normalized Grad-CAM heatmap in [0.0, 1.0].

        Args:
            input_tensor: Image tensor of shape (1, 3, H, W) or (3, H, W).
            target_class: Target class index. If None, uses top predicted class.

        Returns:
            2D NumPy array of shape (H, W) with values in [0.0, 1.0].
        """
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)

        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device).requires_grad_(True)

        # Forward pass
        logits, _ = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()

        score = logits[0, target_class]

        # Zero grads and backward pass
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # Compute channel-wise weights via global average pooling of gradients
        # activations: (1, C, H_feat, W_feat), gradients: (1, C, H_feat, W_feat)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of feature maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H_feat, W_feat)

        # Apply ReLU (only positive contributions)
        cam = F.relu(cam)

        # Resize to input spatial resolution (H, W)
        H, W = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)

        # Normalize to [0.0, 1.0]
        cam_np = cam.squeeze().cpu().numpy()
        cam_min, cam_max = np.min(cam_np), np.max(cam_np)
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np.astype(np.float32)

    def generate_overlay(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        alpha: float = 0.5,
        colormap: str = "jet",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates overlaid visualization: original radiograph with semi-transparent heatmap.

        Args:
            image_tensor: Normalized image tensor of shape (3, H, W) or (1, 3, H, W).
            target_class: Target class ID for Grad-CAM.
            alpha: Opacity blending factor between original image and heatmap.
            colormap: Matplotlib colormap name ('jet', 'plasma', 'turbo', etc.).

        Returns:
            Tuple of:
                - overlay_rgb: Blended uint8 array of shape (H, W, 3).
                - heatmap: 2D float array in [0.0, 1.0].
        """
        heatmap = self.generate_heatmap(image_tensor, target_class=target_class)
        base_rgb = denormalize_image(image_tensor)

        # Apply colormap to heatmap
        cmap_func = cm.get_cmap(colormap)
        heatmap_colored = (cmap_func(heatmap)[:, :, :3] * 255.0).astype(np.uint8)

        # Blend base image and heatmap
        blended = (alpha * heatmap_colored + (1.0 - alpha) * base_rgb).astype(np.uint8)
        return blended, heatmap

    def remove_hooks(self):
        """Removes registered PyTorch hooks to avoid memory leaks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
