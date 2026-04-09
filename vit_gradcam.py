"""
Grad-CAM visualization for Vision Transformer.
Implements gradient-based class activation mapping for ViT.
Used to understand which image regions influence ViT predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import cv2
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ViTGradCAM:
    """
    Grad-CAM implementation for Vision Transformers.
    Visualizes which patches in the image influence the model's prediction.
    """

    def __init__(self, model: nn.Module, device: torch.device, target_layer: str = None):
        """
        Initialize Grad-CAM.

        Args:
            model: ViT model
            device: torch device
            target_layer: Name of target layer (optional, uses encoder by default)
        """
        self.model = model
        self.device = device
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks."""
        # Find the target layer (encoder layers in ViT)
        for name, module in self.model.named_modules():
            if "encoder" in name and "layer" in name:
                module.register_forward_hook(self._save_activation)
                module.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Hook to save activations."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save gradients."""
        self.gradients = grad_output[0].detach()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        patch_size: int = 16,
    ) -> np.ndarray:
        """
        Generate Class Activation Map.

        Args:
            input_tensor: Input image tensor (B, 3, H, W)
            target_class: Target class (0=real, 1=AI-generated). If None, uses predicted class.
            patch_size: ViT patch size (default 16)

        Returns:
            CAM heatmap (H, W)
        """
        self.model.eval()
        batch_size, _, height, width = input_tensor.shape

        # Forward pass
        with torch.enable_grad():
            input_tensor.requires_grad_(True)
            output = self.model(input_tensor)

            # Use predicted class if target not specified
            if target_class is None:
                target_class = output.argmax(dim=1)[0].item()

            # Compute gradient
            self.model.zero_grad()
            score = output[0, target_class]
            score.backward()

        # Compute CAM
        gradients = self.gradients[0].mean(dim=(0, 1))  # Average over spatial dims
        activations = self.activations[0, 1:, :]  # Exclude class token

        cam = (gradients.unsqueeze(0) * activations).mean(dim=1)
        cam = F.relu(cam)

        # Reshape to spatial dimensions
        num_patches_h = height // patch_size
        num_patches_w = width // patch_size
        cam = cam.reshape(num_patches_h, num_patches_w)

        # Upsample to original image size
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (width, height), interpolation=cv2.INTER_LINEAR)

        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def generate_batch_cams(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        patch_size: int = 16,
    ) -> np.ndarray:
        """
        Generate CAMs for a batch of images.

        Args:
            input_tensor: Input batch (B, 3, H, W)
            target_class: Target class
            patch_size: ViT patch size

        Returns:
            Array of CAMs (B, H, W)
        """
        cams = []
        for i in range(input_tensor.shape[0]):
            cam = self.generate_cam(
                input_tensor[i : i + 1], target_class=target_class, patch_size=patch_size
            )
            cams.append(cam)

        return np.stack(cams, axis=0)


class AttentionVisualization:
    """Visualize attention weights from ViT layers."""

    @staticmethod
    def create_heatmap(
        image: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlay CAM heatmap on image.

        Args:
            image: Original image (H, W, 3), values in [0, 255]
            cam: CAM heatmap (H, W), values in [0, 1]
            alpha: Blending factor
            colormap: OpenCV colormap

        Returns:
            Overlay image (H, W, 3)
        """
        # Normalize CAM to 0-255
        cam_scaled = (cam * 255).astype(np.uint8)

        # Apply colormap
        heatmap = cv2.applyColorMap(cam_scaled, colormap)

        # Blend with original image
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

        return overlay

    @staticmethod
    def compare_real_vs_fake_attention(
        real_image: np.ndarray,
        real_cam: np.ndarray,
        fake_image: np.ndarray,
        fake_cam: np.ndarray,
        alpha: float = 0.4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create side-by-side comparison of attention maps.

        Args:
            real_image: Real face image (H, W, 3)
            real_cam: CAM for real face (H, W)
            fake_image: AI-generated face image (H, W, 3)
            fake_cam: CAM for AI-generated face (H, W)
            alpha: Blending factor

        Returns:
            (real_overlay, fake_overlay)
        """
        real_overlay = AttentionVisualization.create_heatmap(real_image, real_cam, alpha)
        fake_overlay = AttentionVisualization.create_heatmap(fake_image, fake_cam, alpha)

        return real_overlay, fake_overlay

    @staticmethod
    def visualize_patch_importance(
        cam: np.ndarray,
        patch_size: int = 16,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Visualize which patches are most important to ViT decision.

        Args:
            cam: CAM heatmap (H, W)
            patch_size: Size of ViT patches
            threshold: Only highlight patches above this value

        Returns:
            Binary mask of important patches (H, W)
        """
        # Normalize CAM
        cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # Create patch grid
        h, w = cam.shape
        num_patches_h = h // patch_size
        num_patches_w = w // patch_size

        # Reshape to patches
        patches = cam_norm.reshape(num_patches_h, patch_size, num_patches_w, patch_size)
        patches = patches.transpose(0, 2, 1, 3)
        patches_reshaped = patches.reshape(num_patches_h * num_patches_w, patch_size * patch_size)

        # Compute importance per patch
        patch_importance = patches_reshaped.mean(axis=1)

        # Create binary mask
        high_importance_mask = (patch_importance > threshold).astype(np.uint8)

        # Expand to full image
        mask = np.zeros_like(cam)
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                patch_idx = i * num_patches_w + j
                mask[
                    i * patch_size : (i + 1) * patch_size, j * patch_size : (j + 1) * patch_size
                ] = high_importance_mask[patch_idx]

        return mask


def save_attention_analysis(
    model: nn.Module,
    dataloader: DataLoader,
    save_dir: str,
    device: torch.device,
    num_samples: int = 5,
):
    """
    Generate and save Grad-CAM visualizations for real and fake faces.

    Args:
        model: ViT model
        dataloader: DataLoader to sample from
        save_dir: Directory to save visualizations
        device: torch device
        num_samples: Number of samples to visualize
    """
    import os
    from torchvision.transforms import Normalize

    os.makedirs(save_dir, exist_ok=True)

    grad_cam = ViTGradCAM(model, device)
    visualizer = AttentionVisualization()

    # Inverse normalization for visualization
    inv_normalize = Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )

    sample_count = {"real": 0, "fake": 0}

    for images, labels in dataloader:
        if sample_count["real"] >= num_samples and sample_count["fake"] >= num_samples:
            break

        images = images.to(device)

        for i, (img, label) in enumerate(zip(images, labels)):
            class_name = "real" if label.item() == 0 else "fake"

            if sample_count[class_name] >= num_samples:
                continue

            # Generate CAM
            cam = grad_cam.generate_cam(img.unsqueeze(0), target_class=label.item())

            # Denormalize image for visualization
            img_vis = inv_normalize(img).cpu().numpy().transpose(1, 2, 0)
            img_vis = np.clip(img_vis * 255, 0, 255).astype(np.uint8)

            # Create overlay
            overlay = visualizer.create_heatmap(img_vis, cam)

            # Save
            filename = os.path.join(
                save_dir, f"{class_name}_{sample_count[class_name]}_grad_cam.jpg"
            )
            cv2.imwrite(filename, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            sample_count[class_name] += 1
            logger.info(f"Saved {filename}")
