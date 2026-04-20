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
import matplotlib.pyplot as plt
import logging
import random
from PIL import Image, ImageFilter
from vit_helpers.vit_data_utils import *
import os
from torchvision.transforms import Normalize
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
        # Handle both tensor and tuple outputs (e.g., self-attention returns (output, weights))
        if isinstance(output, tuple):
            self.activations = output[0].detach()
        else:
            self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save gradients."""
        # Handle both tensor and tuple grad_output
        if isinstance(grad_output, tuple):
            self.gradients = grad_output[0].detach()
        else:
            self.gradients = grad_output.detach()

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
        colormap: int = cv2.COLORMAP_TURBO,
        clip_percentile: Tuple[float, float] = (2.0, 98.0),
        gamma: float = 0.9,
    ) -> np.ndarray:
        """
        Overlay CAM heatmap on image.

        Args:
            image: Original image (H, W, 3), values in [0, 255]
            cam: CAM heatmap (H, W), values in [0, 1]
            alpha: Blending factor
            colormap: OpenCV colormap
            clip_percentile: Lower/upper percentiles used to stretch CAM contrast.
            gamma: Gamma adjustment for CAM contrast (<1 brightens, >1 darkens).

        Returns:
            Overlay image (H, W, 3)
        """
        # Robustly normalize CAM before applying color so maps do not saturate into red.
        cam_vis = np.asarray(cam, dtype=np.float32)
        p_low, p_high = clip_percentile
        lo, hi = np.percentile(cam_vis, [p_low, p_high])
        if hi > lo:
            cam_vis = (cam_vis - lo) / (hi - lo)
        cam_vis = np.clip(cam_vis, 0.0, 1.0)
        cam_vis = np.power(cam_vis, gamma)

        # Normalize CAM to 0-255
        cam_scaled = (cam_vis * 255).astype(np.uint8)

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


# ============================================================================
# Utility Functions for Attention Map Comparison
# ============================================================================


def compute_average_cam(
    image_tensors: np.ndarray,
    grad_cam: ViTGradCAM,
    patch_size: int = 16,
) -> np.ndarray:
    """
    Compute average CAM across multiple image tensors.

    Args:
        image_tensors: Array of image tensors (B, 3, H, W)
        grad_cam: ViTGradCAM instance
        patch_size: ViT patch size

    Returns:
        Average CAM (H, W)
    """
    cams = grad_cam.generate_batch_cams(image_tensors, patch_size=patch_size)
    avg_cam = np.mean(cams, axis=0)
    return avg_cam


def visualize_attention_comparison(
    fig_ax,
    real_cam: np.ndarray,
    source_cam: np.ndarray,
    source_name: str,
    image_size: int = 224,
    real_overlay: np.ndarray = None,
    source_overlay: np.ndarray = None,
) -> None:
    axes = fig_ax

    diff_cam = source_cam - real_cam
    diff_cam_resized = cv2.resize(
        diff_cam, (image_size, image_size), interpolation=cv2.INTER_CUBIC
    )
    vmax = np.max(np.abs(diff_cam)) + 1e-10

    # Real panel (prefer overlay, fallback to raw CAM)
    if real_overlay is not None:
        axes[0].imshow(real_overlay)
    else:
        im0 = axes[0].imshow(real_cam, cmap="hot")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].set_title("Average Grad-CAM\nReal Faces", fontsize=12)
    axes[0].axis("off")

    # Source panel (prefer overlay, fallback to raw CAM)
    if source_overlay is not None:
        axes[1].imshow(source_overlay)
    else:
        im1 = axes[1].imshow(source_cam, cmap="hot")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_title(f"Average Grad-CAM\n{source_name} Faces", fontsize=12)
    axes[1].axis("off")

    # Difference panel (ResNet-style signed map)
    im2 = axes[2].imshow(diff_cam_resized, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[2].set_title(f"Grad-CAM Difference\n{source_name} - Real", fontsize=12)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Δ Grad-CAM")


def analyze_gradcam_by_source(
    source_paths,
    source_name,
    real_cam,
    real_overlay,
    grad_cam,
    vit_model,
    inv_normalize,
    device,
    image_size=224,
    max_samples=None,
):
    """
    Consolidated helper function for Grad-CAM analysis across different sources.

    Args:
        source_paths: List of image paths for the source (e.g., GAN, Diffusion)
        source_name: Name of the source (e.g., "GAN", "Diffusion")
        real_cam: Pre-computed average CAM for real faces
        real_overlay: Pre-computed overlay for real faces
        grad_cam: ViTGradCAM instance
        vit_model: ViT model
        inv_normalize: Inverse normalization transform
        device: torch device
        image_size: Image size (default 224)
        max_samples: Maximum number of samples to process (None = use all)
    """
    if max_samples is None:
        max_samples = MAX_SAMPLES_FOR_ANALYSIS

    print(f"\n" + "=" * 70)
    print(f"ANALYZING REAL vs {source_name.upper()}")
    print("=" * 70 + "\n")

    # Sample images
    num_samples = min(max_samples, len(source_paths))
    fake_sample = random.sample(source_paths, num_samples)

    fake_cams = []
    fake_imgs = []

    print(f"Processing {num_samples} {source_name} images...")

    # Compute attention maps
    for i, img_path in enumerate(fake_sample):
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{num_samples}")

        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = (
                get_data_transforms(image_size=image_size)["val"](img).unsqueeze(0).to(device)
            )

            with torch.no_grad():
                vit_model.eval()
                outputs = vit_model(img_tensor)
                pred_class = outputs.argmax(dim=1).item()

            cam = grad_cam.generate_cam(img_tensor, target_class=pred_class, patch_size=16)
            fake_cams.append(cam)

            # Store normalized image
            img_vis = inv_normalize(img_tensor[0].cpu()).detach().numpy().transpose(1, 2, 0)
            img_vis = np.clip(img_vis, 0, 1)
            fake_imgs.append(img_vis)
        except Exception as e:
            logger.warning(f"Error processing {img_path}: {e}")
            continue

    if not fake_cams:
        print(f"Failed to compute {source_name} attention maps")
        return None, None, None

    # Compute averages
    avg_fake_cam = np.mean(fake_cams, axis=0)
    avg_fake_img = np.mean(fake_imgs, axis=0)
    fake_overlay = AttentionVisualization.create_heatmap(
        (avg_fake_img * 255).astype(np.uint8), avg_fake_cam
    )

    # Compute statistics
    print(f"\nStatistics:")
    stats = print_attention_comparison_stats(real_cam, avg_fake_cam, "Real", source_name)

    # Visualize comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    visualize_attention_comparison(
        axes,
        real_cam,
        avg_fake_cam,
        source_name,
        image_size=image_size,
        real_overlay=real_overlay,
        source_overlay=fake_overlay,
    )

    plt.suptitle(
        f"ViT Grad-CAM Pattern Comparison: Real vs {source_name}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    save_path = f"./vit_model_outputs/gradcam_comparison_real_vs_{source_name.lower()}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Saved to {save_path}\n")

    return avg_fake_cam, fake_overlay, stats


def print_attention_comparison_stats(
    real_cam: np.ndarray,
    source_cam: np.ndarray,
    real_name: str = "Real",
    source_name: str = "Source",
) -> dict:
    """
    Print and return statistics comparing two attention maps.

    Args:
        real_cam: CAM for real faces (H, W)
        source_cam: CAM for source faces (H, W)
        real_name: Name of real class
        source_name: Name of source class

    Returns:
        Dictionary with comparison statistics
    """
    stats = {
        "real_mean": float(real_cam.mean()),
        "real_std": float(real_cam.std()),
        "source_mean": float(source_cam.mean()),
        "source_std": float(source_cam.std()),
        "mean_diff": float(source_cam.mean() - real_cam.mean()),
    }

    print(f"\n{real_name} vs {source_name} Attention Comparison:")
    print(f"  {real_name:20} - Mean: {stats['real_mean']:.4f} | Std: {stats['real_std']:.4f}")
    print(
        f"  {source_name:20} - Mean: {stats['source_mean']:.4f} | Std: {stats['source_std']:.4f}"
    )
    print(f"  Mean Difference: {stats['mean_diff']:+.4f}")

    return stats
