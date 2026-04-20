"""
Attention Rollout visualization for Vision Transformer.
Alternative to Grad-CAM that uses attention weights directly —
no gradients needed, more reliable for ViTs.

Implements the algorithm from:
"Quantifying Attention Flow in Transformers" (Abnar & Zuidema, 2020)

Usage:
    from attention_rollout import ViTAttentionRollout, visualize_comparison

    rollout = ViTAttentionRollout(vit_model, device, discard_ratio=0.9)
    cams = rollout.generate_batch_cams(images, patch_size=16)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional
import logging
from PIL import Image, ImageFilter
from torchvision import transforms
import random
import matplotlib.pyplot as plt
import os
from vit_helpers.vit_data_utils import get_data_transforms
from vit_helpers.vit_gradcam import AttentionVisualization

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ViTAttentionRollout:
    """
    Attention Rollout for Vision Transformers.

    How it differs from Grad-CAM:
    - Grad-CAM: uses gradients of the output w.r.t. a target layer to weight activations.
      Tells you which regions most influence the final classification score.
    - Attention Rollout: propagates raw attention weights across all layers mathematically.
      Tells you where the CLS token is "looking" after accounting for all attention hops.

    In practice:
    - Grad-CAM is class-discriminative (different maps per class)
    - Attention Rollout is class-agnostic (same map regardless of predicted class)
    - Rollout tends to produce smoother, more spatially coherent maps
    - Grad-CAM tends to highlight more specific, localized regions
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        discard_ratio: float = 0.9,
    ):
        """
        Initialize Attention Rollout.

        Args:
            model: ViT model (ViTFaceDetector instance)
            device: torch device
            discard_ratio: Fraction of lowest attention weights to zero out (0.0-1.0).
                           Higher = more focused map. Default 0.9 works well for faces.
        """
        self.model = model
        self.device = device
        self.discard_ratio = discard_ratio
        self.attention_weights = []
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Hook every MultiheadAttention module to capture attention weights."""
        count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                hook = module.register_forward_hook(self._save_attention)
                self._hooks.append(hook)
                count += 1
        logger.info(f"Registered hooks on {count} attention layers")

    def _save_attention(self, module, input, output):
        """Extract attention weights manually from Q, K projections."""
        with torch.no_grad():
            x = input[0]  # (B, seq_len, embed_dim)
            B, N, C = x.shape

            # Project to Q, K, V using the module's in_proj_weight
            qkv = F.linear(x, module.in_proj_weight, module.in_proj_bias)
            # Split into Q, K, V
            qkv = qkv.reshape(B, N, 3, module.num_heads, C // module.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, seq, head_dim)
            q, k, v = qkv.unbind(0)  # each: (B, heads, seq, head_dim)

            # Compute scaled dot-product attention scores
            scale = (C // module.num_heads) ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale  # (B, heads, seq, seq)
            attn = attn.softmax(dim=-1)

            # Average over heads → (B, seq, seq)
            attn = attn.mean(dim=1)

        self.attention_weights.append(attn.detach().cpu())

    # def _save_attention(self, module, input, output):
    #     """
    #     Extract attention weights using F.multi_head_attention_forward directly.
    #     We bypass module.__call__ entirely to avoid infinite recursion —
    #     calling module(q,k,v) inside a hook on that same module loops forever.
    #     """
    #     with torch.no_grad():
    #         q, k, v = input[0], input[1], input[2]

    #         _, attn_weights = F.multi_head_attention_forward(
    #             q, k, v,
    #             embed_dim_to_check=module.embed_dim,
    #             num_heads=module.num_heads,
    #             in_proj_weight=module.in_proj_weight,
    #             in_proj_bias=module.in_proj_bias,
    #             bias_k=module.bias_k,
    #             bias_v=module.bias_v,
    #             add_zero_attn=module.add_zero_attn,
    #             dropout_p=0.0,
    #             out_proj_weight=module.out_proj.weight,
    #             out_proj_bias=module.out_proj.bias,
    #             training=False,
    #             need_weights=True,
    #             average_attn_weights=True,  # average over heads → (B, seq, seq)
    #         )
    #     attn_weights = attn_weights.mean(dim=1)
    #     print(f"  num weights captured: {len(self.attention_weights)}")
    #     # print(f"  first weight shape: {self.attention_weights[0].shape}")
    #     print(f"    attn shape: {attn_weights.shape}")
    #     self.attention_weights.append(attn_weights.detach().cpu())

    def remove_hooks(self):
        """Remove all registered hooks. Call when done to free memory."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        logger.info("Removed all hooks")

    def reinitialize_hooks(self):
        """Re-register hooks after they have been removed. Useful for multiple processing runs."""
        self.remove_hooks()  # Clear any existing hooks first
        self._register_hooks()
        logger.info("Re-registered all hooks")

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        patch_size: int = 16,
        target_class: Optional[int] = None,  # unused — rollout is class-agnostic
    ) -> np.ndarray:
        """
        Generate Attention Rollout map for a single image.

        Args:
            input_tensor: Input image tensor (1, 3, H, W)
            patch_size: ViT patch size (default 16)
            target_class: Ignored — included for API compatibility with ViTGradCAM

        Returns:
            Attention rollout heatmap (H, W), values in [0, 1]
        """
        self.attention_weights = []
        self.model.eval()

        _, _, height, width = input_tensor.shape
        num_patches_h = height // patch_size
        num_patches_w = width // patch_size

        with torch.no_grad():
            _ = self.model(input_tensor)
        # print(f"Num weights captured: {len(self.attention_weights)}")
        # for i, w in enumerate(self.attention_weights):
        #     print(f"  Layer {i}: {w.shape}")

        if len(self.attention_weights) == 0:
            raise RuntimeError("No attention weights captured — hooks did not fire.")

        logger.debug(f"Captured {len(self.attention_weights)} attention layers")

        # --- Attention Rollout Algorithm ---
        seq_len = self.attention_weights[0].shape[-1]  # num_patches + 1 (CLS token)
        rollout = torch.eye(seq_len)

        for attn in self.attention_weights:
            attn = attn[0]  # drop batch dim → (seq_len, seq_len)

            # Discard the lowest attention weights to reduce noise
            flat = attn.view(attn.size(0), -1)
            threshold = torch.quantile(flat, self.discard_ratio, dim=-1, keepdim=True)
            attn = torch.where(flat < threshold, torch.zeros_like(flat), flat)
            attn = attn.view(seq_len, seq_len)

            # Re-normalize rows after discarding
            row_sums = attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            attn = attn / row_sums

            # Add residual connection
            attn = attn + torch.eye(seq_len)
            attn = attn / attn.sum(dim=-1, keepdim=True)

            # Accumulate rollout across layers
            rollout = torch.matmul(attn, rollout)

        # Row 0 = CLS token; columns 1: = patch tokens
        mask = rollout[0, 1:]  # (num_patches,)

        mask = mask.reshape(num_patches_h, num_patches_w).numpy()
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

        return mask

    def generate_batch_cams(
        self,
        input_tensor: torch.Tensor,
        patch_size: int = 16,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Attention Rollout maps for a batch of images.

        Args:
            input_tensor: Input batch (B, 3, H, W)
            patch_size: ViT patch size
            target_class: Ignored — included for API compatibility with ViTGradCAM

        Returns:
            Array of rollout maps (B, H, W)
        """
        cams = []
        for i in range(input_tensor.shape[0]):
            # print(f"  Image {i+1}/{input_tensor.shape[0]}")
            cam = self.generate_cam(input_tensor[i : i + 1], patch_size=patch_size)
            cams.append(cam)
        return np.stack(cams, axis=0)


def visualize_comparison(
    images: torch.Tensor,
    labels: torch.Tensor,
    gradcam_maps: np.ndarray,
    rollout_maps: np.ndarray,
    inv_normalize,
    num_images: int = 4,
    alpha: float = 0.5,
    save_path: str = "./results/gradcam_vs_rollout.png",
):
    """
    Side-by-side comparison of Grad-CAM vs Attention Rollout.

    Args:
        images: Image batch tensor (B, 3, H, W)
        labels: Label tensor (B,)
        gradcam_maps: Grad-CAM heatmaps (B, H, W)
        rollout_maps: Attention Rollout heatmaps (B, H, W)
        inv_normalize: Inverse normalization transform
        num_images: How many images to show
        alpha: Heatmap blend strength
        save_path: Where to save the figure
    """
    import matplotlib.pyplot as plt
    import os

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    n = min(num_images, images.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Original", "Grad-CAM", "Attention Rollout"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=13, fontweight="bold")

    for idx in range(n):
        img = inv_normalize(images[idx]).cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        label_name = "Real" if labels[idx].item() == 0 else "AI-Gen"

        def make_overlay(img, cam):
            cam_scaled = (cam * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(cam_scaled, cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            return cv2.addWeighted(img, 1 - alpha, heatmap_rgb, alpha, 0)

        gradcam_overlay = make_overlay(img, gradcam_maps[idx])
        rollout_overlay = make_overlay(img, rollout_maps[idx])

        axes[idx, 0].imshow(img)
        axes[idx, 0].set_ylabel(f"[{label_name}]", fontsize=11)
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(gradcam_overlay)
        axes[idx, 1].axis("off")

        axes[idx, 2].imshow(rollout_overlay)
        axes[idx, 2].axis("off")

    plt.suptitle("Grad-CAM vs Attention Rollout", fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    logger.info(f"Saved comparison to {save_path}")


def compute_attention_rollout_for_paths(
    image_paths,
    rollout,
    device,
    vit_model,
    inv_normalize,
    IMAGE_SIZE=224,
    label_for_batch=None,
):
    """
    Compute attention rollout maps for a list of image paths.

    Args:
        image_paths: List of image file paths
        rollout: ViTAttentionRollout instance
        device: torch device
        vit_model: ViT model
        inv_normalize: Inverse normalization transform
        IMAGE_SIZE: Image size (default 224)
        label_for_batch: Optional class label for all images

    Returns:
        rollout_maps: List of rollout maps (numpy arrays)
        images_vis: List of denormalized images (numpy arrays)
    """
    # Re-register hooks in case they were removed previously
    rollout.reinitialize_hooks()

    rollout_maps = []
    images_vis = []

    for i, img_path in enumerate(image_paths):
        if i % 500 == 0 and i > 0:
            print(f"  {i}/{len(image_paths)}")

        try:
            img = Image.open(img_path).convert("RGB")

            

            img_tensor = (
                get_data_transforms(image_size=IMAGE_SIZE)["val"](img).unsqueeze(0).to(device)
            )

            vit_model.eval()

            # Generate rollout map
            with torch.no_grad():
                cam = rollout.generate_batch_cams(img_tensor, patch_size=16)[0]

            rollout_maps.append(cam)

            # Denormalize image
            img_vis = inv_normalize(img_tensor[0].cpu()).numpy().transpose(1, 2, 0)
            img_vis = np.clip(img_vis, 0, 1)
            images_vis.append(img_vis)

        except Exception as e:
            logger.warning(f"Error processing {img_path}: {e}")
            continue

    return rollout_maps, images_vis

def compute_and_visualize_rollout_comparison(
    image_paths,
    rollout,
    device,
    vit_model,
    inv_normalize,
    avg_real_rollout_normalized,
    real_rollout_overlay,
    method_name="Method",
    num_samples=500,
    IMAGE_SIZE=224,
    save_path="./results/",
    alpha_difference=0.4,
):
    """
    Compute attention rollout for a set of images and create comparison visualization.

    Args:
        image_paths: List of image file paths
        rollout: ViTAttentionRollout instance
        device: torch device
        vit_model: ViT model
        inv_normalize: Inverse normalization transform
        avg_real_rollout_normalized: Pre-computed real rollout map (for comparison)
        real_rollout_overlay: Pre-computed real overlay (for comparison)
        method_name: Name of the method (e.g., "GAN", "Diffusion", "FaceSwap")
        num_samples: Number of samples to process
        IMAGE_SIZE: Image size
        save_path: Directory to save the result image
        alpha_difference: Transparency of difference overlay (0-1, lower = more transparent)

    Returns:
        None (saves visualization to disk)
    """
    # Sample and process images
    num_samples = min(num_samples, len(image_paths))
    sample = random.sample(image_paths, num_samples)

    print(f"\nProcessing {num_samples} {method_name} images...")
    method_rollouts, method_imgs = compute_attention_rollout_for_paths(
        sample, rollout, device, vit_model, inv_normalize, IMAGE_SIZE
    )

    # Average
    avg_method_rollout = np.mean(method_rollouts, axis=0)
    avg_method_img = np.mean(method_imgs, axis=0)

    # Normalize, resize, blur
    cls_attention_method = (avg_method_rollout - avg_method_rollout.min()) / (
        avg_method_rollout.max() - avg_method_rollout.min() + 1e-10
    )
    cls_attention_method_resized = Image.fromarray(
        (cls_attention_method * 255).astype(np.uint8)
    ).resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.BICUBIC)
    cls_attention_method_resized = cls_attention_method_resized.filter(
        ImageFilter.GaussianBlur(radius=2)
    )
    avg_method_rollout_normalized = np.array(cls_attention_method_resized) / 255.0

    

    method_rollout_overlay = AttentionVisualization.create_heatmap(
        (avg_method_img * 255).astype(np.uint8), avg_method_rollout_normalized
    )

    # Compute signed difference (method - real), aligned with ResNet comparison pattern
    diff_rollout_map = avg_method_rollout_normalized - avg_real_rollout_normalized
    diff_rollout_smooth = cv2.resize(
        diff_rollout_map,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_CUBIC,
    )

    # Print statistics
    print(
        f"\nReal faces       - Mean rollout: {avg_real_rollout_normalized.mean():.4f} | Std: {avg_real_rollout_normalized.std():.4f}"
    )
    print(
        f"{method_name} faces  - Mean rollout: {avg_method_rollout_normalized.mean():.4f} | Std: {avg_method_rollout_normalized.std():.4f}"
    )

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].imshow(real_rollout_overlay)
    axes[0].set_title("Average Attention Rollout\nReal Faces", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(method_rollout_overlay)
    axes[1].set_title(f"Average Attention Rollout\n{method_name} Faces", fontsize=12)
    axes[1].axis("off")

    max_abs = np.max(np.abs(diff_rollout_map)) + 1e-10
    im = axes[2].imshow(
        diff_rollout_smooth,
        cmap="RdBu_r",
        vmin=-max_abs,
        vmax=max_abs,
    )
    axes[2].set_title(f"Attention Difference\n{method_name} - Real", fontsize=12)
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], label="Δ Attention Rollout")

    plt.suptitle(
        f"ViT Attention Rollout Pattern Comparison: Real vs {method_name}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    # Ensure save_path directory exists and create filename
    os.makedirs(save_path, exist_ok=True)
    filename = f"vit_rollout_real_vs_{method_name.lower()}.png"
    filepath = os.path.join(save_path, filename)

    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\n✓ Saved to {filepath}")
