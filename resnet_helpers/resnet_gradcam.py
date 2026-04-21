import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class ResNetGradCAM:
    """
    Grad-CAM for ResNet-based binary face classifier.
    Assumes labels:
        0 -> real
        1 -> fake
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self.forward_handle = self.target_layer.register_forward_hook(self._save_activations)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate_cam(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        input_tensor: shape (1, 3, H, W)
        returns: CAM heatmap as numpy array (H, W), normalized to [0,1]
        """
        self.model.eval()

        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[:, target_class]
        score.backward()

        gradients = self.gradients[0]          # (C, H, W)
        activations = self.activations[0]      # (C, H, W)

        weights = gradients.mean(dim=(1, 2))   # (C,)

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        cam = cam.cpu().numpy()
        return cam


class GradCAMVisualization:
    @staticmethod
    def create_heatmap(image: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:

        h, w = image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))

        cam_vis = np.asarray(cam_resized, dtype=np.float32)

        p_low, p_high = np.percentile(cam_vis, [2, 98])
        if p_high > p_low:
            cam_vis = (cam_vis - p_low) / (p_high - p_low)

        cam_vis = np.clip(cam_vis, 0.0, 1.0)

        cam_vis = np.power(cam_vis, 0.9)  # gamma

        heatmap = (cam_vis * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        return overlay


def save_resnet_gradcam_samples(model, dataloader, save_dir, device, num_samples=5):
    """
    Saves Grad-CAM overlays for a few real and fake images.
    """
    from torchvision.transforms import Normalize

    os.makedirs(save_dir, exist_ok=True)

    grad_cam = ResNetGradCAM(model, target_layer=model.layer4[-1].conv2)

    inv_normalize = Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )

    counts = {"real": 0, "fake": 0}

    for images, labels in dataloader:
        if counts["real"] >= num_samples and counts["fake"] >= num_samples:
            break

        images = images.to(device)

        for img, label in zip(images, labels):
            class_name = "real" if label.item() == 0 else "fake"

            if counts[class_name] >= num_samples:
                continue

            cam = grad_cam.generate_cam(img.unsqueeze(0), target_class=label.item())

            img_vis = inv_normalize(img).cpu().numpy().transpose(1, 2, 0)
            img_vis = np.clip(img_vis * 255, 0, 255).astype(np.uint8)

            overlay = GradCAMVisualization.create_heatmap(img_vis, cam)

            out_path = os.path.join(save_dir, f"{class_name}_{counts[class_name]}_gradcam.jpg")
            cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            counts[class_name] += 1

def compute_average_cams(
    image_paths,
    model,
    grad_cam,
    eval_transform,
    inv_normalize,
    device
):
    cams = []
    imgs = []

    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        img_tensor = eval_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            pred_class = outputs.argmax(dim=1).item()

        cam = grad_cam.generate_cam(img_tensor, target_class=pred_class)
        cams.append(cam)

        img_vis = inv_normalize(img_tensor[0].cpu()).numpy().transpose(1, 2, 0)
        img_vis = np.clip(img_vis, 0, 1)
        imgs.append(img_vis)

    avg_cam = np.mean(cams, axis=0)
    avg_img = np.mean(imgs, axis=0)

    return avg_cam, avg_img


    
def compare_gradcam_patterns(
    real_sample,
    fake_sample,
    fake_name,
    model,
    eval_transform,
    inv_normalize,
    device,
    output_dir="resnet_gradcam_outputs"
):
    os.makedirs(output_dir, exist_ok=True)

    target_layers = {
        "layer1": model.layer1[-1].conv2,
        "layer2": model.layer2[-1].conv2,
        "layer3": model.layer3[-1].conv2,
        "layer4": model.layer4[-1].conv2,
    }

    for layer_name, layer_module in target_layers.items():
        print(f"\nProcessing {fake_name} with {layer_name}...")

        grad_cam = ResNetGradCAM(model, target_layer=layer_module)

        avg_real_cam, avg_real_img = compute_average_cams(
            real_sample, model, grad_cam, eval_transform, inv_normalize, device
        )
        avg_fake_cam, avg_fake_img = compute_average_cams(
            fake_sample, model, grad_cam, eval_transform, inv_normalize, device
        )




        real_overlay = GradCAMVisualization.create_heatmap(
            (avg_real_img * 255).astype(np.uint8), avg_real_cam
        )

        fake_overlay = GradCAMVisualization.create_heatmap(
            (avg_fake_img * 255).astype(np.uint8), avg_fake_cam
        )

        diff_cam = avg_fake_cam - avg_real_cam
        diff_cam_smooth = cv2.resize(
            diff_cam,
            (224, 224),
            interpolation=cv2.INTER_CUBIC
        )

        print(
            f"{layer_name} | Real mean: {avg_real_cam.mean():.4f}, std: {avg_real_cam.std():.4f}"
        )
        print(
            f"{layer_name} | {fake_name} mean: {avg_fake_cam.mean():.4f}, std: {avg_fake_cam.std():.4f}"
        )

        print(
            f"Mean Difference:{(avg_fake_cam.mean() - avg_real_cam.mean()):.4f}"
        )

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].imshow(real_overlay)
        axes[0].set_title(f"AverageGrad-CAM\nReal Faces\n({layer_name})")
        axes[0].axis("off")

        axes[1].imshow(fake_overlay)
        axes[1].set_title(f"Average Grad-CAM\n{fake_name} Faces\n({layer_name})")
        axes[1].axis("off")

        im = axes[2].imshow(
            diff_cam_smooth,
            cmap="RdBu_r",
            vmin=-np.max(np.abs(diff_cam)),
            vmax=np.max(np.abs(diff_cam))
        )
        axes[2].set_title(f"Grad-CAM Difference\n{fake_name} - Real\n({layer_name})")
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], label="Δ Grad-CAM")

        plt.suptitle(f"ResNet Grad-CAM Pattern Comparison: Real vs {fake_name} ({layer_name})", fontsize=14)
        plt.tight_layout()

        save_path = os.path.join(
            output_dir,
            f"resnet_gradcam_real_vs_{fake_name.lower()}_{layer_name}.png"
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()

        grad_cam.remove_hooks()