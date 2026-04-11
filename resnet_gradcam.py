import os
import cv2
import torch
import torch.nn as nn
import numpy as np


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

        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

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
        """
        image: original image as uint8 RGB, shape (H, W, 3)
        cam: heatmap in [0,1], shape (H, W)
        """
        h, w = image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))

        heatmap = (cam_resized * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
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