"""
Vision Transformer model wrapper for AI-generated face detection.
Implements ViT-based classifier with training and evaluation utilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, Dict, Optional
import logging
from tqdm import tqdm
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ViTFaceDetector(nn.Module):
    """
    Vision Transformer-based face detection model.
    Classifies faces as real (0) or AI-generated (1).
    """

    def __init__(self, model_name: str = "vit_base_patch16_224", pretrained: bool = True):
        """
        Initialize Vision Transformer model.

        Args:
            model_name: Name of ViT architecture from torchvision
            pretrained: Whether to use ImageNet pretrained weights
        """
        super(ViTFaceDetector, self).__init__()

        try:
            from torchvision.models import vision_transformer

            if model_name == "vit_base_patch16_224":
                self.backbone = vision_transformer.vit_b_16(
                    weights="IMAGENET1K_V1" if pretrained else None
                )
            elif model_name == "vit_large_patch16_224":
                self.backbone = vision_transformer.vit_l_16(
                    weights="IMAGENET1K_V1" if pretrained else None
                )
            else:
                raise ValueError(f"Unknown model: {model_name}")

            # Replace classification head for binary classification
            hidden_dim = self.backbone.heads[0].in_features
            self.backbone.heads = nn.Sequential(
                nn.Linear(hidden_dim, 512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, 2),  # Binary classification
            )

            self.model_name = model_name
            logger.info(f"Initialized {model_name} with pretrained={pretrained}")

        except ImportError:
            logger.error("torchvision not installed. Using simple ViT implementation.")
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ViT."""
        return self.backbone(x)

    def get_attention_maps(self, x: torch.Tensor):
        """
        Extract attention maps from all transformer blocks.
        Returns attention weights for Grad-CAM visualization.
        """
        attention_maps = []
        # Process through patch embedding
        x = self.backbone._process_input(x)
        n, _, c = x.shape

        # Expand class token
        batch_class_token = self.backbone.class_token.expand(n, -1, -1)
        x = torch.cat((batch_class_token, x), dim=1)

        # Apply positional embedding
        x = x + self.backbone.encoder.pos_embedding

        # Pass through transformer blocks and collect attention
        for layer in self.backbone.encoder.layers:
            attention = layer.ln_1(x)
            attention_maps.append(attention)
            x = layer(x)

        return attention_maps


class ViTTrainer:
    """Train and evaluate Vision Transformer face detector."""

    def __init__(
        self,
        model: ViTFaceDetector,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        """
        Initialize trainer.

        Args:
            model: ViTFaceDetector instance
            device: torch device (cuda or cpu)
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization
        """
        self.model = model.to(device)
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

        self.training_history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        logger.info(f"Trainer initialized on device: {device}")

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.

        Returns:
            (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc="Training")
        for images, labels in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix({"loss": loss.item(), "acc": 100.0 * correct / total})

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate model.

        Returns:
            (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validating")
            for images, labels in progress_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

                progress_bar.set_postfix({"loss": loss.item(), "acc": 100.0 * correct / total})

        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        checkpoint_dir: str = "./checkpoints",
    ) -> Dict:
        """
        Full training loop.

        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            num_epochs: Number of epochs
            checkpoint_dir: Directory to save checkpoints

        Returns:
            Training history dictionary
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_val_acc = 0.0
        patience = 10
        patience_counter = 0

        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")

            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.training_history["train_loss"].append(train_loss)
            self.training_history["train_acc"].append(train_acc)

            # Validate
            val_loss, val_acc = self.validate(val_loader)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["val_acc"].append(val_acc)

            # Learning rate scheduler
            self.scheduler.step()

            logger.info(
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
                self.save_checkpoint(checkpoint_path, epoch)
                logger.info(f"Saved best model with val_acc: {val_acc:.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break

        return self.training_history

    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Evaluate model on test set.

        Returns:
            Dictionary with metrics
        """
        self.model.eval()
        correct = 0
        total = 0
        predictions = []
        ground_truth = []

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                _, predicted = outputs.max(1)

                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

                predictions.extend(predicted.cpu().numpy())
                ground_truth.extend(labels.cpu().numpy())

        accuracy = 100.0 * correct / total

        return {
            "accuracy": accuracy,
            "predictions": np.array(predictions),
            "ground_truth": np.array(ground_truth),
        }

    def save_checkpoint(self, path: str, epoch: int):
        """Save model checkpoint."""
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "training_history": self.training_history,
            },
            path,
        )

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        if not os.path.exists(path):
            logger.warning(f"Checkpoint not found: {path}")
            return

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.training_history = checkpoint["training_history"]

        logger.info(f"Loaded checkpoint from {path}")
