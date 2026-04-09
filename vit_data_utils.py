"""
Data utilities for face classification dataset.
Handles loading, preprocessing, and batching of real vs AI-generated face images.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceClassificationDataset(Dataset):
    """
    Dataset class for real vs AI-generated face classification.

    Expects directory structure:
    dataset_root/
        real/
            image1.jpg
            image2.jpg
            ...
        fake/
            image1.jpg
            image2.jpg
            ...
    """

    def __init__(self, root_dir: str, split: str = "train", transform=None):
        """
        Args:
            root_dir: Root directory containing 'real' and 'fake' subdirectories
            split: 'train', 'val', or 'test'
            transform: torchvision transforms to apply to images
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.images = []
        self.labels = []

        # Load image paths and labels
        self._load_images()

    def _load_images(self):
        """Load image paths from directory structure."""
        real_dir = os.path.join(self.root_dir, "real")
        fake_dir = os.path.join(self.root_dir, "fake")

        # Load real faces (label 0)
        if os.path.exists(real_dir):
            for img_name in os.listdir(real_dir):
                if img_name.endswith((".jpg", ".jpeg", ".png", ".JPG", ".PNG")):
                    self.images.append(os.path.join(real_dir, img_name))
                    self.labels.append(0)

        # Load AI-generated faces (label 1)
        if os.path.exists(fake_dir):
            for img_name in os.listdir(fake_dir):
                if img_name.endswith((".jpg", ".jpeg", ".png", ".JPG", ".PNG")):
                    self.images.append(os.path.join(fake_dir, img_name))
                    self.labels.append(1)

        logger.info(f"Loaded {len(self.images)} images for {self.split} split")
        if len(self.images) == 0:
            logger.warning(f"No images found in {self.root_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        try:
            img_path = self.images[idx]
            image = Image.open(img_path).convert("RGB")
            label = self.labels[idx]

            if self.transform:
                image = self.transform(image)

            return image, label
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a black image on error
            return torch.zeros(3, 224, 224), self.labels[idx]


def get_data_transforms(image_size: int = 224) -> dict:
    """
    Get standard transforms for train/val/test splits.
    Vision Transformers typically use consistent preprocessing across splits.
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )

    val_test_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return {
        "train": train_transform,
        "val": val_test_transform,
        "test": val_test_transform,
    }


def create_dataloaders(
    dataset_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, and test dataloaders.

    Args:
        dataset_root: Root directory containing data splits
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        image_size: Image size for ViT (default 224 for ViT-base)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    transforms_dict = get_data_transforms(image_size)

    train_dataset = FaceClassificationDataset(
        root_dir=os.path.join(dataset_root, "train"),
        split="train",
        transform=transforms_dict["train"],
    )

    val_dataset = FaceClassificationDataset(
        root_dir=os.path.join(dataset_root, "val"), split="val", transform=transforms_dict["val"]
    )

    test_dataset = FaceClassificationDataset(
        root_dir=os.path.join(dataset_root, "test"),
        split="test",
        transform=transforms_dict["test"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def get_class_distribution(dataloader: DataLoader) -> dict:
    """Get class distribution in a dataloader."""
    real_count = 0
    fake_count = 0

    for _, labels in dataloader:
        real_count += (labels == 0).sum().item()
        fake_count += (labels == 1).sum().item()

    return {"real": real_count, "fake": fake_count, "total": real_count + fake_count}
