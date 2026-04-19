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
        real_dir = os.path.join(self.root_dir, "0")
        fake_dir = os.path.join(self.root_dir, "1")

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
        root_dir=os.path.join(dataset_root, "validate"), split="val", transform=transforms_dict["val"]
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
    """Get class distribution in a dataloader without iterating through all batches."""
    # Fast path: if dataset has labels attribute, count directly
    dataset = dataloader.dataset
    if hasattr(dataset, "labels"):
        labels = torch.tensor(dataset.labels)
        real_count = (labels == 0).sum().item()
        fake_count = (labels == 1).sum().item()
        return {"real": real_count, "fake": fake_count, "total": real_count + fake_count}

    # Fallback: iterate through dataloader (slow)
    real_count = 0
    fake_count = 0
    for _, labels in dataloader:
        real_count += (labels == 0).sum().item()
        fake_count += (labels == 1).sum().item()

    return {"real": real_count, "fake": fake_count, "total": real_count + fake_count}


def print_dataset_statistics(
    train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, batch_size: int
) -> None:
    """
    Print detailed statistics about train/val/test splits including class distribution.

    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
        batch_size: Batch size used for loaders
    """
    train_dist = get_class_distribution(train_loader)
    val_dist = get_class_distribution(val_loader)
    test_dist = get_class_distribution(test_loader)

    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)

    print(f"\n{'Split':<15} {'Total':<12} {'Real':<12} {'Fake':<12} {'Real %':<12} {'Fake %':<12}")
    print("-" * 70)

    for name, dist in [("Train", train_dist), ("Validation", val_dist), ("Test", test_dist)]:
        real_pct = 100 * dist["real"] / dist["total"] if dist["total"] > 0 else 0
        fake_pct = 100 * dist["fake"] / dist["total"] if dist["total"] > 0 else 0
        print(
            f"{name:<15} {dist['total']:<12} {dist['real']:<12} {dist['fake']:<12} {real_pct:<12.1f} {fake_pct:<12.1f}"
        )

    print("-" * 70)
    total_real = train_dist["real"] + val_dist["real"] + test_dist["real"]
    total_fake = train_dist["fake"] + val_dist["fake"] + test_dist["fake"]
    total_all = total_real + total_fake
    print(
        f"{'TOTAL':<15} {total_all:<12} {total_real:<12} {total_fake:<12} {100*total_real/total_all:<12.1f} {100*total_fake/total_all:<12.1f}"
    )
    print("=" * 70 + "\n")


def create_dataloaders_from_source(
    data_source_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    train_split: float = 0.6,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders from raw data_source directory with automatic train/val/test splits.

    This function loads images from:
    - data_source_root/ffhq/ (real images)
    - data_source_root/fake/ (AI-generated images including subdirectories)

    Args:
        data_source_root: Root directory containing 'ffhq' and 'fake' subdirectories
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        image_size: Image size for ViT (default 224)
        train_split: Fraction for training (default 0.6 = 60%)
        val_split: Fraction for validation (default 0.15 = 15%)
        test_split: Fraction for testing (default 0.15 = 15%)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Collect all image paths
    all_images = []
    all_labels = []

    # Load real images from FFHQ
    real_dir = os.path.join(data_source_root, "ffhq")
    if os.path.exists(real_dir):
        for img_name in os.listdir(real_dir):
            if img_name.startswith("."):  # Skip hidden files
                continue
            if img_name.endswith((".jpg", ".jpeg", ".png", ".JPG", ".PNG")):
                all_images.append(os.path.join(real_dir, img_name))
                all_labels.append(0)  # 0 = real
        logger.info(f"Loaded {len([l for l in all_labels if l == 0])} real images from FFHQ")

    # Load AI-generated images from fake directory (including subdirectories)
    fake_dir = os.path.join(data_source_root, "fake")
    if os.path.exists(fake_dir):
        for root, dirs, files in os.walk(fake_dir):
            for img_name in files:
                if img_name.startswith("."):  # Skip hidden files
                    continue
                if img_name.endswith((".jpg", ".jpeg", ".png", ".JPG", ".PNG")):
                    all_images.append(os.path.join(root, img_name))
                    all_labels.append(1)  # 1 = fake
        logger.info(f"Loaded {len([l for l in all_labels if l == 1])} fake images")

    # Shuffle and split
    indices = np.arange(len(all_images))
    np.random.shuffle(indices)

    total_samples = len(all_images)
    train_idx = int(total_samples * train_split)
    val_idx = train_idx + int(total_samples * val_split)

    train_indices = indices[:train_idx]
    val_indices = indices[train_idx:val_idx]
    test_indices = indices[val_idx:]

    logger.info(
        f"\nDataset split (train, val, test): {len(train_indices)}, {len(val_indices)}, {len(test_indices)}"
    )

    # Create dataset instances
    transforms_dict = get_data_transforms(image_size)

    train_image_list = [all_images[i] for i in train_indices]
    train_label_list = [all_labels[i] for i in train_indices]
    train_dataset = RawFileDataset(
        image_paths=train_image_list,
        labels=train_label_list,
        transform=transforms_dict["train"],
        image_size=image_size,
    )

    val_image_list = [all_images[i] for i in val_indices]
    val_label_list = [all_labels[i] for i in val_indices]
    val_dataset = RawFileDataset(
        image_paths=val_image_list,
        labels=val_label_list,
        transform=transforms_dict["val"],
        image_size=image_size,
    )

    test_image_list = [all_images[i] for i in test_indices]
    test_label_list = [all_labels[i] for i in test_indices]
    test_dataset = RawFileDataset(
        image_paths=test_image_list,
        labels=test_label_list,
        transform=transforms_dict["test"],
        image_size=image_size,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


class RawFileDataset(Dataset):
    """
    Dataset that loads images from a list of file paths with labels.
    Useful for loading from data_source directories.
    """

    def __init__(
        self, image_paths: List[str], labels: List[int], transform=None, image_size: int = 224
    ):
        """
        Args:
            image_paths: List of full file paths to images
            labels: List of integer labels (0=real, 1=fake)
            transform: torchvision transforms to apply
            image_size: Size of fallback tensor on error (default 224)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        try:
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert("RGB")
            label = self.labels[idx]

            if self.transform:
                image = self.transform(image)

            return image, label
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a black image on error
            return torch.zeros(3, self.image_size, self.image_size), self.labels[idx]


# ============================================================================
# Attention Analysis Helpers
# ============================================================================


def load_image_paths_from_directory(directory: str, limit: int = None) -> List[str]:
    """
    Load all image file paths from a directory.

    Args:
        directory: Path to directory containing images
        limit: Maximum number of paths to return (None = return all)

    Returns:
        List of image file paths
    """
    valid_extensions = (".png", ".jpg", ".jpeg", ".webp", ".PNG", ".JPG", ".JPEG")

    if not os.path.exists(directory):
        logger.warning(f"Directory not found: {directory}")
        return []

    image_paths = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(valid_extensions)
    ]

    if limit is not None:
        image_paths = image_paths[:limit]

    logger.info(f"Loaded {len(image_paths)} images from {directory}")
    return image_paths


def compute_and_print_attention_statistics(
    cams: List[np.ndarray], class_name: str = "Class"
) -> dict:
    """
    Compute and print statistics for a list of attention maps (CAMs).

    Args:
        cams: List of CAM arrays
        class_name: Name of the class for printing

    Returns:
        Dictionary with statistics (mean, std, min, max)
    """
    if not cams:
        logger.warning(f"No CAMs provided for {class_name}")
        return {}

    avg_cam = np.mean(cams, axis=0)
    stats = {
        "mean": float(avg_cam.mean()),
        "std": float(avg_cam.std()),
        "min": float(avg_cam.min()),
        "max": float(avg_cam.max()),
    }

    print(f"\n{class_name} Attention Statistics:")
    print(f"  Mean: {stats['mean']:.4f} | Std: {stats['std']:.4f}")
    print(f"  Min:  {stats['min']:.4f} | Max: {stats['max']:.4f}")

    return stats
