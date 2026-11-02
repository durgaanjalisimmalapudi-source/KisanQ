"""
Dataset Loader and Preprocessing Module for KisanQ.
Supports Multi-Crop Datasets: Mango (Kaggle aryashah2k) & Hibiscus (Kaggle sayooj98).
"""

import os
import sys
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


class LeafDataset(Dataset):
    """PyTorch Dataset for plant leaf images."""
    def __init__(self, image_paths: List[Path], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", config.IMAGE_SIZE, color=(0, 0, 0))

        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


class DatasetManager:
    """Manages multi-crop dataset discovery, validation, cleaning, and loader generation."""

    def __init__(self, crop: str = "mango", target_classes: Optional[List[str]] = None):
        self.crop = crop.lower()
        if self.crop == "hibiscus":
            self.dataset_dir = config.HIBISCUS_DATA_DIR
            self.target_classes = target_classes or config.HIBISCUS_CLASSES
        else:
            self.dataset_dir = config.MANGO_DATA_DIR
            self.target_classes = target_classes or config.PRIMARY_CLASSES

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.target_classes)}
        self.idx_to_class = {i: cls_name for i, cls_name in enumerate(self.target_classes)}

        self.train_paths: List[Path] = []
        self.train_labels: List[int] = []
        self.val_paths: List[Path] = []
        self.val_labels: List[int] = []
        self.test_paths: List[Path] = []
        self.test_labels: List[int] = []

    def check_dataset_exists(self) -> bool:
        """Check if dataset directory exists and is populated."""
        return bool(self.dataset_dir and self.dataset_dir.exists())

    def scan_and_clean_dataset(self, max_samples_per_class: int = 300) -> Dict[str, Dict]:
        """
        Scans class directories, validates images, deduplicates, and splits.
        Handles both single-folder structures (Mango) and train/test subfolders (Hibiscus).
        """
        stats = {}
        all_samples: Dict[str, List[Path]] = {}
        seen_hashes = set()

        print(f"\n[DatasetManager] Scanning '{self.crop}' dataset in: {self.dataset_dir}")
        print(f"[DatasetManager] Target Classes: {self.target_classes}")

        if self.crop == "hibiscus":
            # Hibiscus has train/fresh, train/diseased, test/fresh, test/diseased
            class_folder_map = {
                "Healthy": ["train/fresh", "test/fresh", "fresh"],
                "Leaf Spot / Blight": ["train/diseased", "test/diseased", "diseased"]
            }
            for cls_name in self.target_classes:
                valid_images: List[Path] = []
                subfolders = class_folder_map.get(cls_name, [cls_name])
                for sf in subfolders:
                    sf_path = self.dataset_dir / sf
                    if not sf_path.exists():
                        continue
                    for img_path in sorted(sf_path.glob("*")):
                        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                            continue
                        try:
                            with open(img_path, "rb") as f:
                                file_hash = hashlib.md5(f.read()).hexdigest()
                            if file_hash in seen_hashes:
                                continue
                            seen_hashes.add(file_hash)
                            valid_images.append(img_path)
                        except Exception:
                            continue

                all_samples[cls_name] = valid_images
                stats[cls_name] = {"valid": len(valid_images)}
                print(f"  - {cls_name:20s}: {len(valid_images)} valid images")

        else:
            # Mango dataset
            for cls_name in self.target_classes:
                cls_dir = self.dataset_dir / cls_name
                if not cls_dir.exists() or not cls_dir.is_dir():
                    continue

                valid_images: List[Path] = []
                for img_path in sorted(cls_dir.glob("*")):
                    if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                        continue
                    try:
                        with open(img_path, "rb") as f:
                            file_hash = hashlib.md5(f.read()).hexdigest()
                        if file_hash in seen_hashes:
                            continue
                        seen_hashes.add(file_hash)
                        valid_images.append(img_path)
                    except Exception:
                        continue

                if max_samples_per_class and len(valid_images) > max_samples_per_class:
                    np.random.seed(config.RANDOM_SEED)
                    valid_images = list(np.random.choice(valid_images, size=max_samples_per_class, replace=False))

                all_samples[cls_name] = valid_images
                stats[cls_name] = {"valid": len(valid_images)}
                print(f"  - {cls_name:20s}: {len(valid_images)} valid images")

        # Stratified Splitting: 70% Train, 15% Val, 15% Test
        paths_list = []
        labels_list = []
        for cls_name, paths in all_samples.items():
            if not paths:
                continue
            cls_idx = self.class_to_idx[cls_name]
            paths_list.extend(paths)
            labels_list.extend([cls_idx] * len(paths))

        paths_np = np.array(paths_list)
        labels_np = np.array(labels_list)

        train_p, temp_p, train_l, temp_l = train_test_split(
            paths_np, labels_np, test_size=0.30, random_state=config.RANDOM_SEED, stratify=labels_np
        )
        val_p, test_p, val_l, test_l = train_test_split(
            temp_p, temp_l, test_size=0.50, random_state=config.RANDOM_SEED, stratify=temp_l
        )

        self.train_paths = list(train_p)
        self.train_labels = list(train_l)
        self.val_paths = list(val_p)
        self.val_labels = list(val_l)
        self.test_paths = list(test_p)
        self.test_labels = list(test_l)

        print(f"[DatasetManager] Split for '{self.crop}': Train={len(self.train_paths)}, Val={len(self.val_paths)}, Test={len(self.test_paths)}")
        return stats

    def get_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        train_transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.RandomRotation(degrees=25),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
        ])

        eval_transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
        ])
        return train_transform, eval_transform

    def get_dataloaders(self, batch_size: int = 32, num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
        if not self.train_paths:
            self.scan_and_clean_dataset()

        train_transform, eval_transform = self.get_transforms()
        train_ds = LeafDataset(self.train_paths, self.train_labels, transform=train_transform)
        val_ds = LeafDataset(self.val_paths, self.val_labels, transform=eval_transform)
        test_ds = LeafDataset(self.test_paths, self.test_labels, transform=eval_transform)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, val_loader, test_loader


def preprocess_single_image(image_input, eval_transform=None) -> torch.Tensor:
    """Preprocesses a single image input for ResNet18 and returns tensor (1, 3, 224, 224)."""
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 3 and image_input.shape[2] == 3:
            img = Image.fromarray(image_input[..., ::-1])
        else:
            img = Image.fromarray(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        raise ValueError(f"Unsupported image type: {type(image_input)}")

    if eval_transform is None:
        eval_transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
        ])

    return eval_transform(img).unsqueeze(0)


if __name__ == "__main__":
    print("Testing Mango...")
    dm_mango = DatasetManager(crop="mango")
    dm_mango.scan_and_clean_dataset(max_samples_per_class=100)

    print("\nTesting Hibiscus...")
    dm_hib = DatasetManager(crop="hibiscus")
    dm_hib.scan_and_clean_dataset()
