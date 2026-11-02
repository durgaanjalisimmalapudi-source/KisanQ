"""
ResNet18 Classical CNN Training Pipeline for KisanQ.
Supports transfer learning across Mango and Hibiscus datasets.
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from preprocessing.dataset_loader import DatasetManager
from ai_model.resnet_classifier import LeafResNet18, get_resnet_model


def train_resnet_model(
    crop: str = "mango",
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device: str = "cpu",
    samples_per_class: int = 250
) -> Dict[str, Any]:
    """Train ResNet18 on specified crop/flower dataset."""
    crop_info = config.CROP_CATALOG.get(crop, config.CROP_CATALOG["mango"])
    target_classes = crop_info["classes"]

    print(f"\n=======================================================")
    print(f"   STARTING RESNET18 TRAINING: CROP = {crop.upper()}")
    print(f"=======================================================\n")

    dm = DatasetManager(crop=crop, target_classes=target_classes)
    dm.scan_and_clean_dataset(max_samples_per_class=samples_per_class)
    train_loader, val_loader, test_loader = dm.get_dataloaders(batch_size=batch_size)

    model = get_resnet_model(num_classes=len(target_classes), pretrained=True, device=device)

    # Freeze initial layers for fast transfer learning
    for param in [model.conv1, model.bn1, model.layer1]:
        for p in param.parameters():
            p.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_model_path = config.SAVED_MODELS_DIR / f"resnet18_leaf_{crop}.pth"

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

        epoch_train_loss = running_loss / total
        epoch_train_acc = (correct / total) * 100

        # Val Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels).item()
                val_total += labels.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)

    print(f"[ResNet18] Finished training {crop} in {time.time() - start_time:.2f}s. Saved to {best_model_path}")

    # Evaluate on held-out test split
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    report = classification_report(all_targets, all_preds, target_names=target_classes, output_dict=True, zero_division=0)
    print(f"Test Accuracy: {report['accuracy']*100:.2f}%")
    return {"crop": crop, "accuracy": report["accuracy"] * 100, "report": report}


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training on device: {device}")
    train_resnet_model(crop="mango", epochs=4, device=device, samples_per_class=200)
    train_resnet_model(crop="hibiscus", epochs=4, device=device)
