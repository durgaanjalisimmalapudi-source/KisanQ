"""
Classical CNN Training Pipeline for KisanQ.
Trains MobileNetV2 with transfer learning on mango leaf disease dataset.
Implements early stopping, checkpointing, loss/accuracy curves, confusion matrix, and classification report.
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
from ai_model.mobilenet_classifier import LeafMobileNetV2, get_model


def train_classical_model(
    epochs: int = 8,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device: str = "cpu",
    samples_per_class: int = 250
) -> Dict[str, Any]:
    """Train MobileNetV2 on leaf dataset and evaluate on test set."""
    print(f"\n=======================================================")
    print(f"   STARTING CLASSICAL CNN TRAINING (MobileNetV2)")
    print(f"=======================================================\n")
    
    # 1. Dataset setup
    dm = DatasetManager(target_classes=config.PRIMARY_CLASSES)
    dm.scan_and_clean_dataset(max_samples_per_class=samples_per_class)
    train_loader, val_loader, test_loader = dm.get_dataloaders(batch_size=batch_size)

    # 2. Model setup
    model = get_model(num_classes=len(config.PRIMARY_CLASSES), pretrained=True, device=device)
    
    # Freeze initial early feature layers for transfer learning efficiency
    for param in model.features[:10].parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_model_path = config.SAVED_MODELS_DIR / "mobilenet_v2_leaf.pth"

    start_train_time = time.time()

    for epoch in range(1, epochs + 1):
        # --- Training Phase ---
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

        # --- Validation Phase ---
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
        scheduler.step(epoch_val_loss)

        history["train_loss"].append(round(epoch_train_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 2))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["val_acc"].append(round(epoch_val_acc, 2))

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

        # Model Checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Checkpoint saved to {best_model_path}")

    total_time = time.time() - start_train_time
    print(f"\n[Training Complete] Total time: {total_time:.2f}s. Best Val Loss: {best_val_loss:.4f}")

    # Load best checkpoint for final evaluation
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # --- Test Evaluation ---
    all_preds = []
    all_targets = []
    inference_times = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            t0 = time.perf_counter()
            outputs = model(images)
            t1 = time.perf_counter()
            inference_times.append((t1 - t0) / images.size(0) * 1000)  # ms per image

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    avg_latency = float(np.mean(inference_times))

    report = classification_report(
        all_targets, all_preds, target_names=config.PRIMARY_CLASSES, output_dict=True, zero_division=0
    )
    conf_mat = confusion_matrix(all_targets, all_preds)

    # Plot & Save Learning Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(range(1, epochs + 1), history["train_loss"], label="Train Loss", marker="o", color="#2563eb")
    ax1.plot(range(1, epochs + 1), history["val_loss"], label="Val Loss", marker="s", color="#dc2626")
    ax1.set_title("Classical MobileNetV2 Loss Curve", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(range(1, epochs + 1), history["train_acc"], label="Train Acc (%)", marker="o", color="#16a34a")
    ax2.plot(range(1, epochs + 1), history["val_acc"], label="Val Acc (%)", marker="s", color="#9333ea")
    ax2.set_title("Classical MobileNetV2 Accuracy Curve", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    metrics_plot_path = config.REPORTS_DIR / "classical_training_curves.png"
    plt.savefig(metrics_plot_path, dpi=200)
    plt.close()

    # Plot Confusion Matrix
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        conf_mat, annot=True, fmt="d", cmap="Blues",
        xticklabels=config.PRIMARY_CLASSES, yticklabels=config.PRIMARY_CLASSES
    )
    plt.title("Classical MobileNetV2 Test Confusion Matrix", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Disease")
    plt.ylabel("Ground Truth")
    plt.tight_layout()
    conf_mat_path = config.REPORTS_DIR / "classical_confusion_matrix.png"
    plt.savefig(conf_mat_path, dpi=200)
    plt.close()

    # Save metrics report JSON
    results = {
        "model_name": "MobileNetV2 (Classical Baseline)",
        "epochs": epochs,
        "test_accuracy": round(float(report["accuracy"]) * 100, 2),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "avg_latency_ms": round(avg_latency, 2),
        "classification_report": report,
        "confusion_matrix": conf_mat.tolist(),
        "history": history
    }

    report_json_path = config.REPORTS_DIR / "classical_evaluation.json"
    with open(report_json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Classical Evaluation Summary]")
    print(f"  Test Accuracy  : {results['test_accuracy']}%")
    print(f"  Macro F1-Score : {results['macro_f1']}")
    print(f"  Average Latency: {results['avg_latency_ms']} ms/image")
    print(f"  Saved Reports  : {report_json_path}, {metrics_plot_path}, {conf_mat_path}")
    return results


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    train_classical_model(epochs=6, batch_size=32, device=device, samples_per_class=200)
