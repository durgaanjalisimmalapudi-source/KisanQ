"""
Quantum VQC Training and Feature Reduction Pipeline for KisanQ.
Connects:
CNN Embeddings (1280D) -> PCA (4D) -> 4-Qubit VQC -> Pauli-Z Expectation Values -> Decision Softmax.
Trains variational rotation parameters and evaluates on held-out test data.
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from sklearn.metrics import classification_report, confusion_matrix
import torch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from preprocessing.dataset_loader import DatasetManager
from ai_model.mobilenet_classifier import LeafMobileNetV2, get_model
from quantum_ml.pca_reducer import QuantumPCAReducer
from quantum_ml.vqc_circuit import FourQubitVQC
from quantum_ml.hybrid_model import HybridCNN_VQC


def extract_all_features(model: LeafMobileNetV2, loader, device: str = "cpu") -> Tuple[np.ndarray, np.ndarray]:
    """Extracts 1280-D CNN features and labels from a PyTorch DataLoader."""
    model.eval()
    features_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            feats = model.extract_features(images)
            features_list.append(feats.cpu().numpy())
            labels_list.append(labels.numpy())
    return np.vstack(features_list), np.concatenate(labels_list)


def train_quantum_hybrid_model(
    max_iter: int = 40,
    device: str = "cpu",
    samples_per_class: int = 200
) -> Dict[str, Any]:
    """Train PCA, 4-Qubit VQC parameters, and classical decision head."""
    print(f"\n=======================================================")
    print(f"   STARTING 4-QUBIT HYBRID QUANTUM VQC TRAINING")
    print(f"=======================================================\n")

    # 1. Dataset setup
    dm = DatasetManager(target_classes=config.PRIMARY_CLASSES)
    dm.scan_and_clean_dataset(max_samples_per_class=samples_per_class)
    train_loader, val_loader, test_loader = dm.get_dataloaders(batch_size=32)

    # 2. Load trained or pretrained MobileNetV2
    weights_path = config.SAVED_MODELS_DIR / "mobilenet_v2_leaf.pth"
    cnn_model = get_model(num_classes=len(config.PRIMARY_CLASSES), pretrained=True, weights_path=weights_path, device=device)

    # 3. Extract 1280-D CNN embeddings
    print("[QuantumPipeline] Extracting high-dimensional features from CNN...")
    train_feats, train_labels = extract_all_features(cnn_model, train_loader, device=device)
    val_feats, val_labels = extract_all_features(cnn_model, val_loader, device=device)
    test_feats, test_labels = extract_all_features(cnn_model, test_loader, device=device)

    # 4. PCA Feature Reduction (1280 -> 4)
    pca_reducer = QuantumPCAReducer(n_components=config.NUM_PCA_FEATURES)
    train_4d = pca_reducer.fit_transform(train_feats)
    val_4d = pca_reducer.transform(val_feats)
    test_4d = pca_reducer.transform(test_feats)
    pca_reducer.save()

    # 5. Initialize 4-Qubit VQC and Hybrid Model
    vqc = FourQubitVQC(num_qubits=config.NUM_QUBITS, num_layers=1)
    num_classes = len(config.PRIMARY_CLASSES)
    num_vqc_weights = vqc.num_weights  # 8
    num_decision_weights = num_classes * config.NUM_QUBITS  # 3 * 4 = 12
    num_bias = num_classes  # 3
    total_trainable = num_vqc_weights + num_decision_weights + num_bias

    print(f"[QuantumPipeline] 4-Qubit VQC created with {vqc.num_qubits} qubits, {num_vqc_weights} variational angles.")
    print(f"[QuantumPipeline] Total trainable parameters in hybrid decision: {total_trainable}")

    # Subsample training set if large to speed up quantum simulator optimization
    train_sub_n = min(150, len(train_4d))
    np.random.seed(config.RANDOM_SEED)
    sub_indices = np.random.choice(len(train_4d), size=train_sub_n, replace=False)
    x_train_sub = train_4d[sub_indices]
    y_train_sub = train_labels[sub_indices]

    def unpack_params(p: np.ndarray):
        w_vqc = p[:num_vqc_weights]
        w_dec = p[num_vqc_weights:num_vqc_weights + num_decision_weights].reshape(num_classes, config.NUM_QUBITS)
        b_dec = p[num_vqc_weights + num_decision_weights:]
        return w_vqc, w_dec, b_dec

    loss_history = []

    def loss_func(p: np.ndarray) -> float:
        w_vqc, w_dec, b_dec = unpack_params(p)
        exp_vals = vqc.compute_expectation_values(x_train_sub, w_vqc)  # (N, 4)
        logits = np.dot(exp_vals, w_dec.T) + b_dec  # (N, num_classes)
        # Softmax & Cross-entropy
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_l = np.exp(logits - max_logits)
        probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
        
        # Cross entropy loss with L2 regularization
        epsilon = 1e-12
        correct_logprobs = -np.log(probs[np.arange(len(y_train_sub)), y_train_sub] + epsilon)
        loss = np.mean(correct_logprobs) + 1e-4 * np.sum(p ** 2)
        loss_history.append(float(loss))
        return loss

    # Initialize weights
    np.random.seed(config.RANDOM_SEED)
    init_params = np.random.randn(total_trainable) * 0.1

    print(f"[QuantumPipeline] Optimizing Variational Quantum Parameters via L-BFGS-B...")
    start_vqc_train = time.time()
    res = minimize(loss_func, init_params, method="L-BFGS-B", options={"maxiter": max_iter, "disp": True})
    vqc_train_time = time.time() - start_vqc_train
    print(f"[QuantumPipeline] Optimization finished in {vqc_train_time:.2f}s. Final Train Loss: {res.fun:.4f}")

    opt_w_vqc, opt_w_dec, opt_b_dec = unpack_params(res.x)

    # 6. Evaluation on Test Set
    t0 = time.perf_counter()
    test_exp_vals = vqc.compute_expectation_values(test_4d, opt_w_vqc)
    test_logits = np.dot(test_exp_vals, opt_w_dec.T) + opt_b_dec
    t1 = time.perf_counter()
    quantum_latency_ms = (t1 - t0) / len(test_4d) * 1000

    test_preds = np.argmax(test_logits, axis=1)
    report = classification_report(test_labels, test_preds, target_names=config.PRIMARY_CLASSES, output_dict=True, zero_division=0)
    conf_mat = confusion_matrix(test_labels, test_preds)

    # 7. Save Hybrid Model Checkpoint
    hybrid = HybridCNN_VQC(cnn_model=cnn_model, pca_reducer=pca_reducer, classes=config.PRIMARY_CLASSES, device=device)
    hybrid.vqc_weights = opt_w_vqc
    hybrid.decision_weights = opt_w_dec
    hybrid.decision_bias = opt_b_dec
    hybrid.is_trained = True
    hybrid.save_quantum_checkpoint()

    # 8. Generate Visualizations
    # Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, color="#8b5cf6", linewidth=2, label="VQC Cross-Entropy Loss")
    plt.title("4-Qubit Variational Quantum Circuit Training Loss", fontsize=12, fontweight="bold")
    plt.xlabel("Optimization Iteration")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    loss_curve_path = config.REPORTS_DIR / "quantum_training_loss.png"
    plt.savefig(loss_curve_path, dpi=200)
    plt.close()

    # Confusion Matrix
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        conf_mat, annot=True, fmt="d", cmap="Purples",
        xticklabels=config.PRIMARY_CLASSES, yticklabels=config.PRIMARY_CLASSES
    )
    plt.title("Hybrid CNN-VQC Test Confusion Matrix", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Disease")
    plt.ylabel("Ground Truth")
    plt.tight_layout()
    conf_mat_path = config.REPORTS_DIR / "quantum_confusion_matrix.png"
    plt.savefig(conf_mat_path, dpi=200)
    plt.close()

    # Save Quantum Evaluation JSON
    results = {
        "model_name": "Hybrid CNN-VQC (4-Qubit Variational Quantum Circuit)",
        "num_qubits": config.NUM_QUBITS,
        "num_variational_params": num_vqc_weights,
        "total_hybrid_params": total_trainable,
        "test_accuracy": round(float(report["accuracy"]) * 100, 2),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "avg_latency_ms": round(quantum_latency_ms, 2),
        "training_time_s": round(vqc_train_time, 2),
        "classification_report": report,
        "confusion_matrix": conf_mat.tolist(),
        "circuit_ascii": vqc.get_circuit_ascii()
    }

    report_json_path = config.REPORTS_DIR / "quantum_evaluation.json"
    with open(report_json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Quantum Evaluation Summary]")
    print(f"  Test Accuracy  : {results['test_accuracy']}%")
    print(f"  Macro F1-Score : {results['macro_f1']}")
    print(f"  Latency/Image  : {results['avg_latency_ms']} ms")
    print(f"  Saved Reports  : {report_json_path}, {loss_curve_path}, {conf_mat_path}")
    return results


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    train_quantum_hybrid_model(max_iter=30, device=device, samples_per_class=200)
