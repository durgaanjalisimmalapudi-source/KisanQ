"""
PennyLane 4-Qubit VQC Training Pipeline for KisanQ.
Connects:
Leaf Image -> ResNet18 Backbone (512D) -> PCA (4D) -> PennyLane VQC -> Decision Softmax.
Trains variational quantum rotation parameters for both Mango and Hibiscus.
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
from ai_model.resnet_classifier import LeafResNet18, get_resnet_model
from quantum_ml.pca_reducer import QuantumPCAReducer
from quantum_ml.pennylane_vqc import PennyLane4QubitVQC
from quantum_ml.hybrid_model import HybridCNN_VQC


def extract_resnet_features(model: LeafResNet18, loader, device: str = "cpu") -> Tuple[np.ndarray, np.ndarray]:
    """Extracts 512-D ResNet18 features from a PyTorch DataLoader."""
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


def train_pennylane_vqc(
    crop: str = "mango",
    max_iter: int = 12,
    device: str = "cpu",
    samples_per_class: int = 200
) -> Dict[str, Any]:
    """Train PennyLane 4-Qubit VQC on specified crop dataset."""
    crop_info = config.CROP_CATALOG.get(crop, config.CROP_CATALOG["mango"])
    target_classes = crop_info["classes"]
    num_classes = len(target_classes)

    print(f"\n=======================================================")
    print(f"   PENNYLANE 4-QUBIT VQC TRAINING: CROP = {crop.upper()}")
    print(f"=======================================================\n")

    # 1. Dataset
    dm = DatasetManager(crop=crop, target_classes=target_classes)
    dm.scan_and_clean_dataset(max_samples_per_class=samples_per_class)
    train_loader, val_loader, test_loader = dm.get_dataloaders(batch_size=32)

    # 2. ResNet18 Backbone
    weights_path = config.SAVED_MODELS_DIR / f"resnet18_leaf_{crop}.pth"
    resnet = get_resnet_model(num_classes=num_classes, pretrained=True, weights_path=weights_path, device=device)

    # 3. Extract 512-D features
    print(f"[PennyLane-{crop}] Extracting 512-D features from ResNet18...")
    train_feats, train_labels = extract_resnet_features(resnet, train_loader, device=device)
    val_feats, val_labels = extract_resnet_features(resnet, val_loader, device=device)
    test_feats, test_labels = extract_resnet_features(resnet, test_loader, device=device)

    # 4. Fit PCA (512 -> 4)
    pca_reducer = QuantumPCAReducer(n_components=config.NUM_PCA_FEATURES)
    train_4d = pca_reducer.fit_transform(train_feats)
    val_4d = pca_reducer.transform(val_feats)
    test_4d = pca_reducer.transform(test_feats)
    pca_save_path = config.SAVED_MODELS_DIR / f"pca_reducer_resnet_{crop}.pkl"
    pca_reducer.save(pca_save_path)

    # Also save as default if crop is mango
    if crop == "mango":
        pca_reducer.save(config.SAVED_MODELS_DIR / "pca_reducer_resnet.pkl")

    # 5. PennyLane 4-Qubit VQC
    vqc = PennyLane4QubitVQC(num_qubits=config.NUM_QUBITS, num_layers=1)
    num_vqc_weights = vqc.num_weights  # 8
    num_decision_weights = num_classes * config.NUM_QUBITS  # num_classes * 4
    num_bias = num_classes
    total_trainable = num_vqc_weights + num_decision_weights + num_bias

    # Subsample training set for fast optimization
    train_sub_n = min(45, len(train_4d))
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
        exp_vals = vqc.compute_expectation_values(x_train_sub, w_vqc)
        logits = np.dot(exp_vals, w_dec.T) + b_dec
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_l = np.exp(logits - max_logits)
        probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)

        epsilon = 1e-12
        correct_logprobs = -np.log(probs[np.arange(len(y_train_sub)), y_train_sub] + epsilon)
        loss = np.mean(correct_logprobs) + 1e-4 * np.sum(p ** 2)
        loss_history.append(float(loss))
        return loss

    np.random.seed(config.RANDOM_SEED)
    init_params = np.random.randn(total_trainable) * 0.1

    print(f"[PennyLane-{crop}] Optimizing PennyLane VQC Variational Parameters...")
    t0 = time.time()
    res = minimize(loss_func, init_params, method="L-BFGS-B", options={"maxiter": max_iter})
    vqc_train_time = time.time() - t0
    print(f"[PennyLane-{crop}] Optimization finished in {vqc_train_time:.2f}s. Final Loss: {res.fun:.4f}")

    opt_w_vqc, opt_w_dec, opt_b_dec = unpack_params(res.x)

    # 6. Evaluation on Held-Out Test Set
    t0_eval = time.perf_counter()
    test_exp_vals = vqc.compute_expectation_values(test_4d, opt_w_vqc)
    test_logits = np.dot(test_exp_vals, opt_w_dec.T) + opt_b_dec
    t1_eval = time.perf_counter()
    latency_ms = (t1_eval - t0_eval) / len(test_4d) * 1000

    test_preds = np.argmax(test_logits, axis=1)
    report = classification_report(test_labels, test_preds, target_names=target_classes, output_dict=True, zero_division=0)
    conf_mat = confusion_matrix(test_labels, test_preds)

    # 7. Save Hybrid Model Checkpoint
    hybrid = HybridCNN_VQC(resnet_model=resnet, pca_reducer=pca_reducer, crop=crop, classes=target_classes, device=device)
    hybrid.vqc_weights = opt_w_vqc
    hybrid.decision_weights = opt_w_dec
    hybrid.decision_bias = opt_b_dec
    hybrid.is_trained = True
    hybrid.save_quantum_checkpoint(config.SAVED_MODELS_DIR / f"hybrid_pennylane_{crop}.pkl")

    # Also save as default
    if crop == "mango":
        hybrid.save_quantum_checkpoint(config.SAVED_MODELS_DIR / "hybrid_pennylane_default.pkl")

    print(f"\n[PennyLane-{crop} Evaluation]")
    print(f"  Test Accuracy : {report['accuracy']*100:.2f}%")
    print(f"  Macro F1-Score: {report['macro avg']['f1-score']:.4f}")
    print(f"  Simulator     : PennyLane default.qubit")
    print(f"  Latency/Image : {latency_ms:.2f} ms")
    return {"crop": crop, "accuracy": report["accuracy"] * 100, "latency_ms": latency_ms, "report": report}


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    train_pennylane_vqc(crop="mango", max_iter=25, device=device)
    train_pennylane_vqc(crop="hibiscus", max_iter=25, device=device)
