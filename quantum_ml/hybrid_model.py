"""
End-to-End Hybrid Quantum-Classical Model for KisanQ.
Pipelined Architecture:
Leaf Image -> ResNet18 Backbone (512D) -> PCA Reduction (4D) ->
PennyLane 4-Qubit VQC -> Quantum Measurement <Z_i> -> Decision Layer -> Prediction.
Supports Multi-Crop Diagnosis: Mango, Hibiscus, and extensible flower/leaf species.
"""

import sys
import time
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from preprocessing.dataset_loader import preprocess_single_image
from ai_model.resnet_classifier import LeafResNet18, get_resnet_model
from quantum_ml.pca_reducer import QuantumPCAReducer
from quantum_ml.pennylane_vqc import PennyLane4QubitVQC


class HybridCNN_VQC:
    """
    Connected Hybrid ResNet18 + PennyLane 4-Qubit VQC classification pipeline.
    """
    def __init__(
        self,
        resnet_model: Optional[LeafResNet18] = None,
        pca_reducer: Optional[QuantumPCAReducer] = None,
        crop: str = "mango",
        classes: Optional[List[str]] = None,
        device: str = "cpu"
    ):
        self.device = device
        self.crop = crop.lower()
        if classes:
            self.classes = classes
        else:
            self.classes = config.CROP_CATALOG.get(self.crop, config.CROP_CATALOG["mango"])["classes"]
        self.num_classes = len(self.classes)

        # 1. ResNet18 Classical Backbone
        self.cnn = resnet_model if resnet_model else get_resnet_model(
            num_classes=self.num_classes, pretrained=True, device=self.device
        )
        self.cnn.eval()

        # 2. PCA Reducer (512-D -> 4-D)
        self.pca = pca_reducer if pca_reducer else self._load_or_create_pca()

        # 3. PennyLane 4-Qubit VQC
        self.vqc = PennyLane4QubitVQC(num_qubits=config.NUM_QUBITS, num_layers=1)

        # 4. Trainable Parameters
        self.vqc_weights = np.zeros(self.vqc.num_weights, dtype=float)
        self.decision_weights = np.zeros((self.num_classes, config.NUM_QUBITS), dtype=float)
        self.decision_bias = np.zeros(self.num_classes, dtype=float)
        self.is_trained = False

    def _load_or_create_pca(self) -> QuantumPCAReducer:
        pca_path = config.SAVED_MODELS_DIR / f"pca_reducer_resnet_{self.crop}.pkl"
        if pca_path.exists():
            return QuantumPCAReducer.load(pca_path)
        # Fallback to general PCA
        general_pca = config.SAVED_MODELS_DIR / "pca_reducer_resnet.pkl"
        if general_pca.exists():
            return QuantumPCAReducer.load(general_pca)
        return QuantumPCAReducer(n_components=config.NUM_PCA_FEATURES)

    def set_crop(self, crop: str):
        """Switch active crop / flower mode (e.g. 'mango' or 'hibiscus')."""
        self.crop = crop.lower()
        crop_info = config.CROP_CATALOG.get(self.crop, config.CROP_CATALOG["mango"])
        self.classes = crop_info["classes"]
        self.num_classes = len(self.classes)
        # Attempt to load crop-specific weights if available
        self.load_quantum_checkpoint(config.SAVED_MODELS_DIR / f"hybrid_pennylane_{self.crop}.pkl")

    def forward_quantum(self, four_features: np.ndarray) -> np.ndarray:
        """Evaluates PennyLane VQC and classical decision layer."""
        exp_vals = self.vqc.compute_expectation_values(four_features, self.vqc_weights)
        if exp_vals.ndim == 1:
            logits = np.dot(self.decision_weights, exp_vals) + self.decision_bias
        else:
            logits = np.dot(exp_vals, self.decision_weights.T) + self.decision_bias
        return logits

    def predict_image(self, image_input: Any, run_shots: bool = True) -> Dict[str, Any]:
        """
        Full end-to-end inference:
        Leaf -> ResNet18 (512D) -> PCA (4D) -> PennyLane 4-Qubit VQC -> <Z_i> -> Decision Softmax.
        """
        start_time = time.perf_counter()

        # Step 1: Preprocess (224x224 RGB, Normalized)
        tensor = preprocess_single_image(image_input).to(self.device)

        # Step 2: ResNet18 Feature Extraction (512D)
        with torch.no_grad():
            resnet_features = self.cnn.extract_features(tensor).cpu().numpy()  # (1, 512)

        # Step 3: PCA Reduction (512D -> 4D in [-pi, pi])
        four_features = self.pca.transform(resnet_features)[0]  # (4,)

        # Step 4: PennyLane 4-Qubit VQC Expectation Values <Z_i>
        exp_vals = self.vqc.compute_expectation_values(four_features, self.vqc_weights)  # (4,)

        # Step 5: Decision Layer & Softmax
        logits = np.dot(self.decision_weights, exp_vals) + self.decision_bias
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)

        pred_idx = int(np.argmax(probabilities))
        pred_class = self.classes[pred_idx]
        confidence = float(probabilities[pred_idx])
        is_uncertain = bool(confidence < config.CONFIDENCE_THRESHOLD)

        inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        circuit_ascii = self.vqc.get_circuit_ascii(four_features, self.vqc_weights)

        return {
            "crop": self.crop,
            "predicted_class": pred_class,
            "predicted_index": pred_idx,
            "confidence": round(confidence * 100, 2),
            "confidence_ratio": round(confidence, 4),
            "is_uncertain": is_uncertain,
            "class_probabilities": {
                cls_name: round(float(prob) * 100, 2)
                for cls_name, prob in zip(self.classes, probabilities)
            },
            "quantum_measurements": [round(float(v), 4) for v in exp_vals],
            "pca_features": [round(float(f), 4) for f in four_features],
            "vqc_weights": [round(float(w), 4) for w in self.vqc_weights],
            "quantum_simulator": "PennyLane (default.qubit)",
            "cnn_backbone": "ResNet18 (Residual Transfer Learning)",
            "inference_time_ms": inference_time_ms,
            "circuit_ascii": circuit_ascii,
            "pipeline_stages": [
                "1. Leaf Image Preprocessing (224x224 RGB)",
                "2. ResNet18 Feature Extraction (512-D Residual Embedding)",
                "3. PCA Dimensionality Reduction (4-D [-π, π])",
                "4. PennyLane 4-Qubit State Preparation (RY Rotations)",
                "5. Quantum Entanglement Loop (CNOT Ring)",
                "6. Parameterized Variational Rotation (Trainable RY & RZ)",
                "7. Pauli-Z Expectation Measurement (<Z_0> ... <Z_3>)",
                "8. Classical Decision Softmax"
            ]
        }

    def save_quantum_checkpoint(self, filepath: Optional[Path] = None) -> Path:
        if filepath is None:
            filepath = config.SAVED_MODELS_DIR / f"hybrid_pennylane_{self.crop}.pkl"
        data = {
            "crop": self.crop,
            "vqc_weights": self.vqc_weights,
            "decision_weights": self.decision_weights,
            "decision_bias": self.decision_bias,
            "classes": self.classes,
            "is_trained": self.is_trained
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"[HybridModel] Saved PennyLane VQC checkpoint to {filepath}")
        return Path(filepath)

    def load_quantum_checkpoint(self, filepath: Optional[Path] = None) -> bool:
        if filepath is None:
            filepath = config.SAVED_MODELS_DIR / f"hybrid_pennylane_{self.crop}.pkl"
        if not Path(filepath).exists():
            return False
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.vqc_weights = data["vqc_weights"]
        self.decision_weights = data["decision_weights"]
        self.decision_bias = data["decision_bias"]
        self.classes = data.get("classes", self.classes)
        self.crop = data.get("crop", self.crop)
        self.is_trained = data.get("is_trained", True)
        print(f"[HybridModel] Loaded PennyLane VQC checkpoint from {filepath}")
        return True


if __name__ == "__main__":
    hybrid = HybridCNN_VQC(crop="mango")
    test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    if not hybrid.pca.is_fitted:
        hybrid.pca.fit(np.random.randn(20, 512))
    res = hybrid.predict_image(test_img)
    print("Test Prediction:", res["predicted_class"], f"({res['confidence']}%)")
    print("Quantum Measurements:", res["quantum_measurements"])
    print("Simulator:", res["quantum_simulator"])
    print("Backbone:", res["cnn_backbone"])
