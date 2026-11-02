"""
PennyLane 4-Qubit Variational Quantum Circuit (VQC) for KisanQ.
Replaces Qiskit Aer simulator with high-performance PennyLane default.qubit simulator.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pennylane as qml

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


class PennyLane4QubitVQC:
    """
    4-Qubit Parameterized Variational Quantum Circuit using PennyLane.
    Encodes 4 classical features via RY rotation gates, creates entanglement with CNOTs,
    and measures Pauli-Z expectation values.
    """
    def __init__(self, num_qubits: int = config.NUM_QUBITS, num_layers: int = 1):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.num_weights = self.num_qubits * 2 * self.num_layers  # 4 * 2 = 8
        
        # Initialize PennyLane statevector simulator device
        self.dev = qml.device("default.qubit", wires=self.num_qubits)
        
        # Define QNode
        @qml.qnode(self.dev, interface="autograd")
        def _circuit(features, weights):
            # 1. Feature Encoding Layer (RY Rotations)
            for i in range(self.num_qubits):
                qml.RY(features[i], wires=i)

            # 2. Entanglement Ring (CNOTs)
            for i in range(self.num_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.num_qubits])

            # 3. Variational Rotation Layer (RY followed by RZ)
            for i in range(self.num_qubits):
                qml.RY(weights[i], wires=i)
                qml.RZ(weights[i + self.num_qubits], wires=i)

            # 4. Pauli-Z Observable Measurements
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        self.qnode = _circuit

    def get_circuit_ascii(self, features: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None) -> str:
        """Draws the circuit in ASCII format using PennyLane drawer."""
        if features is None:
            features = np.zeros(self.num_qubits)
        if weights is None:
            weights = np.zeros(self.num_weights)
        try:
            return str(qml.draw(self.qnode)(features, weights))
        except Exception:
            return "PennyLane 4-Qubit VQC: RY(x) -> CNOT Loop -> RY(θ)/RZ(θ) -> <Z>"

    def compute_expectation_values(self, features: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Computes Pauli-Z expectation values <Z_i> for each qubit.
        features: shape (4,) or (batch_size, 4) in [-pi, pi]
        weights: shape (8,)
        Returns: expectation values of shape (4,) or (batch_size, 4) in [-1.0, 1.0]
        """
        features = np.asarray(features, dtype=float)
        weights = np.asarray(weights, dtype=float)

        if features.ndim == 1:
            exp_vals = self.qnode(features, weights)
            return np.array([float(v) for v in exp_vals])

        batch_size = features.shape[0]
        results = np.zeros((batch_size, self.num_qubits), dtype=float)
        for b in range(batch_size):
            exp_vals = self.qnode(features[b], weights)
            results[b] = [float(v) for v in exp_vals]
        return results


if __name__ == "__main__":
    vqc = PennyLane4QubitVQC()
    x = np.array([0.5, 1.2, -0.8, 0.3])
    w = np.zeros(8)
    exp_vals = vqc.compute_expectation_values(x, w)
    print("PennyLane Expectation Values <Z_i>:", exp_vals)
    print("\nPennyLane ASCII Circuit:\n", vqc.get_circuit_ascii(x, w))
