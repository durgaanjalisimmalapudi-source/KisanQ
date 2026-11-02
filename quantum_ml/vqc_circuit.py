"""
4-Qubit Variational Quantum Circuit (VQC) in Qiskit for KisanQ.
Genuine executable quantum simulation using Qiskit Aer.
"""

import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import numpy as np

# Qiskit 2.x Imports
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, SparsePauliOp

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


class FourQubitVQC:
    """
    4-Qubit Parameterized Variational Quantum Circuit.
    Explicitly separates feature encoding parameters and variational weights.
    """
    def __init__(self, num_qubits: int = config.NUM_QUBITS, num_layers: int = 1):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        
        # 4 Feature encoding parameters (x_0, x_1, x_2, x_3)
        self.feature_params = ParameterVector("x", self.num_qubits)
        
        # Variational parameters: per layer 4 RY + 4 RZ = 8 parameters
        self.num_weights = self.num_qubits * 2 * self.num_layers
        self.weight_params = ParameterVector("θ", self.num_weights)
        
        # Build base template circuit
        self.circuit = self._build_circuit()
        
        # Aer simulator backend
        self.simulator = AerSimulator()
        
        # Observables: Pauli-Z on each qubit (Z_0, Z_1, Z_2, Z_3)
        self.observables = [
            SparsePauliOp.from_list([(f"{'I' * (self.num_qubits - 1 - i)}Z{'I' * i}", 1.0)])
            for i in range(self.num_qubits)
        ]

    def _build_circuit(self) -> QuantumCircuit:
        """Construct the 4-qubit parameterized circuit."""
        qc = QuantumCircuit(self.num_qubits, name="KisanQ_4Qubit_VQC")
        
        # --- Stage 1: Feature Encoding Layer (RY Rotations) ---
        for i in range(self.num_qubits):
            qc.ry(self.feature_params[i], i)
        
        qc.barrier(label="Encoding")

        # --- Stage 2: Variational & Entangling Layers ---
        param_idx = 0
        for layer in range(self.num_layers):
            # Entanglement Ring (CNOT Gates)
            for i in range(self.num_qubits):
                qc.cx(i, (i + 1) % self.num_qubits)
            
            # Trainable Rotation Gates: RY followed by RZ
            for i in range(self.num_qubits):
                qc.ry(self.weight_params[param_idx], i)
                param_idx += 1
                qc.rz(self.weight_params[param_idx], i)
                param_idx += 1
                
            qc.barrier(label=f"Layer_{layer+1}")
            
        return qc

    def get_circuit_ascii(self) -> str:
        """Return ASCII diagram of the quantum circuit."""
        return str(self.circuit.draw(output="text"))

    def compute_expectation_values(self, features: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Computes Pauli-Z expectation values <Z_i> for each qubit.
        features: shape (4,) or (batch_size, 4)
        weights: shape (num_weights,)
        Returns: expectation values of shape (batch_size, 4) in range [-1.0, 1.0]
        """
        single_sample = False
        features = np.asarray(features, dtype=float)
        if features.ndim == 1:
            features = features[np.newaxis, :]
            single_sample = True

        batch_size = features.shape[0]
        results = np.zeros((batch_size, self.num_qubits), dtype=float)

        # Dictionary of weights
        weight_dict = {self.weight_params[i]: float(weights[i]) for i in range(self.num_weights)}

        for b in range(batch_size):
            feat_dict = {self.feature_params[i]: float(features[b, i]) for i in range(self.num_qubits)}
            full_dict = {**feat_dict, **weight_dict}
            
            # Assign parameters and compute statevector
            bound_qc = self.circuit.assign_parameters(full_dict)
            sv = Statevector.from_instruction(bound_qc)
            
            for q_idx in range(self.num_qubits):
                results[b, q_idx] = sv.expectation_value(self.observables[q_idx]).real

        if single_sample:
            return results[0]
        return results

    def run_shots_simulation(self, features: np.ndarray, weights: np.ndarray, shots: int = 1024) -> Dict[str, int]:
        """
        Runs quantum measurement with discrete shots on Qiskit Aer simulator.
        Returns bitstring measurement frequencies (e.g., {'0000': 312, '1010': 201, ...}).
        """
        feat_dict = {self.feature_params[i]: float(features[i]) for i in range(self.num_qubits)}
        weight_dict = {self.weight_params[i]: float(weights[i]) for i in range(self.num_weights)}
        full_dict = {**feat_dict, **weight_dict}

        # Create measurement circuit
        meas_qc = self.circuit.copy()
        meas_qc.measure_all()
        bound_qc = meas_qc.assign_parameters(full_dict)

        result = self.simulator.run(bound_qc, shots=shots).result()
        counts = result.get_counts()
        return counts


if __name__ == "__main__":
    vqc = FourQubitVQC(num_qubits=4, num_layers=1)
    print("4-Qubit Variational Quantum Circuit Architecture:")
    print(vqc.get_circuit_ascii())
    
    test_x = np.array([0.785, -1.57, 2.35, -0.45])  # In [-pi, pi]
    test_w = np.zeros(vqc.num_weights)
    
    exp_vals = vqc.compute_expectation_values(test_x, test_w)
    print(f"\nPauli-Z Expectation Values <Z_i>: {exp_vals}")
    
    counts = vqc.run_shots_simulation(test_x, test_w, shots=500)
    print(f"Aer Shots Simulation (Sample 5 counts): {dict(list(counts.items())[:5])}")
