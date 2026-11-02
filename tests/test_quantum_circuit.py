"""
Unit Tests for PennyLane 4-Qubit Variational Quantum Circuit (VQC).
"""

import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from quantum_ml.pennylane_vqc import PennyLane4QubitVQC


class TestPennyLaneQuantumCircuit(unittest.TestCase):

    def setUp(self):
        self.vqc = PennyLane4QubitVQC(num_qubits=4, num_layers=1)

    def test_circuit_properties(self):
        """Verify circuit has 4 qubits and expected parameter counts."""
        self.assertEqual(self.vqc.num_qubits, 4)
        self.assertEqual(self.vqc.num_weights, 8)

    def test_expectation_values(self):
        """Verify Pauli-Z expectation values are in [-1.0, 1.0]."""
        test_x = np.array([0.5, -1.2, 2.1, -0.4])
        test_w = np.zeros(8)
        exp_vals = self.vqc.compute_expectation_values(test_x, test_w)
        self.assertEqual(len(exp_vals), 4)
        for val in exp_vals:
            self.assertTrue(-1.0001 <= val <= 1.0001)

    def test_batch_expectation_values(self):
        """Verify batch evaluation returns shape (batch_size, 4)."""
        batch_x = np.array([
            [0.1, 0.2, 0.3, 0.4],
            [-0.5, 0.8, -1.1, 2.0]
        ])
        test_w = np.zeros(8)
        results = self.vqc.compute_expectation_values(batch_x, test_w)
        self.assertEqual(results.shape, (2, 4))


if __name__ == "__main__":
    unittest.main()
