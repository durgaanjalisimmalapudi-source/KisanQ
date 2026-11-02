"""
End-to-End Pipeline Integration Test for KisanQ (ResNet18 + PennyLane Edition).
"""

import sys
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from quantum_ml.hybrid_model import HybridCNN_VQC


class TestResNetPennyLanePipeline(unittest.TestCase):

    def test_mango_prediction(self):
        """Verify Mango prediction with ResNet18 + PennyLane."""
        hybrid = HybridCNN_VQC(crop="mango", device="cpu")
        loaded = hybrid.load_quantum_checkpoint()
        self.assertTrue(loaded, "Mango PennyLane checkpoint must exist.")

        sample_path = Path("interface/static/samples/healthy_sample_1.jpg")
        test_img = Image.open(sample_path) if sample_path.exists() else Image.new("RGB", (224, 224), color=(34, 139, 34))

        res = hybrid.predict_image(test_img)
        self.assertEqual(res["crop"], "mango")
        self.assertEqual(res["quantum_simulator"], "PennyLane (default.qubit)")
        self.assertIn(res["predicted_class"], config.CROP_CATALOG["mango"]["classes"])
        self.assertEqual(len(res["quantum_measurements"]), 4)

    def test_hibiscus_prediction(self):
        """Verify Hibiscus prediction with ResNet18 + PennyLane."""
        hybrid = HybridCNN_VQC(crop="hibiscus", device="cpu")
        loaded = hybrid.load_quantum_checkpoint()
        self.assertTrue(loaded, "Hibiscus PennyLane checkpoint must exist.")

        sample_path = Path("interface/static/samples/hibiscus_healthy_sample_1.jpg")
        test_img = Image.open(sample_path) if sample_path.exists() else Image.new("RGB", (224, 224), color=(200, 50, 50))

        res = hybrid.predict_image(test_img)
        self.assertEqual(res["crop"], "hibiscus")
        self.assertEqual(res["quantum_simulator"], "PennyLane (default.qubit)")
        self.assertIn(res["predicted_class"], config.CROP_CATALOG["hibiscus"]["classes"])
        self.assertEqual(len(res["quantum_measurements"]), 4)


if __name__ == "__main__":
    unittest.main()
