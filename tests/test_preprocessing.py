"""
Unit Tests for KisanQ Preprocessing and Dataset Loading.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from preprocessing.dataset_loader import DatasetManager, preprocess_single_image


class TestPreprocessing(unittest.TestCase):

    def test_single_image_preprocessing(self):
        """Verify image preprocessing outputs tensor of shape (1, 3, 224, 224)."""
        dummy_img = Image.new("RGB", (320, 240), color=(120, 180, 70))
        tensor = preprocess_single_image(dummy_img)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        self.assertIsInstance(tensor, torch.Tensor)

    def test_numpy_array_preprocessing(self):
        """Verify OpenCV BGR numpy array conversion."""
        dummy_arr = np.zeros((300, 300, 3), dtype=np.uint8)
        tensor = preprocess_single_image(dummy_arr)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))

    def test_dataset_manager_detection(self):
        """Verify dataset detection checks cache and local paths."""
        dm = DatasetManager()
        exists = dm.check_dataset_exists()
        self.assertTrue(exists, "Dataset should be detected in cache or local directory")


if __name__ == "__main__":
    unittest.main()
