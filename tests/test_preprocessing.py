import sys
import unittest
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from utils import preprocess_face


class PreprocessingTests(unittest.TestCase):
    def test_resize_produces_expected_model_shape(self):
        face = np.arange(80 * 60, dtype=np.uint8).reshape(80, 60)
        result = preprocess_face(face)
        self.assertEqual(result.shape, (1, 48, 48, 1))
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)

    def test_normalized_input_remains_normalized(self):
        face = np.full((48, 48), 0.5, dtype=np.float32)
        result = preprocess_face(face)
        self.assertAlmostEqual(float(result.mean()), 0.5, places=4)

    def test_non_grayscale_input_is_rejected(self):
        face = np.zeros((48, 48, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            preprocess_face(face)

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            preprocess_face(np.empty((0, 0), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
