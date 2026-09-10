import unittest

import numpy as np

from src.smoothing import PredictionSmoother


class PredictionSmootherTests(unittest.TestCase):
    def test_first_update_preserves_probability_distribution(self):
        smoother = PredictionSmoother(num_classes=3, alpha=0.35, window_size=7)
        result = smoother.update([0.1, 0.7, 0.2])

        np.testing.assert_allclose(result, [0.1, 0.7, 0.2])
        self.assertAlmostEqual(float(result.sum()), 1.0)

    def test_repeated_updates_are_normalized(self):
        smoother = PredictionSmoother(num_classes=3)
        for probabilities in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
            result = smoother.update(probabilities)
            self.assertAlmostEqual(float(result.sum()), 1.0)

    def test_invalid_shape_is_rejected(self):
        smoother = PredictionSmoother(num_classes=3)
        with self.assertRaises(ValueError):
            smoother.update([0.5, 0.5])

    def test_reset_clears_previous_state(self):
        smoother = PredictionSmoother(num_classes=2)
        smoother.update([0.1, 0.9])
        smoother.reset()
        np.testing.assert_allclose(smoother.update([0.8, 0.2]), [0.8, 0.2])


if __name__ == "__main__":
    unittest.main()
