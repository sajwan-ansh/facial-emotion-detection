"""Temporal smoothing for frame-by-frame emotion probabilities."""

from collections import deque

import numpy as np


class PredictionSmoother:
    """Blend recent predictions to reduce frame-to-frame label flicker."""

    def __init__(self, num_classes, alpha=0.35, window_size=7):
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.num_classes = num_classes
        self.alpha = alpha
        self.history = deque(maxlen=window_size)
        self.state = None

    def update(self, probabilities):
        probabilities = np.asarray(probabilities, dtype=np.float32)
        if (
            probabilities.ndim != 1
            or probabilities.size != self.num_classes
            or not np.isfinite(probabilities).all()
            or probabilities.sum() <= 0
            or np.any(probabilities < 0)
        ):
            raise ValueError(
                "probabilities must be a finite non-negative 1-D vector with "
                "the configured number of classes"
            )

        probabilities = probabilities / probabilities.sum()
        self.history.append(probabilities)
        recent = np.mean(np.stack(self.history), axis=0)

        if self.state is None:
            self.state = recent.copy()
        else:
            self.state = self.alpha * recent + (1 - self.alpha) * self.state
            self.state /= self.state.sum()

        return self.state.copy()

    def reset(self):
        self.history.clear()
        self.state = None
