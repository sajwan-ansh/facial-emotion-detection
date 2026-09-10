from collections import deque

import numpy as np


class PredictionSmoother:
    def __init__(self, num_classes, alpha=0.35, window_size=7):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.alpha = alpha
        self.history = deque(maxlen=window_size)
        self.state = np.zeros(num_classes, dtype=np.float32)

    def update(self, probabilities):
        probabilities = np.asarray(probabilities, dtype=np.float32)
        if probabilities.ndim != 1 or not np.isfinite(probabilities).all() or probabilities.sum() <= 0:
            raise ValueError("probabilities must be a finite 1-D vector with positive sum")
        probabilities /= probabilities.sum()
        self.history.append(probabilities)
        recent = np.mean(np.stack(self.history), axis=0)
        self.state = self.alpha * recent + (1 - self.alpha) * self.state
        self.state /= self.state.sum()
        return self.state.copy()

    def reset(self):
        self.history.clear()
        self.state.fill(0)
