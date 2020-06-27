import numpy as np

from .normalizers import MeanNormalizer


class LinearRegressor:
    def __init__(self, coefficients, normalizer: MeanNormalizer):
        self._coefficients = coefficients
        self._normalizer = normalizer

    def predict(self, x: np.ndarray) -> np.ndarray:
        normalized_x = self._normalizer.normalize(x)
        return np.dot(normalized_x, self._coefficients[1:]) + self._coefficients[0]
