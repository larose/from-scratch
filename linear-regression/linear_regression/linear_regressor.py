import numpy as np


class LinearRegressor:
    def __init__(self, coefficients, normalize):
        self._coefficients = coefficients
        self._normalize = normalize

    def predict(self, x: np.ndarray) -> np.ndarray:
        normalized_x = self._normalize(x)
        return np.dot(normalized_x, self._coefficients[1:]) + self._coefficients[0]
