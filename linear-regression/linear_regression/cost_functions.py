import abc

import numpy as np


class CostFunction(abc.ABC):
    def gradient(self, coefficients: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def value(self, coefficients: np.ndarray) -> float:
        raise NotImplementedError()


class MeanSquareError(CostFunction):
    def __init__(self, train_y: np.ndarray, normalized_x_with_intercept: np.ndarray):
        self._train_y = train_y
        self._normalized_x_with_intercept = normalized_x_with_intercept

    def gradient(self, coefficients) -> np.ndarray:
        predicted_values = np.dot(self._normalized_x_with_intercept, coefficients)
        deltas = predicted_values - self._train_y
        gradient_sum_term = self._normalized_x_with_intercept.T.dot(deltas)
        return gradient_sum_term / self._normalized_x_with_intercept.shape[0]

    def value(self, coefficients: np.ndarray) -> float:
        predictions = np.dot(self._normalized_x_with_intercept, coefficients)
        deltas = predictions - self._train_y
        deltas_squared = deltas ** 2
        deltas_squared_sum = np.sum(deltas_squared)
        return deltas_squared_sum / (2 * self._normalized_x_with_intercept.shape[0])
