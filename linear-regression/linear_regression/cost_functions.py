import abc

import numpy as np


class CostFunction(abc.ABC):
    def gradient(self, coefficients: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def evaluate(self, coefficients: np.ndarray) -> float:
        raise NotImplementedError()


class MeanSquareError(CostFunction):
    def __init__(
        self,
        train_y: np.ndarray,
        normalized_x_with_intercept: np.ndarray,
        num_train_data: int,
    ):
        assert normalized_x_with_intercept.shape[0] == num_train_data
        self._train_y = train_y
        self._normalized_x_with_intercept = normalized_x_with_intercept
        self._num_train_data = num_train_data

    def gradient(self, coefficients) -> np.ndarray:
        predicted_values = np.dot(self._normalized_x_with_intercept, coefficients)
        deltas = predicted_values - self._train_y
        gradient_sum_term = self._normalized_x_with_intercept.T.dot(deltas)
        return gradient_sum_term / self._num_train_data

    def evaluate(self, coefficients: np.ndarray) -> float:
        predictions = np.dot(self._normalized_x_with_intercept, coefficients)
        deltas = predictions - self._train_y
        deltas_squared = deltas ** 2
        deltas_squared_sum = np.sum(deltas_squared)
        return deltas_squared_sum / (2 * self._num_train_data)
