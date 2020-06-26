from dataclasses import dataclass

import numpy as np

from .linear_regressor import LinearRegressor
from .normalization import MeanNormalization


@dataclass
class GradientDescentParameters:
    learning_rate: float = 0.01
    max_num_iterations: int = 1_000_000
    convergence_threshold: float = 0.00001
    check_convergence_iteration_step: int = 10_000


class GradientDescent:
    def __init__(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        parameters: GradientDescentParameters,
    ):
        self._train_x = train_x
        self._train_y = train_y
        self._parameters = parameters
        self._iteration_count = 0

        self._init_normalizer()
        self._normalize_x()
        self._init_coefficients()
        self._init_stop_conditions()

    def fit(self):
        while not self._stop():
            self._update_coefficients()
            self._iteration_count += 1

        return self._create_linear_regressor()

    def _init_normalizer(self):
        self._normalize = MeanNormalization.from_data(self._train_x)

    def _normalize_x(self):
        normalized_x = self._normalize(self._train_x)
        self._normalized_x_with_intercept = np.hstack(
            (np.ones((normalized_x.shape[0], 1)), normalized_x)
        )

    def _init_coefficients(self):
        self._coefficients = np.zeros((self._normalized_x_with_intercept.shape[1], 1))

    def _init_stop_conditions(self):
        self._stop_conditions = [
            MaxNumIterationsStopCondition(self._parameters.max_num_iterations),
            ConvergenceStopCondition(
                self._parameters.check_convergence_iteration_step,
                self._parameters.convergence_threshold,
                lambda: self._cost(
                    self._coefficients, self._normalized_x_with_intercept, self._train_y
                ),
            ),
        ]

    def _create_linear_regressor(self):
        return LinearRegressor(self._coefficients, self._normalize)

    def _stop(self):
        return any(
            stop_condition(self._iteration_count)
            for stop_condition in self._stop_conditions
        )

    def _update_coefficients(self):
        coefficient_deltas = self._coefficient_deltas()
        self._coefficients += coefficient_deltas

    def _coefficient_deltas(self):
        predicted_values = np.dot(self._normalized_x_with_intercept, self._coefficients)
        deltas = predicted_values - self._train_y
        gradient_sum_term = self._normalized_x_with_intercept.T.dot(deltas)
        gradient_coefficient = (
            self._parameters.learning_rate / self._normalized_x_with_intercept.shape[0]
        )
        coefficient_deltas = -gradient_coefficient * gradient_sum_term
        return coefficient_deltas

    @staticmethod
    def _cost(coefficients, x, y):
        predictions = np.dot(x, coefficients)
        deltas = predictions - y
        deltas_squared = deltas ** 2
        deltas_squared_sum = np.sum(deltas_squared)
        return deltas_squared_sum / (2 * x.shape[0])


class MaxNumIterationsStopCondition:
    def __init__(self, max_num_iterations):
        self._max_num_iterations = max_num_iterations

    def __call__(self, iteration_count):
        return iteration_count >= self._max_num_iterations


class ConvergenceStopCondition:
    def __init__(
        self,
        check_convergence_iteration_step: int,
        convergence_threshold: float,
        cost_fn,
    ):
        self._check_convergence_iteration_step = check_convergence_iteration_step
        self._convergence_threshold = convergence_threshold
        self._cost_fn = cost_fn
        self._previous_cost = float("inf")

    def __call__(self, iteration_count):
        return self._check_convergence(iteration_count) and self._has_converged(
            self._cost_fn
        )

    def _check_convergence(self, iteration_count):
        return iteration_count % self._check_convergence_iteration_step == 0

    def _has_converged(self, cost_fn):
        current_cost = cost_fn()
        has_converged = self._previous_cost - current_cost < self._convergence_threshold
        self._previous_cost = current_cost
        return has_converged
