import typing
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


@dataclass
class GradientDescentState:
    iteration_count: int
    coefficients: np.ndarray


def init_gradient_descent_state(
    normalized_x_with_intercept: np.ndarray
) -> GradientDescentState:
    coefficients = init_coefficients(normalized_x_with_intercept)

    return GradientDescentState(iteration_count=0, coefficients=coefficients)


def init_coefficients(normalized_x_with_intercept):
    return np.zeros((normalized_x_with_intercept.shape[1], 1))


def init_stop_conditions(parameters, normalized_x_with_intercept, train_y):
    return [
        MaxNumIterationsStopCondition(parameters),
        ConvergenceStopCondition(parameters, normalized_x_with_intercept, train_y),
    ]


def gradient_descent(
    parameters: GradientDescentParameters, train_x: np.ndarray, train_y: np.ndarray
) -> LinearRegressor:

    normalize = MeanNormalization.from_data(train_x)

    normalized_x = normalize(train_x)
    normalized_x_with_intercept = np.hstack(
        (np.ones((normalized_x.shape[0], 1)), normalized_x)
    )

    stop_conditions = init_stop_conditions(
        parameters, normalized_x_with_intercept, train_y
    )

    state = init_gradient_descent_state(normalized_x_with_intercept)

    while not stop(state, stop_conditions):
        cost_function_gradient = compute_cost_function_gradient(
            normalized_x_with_intercept, state.coefficients, train_y
        )
        coefficient_deltas = -parameters.learning_rate * cost_function_gradient
        state.coefficients += coefficient_deltas
        state.iteration_count += 1

    return LinearRegressor(state.coefficients, normalize)


def stop(state, stop_conditions) -> bool:
    return any(stop_condition(state) for stop_condition in stop_conditions)


def compute_cost_function_gradient(
    normalized_x_with_intercept, coefficients, train_y
) -> np.ndarray:
    predicted_values = np.dot(normalized_x_with_intercept, coefficients)
    deltas = predicted_values - train_y
    gradient_sum_term = normalized_x_with_intercept.T.dot(deltas)
    return gradient_sum_term / normalized_x_with_intercept.shape[0]


class MaxNumIterationsStopCondition:
    def __init__(self, parameters: GradientDescentParameters):
        self._max_num_iterations = parameters.max_num_iterations

    def __call__(self, state: GradientDescentState):
        return state.iteration_count >= self._max_num_iterations


class ConvergenceStopCondition:
    def __init__(
        self,
        parameters: GradientDescentParameters,
        normalized_x_with_intercept,
        train_y,
    ):
        self._check_convergence_iteration_step = (
            parameters.check_convergence_iteration_step
        )
        self._convergence_threshold = parameters.convergence_threshold
        self._normalized_x_with_intercept = normalized_x_with_intercept
        self._train_y = train_y
        self._previous_cost_value = float("inf")

    def __call__(self, state: GradientDescentState):
        return self._check_convergence(state.iteration_count) and self._has_converged(
            state.coefficients
        )

    def _check_convergence(self, iteration_count):
        return iteration_count % self._check_convergence_iteration_step == 0

    def _has_converged(self, coefficient):
        current_cost_value = self._cost(coefficient)
        has_converged = (
            self._previous_cost_value - current_cost_value < self._convergence_threshold
        )
        self._previous_cost_value = current_cost_value

        return has_converged

    def _cost(self, coefficients):
        predictions = np.dot(self._normalized_x_with_intercept, coefficients)
        deltas = predictions - self._train_y
        deltas_squared = deltas ** 2
        deltas_squared_sum = np.sum(deltas_squared)
        return deltas_squared_sum / (2 * self._normalized_x_with_intercept.shape[0])

