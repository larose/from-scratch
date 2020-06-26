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
    parameters: GradientDescentParameters
    train_x: np.ndarray
    train_y: np.ndarray
    iteration_count: int
    normalize: typing.Callable[[np.ndarray], np.ndarray]
    normalized_x_with_intercept: np.ndarray
    coefficients: np.ndarray
    stop_conditions: typing.List[typing.Callable[[int], bool]]


def init_gradient_descent_state(
    parameters: GradientDescentParameters, train_x: np.ndarray, train_y=np.ndarray
):
    normalize = init_normalize(train_x)
    normalized_x_with_intercept = normalize_x(normalize, train_x)
    coefficients = init_coefficients(normalized_x_with_intercept)

    return GradientDescentState(
        parameters=parameters,
        train_x=train_x,
        train_y=train_y,
        iteration_count=0,
        normalize=normalize,
        normalized_x_with_intercept=normalized_x_with_intercept,
        coefficients=coefficients,
        stop_conditions=init_stop_conditions(
            parameters.max_num_iterations,
            parameters.check_convergence_iteration_step,
            parameters.convergence_threshold,
            coefficients,
            normalized_x_with_intercept,
            train_y,
        ),
    )


def init_normalize(train_x: np.ndarray):
    return MeanNormalization.from_data(train_x)


def normalize_x(normalize, train_x):
    normalized_x = normalize(train_x)
    return np.hstack((np.ones((normalized_x.shape[0], 1)), normalized_x))


def init_coefficients(normalized_x_with_intercept):
    return np.zeros((normalized_x_with_intercept.shape[1], 1))


def init_stop_conditions(
    max_num_iterations: int,
    check_convergence_iteration_step: int,
    convergence_threshold: int,
    coefficients,
    normalized_x_with_intercept,
    train_y,
):
    return [
        MaxNumIterationsStopCondition(max_num_iterations),
        ConvergenceStopCondition(
            check_convergence_iteration_step,
            convergence_threshold,
            lambda: cost(coefficients, normalized_x_with_intercept, train_y),
        ),
    ]


def cost(coefficients, x, y):
    predictions = np.dot(x, coefficients)
    deltas = predictions - y
    deltas_squared = deltas ** 2
    deltas_squared_sum = np.sum(deltas_squared)
    return deltas_squared_sum / (2 * x.shape[0])


def gradient_descent(
    parameters: GradientDescentParameters, train_x: np.ndarray, train_y: np.ndarray
):
    state = init_gradient_descent_state(parameters, train_x, train_y)

    while not stop(state.iteration_count, state.stop_conditions):
        cost_function_gradient = compute_cost_function_gradient(
            state.normalized_x_with_intercept, state.coefficients, train_y
        )
        coefficient_deltas = -parameters.learning_rate * cost_function_gradient
        state.coefficients += coefficient_deltas
        state.iteration_count += 1

    return LinearRegressor(state.coefficients, state.normalize)


def stop(iteration_count, stop_conditions):
    return any(stop_condition(iteration_count) for stop_condition in stop_conditions)


def compute_cost_function_gradient(normalized_x_with_intercept, coefficients, train_y):
    predicted_values = np.dot(normalized_x_with_intercept, coefficients)
    deltas = predicted_values - train_y
    gradient_sum_term = normalized_x_with_intercept.T.dot(deltas)
    return gradient_sum_term / normalized_x_with_intercept.shape[0]


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
