import typing
from dataclasses import dataclass

import numpy as np

from .cost_functions import CostFunction


@dataclass
class GradientDescentState:
    iteration_count: int
    coefficients: np.ndarray


def init_gradient_descent_state(normalized_x_with_intercept: np.ndarray):
    coefficients = init_coefficients(normalized_x_with_intercept)

    return GradientDescentState(iteration_count=0, coefficients=coefficients)


def init_coefficients(normalized_x_with_intercept):
    return np.zeros((normalized_x_with_intercept.shape[1], 1))


def gradient_descent(
    learning_rate: float,
    cost_funtion: CostFunction,
    stop_conditions: typing.List[typing.Callable[[GradientDescentState], bool]],
    normalized_x_with_intercept: np.ndarray,
):
    state = init_gradient_descent_state(normalized_x_with_intercept)

    while not stop(state, stop_conditions):
        gradient = cost_funtion.gradient(state.coefficients)
        coefficient_deltas = -learning_rate * gradient
        state.coefficients += coefficient_deltas
        state.iteration_count += 1

    return state.coefficients


def stop(state, stop_conditions) -> bool:
    return any(stop_condition(state) for stop_condition in stop_conditions)
