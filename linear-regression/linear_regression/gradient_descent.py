import numpy as np

from .cost_functions import CostFunction
from .gradient_descent_state import GradientDescentState
from .stop_conditions import StopCondition


def gradient_descent(
    learning_rate: float,
    cost_funtion: CostFunction,
    stop_condition: StopCondition,
    num_features: int,
):
    state = GradientDescentState(
        iteration_count=0, coefficients=np.zeros((num_features + 1, 1))
    )

    while not stop_condition.evaluate(state):
        gradient = cost_funtion.gradient(state.coefficients)
        coefficient_deltas = -learning_rate * gradient
        state.coefficients += coefficient_deltas
        state.iteration_count += 1

    return state.coefficients


def stop(state, stop_conditions) -> bool:
    return any(stop_condition(state) for stop_condition in stop_conditions)
