import abc
import typing

from linear_regression.cost_functions import CostFunction
from linear_regression.gradient_descent_state import GradientDescentState


class StopCondition(abc.ABC):
    def evaluate(self, state: GradientDescentState) -> bool:
        raise NotImplementedError()


class AnyStopCondition(StopCondition):
    def __init__(self, stop_conditions=typing.List[StopCondition]):
        self._stop_conditions = stop_conditions

    def evaluate(self, state: GradientDescentState):
        return any(
            stop_condition.evaluate(state) for stop_condition in self._stop_conditions
        )


class MaxNumIterationsStopCondition(StopCondition):
    def __init__(self, max_num_iterations: int):
        self._max_num_iterations = max_num_iterations

    def evaluate(self, state: GradientDescentState):
        return state.iteration_count >= self._max_num_iterations


class ConvergenceStopCondition(StopCondition):
    def __init__(
        self,
        check_convergence_iteration_step: int,
        convergence_threshold: float,
        cost_function: CostFunction,
    ):
        self._check_convergence_iteration_step = check_convergence_iteration_step
        self._convergence_threshold = convergence_threshold
        self._previous_cost_value = float("inf")
        self._cost_function = cost_function

    def evaluate(self, state: GradientDescentState):
        return self._check_convergence(state.iteration_count) and self._has_converged(
            state.coefficients
        )

    def _check_convergence(self, iteration_count):
        return iteration_count % self._check_convergence_iteration_step == 0

    def _has_converged(self, coefficient):
        current_cost_value = self._cost_function.evaluate(coefficient)
        has_converged = (
            self._previous_cost_value - current_cost_value < self._convergence_threshold
        )
        self._previous_cost_value = current_cost_value

        return has_converged
