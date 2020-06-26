from .gradient_descent import CostFunction, GradientDescentState


class MaxNumIterationsStopCondition:
    def __init__(self, max_num_iterations: int):
        self._max_num_iterations = max_num_iterations

    def __call__(self, state: GradientDescentState):
        return state.iteration_count >= self._max_num_iterations


class ConvergenceStopCondition:
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

    def __call__(self, state: GradientDescentState):
        return self._check_convergence(state.iteration_count) and self._has_converged(
            state.coefficients
        )

    def _check_convergence(self, iteration_count):
        return iteration_count % self._check_convergence_iteration_step == 0

    def _has_converged(self, coefficient):
        current_cost_value = self._cost_function.value(coefficient)
        has_converged = (
            self._previous_cost_value - current_cost_value < self._convergence_threshold
        )
        self._previous_cost_value = current_cost_value

        return has_converged
