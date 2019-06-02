import numpy as np

from linear_regression.gradient_descent import (
    GradientDescent,
    GradientDescentParameters,
)
from linear_regression.linear_regressor import LinearRegressor


def create_linear_regressor(
    train_x: np.ndarray,
    train_y: np.ndarray,
    gradient_descent_parameters: GradientDescentParameters,
) -> LinearRegressor:
    gradient_descent = GradientDescent(train_x, train_y, gradient_descent_parameters)
    return gradient_descent.run()
