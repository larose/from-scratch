import csv
import os.path
from dataclasses import dataclass

import numpy as np
import pytest
from sklearn import linear_model

from linear_regression.cost_functions import MeanSquareError
from linear_regression.gradient_descent import gradient_descent
from linear_regression.linear_regressor import LinearRegressor
from linear_regression.normalization import MeanNormalization
from linear_regression.stop_conditions import (
    ConvergenceStopCondition,
    MaxNumIterationsStopCondition,
)

DATASETS_DIRECTORY = "datasets"


@dataclass
class Dataset:
    num_train_data: int
    num_test_data: int
    num_features: int

    # Matrix: num_train_data x num_features
    train_independant_variables: np.ndarray

    # Vector column: num_train_data x 1
    train_dependent_variables: np.array

    # Matrix: num_test_data x num_features
    test_independent_variables: np.array


@pytest.fixture(scope="module", params=["toy", "petrol"])
def dataset(request):
    train_filename = f"{request.param}_train.csv"
    x = []
    y = []
    with open(os.path.join(DATASETS_DIRECTORY, train_filename)) as train_file:
        for row in csv.reader(train_file):
            x.append([float(element) for element in row[:-1]])
            y.append(float(row[-1]))

    test_filename = f"{request.param}_test.csv"
    test = []
    with open(os.path.join(DATASETS_DIRECTORY, test_filename)) as test_file:
        for row in csv.reader(test_file):
            test.append([float(element) for element in row])

    train_independant_variables = np.array(x)
    test_independant_variables = np.array(test)
    return Dataset(
        num_train_data=train_independant_variables.shape[0],
        num_test_data=test_independant_variables.shape[0],
        num_features=train_independant_variables.shape[1],

        train_independant_variables=train_independant_variables,
        train_dependent_variables=np.array(y, ndmin=2).T,
        test_independent_variables=test_independant_variables,
    )


def test_linear_regression_output(dataset: Dataset):
    reference_linear_regressor = linear_model.LinearRegression()
    reference_linear_regressor.fit(dataset.train_independant_variables, dataset.train_dependent_variables)
    reference_prediction = reference_linear_regressor.predict(dataset.test_independent_variables)

    normalize = MeanNormalization.from_data(dataset.train_independant_variables)

    normalized_x = normalize(dataset.train_independant_variables)
    normalized_x_with_intercept = np.hstack(
        (np.ones((normalized_x.shape[0], 1)), normalized_x)
    )

    cost_function = MeanSquareError(dataset.train_dependent_variables, normalized_x_with_intercept)
    stop_conditions = [
        MaxNumIterationsStopCondition(1_000_000),
        ConvergenceStopCondition(10_000, 0.00001, cost_function),
    ]

    coefficients = gradient_descent(
        learning_rate=0.01,
        cost_funtion=cost_function,
        stop_conditions=stop_conditions,
        normalized_x_with_intercept=normalized_x_with_intercept,
    )
    linear_regressor = LinearRegressor(coefficients, normalize)
    prediction = linear_regressor.predict(dataset.test_independent_variables)

    assert prediction == pytest.approx(reference_prediction, rel=0.1)
