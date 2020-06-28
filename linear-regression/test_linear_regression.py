import csv
import os.path
from dataclasses import dataclass

import numpy as np
import pytest
from sklearn import linear_model

from linear_regression.cost_functions import MeanSquareError
from linear_regression.gradient_descent import gradient_descent
from linear_regression.linear_regressor import LinearRegressor
from linear_regression.normalizers import MeanNormalizer
from linear_regression.stop_conditions import (
    AnyStopCondition,
    ConvergenceStopCondition,
    MaxNumIterationsStopCondition,
)

DATASETS_DIRECTORY = "datasets"


@dataclass
class Dataset:
    num_train_data: int
    num_test_data: int
    num_features: int
    train_x: np.ndarray  # Matrix: num_train_data x num_features
    train_y: np.array  # Vector column: num_train_data x 1
    test_x: np.array  # Matrix: num_test_data x num_features


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

    train_x = np.array(x)
    train_y = np.array(y, ndmin=2).T
    test_x = np.array(test)
    return Dataset(
        num_train_data=train_x.shape[0],
        num_test_data=test_x.shape[0],
        num_features=train_x.shape[1],
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
    )


def create_normalizer(train_x: np.ndarray):
    return MeanNormalizer.from_data(train_x)


def create_normalized_x_with_intercept(
        normalizer: MeanNormalizer, train_x: np.ndarray, num_train_data: int
):
    normalized_x = normalizer.normalize(train_x)
    return np.hstack((np.ones((num_train_data, 1)), normalized_x))


def test_linear_regression_output(dataset: Dataset):
    reference_linear_regressor = linear_model.LinearRegression()
    reference_linear_regressor.fit(dataset.train_x, dataset.train_y)
    reference_prediction = reference_linear_regressor.predict(dataset.test_x)

    normalizer = create_normalizer(dataset.train_x)

    normalized_x_with_intercept = create_normalized_x_with_intercept(
        normalizer, dataset.train_x, dataset.num_train_data
    )

    cost_function = MeanSquareError(
        dataset.train_y, normalized_x_with_intercept, dataset.num_train_data
    )
    stop_condition = AnyStopCondition(
        [
            MaxNumIterationsStopCondition(1_000_000),
            ConvergenceStopCondition(10_000, 0.00001, cost_function),
        ]
    )

    coefficients = gradient_descent(
        learning_rate=0.01,
        cost_funtion=cost_function,
        stop_condition=stop_condition,
        num_features=dataset.num_features,
    )
    linear_regressor = LinearRegressor(coefficients, normalizer)
    prediction = linear_regressor.predict(dataset.test_x)

    assert prediction == pytest.approx(reference_prediction, rel=0.1)
