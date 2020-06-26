import csv
import os.path

import numpy as np
import pytest
from sklearn import linear_model

from linear_regression.gradient_descent import (
    GradientDescent,
    GradientDescentParameters,
)

DATASETS_DIRECTORY = "datasets"


@pytest.fixture(scope="module", params=["toy", "petrol"])
def data(request):
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

    # m: number of train data
    # n: number of features
    # r: number of test data
    return (
        np.array(x),  # Matrix (m x n)
        np.array(y, ndmin=2).T,  # Vector column (m x 1)
        np.array(test),  # Matrix (r x n)
    )


def test_linear_regression_output(data):
    train_x, train_y, test = data

    reference_linear_regressor = linear_model.LinearRegression()
    reference_linear_regressor.fit(train_x, train_y)
    reference_prediction = reference_linear_regressor.predict(test)

    gradient_descent = GradientDescent(train_x, train_y, GradientDescentParameters())
    linear_regressor = gradient_descent.fit()
    prediction = linear_regressor.predict(test)

    assert prediction == pytest.approx(reference_prediction, rel=0.1)
