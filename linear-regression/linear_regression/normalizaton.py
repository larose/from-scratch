import numpy as np


class MeanNormalization:
    def __init__(self, average, range):
        self._average = average
        self._range = range

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self._average) / self._range

    @staticmethod
    def from_data(x):
        min = x.min(axis=0)
        max = x.max(axis=0)
        average = x.mean(axis=0)
        range = max - min

        return MeanNormalization(average=average, range=range)
