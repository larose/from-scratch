import numpy as np


class MeanNormalizer:
    def __init__(self, average: float, range: float):
        self._average = average
        self._range = range

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self._average) / self._range

    @staticmethod
    def from_data(x: np.ndarray):
        min = x.min(axis=0)
        max = x.max(axis=0)
        range = max - min
        average = x.mean(axis=0)

        return MeanNormalizer(average=average, range=range)