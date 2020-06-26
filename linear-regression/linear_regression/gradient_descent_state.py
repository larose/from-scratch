from dataclasses import dataclass

import numpy as np


@dataclass
class GradientDescentState:
    iteration_count: int
    coefficients: np.ndarray  # Vector column: (num features + 1) x 1
