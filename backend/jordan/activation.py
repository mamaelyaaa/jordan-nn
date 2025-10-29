from math import exp
from typing import Protocol

import numpy as np


class ActivationProtocol(Protocol):
    """Прототип, описывающий функции активации"""

    def calculate(self, x: float) -> float:
        pass


class SigmoidActivation:
    """Сигмоидная функция активации"""

    def __init__(self, saturation: float = 0.5, t: float = 0):
        """
        :param saturation: Параметр насыщения
        :param t:
        """
        self.t = t
        self.saturation = saturation

    def calculate(self, x: np.ndarray) -> np.ndarray:
        return np.array(
            [
                1 / (1 + np.exp(-self.saturation * (x[i] - self.t)))
                for i in range(len(x))
            ]
        )


class ReLUActivation:
    """Функция активации ReLU"""

    def __init__(self, k: float = 1):
        self.k = k

    def calculate(self, x: float) -> float:
        return 0 if x <= 0 else self.k * x
