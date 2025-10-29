from typing import Protocol

import numpy as np


class ActivationProtocol(Protocol):
    """Прототип, описывающий функции активации"""

    def calculate(self, x: np.ndarray) -> np.ndarray:
        """
        Рассчитывает выходы нейрона

        :param x: Входные данные
        :return: Выходы нейрона
        """
        pass

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Рассчитывает производную от функции активации
        :param x: Входные данные
        :return: Производная
        """
        pass


class SigmoidActivation:
    """Сигмоидная функция активации"""

    def __init__(self, saturation: float = 0.5, t: float = 0):
        """
        :param saturation: Параметр насыщения
        :param t:
        """
        self._t = t
        self._saturation = saturation

    def calculate(self, x: np.ndarray) -> np.ndarray:
        return np.array(
            [
                1 / (1 + np.exp(-self._saturation * (x[i] - self._t)))
                for i in range(len(x))
            ]
        )

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return self._saturation * self.calculate(x) * (1 - self.calculate(x))

    @property
    def saturation(self) -> float:
        """Параметр насыщения"""
        return self._saturation


class ReLUActivation:
    """Функция активации ReLU"""

    def __init__(self, k: float = 1):
        self._k = k

    def calculate(self, x: np.ndarray) -> np.ndarray:
        return 0 if x <= 0 else self._k * x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 0 if x <= 0 else self._k
