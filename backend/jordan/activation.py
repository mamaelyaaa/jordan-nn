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


class TanhActivation:
    """Функция активации гиперболический тангенс"""

    def __init__(self, saturation: float = 0.5, t: float = 0):
        self._saturation = saturation
        self._t = t

    def calculate(self, x: np.ndarray) -> np.ndarray:
        return np.array(
            [
                2 / (1 + np.exp(-self._saturation * (x[i] - self._t))) - 1
                for i in range(len(x))
            ]
        )

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return self._saturation / 2 * (1 - self.calculate(x)) ** 2


class ReLUActivation:
    """Функция активации ReLU"""

    def __init__(self, k: float = 1):
        self._k = k

    def calculate(self, x: np.ndarray) -> np.ndarray:
        return np.array([0 if sample <= 0 else self._k * sample for sample in x])

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.array([0 if sample <= 0 else self._k for sample in x])
