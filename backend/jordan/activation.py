from math import exp
from typing import Protocol


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
        self.validate_saturation(saturation)
        self.saturation = saturation

    def calculate(self, x: float) -> float:
        return 1 / (1 + exp(-self.saturation * (x - self.t)))

    @staticmethod
    def validate_saturation(saturation: float) -> None:
        if not 0.5 <= saturation <= 2.5:
            raise ValueError("Не рекомендованный параметр сатурации. От 0.5 до 2.5")
        return


class ReLUActivation:
    """Функция активации ReLU"""

    def __init__(self, k: float = 1):
        self.k = k

    def calculate(self, x: float) -> float:
        return 0 if x <= 0 else self.k * x
