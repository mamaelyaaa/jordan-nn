import random
from typing import Optional

import numpy as np


class Neuron:
    """Класс, имплементирующий работу нейрона в НС"""

    def __init__(
        self,
        inputs: list[float] | np.ndarray,
        weights: Optional[np.ndarray] = None,
    ):
        """
        :param inputs: Входящие данные для нейрона
        :param weights: Веса входящие в текущий нейрон
        """
        self.inputs = inputs
        self.weights = (
            [1.0] + weights
            if weights
            else [1.0] + [random.uniform(-1, 1) for _ in range(len(inputs))]
        )

    def __repr__(self):
        return f"Нейрон <Веса: {[float(f"{weight:.2f}") for weight in self.weights]}, входы: {self.inputs}>"


# class NeuronResidual(Neuron):
#     """Класс, расширяющий работу обычного нейрона в расчете невязки"""
#
#     def __init__(self, inputs: list[float], weights: Optional[list[float]] = None):
#         super().__init__(inputs, weights)
#         self.residuals: list[float]

# class Neuron:
#
#     def __init__(
#         self,
#         inputs: list[float],
#         weights: Optional[list[float]] = None,
#         fictive: bool = False,
#     ):
#         self._inputs = inputs
#         self._fictive = fictive
#         if weights:
#             self._weights = weights
#         else:
#             if fictive:
#                 self._weights = [1] + [
#                     random.uniform(-1, 1) for _ in range(len(inputs))
#                 ]
#             else:
#                 self._weights = [random.uniform(-1, 1) for _ in range(len(inputs))]
#
#     def __repr__(self):
#         return f"Веса нейрона: {self._weights}"
#
#     @staticmethod
#     def _is_valid_weight(value) -> bool:
#         return -1 <= value <= 1
#
#     @property
#     def weights(self) -> list[float]:
#         return self._weights
#
#     @weights.setter
#     def weights(self, inputs: list[float]) -> None:
#         self._weights = inputs
#
#     @property
#     def inputs(self) -> list[float]:
#         return self._inputs
#
#     @inputs.setter
#     def inputs(self, inputs: list[float]) -> None:
#         self._inputs = inputs
