from enum import Enum
from typing import Protocol

import numpy as np


class RegularizerEnum(str, Enum):
    L1 = "L1"
    L2 = "L2"


class RegularizerProtocol(Protocol):
    """Протокол для регуляризаторов"""

    def compute_gradient(self, weights: np.ndarray) -> np.ndarray:
        pass


class NoRegularizer:
    """Пустой регуляризатор (по умолчанию)"""

    @staticmethod
    def compute_gradient(weights: np.ndarray) -> np.ndarray:
        return np.zeros_like(weights)


class L2:
    """L2 регуляризатор"""

    def __init__(self, lm: float):
        self.lm = lm

    def compute_gradient(self, weights: np.ndarray) -> np.ndarray:
        return self.lm * weights


class L1:
    """L1 регуляризация"""

    def __init__(self, lm):
        self.lm = lm

    def compute_gradient(self, weights: np.ndarray) -> np.ndarray:
        return self.lm * np.sign(weights)
