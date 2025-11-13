from typing import Protocol, Optional

import numpy as np


class DataProcessorProtocol(Protocol):
    """Протокол для обработки данных (нормализация, денормализация)"""

    def fit(self, data: np.ndarray) -> None:
        """Обучение процессора на данных (вычисление параметров)"""
        pass

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Нормализация данных"""
        pass

    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """Денормализация данных"""
        pass


class MinMaxScaler:

    def __init__(self):
        self.data_min: Optional = None
        self.data_max: Optional = None

    def fit(self, data: np.ndarray) -> None:
        """
        Обучение процессора на данных (вычисление параметров)
        Запоминает min и max
        """
        pass

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Нормализация данных"""
        pass

    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """Денормализация данных"""
        pass
