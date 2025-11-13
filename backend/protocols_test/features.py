from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from protocols import RawData


@dataclass
class Dataset:
    """Датасет"""

    features: np.ndarray
    targets: np.ndarray


class IFeatureEngine(ABC):

    def __init__(self, raw_data: RawData):
        self.raw_data = raw_data
        self.base_feature_columns: list[str] = []
        self.base_targets_columns: list[str] = []
        self.indicators: list

    @abstractmethod
    def set_base_features(self, columns: list[str]) -> None:
        """
        Устанавливает базовые колонки из сырых данных парсера для входов
        :param columns: Список имен столбцов
        :return:
        """
        pass

    @abstractmethod
    def set_base_targets(self, columns: list[str]) -> None:
        """
        Устанавливает базовые колонки из сырых данных парсера для целевых значений
        :param columns: Список имен столбцов
        :return:
        """
        pass

    @abstractmethod
    def transform(self) -> "Dataset":
        """
        Закрепляет входы и выходы для модели
        :return: Структура Dataset
        """
        pass

    @abstractmethod
    def add_rsi(self, period: int = 14) -> None:
        """
        Добавление индикатора RSI
        :param period: Период для RSI
        :return:
        """
        pass


class FeatureEngine:
    """Двигатель для создания и управления фичами"""

    pass
