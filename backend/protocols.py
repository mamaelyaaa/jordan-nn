from typing import Protocol, Iterator, Optional

import numpy as np


# ==== Парсинг ====


class DataParserProtocol(Protocol):
    """Протокол парсинга данных (CSV, text, json, parquet, ...)"""

    @classmethod
    def load(cls, source: str) -> "RawData":
        """Подгрузка данных"""
        pass


class CSVParser:
    """Реализация парсинга данных из CSV"""

    @classmethod
    def load(cls, source: str) -> "RawData":
        """Подгрузка данных из CSV"""
        pass


class RawData:
    """Контейнер сырых данных"""

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = features
        self.targets = targets


# ==== Генератор батчей ====


class BatchGeneratorProtocol(Protocol):
    """Протокол генератора батчей"""

    def __len__(self) -> int:
        """Количество батчей в эпохе"""
        pass

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Итератор по батчам"""
        pass

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Получить батч по индексу"""
        pass


class TimeSeriesBatchGenerator:
    """Генератор батчей для временных рядов с окнами и горизонтом"""

    def __init__(
        self,
        lookback: int,
        horizon: int = 1,
        batch_size_per_epoch: int = 32,
    ):
        """
        :param lookback: Размер окна истории: сколько прошлых точек используем для прогноза
        :param horizon: Горизонт прогноза: на сколько шагов вперед предсказываем
        :param batch_size_per_epoch: Размер батча: количество окон в одном батче
        """
        self.lookback = lookback
        self.horizon = horizon
        self.batch_size_per_epoch = batch_size_per_epoch
        self.windows = self._create_all_windows()

    def _create_all_windows(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Создает все окна для итерации"""
        pass

    def __len__(self) -> int:
        """Количество батчей в эпохе"""
        pass

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Итератор по батчам"""
        pass

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Получить батч по индексу"""
        pass


# ==== Датасет ====


class DatasetProtocol(Protocol):
    """Протокол датасета"""

    def __len__(self) -> int:
        """Размер датасета"""
        pass

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Возвращает образ (features, target)"""
        pass

    def split(self, test_size) -> tuple["DatasetProtocol", "DatasetProtocol"]:
        """Разделение выборки на обучающую и тестовую"""
        pass

    def create_batch_generator(
        self,
        lookback: int,
        horizon: int = 1,
        batch_size: int = 32,
    ) -> BatchGeneratorProtocol:
        """Создает batch generator для этого датасета"""
        pass

    @property
    def features(self) -> np.ndarray:
        """Получить фичи"""
        pass

    @property
    def targets(self) -> np.ndarray:
        """Получить таргеты"""
        pass

    @property
    def batch_generator(self):
        """Получить генератор батчей"""
        pass


# Будущая имплементация
class TimeSeriesDataset:
    """Реализация датасета временного ряда"""

    def __init__(self, raw_data: RawData):
        """
        :param raw_data: Сырые данные
        """
        self.raw_data = raw_data

        # Генератор батчей
        self._batch_gen: Optional[BatchGeneratorProtocol] = None

    def __len__(self) -> int:
        """Размер датасета"""
        pass

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Возвращает образ (features, target)"""
        pass

    def split(self, test_size) -> tuple["DatasetProtocol", "DatasetProtocol"]:
        """Разделение выборки на обучающую и тестовую"""
        pass

    def create_batch_generator(
        self,
        lookback: int,
        horizon: int = 1,
        batch_size: int = 32,
    ) -> BatchGeneratorProtocol:
        """Создает batch generator для этого датасета"""
        pass

    @property
    def features(self) -> np.ndarray:
        """Получить фичи"""
        pass

    @property
    def targets(self) -> np.ndarray:
        """Получить таргеты"""
        pass

    @property
    def batch_generator(self) -> BatchGeneratorProtocol:
        """Получить генератор батчей"""
        pass
