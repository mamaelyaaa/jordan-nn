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

    def __init__(self, data: np.ndarray, headers: list[str]):
        self.data = data
        self.headers = headers


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
        batch_size: int = 32,
    ):
        """
        :param lookback: Размер окна истории: сколько прошлых точек используем для прогноза
        :param horizon: Горизонт прогноза: на сколько шагов вперед предсказываем
        :param batch_size: Размер батча: количество окон в одном батче
        """
        self.lookback = lookback
        self.horizon = horizon
        self.batch_size = batch_size
        self.windows = self._create_all_windows()

    def _create_all_windows(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Создает все окна для итерации"""
        pass

    def __len__(self) -> int:
        """Количество батчей в эпохе"""
        pass

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Итератор по батчам

        :return: кортеж из (features, targets)
        """
        pass

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Получить батч по индексу"""
        pass


# ==== Датасет ====


class TimeSeriesDatasetProtocol(Protocol):
    """Протокол датасета"""

    def __len__(self) -> int:
        """Размер датасета"""
        pass

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Возвращает образ (features, target)"""
        pass

    def split(
        self, test_size: float = 0.3
    ) -> tuple["TimeSeriesDatasetProtocol", "TimeSeriesDatasetProtocol"]:
        """Разделение выборки на обучающую и тестовую"""
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


# Будущая имплементация
class TimeSeriesDataset:
    """Реализация датасета временного ряда"""

    def __init__(
        self,
        raw_data: RawData,
        batch_gen: BatchGeneratorProtocol,
        feature_headers: list[str],
        targets_headers: list[str],
    ):
        self.raw_data = raw_data
        self.batch_gen = batch_gen

        self._feature_idx = [
            self.raw_data.headers.index(feature) for feature in feature_headers
        ]
        self._targets_idx = [
            self.raw_data.headers.index(targets) for targets in targets_headers
        ]
        self._features = self.raw_data.data[:, self._feature_idx]
        self._targets = self.raw_data.data[:, self._targets_idx]

    def _split_features_targets(self) -> tuple[np.ndarray, np.ndarray]:
        """Разделение данных на features и targets"""
        pass

    def __len__(self) -> int:
        """Размер датасета"""
        pass

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Возвращает образ (features, target)"""
        pass

    def split(
        self, test_size
    ) -> tuple["TimeSeriesDatasetProtocol", "TimeSeriesDatasetProtocol"]:
        """Разделение выборки на обучающую и тестовую"""
        pass

    @property
    def features(self) -> np.ndarray:
        """Получить фичи"""
        return self._features

    @property
    def targets(self) -> np.ndarray:
        """Получить таргеты"""
        return self._targets

    @property
    def batch_generator(self) -> BatchGeneratorProtocol:
        """Получить генератор батчей"""
        pass
