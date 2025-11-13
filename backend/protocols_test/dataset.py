from typing import Protocol

import numpy as np

from protocols_test import DataProcessorProtocol
from protocols_test.batch_generator import BatchGeneratorProtocol
from protocols_test.parser import RawData


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

    @property
    def processor(self) -> DataProcessorProtocol:
        """Получение датапроцессора"""
        pass


# Будущая имплементация
class TimeSeriesDataset:
    """Реализация датасета временного ряда"""

    def __init__(
        self,
        raw_data: RawData,
        batch_generator: BatchGeneratorProtocol,
        processor: DataProcessorProtocol,
        feature_headers: list[str],
        targets_headers: list[str],
    ):
        self.raw_data = raw_data
        self._batch_generator = batch_generator
        self._processor = processor
        self._features, self._targets = self._init_features_and_targets(
            feature_headers, targets_headers
        )

    def _init_features_and_targets(
        self,
        feature_headers: list[str],
        targets_headers: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Разделение данных на features и targets"""

        feature_idx = [
            self.raw_data.headers.index(feature) for feature in feature_headers
        ]
        targets_idx = [
            self.raw_data.headers.index(targets) for targets in targets_headers
        ]

        features = self.raw_data.data[:, feature_idx]
        targets = self.raw_data.data[:, targets_idx]
        return features, targets

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

    @property
    def processor(self) -> DataProcessorProtocol:
        """Получение датапроцессора"""
        pass
