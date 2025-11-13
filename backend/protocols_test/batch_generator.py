from typing import Protocol, Iterator

import numpy as np


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

        :return: кортеж из (features, targets) СО СДВИГОМ
        """
        pass

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Получить батч по индексу"""
        pass
