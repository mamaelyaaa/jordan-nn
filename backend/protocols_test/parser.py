from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class RawData:
    """Контейнер сырых данных"""

    data: np.ndarray
    headers: list[str]


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
