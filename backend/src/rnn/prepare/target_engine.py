from enum import Enum
from typing import Dict, Protocol, Tuple

import pandas as pd

from .scaler import ScalerProtocol, StandardScaler


class TargetEnum(str, Enum):
    CLOSE_1D = "close_1d"


class TargetFunc(Protocol):
    def __call__(self, df: pd.DataFrame) -> Tuple[pd.Series, ScalerProtocol]: ...


class TargetEngine:

    def __init__(self):
        self.__target_registry: Dict[TargetEnum, TargetFunc] = {
            TargetEnum.CLOSE_1D: self.target_close_1d,
        }

    def build_target(
        self, df: pd.DataFrame, target: TargetEnum
    ) -> Tuple[pd.DataFrame, Dict[str, ScalerProtocol]]:

        targets_df = pd.DataFrame(index=df.index)
        scalers_registry: Dict[str, ScalerProtocol] = {}

        series, scaler = self.__target_registry[target](df)
        targets_df[target] = series
        scalers_registry[target] = scaler

        targets_df = targets_df.dropna()

        return targets_df, scalers_registry

    @staticmethod
    def target_close_1d(df: pd.DataFrame) -> Tuple[pd.Series, ScalerProtocol]:
        """Цена закрытия следующего дня"""
        s = df["Close"].shift(-1)
        return s, StandardScaler()
