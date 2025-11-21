from typing import Dict, Protocol, List, Tuple, TypeVar
import numpy as np
import pandas as pd

from rnn.prepare import TargetType
from rnn.prepare.normalizer import PercentScaler
from rnn.prepare.on_review.scaler import (
    ScalerProtocol,
    StandardScaler,
    EmptyScaler,
    MinMaxScaler,
    PercentDevScaler,
)


class TargetFunc(Protocol):
    def __call__(self, df: pd.DataFrame) -> Tuple[pd.Series, ScalerProtocol]: ...


class TargetEngine:

    def __init__(self):
        self.__target_registry: Dict[str, TargetFunc] = {
            # БАЗОВЫЕ ТАРГЕТЫ
            "target_close_1d": self.target_close_1d,
            "target_log_return_1d": self.target_log_return_1d,
            "target_pct_return_1d": self.target_pct_return_1d,
            # БОЛЕЕ СЛОЖНЫЕ ТАРГЕТЫ
            "target_volatility_1d": self.target_volatility_1d,
            "target_direction_1d": self.target_direction_1d,
        }

    def build_target(
        self,
        df: pd.DataFrame,
        target: list[TargetType],
    ) -> Tuple[pd.DataFrame, Dict[str, ScalerProtocol]]:

        targets_df = pd.DataFrame(index=df.index)
        scalers_registry: Dict[str, ScalerProtocol] = {}

        for name in target:
            if name not in self.__target_registry:
                raise ValueError(f"Неизвестный таргет: {name}")

            series, scaler = self.__target_registry[name](df)
            targets_df[name] = series
            scalers_registry[name] = scaler

        targets_df = targets_df.dropna()

        return targets_df, scalers_registry

    @staticmethod
    def target_close_1d(df: pd.DataFrame) -> Tuple[pd.Series, ScalerProtocol]:
        """Цена закрытия следующего дня"""
        s = df["Close"].shift(-1)
        return s, StandardScaler()

    @staticmethod
    def target_log_return_1d(df: pd.DataFrame) -> Tuple[pd.Series, ScalerProtocol]:
        """Лог-доходность следующего дня"""
        s = np.log(df["Close"]).diff().shift(-1)
        return s, EmptyScaler()

    @staticmethod
    def target_pct_return_1d(df: pd.DataFrame) -> Tuple[pd.Series, ScalerProtocol]:
        """Обычная доходность следующего дня"""
        s = df["Close"].pct_change().shift(-1)
        return s, EmptyScaler()

    @staticmethod
    def target_volatility_1d(df: pd.DataFrame) -> Tuple[pd.Series, ScalerProtocol]:
        """Абсолютная волатильность свечи следующего дня"""
        daily_vol = (df["High"] - df["Low"]) / df["Open"]
        return daily_vol.shift(-1), StandardScaler()

    @staticmethod
    def target_direction_1d(df: pd.DataFrame) -> Tuple[pd.Series, ScalerProtocol]:
        """Направление движения цены: -1 / 0 / 1"""
        ret = df["Close"].pct_change().shift(-1)
        direction = np.sign(ret)
        return direction, EmptyScaler()
