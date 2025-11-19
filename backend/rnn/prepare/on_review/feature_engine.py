from typing import Dict, Protocol, List, Tuple
import numpy as np
import pandas as pd

from rnn.prepare.on_review.scaler import (
    ScalerProtocol,
    StandardScaler,
    EmptyScaler,
    MinMaxScaler,
)


class FeatureFunc(Protocol):
    def __call__(self, df: pd.DataFrame) -> Tuple[pd.Series, ScalerProtocol]:
        pass


class FeatureEngine:

    __feature_registry: Dict[str, FeatureFunc]

    def __init__(self):
        self.__feature_registry = {
            "raw": lambda df: self.raw(df, "Close"),
            "log_return": lambda df: self.log_return(df, "Close"),
            "candle_body": self.candle_body,
            "rsi_14": lambda df: self.rsi(df, "Close"),
            "sma_14": lambda df: self.sma(df, "Close"),
            "ema_14": lambda df: self.ema(df, "Close"),
            "hv_14": lambda df: self.hv(df, "Close"),
        }

    def build_features(
        self, df: pd.DataFrame, features: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, ScalerProtocol]]:
        features_df: pd.DataFrame = pd.DataFrame(index=df.index)
        scalers_registry: Dict[str, ScalerProtocol] = {}

        for name in features:
            if name not in self.__feature_registry:
                raise ValueError(f"Неизвестный признак: {name}")
            else:
                result, scaler = self.__feature_registry[name](df)
                features_df[name] = result
                scalers_registry[name] = scaler

        return features_df, scalers_registry

    @staticmethod
    def raw(
        df: pd.DataFrame, price_type: str = "Close"
    ) -> Tuple[pd.Series, ScalerProtocol]:
        """
        Исходные значения
        """
        if price_type not in df.columns:
            raise ValueError(f"Столбец '{price_type}' не найден.")

        raw: pd.Series = df[price_type]
        scaler: StandardScaler = StandardScaler()

        return raw, scaler

    @staticmethod
    def log_return(
        df: pd.DataFrame, price_type: str = "Close"
    ) -> Tuple[pd.Series, ScalerProtocol]:
        """
        Логарифмическая доходность
        """
        if price_type not in df.columns:
            raise ValueError(f"Столбец '{price_type}' не найден.")

        log_return: pd.Series = np.log(df[price_type] / df[price_type].shift(1))
        # scaler: StandardScaler = StandardScaler()
        scaler: EmptyScaler = EmptyScaler()

        return log_return, scaler

    @staticmethod
    def hv(
        df: pd.DataFrame, price_type: str = "Close", period: int = 14
    ) -> Tuple[pd.Series, ScalerProtocol]:
        """
        Историческая волатильность
        """
        log_return = np.log(df[price_type] / df[price_type].shift(1))

        hv_daily = log_return.rolling(window=period).std()

        hv = hv_daily * np.sqrt(250)

        scaler: StandardScaler = StandardScaler()

        return hv, scaler

    @staticmethod
    def candle_body(df: pd.DataFrame) -> Tuple[pd.Series, ScalerProtocol]:
        """
        Относительное тело свечи
        """
        required_cols = ["Open", "High", "Low", "Close"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Столбец '{col}' не найден.")

        body = (df["Close"] - df["Open"]) / (df["High"] - df["Low"])

        scaler: StandardScaler = StandardScaler()
        # scaler: EmptyScaler = EmptyScaler()

        return body, scaler

    @staticmethod
    def rsi(
        df: pd.DataFrame, price_type: str = "Close", period: int = 14
    ) -> Tuple[pd.Series, ScalerProtocol]:
        """
        Относительный индекс силы
        """
        if price_type not in df.columns:
            raise ValueError(f"Столбец '{price_type}' не найден.")
        delta = df[price_type].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        rs = avg_gain / avg_loss
        rsi = 1 - (1 / (1 + rs))

        scaler: EmptyScaler = EmptyScaler()

        return rsi, scaler

    @staticmethod
    def sma(
        df: pd.DataFrame, price_type: str = "Close", period: int = 14
    ) -> Tuple[pd.Series, ScalerProtocol]:
        """
        Простое скользящее среднее
        """
        if price_type not in df.columns:
            raise ValueError(f"Столбец '{price_type}' не найден.")

        # scaler: PercentDevScaler = PercentDevScaler()
        # scaler: StandardScaler = StandardScaler()
        scaler: MinMaxScaler = MinMaxScaler((-1, 1))

        return df[price_type].rolling(window=period).mean(), scaler

    @staticmethod
    def ema(
        df: pd.DataFrame, price_type: str = "Close", period: int = 14
    ) -> Tuple[pd.Series, ScalerProtocol]:
        """
        Экспоненциальное скользящее среднее
        """
        if price_type not in df.columns:
            raise ValueError(f"Столбец '{price_type}' не найден.")

        # scaler: PercentDevScaler = PercentDevScaler()
        # scaler: StandardScaler = StandardScaler()
        scaler: MinMaxScaler = MinMaxScaler((-1, 1))

        return df[price_type].ewm(span=period, adjust=False).mean(), scaler
