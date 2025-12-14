from enum import Enum
from typing import Dict, Protocol, Tuple

import numpy as np
import pandas as pd

from .scaler import ScalerProtocol, EmptyScaler, MinMaxScaler, StandardScaler


class FeaturesEnum(str, Enum):
    # Базовые
    LOG_RETURN = "log_return"
    PCT_RETURN = "pct_return"
    # Свечные
    CLOSE = "close"
    HIGH = "high"
    LOW = "low"
    CANDLE_BODY = "candle_body"
    # Индикаторы
    RSI14 = "rsi_14"
    EMA14 = "ema_14"
    SMA14 = "sma_14"
    HV14 = "hv_14"
    # Волатильность
    VOLATILITY = "volatility"


class FeatureProtocol(Protocol):

    def __call__(self, df: pd.DataFrame) -> Tuple[pd.Series, ScalerProtocol]:
        pass


class FeatureEngine:

    def __init__(self):
        self.__feature_registry: Dict[FeaturesEnum, FeatureProtocol] = {
            # Базовые
            FeaturesEnum.LOG_RETURN: self.log_return,
            FeaturesEnum.PCT_RETURN: self.pct_return,
            # Свечные
            FeaturesEnum.HIGH: self.high_rel,
            FeaturesEnum.LOW: self.low_rel,
            FeaturesEnum.CLOSE: self.close_rel,
            FeaturesEnum.CANDLE_BODY: self.candle_body,
            # Индикаторы
            FeaturesEnum.RSI14: lambda df: self.rsi(df, period=14),
            FeaturesEnum.HV14: lambda df: self.hv(df, period=14),
            # Скользящие
            FeaturesEnum.SMA14: lambda df: self.sma(df, period=14),
            FeaturesEnum.EMA14: lambda df: self.ema(df, period=14),
            # Волатильность
            # "true_range_pct": self.true_range_pct,
            FeaturesEnum.VOLATILITY: self.daily_volatility_abs,
        }

    def build_features(
        self,
        df: pd.DataFrame,
        features: list[FeaturesEnum],
    ) -> tuple[pd.DataFrame, dict[str, ScalerProtocol]]:

        features_df = pd.DataFrame(index=df.index)
        scalers_registry: dict[str, ScalerProtocol] = {}

        # Добавление сдвига
        self.__feature_registry[FeaturesEnum.PCT_RETURN] = self.pct_return

        for name in features:
            if name not in self.__feature_registry:
                raise ValueError(f"Неизвестный признак: {name}")
            series, scaler = self.__feature_registry[name](df)
            features_df[name] = series
            scalers_registry[name] = scaler

        features_df = features_df.dropna()

        return features_df, scalers_registry

    @staticmethod
    def log_return(df: pd.DataFrame) -> Tuple[pd.Series, ScalerProtocol]:
        series = np.log(df["Close"]).diff()
        return series, EmptyScaler()

    @staticmethod
    def pct_return(df: pd.DataFrame) -> Tuple[pd.Series, ScalerProtocol]:
        return df["Close"].pct_change(), EmptyScaler()

    @staticmethod
    def high_rel(df: pd.DataFrame) -> Tuple[pd.Series, ScalerProtocol]:
        return df["High"] / df["Open"] - 1, MinMaxScaler()

    @staticmethod
    def low_rel(df: pd.DataFrame) -> Tuple[pd.Series, ScalerProtocol]:
        return df["Low"] / df["Open"] - 1, MinMaxScaler()

    @staticmethod
    def close_rel(df: pd.DataFrame) -> Tuple[pd.Series, ScalerProtocol]:
        return df["Close"] / df["Open"] - 1, MinMaxScaler()

    @staticmethod
    def candle_body(df: pd.DataFrame) -> Tuple[pd.Series, ScalerProtocol]:
        body = (df["Close"] - df["Open"]) / (df["High"] - df["Low"])
        return body, StandardScaler()

    @staticmethod
    def rsi(df: pd.DataFrame, period=14) -> Tuple[pd.Series, ScalerProtocol]:
        delta = df["Close"].diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / period).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / period).mean()
        rs = gain / loss
        rsi = 1 - (1 / (1 + rs))
        return rsi, EmptyScaler()

    @staticmethod
    def true_range_pct(df: pd.DataFrame) -> Tuple[pd.Series, ScalerProtocol]:
        hl = df["High"] - df["Low"]
        hc = abs(df["High"] - df["Close"].shift(1))
        lc = abs(df["Low"] - df["Close"].shift(1))
        tr = np.maximum(hl, np.maximum(hc, lc))
        return tr / df["Close"], StandardScaler()

    @staticmethod
    def daily_volatility_abs(df: pd.DataFrame) -> Tuple[pd.Series, ScalerProtocol]:
        return (df["High"] - df["Low"]) / df["Open"], StandardScaler()

    @staticmethod
    def hv(df: pd.DataFrame, period=14) -> Tuple[pd.Series, ScalerProtocol]:
        log_return = np.log(df["Close"] / df["Close"].shift(1))
        hv_daily = log_return.rolling(window=period).std()
        hv = hv_daily * np.sqrt(250)
        return hv, StandardScaler()

    @staticmethod
    def sma(df: pd.DataFrame, period=14) -> Tuple[pd.Series, ScalerProtocol]:
        sma = df["Close"].rolling(window=period).mean()
        return sma, MinMaxScaler((-1, 1))

    @staticmethod
    def ema(df: pd.DataFrame, period=14) -> Tuple[pd.Series, ScalerProtocol]:
        ema = df["Close"].ewm(span=period, adjust=False).mean()
        return ema, MinMaxScaler((-1, 1))
