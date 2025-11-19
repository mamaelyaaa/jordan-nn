from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import FeaturesType, TargetType
from .feature_engine import FeatureEngine
from .on_review.scaler import ScalerProtocol
from .target_engine import TargetEngine


@dataclass
class Dataset:
    df: pd.DataFrame
    # Целевые значения

    # Обучающая выборка
    x_train_N: np.ndarray
    y_train_N: np.ndarray
    train_size: int
    close_train: pd.Series

    # Тестовая выборка
    x_test_N: np.ndarray
    y_test_N: np.ndarray
    test_size: int
    close_test: pd.Series


class DataLoader:

    def __init__(self):
        self.features_scalers: dict[str, ScalerProtocol] = {}
        self.feature_engine = FeatureEngine()
        self.target_engine = TargetEngine()
        self.target_scalers = None  # Добавляем для денормализации

    @staticmethod
    def load_raw_data(source: Path | str) -> pd.DataFrame:
        raw_data = pd.read_csv(source)
        df = raw_data.drop(columns=["OpenInt"], errors="ignore")
        return df

    def prepare_data(
        self,
        df: pd.DataFrame,
        features: list[FeaturesType],
        target: list[TargetType],
        test_rate: float = 0.3,
    ) -> "Dataset":
        """Подготовка данных в формате вашего примера"""

        # Создаем признаки
        features_df, feature_scalers = self.feature_engine.build_features(
            df, features=features
        )
        features_df = features_df.dropna()

        # Создаем таргет
        target_df, target_scalers = self.target_engine.build_target(df, target=target)
        self.target_scalers = target_scalers  # Сохраняем для денормализации
        target_df = target_df.dropna()

        n = len(features_df)
        test_size = int(n * test_rate)
        train_size = n - test_size

        close_all = df[["Close"]]
        close_train = df[["Close"]][:train_size][:-1]
        close_test = df[["Close"]][train_size:][:-1]

        # Разделяем исходные данные
        x_train = features_df.iloc[:train_size].copy()
        x_test = features_df.iloc[train_size:].copy()
        y_train = target_df.iloc[:train_size].copy()
        y_test = target_df.iloc[train_size:].copy()

        # Нормализация данных признаков
        for feature, f_scaler in feature_scalers.items():
            x_train.loc[:, feature] = f_scaler.normalize(x_train[feature], fit=True)
            x_test.loc[:, feature] = f_scaler.normalize(x_test[feature], fit=False)

        # Нормализация целевых переменных
        for target_name, t_scaler in target_scalers.items():
            y_train.loc[:, target_name] = t_scaler.normalize(
                y_train[target_name], fit=True
            )
            y_test.loc[:, target_name] = t_scaler.normalize(
                y_test[target_name], fit=False
            )

        # Разбиваем на входы и выходы для временных рядов (t -> t+1)
        x_train_normalized = x_train[:-1].values  # Признаки дня t
        y_train_normalized = y_train[1:].values  # Цель дня t+1
        x_test_normalized = x_test[:-1].values  # Признаки дня t
        y_test_normalized = y_test[1:].values  # Цель дня t+1

        return Dataset(
            df=df,
            close_test=close_test,
            close_train=close_train,
            x_train_N=x_train_normalized,
            y_train_N=y_train_normalized,
            train_size=len(x_train_normalized),
            x_test_N=x_test_normalized,
            y_test_N=y_test_normalized,
            test_size=len(x_test_normalized),
        )

    def denormalize_predictions(
        self, predictions_n: np.ndarray, target_name: str = None
    ) -> np.ndarray:
        """Обратное преобразование предсказаний"""
        if self.target_scalers is None:
            raise ValueError(
                "Скалеры не инициализированы. Сначала вызовите prepare_data"
            )

        if target_name is None:
            # Берем первый доступный скалер
            target_name = next(iter(self.target_scalers.keys()))

        return self.target_scalers[target_name].denormalize(predictions_n)

    def denormalize_targets(
        self, targets_n: np.ndarray, target_name: str = None
    ) -> np.ndarray:
        """Обратное преобразование целей"""
        return self.denormalize_predictions(targets_n, target_name)

    def denormalize_log_returns(
        self, returns_normalized: np.ndarray, target_name: str = None
    ) -> np.ndarray:
        """
        Денормализация лог-доходностей, нормализованных как (x - mean) / std.
        Возвращает массив тех же размеров.
        """
        if self.target_scalers is None:
            raise ValueError(
                "Скалеры не инициализированы. Сначала вызовите prepare_data"
            )

        if target_name is None:
            target_name = next(iter(self.target_scalers.keys()))

        scaler = self.target_scalers[target_name]

        # Предполагаем, что скалер имеет атрибуты _mean и _std
        if hasattr(scaler, "_mean") and hasattr(scaler, "_std"):
            mu = scaler._mean  # среднее таргета (лог доходности)
            sigma = scaler._std  # std таргета
            return returns_normalized * sigma + mu
        else:
            # Если скалер не имеет этих атрибутов, используем общий метод
            return scaler.denormalize(returns_normalized)
