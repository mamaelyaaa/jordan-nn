from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .normalizer import StandardScaler
from .feature_engineering import FeatureEngineering


@dataclass
class Dataset:
    # Обучающая выборка
    x_train_N: np.ndarray
    y_train_N: np.ndarray
    train_size: int

    # Тестовая выборка
    x_test_N: np.ndarray
    y_test_N: np.ndarray
    test_size: int


class DataLoader:

    def __init__(self):
        self.feature_scaler = StandardScaler()
        self.targets_scaler = StandardScaler()
        self.feature_engine = FeatureEngineering()

    @staticmethod
    def load_raw_data(source: Path | str) -> pd.DataFrame:
        raw_data = pd.read_csv(source)
        df = raw_data.drop(columns=["OpenInt"], errors="ignore")
        return df

    def prepare_data(self, df: pd.DataFrame, test_rate: float = 0.3) -> "Dataset":
        """
        Подготовка данных в формате вашего примера
        """

        # Создаем признаки
        df = self.feature_engine.engine_features(df)
        df = df.dropna()

        # Берем только фичи
        feature_columns = [
            col for col in df.columns if col not in ["target_close", "Date"]
        ]
        features = df[feature_columns].values
        targets = df[["target_close"]].values

        n = len(features)
        test_size = int(n * test_rate)
        train_size = n - test_size

        # Разделяем исходные данные
        x_train = features[:train_size]
        x_test = features[train_size:]
        y_train = targets[:train_size]
        y_test = targets[train_size:]

        # Нормализация данных по обучающей выборке
        x_train_normalized = self.feature_scaler.fit_normalize(x_train)
        y_train_normalized = self.targets_scaler.fit_normalize(y_train)

        x_test_normalized = self.feature_scaler.normalize(x_test)
        y_test_normalized = self.targets_scaler.normalize(y_test)

        # Разбиваем на входы и выходы для временных рядов
        x_train_normalized = x_train_normalized[:-1]  # Признаки дня t
        y_train_normalized = y_train_normalized[1:]  # Цель дня t+1

        x_test_normalized = x_test_normalized[:-1]  # Признаки дня t
        y_test_normalized = y_test_normalized[1:]  # Цель дня t+1

        return Dataset(
            x_train_N=x_train_normalized,
            y_train_N=y_train_normalized,
            train_size=train_size,
            x_test_N=x_test_normalized,
            y_test_N=y_test_normalized,
            test_size=test_size,
        )

    def denormalize_predictions(self, predictions_n: np.ndarray) -> np.ndarray:
        """Обратное преобразование предсказаний"""
        return self.targets_scaler.denormalize(predictions_n.reshape(-1, 1))

    def denormalize_targets(self, targets_n: np.ndarray) -> np.ndarray:
        """Обратное преобразование целей"""
        return self.targets_scaler.denormalize(targets_n)
