from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd

from rnn.prepare.on_review.feature_engine import FeatureEngine
from rnn.prepare.on_review.scaler import ScalerProtocol


@dataclass
class Dataset:
    # Целевые значения
    close_all: np.ndarray

    # Обучающая выборка
    close_train: np.ndarray
    x_train_N: np.ndarray
    y_train_N: np.ndarray
    train_size: int

    # Тестовая выборка
    close_test: np.ndarray
    x_test_N: np.ndarray
    y_test_N: np.ndarray
    test_size: int


class DataLoader:

    def __init__(self):
        self.scalers: Dict[str, ScalerProtocol] = {}
        self.engine = FeatureEngine()

    @staticmethod
    def load_raw_data(source: Path | str) -> pd.DataFrame:
        raw_data = pd.read_csv(source)
        df = raw_data.drop(columns=["OpenInt"], errors="ignore")
        return df

    def prepare_data(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        test_rate: float = 0.3,
    ) -> "Dataset":
        """
        Подготовка данных в формате вашего примера
        """
        if not features:
            features = ["candle_body"]
        target = ["log_return"]
        raw = ["raw"]

        # Создаем выборку
        samples_df, self.scalers = self.engine.build_features(
            df, features + target + raw
        )
        samples_df = samples_df.dropna()

        n = len(samples_df)
        test_size = int(n * test_rate)
        train_size = n - test_size

        close_all = samples_df[["raw"]]
        close_train = samples_df[["raw"]][:train_size][:-1]
        close_test = samples_df[["raw"]][train_size:][:-1]

        samples_df = samples_df.drop(columns=["raw"])

        train_samples_df = samples_df[:train_size].copy()
        train_samples_normalized: pd.DataFrame = pd.DataFrame(
            index=train_samples_df.index
        )
        for name in train_samples_df:
            train_samples_normalized[name] = self.scalers[name].normalize(
                train_samples_df[name]
            )

        test_samples_df = samples_df[train_size:].copy()
        test_samples_normalized: pd.DataFrame = pd.DataFrame(
            index=test_samples_df.index
        )
        for name in test_samples_df:
            test_samples_normalized[name] = self.scalers[name].normalize(
                test_samples_df[name], fit=False
            )

        # Разделяем исходные данные
        x_train_normalized = train_samples_normalized[features].values
        x_test_normalized = test_samples_normalized[features].values
        y_train_normalized = train_samples_normalized[target].values
        y_test_normalized = test_samples_normalized[target].values

        # Разбиваем на входы и выходы для временных рядов
        x_train_normalized = x_train_normalized[:-1]  # Признаки дня t
        y_train_normalized = y_train_normalized[1:]  # Цель дня t+1
        x_test_normalized = x_test_normalized[:-1]  # Признаки дня t
        y_test_normalized = y_test_normalized[1:]  # Цель дня t+1]

        return Dataset(
            close_all=close_all,
            close_train=close_train,
            x_train_N=x_train_normalized,
            y_train_N=y_train_normalized,
            train_size=train_size,
            close_test=close_test,
            x_test_N=x_test_normalized,
            y_test_N=y_test_normalized,
            test_size=test_size,
        )

    def denormalize_predictions(self, predictions_n: np.ndarray) -> np.ndarray:
        """Обратное преобразование предсказаний"""
        return self.scalers["log_return"].denormalize(predictions_n)
