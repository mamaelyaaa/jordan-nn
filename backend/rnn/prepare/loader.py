from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .normalizer import StandardScaler, PercentScaler
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
        self.feature_scalers = {}
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
        df.drop(columns=["Date", "Close", "Open", "High", "Low", "Volume"])

        # 2. Разделяем фичи на группы по типам
        feature_groups = self._categorize_features(df)

        # 3. Нормализуем каждую группу отдельно
        normalized_features = []

        for group_name, cols in feature_groups.items():
            if not cols:
                continue

            group_data = df[cols].values

            if group_name not in self.feature_scalers:
                if any("%" in col.lower() for col in cols):
                    # TODO Проверить какие еще столбцы содержат проценты и пометить их "%"

                    self.feature_scalers[group_name] = PercentScaler()
                    normalized_group = self.feature_scalers[group_name].normalize(
                        group_data
                    )
                else:
                    self.feature_scalers[group_name] = StandardScaler()
                    normalized_group = self.feature_scalers[group_name].fit_normalize(
                        group_data
                    )
            else:
                normalized_group = self.feature_scalers[group_name].normalize(
                    group_data
                )

            normalized_features.append(normalized_group)

        # 4. Объединяем все нормализованные фичи
        features_normalized = np.hstack(normalized_features)

        targets = df[["target_close"]].values

        n = len(features_normalized)
        test_size = int(n * test_rate)
        train_size = n - test_size

        # Разделяем исходные данные
        x_train_normalized = features_normalized[:train_size]
        x_test_normalized = features_normalized[train_size:]
        y_train = targets[:train_size]
        y_test = targets[train_size:]

        # Нормализация данных по обучающей выборке
        y_train_normalized = self.targets_scaler.fit_normalize(y_train)
        y_test_normalized = self.targets_scaler.normalize(y_test)

        # Разбиваем на входы и выходы для временных рядов
        x_train_normalized = x_train_normalized[:-1]  # Признаки дня t
        y_train_normalized = y_train_normalized[1:]  # Цель дня t+1
        x_test_normalized = x_test_normalized[:-1]  # Признаки дня t
        y_test_normalized = y_test_normalized[1:]  # Цель дня t+1]

        return Dataset(
            x_train_N=x_train_normalized,
            y_train_N=y_train_normalized,
            train_size=train_size,
            x_test_N=x_test_normalized,
            y_test_N=y_test_normalized,
            test_size=test_size,
        )

    @staticmethod
    def _categorize_features(df: pd.DataFrame) -> dict:
        """Разделение фич на логические группы"""

        all_features = [
            col for col in df.columns if col not in ["target_close", "Date"]
        ]

        groups = {
            "prices": [],  # Абсолютные цены
            "volumes": [],  # Объемы
            "returns": [],  # Доходности, изменения
            "ratios": [],  # Отношения, проценты
            "volatility": [],  # Волатильность
        }

        for feature in all_features:
            feature_lower = feature.lower()

            if any(
                price in feature_lower for price in ["open", "high", "low", "close"]
            ):
                if "change" not in feature_lower and "ratio" not in feature_lower:
                    groups["prices"].append(feature)

            elif "volume" in feature_lower:
                groups["volumes"].append(feature)

            elif any(
                term in feature_lower for term in ["change", "return", "momentum"]
            ):
                groups["returns"].append(feature)

            elif any(
                term in feature_lower for term in ["volatility", "range", "std", "var"]
            ):
                groups["volatility"].append(feature)

            elif any(
                term in feature_lower for term in ["ratio", "pct", "percent", "vs_"]
            ):
                groups["ratios"].append(feature)

            else:
                # По умолчанию в returns
                groups["returns"].append(feature)

        return groups

    def denormalize_predictions(self, predictions_n: np.ndarray) -> np.ndarray:
        """Обратное преобразование предсказаний"""
        return self.targets_scaler.denormalize(predictions_n)

    def denormalize_targets(self, targets_n: np.ndarray) -> np.ndarray:
        """Обратное преобразование целей"""
        return self.targets_scaler.denormalize(targets_n)
