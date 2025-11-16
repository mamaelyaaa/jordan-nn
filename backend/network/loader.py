import numpy as np
import pandas as pd
from normalizer import StandardScaler


class DataLoader:

    def __init__(self):
        self.feature_scaler = StandardScaler()
        self.targets_scaler = StandardScaler()

    @staticmethod
    def load_raw_data(source: str) -> pd.DataFrame:
        raw_data = pd.read_csv(source)
        df = raw_data.drop(columns=["OpenInt"], errors="ignore")
        return df

    def create_minimal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создание минимального набора признаков на основе вашей структуры
        """
        features_df = pd.DataFrame(index=df.index)

        # Базовые признаки (как в вашем примере)
        features_df["Close"] = df["Close"]
        features_df["Open"] = df["Open"]
        features_df["High"] = df["High"]
        features_df["Low"] = df["Low"]
        features_df["Volume"] = df["Volume"]

        # Добавляем несколько эффективных признаков
        features_df["price_change_1d"] = df["Close"].pct_change(1)
        features_df["daily_volatility"] = (df["High"] - df["Low"]) / df["Open"]

        # Целевая переменная - цена закрытия на следующий день
        features_df["target_close"] = df["Close"].shift(-1)

        # Удаляем NaN
        features_df = features_df.dropna()

        return features_df

    def prepare_data(self, df: pd.DataFrame, test_rate: float = 0.3) -> dict:
        """
        Подготовка данных в формате вашего примера
        """
        # Создаем признаки
        features_df = self.create_minimal_features(df)

        # Берем только фичи (исключая цель)
        feature_columns = [col for col in features_df.columns if col != "target_close"]
        features = features_df[feature_columns].values
        targets = features_df[["target_close"]].values

        n = len(features)
        test_size = int(n * test_rate)
        train_size = n - test_size

        # Разделяем исходные данные
        x_train = features[:train_size]
        x_test = features[train_size:]
        y_train = targets[:train_size]
        y_test = targets[train_size:]

        # Нормализация данных по обучающей выборке
        x_train_N = self.feature_scaler.fit_normalize(x_train)
        y_train_N = self.targets_scaler.fit_normalize(y_train)

        x_test_N = self.feature_scaler.normalize(x_test)
        y_test_N = self.targets_scaler.normalize(y_test)

        # Разбиваем на входы и выходы для временных рядов
        x_train_N = x_train_N[:-1]  # Признаки дня t
        y_train_N = y_train_N[1:]  # Цель дня t+1

        x_test_N = x_test_N[:-1]  # Признаки дня t
        y_test_N = y_test_N[1:]  # Цель дня t+1

        return {
            "x_train_N": x_train_N,
            "y_train_N": y_train_N,
            "x_test_N": x_test_N,
            "y_test_N": y_test_N,
            "train_size": train_size,
            "test_size": test_size,
            "feature_names": feature_columns,
        }

    def denormalize_predictions(self, predictions_N: np.ndarray) -> np.ndarray:
        """Обратное преобразование предсказаний"""
        return self.targets_scaler.denormalize(predictions_N.reshape(-1, 1))

    def denormalize_targets(self, targets_N: np.ndarray) -> np.ndarray:
        """Обратное преобразование целей"""
        return self.targets_scaler.denormalize(targets_N)
