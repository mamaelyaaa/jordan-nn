import numpy as np


class StandardScaler:
    """Ручная реализация StandardScaler"""

    def __init__(self):
        self._mean = None
        self._std = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        """Вычисляем среднее и стандартное отклонение для каждого признака"""
        self._mean = np.mean(x, axis=0)  # Среднее по каждому столбцу
        self._std = np.std(x, axis=0)  # Стандартное отклонение по каждому столбцу
        return self

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Применяем нормализацию: (X - mean) / std"""
        if self._mean is None or self._std is None:
            raise ValueError("Сначала нужно вызвать fit()")
        return (x - self._mean) / self._std

    def fit_normalize(self, x: np.ndarray) -> np.ndarray:
        """fit + normalize в одном методе"""
        return self.fit(x).normalize(x)

    def denormalize(self, x_normalized: np.ndarray) -> np.ndarray:
        """Обратное преобразование: X_scaled * std + mean"""
        if self._mean is None or self._std is None:
            raise ValueError("Сначала нужно вызвать fit()")
        return x_normalized * self._std + self._mean
