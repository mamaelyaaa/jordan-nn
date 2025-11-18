import numpy as np


class StandardScaler:
    """Класс нормализации и денормализации"""

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


class PercentScaler:
    """Нормализация процентных изменений: 0 = нет изменений, 1 = изменение в 2 раза"""

    def __init__(self):
        self._epsilon = 1e-8

    @staticmethod
    def normalize(x: np.ndarray) -> np.ndarray:
        """Преобразование процентов в логарифмическую шкалу"""
        # x - процентные изменения в формате [0.01 = 1%, -0.05 = -5% и т.д.]
        # Преобразуем: log(1 + x) где 0 = нет изменений
        return np.log1p(x) / np.log(2)  # делим на log(2) чтобы 1 = удвоение

    @staticmethod
    def denormalize(x_normalized: np.ndarray) -> np.ndarray:
        """Обратное преобразование из логарифмической шкалы"""
        return np.expm1(x_normalized * np.log(2))
