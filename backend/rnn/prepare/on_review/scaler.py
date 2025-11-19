from typing import Protocol, Tuple, Optional
import numpy as np


class ScalerProtocol(Protocol):
    def fit(self, series: np.ndarray) -> None:
        pass

    def normalize(self, series: np.ndarray, fit: bool = True) -> np.ndarray:
        pass

    def denormalize(self, series: np.ndarray) -> np.ndarray:
        pass


class EmptyScaler:

    def fit(self, series: np.ndarray) -> None:
        return None

    def normalize(self, series: np.ndarray, fit: bool = True) -> np.ndarray:
        return series

    def denormalize(self, series: np.ndarray) -> np.ndarray:
        return series


class MinMaxScaler:
    def __init__(self, diapason: Tuple[float, float] = (0, 1)):
        self._diapason_min: float = min(diapason)
        self._diapason_max: float = max(diapason)
        self._min: Optional[float] = None
        self._max: Optional[float] = None
        self._fitted: bool = False

    def fit(self, series: np.ndarray) -> None:
        self._min = series.min(axis=0)
        self._max = series.max(axis=0)
        self._fitted = True

    def normalize(self, series: np.ndarray, fit: bool = True) -> np.ndarray:
        min_val: float
        max_val: float

        if fit:
            self.fit(series)

        if self._fitted:
            min_val = self._min
            max_val = self._max
        else:
            raise ValueError("Scaler must be fitted before normalization")

        series_range = max_val - min_val

        if np.any(series_range == 0):
            midpoint = (self._diapason_min + self._diapason_max) / 2
            return np.full_like(series, midpoint, dtype=float)

        normalized = (series - min_val) / series_range
        normalized = (
            normalized * (self._diapason_max - self._diapason_min) + self._diapason_min
        )

        return normalized

    def denormalize(self, series: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ValueError("Scaler must be fitted before denormalization")

        series_range = self._max - self._min

        if np.any(series_range == 0):
            midpoint = (self._diapason_min + self._diapason_max) / 2
            return np.full_like(series, midpoint, dtype=float)

        denormalized = (series - self._diapason_min) / (
            self._diapason_max - self._diapason_min
        )
        denormalized = denormalized * series_range + self._min

        return denormalized

    @property
    def minmax(self):
        return self._min, self._max


class StandardScaler:
    def __init__(self):
        self._mean: Optional[float] = None
        self._std: Optional[float] = None
        self._fitted: bool = False

    def fit(self, series: np.ndarray) -> None:
        self._mean = series.mean(axis=0)
        self._std = series.std(axis=0)
        self._fitted = True

    def normalize(self, series: np.ndarray, fit: bool = True) -> np.ndarray:
        mean_val: float
        std_val: float

        if fit:
            self.fit(series)

        if self._fitted:
            mean_val = self._mean
            std_val = self._std

        else:
            raise ValueError("Scaler must be fitted before normalization")

        normalized = (series - mean_val) / std_val

        return normalized

    def denormalize(self, series: np.ndarray) -> np.ndarray:
        mean_val: float
        std_val: float

        if not self._fitted:
            raise ValueError("Scaler must be fitted before denormalization")

        denormalized = series * self._std + self._mean

        return denormalized

    @property
    def mean_std(self):
        return self._mean, self._std


class PercentDevScaler:

    def __init__(self, series: Optional[np.ndarray] = None):
        self._series: Optional[np.ndarray] = series
        self._fitted: bool = series is not None

    def fit(self, series: np.ndarray) -> None:
        self._series = series.astype(float)
        self._fitted = True

    def normalize(self, series: np.ndarray, fit: bool = True) -> np.ndarray:

        if fit:
            self._series = series
            self._fitted = True

        if not self._fitted:
            raise ValueError(
                "Scaler must be fitted with price_series before normalization."
            )

        result = (series - self._series) / self._series
        return result

    def denormalize(self, series: np.ndarray) -> np.ndarray:

        if not self._fitted:
            raise ValueError("Scaler must be fitted before denormalization")

        return self._series * (1.0 + series)
