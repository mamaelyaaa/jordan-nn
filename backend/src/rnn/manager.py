import asyncio
import dataclasses
from pathlib import Path
from typing import Optional, Callable, Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.rnn.jordan import JordanRNN
from src.rnn.prepare import FeaturesType, TargetType
from src.rnn.prepare.loader import Dataset, DataLoader
from src.rnn.structure.activation import (
    LinearActivation,
    TanhActivation,
    ActivationProtocol,
)
from src.rnn.structure.layers import HiddenLayer, OutputLayer
from src.rnn.structure.regularizer import RegularizerProtocol, NoRegularizer


class RNNManager:
    """Класс, объединяющий подготовку данных, обучение и предсказания."""

    def __init__(
        self,
        target: Optional[list[TargetType | str]] = None,
        features: Optional[list[FeaturesType | str]] = None,
        learning_rate: float = 0.003,
        hidden_activation: Optional[ActivationProtocol] = None,
        hidden_neurons: int = 128,
        regularization: Optional[RegularizerProtocol] = None,
    ):
        self.features = features
        self.target = target
        self.learning_rate = learning_rate
        self.hidden_activation = hidden_activation or TanhActivation()
        self.hidden_neurons = hidden_neurons
        self.regularization = regularization or NoRegularizer()

        # Компоненты
        self.loader = DataLoader()
        self.dataset: Optional[Dataset] = None
        self._raw_data: Optional[pd.DataFrame] = None

        # Модель
        self.model: Optional[JordanRNN] = None

        # Модель
        # self.model: Optional[JordanRNN] = JordanRNN(
        #     hidden_layer=HiddenLayer(
        #         neurons=hidden_neurons, activation=hidden_activation
        #     ),
        #     output_layer=OutputLayer(activation=LinearActivation()),
        #     learning_rate=learning_rate,
        #     regularization=regularization,
        # )

        # TODO Сделать метод инициализации модели

    def set_config(
        self,
        target: Optional[list[TargetType | str]] = None,
        features: Optional[list[FeaturesType | str]] = None,
        learning_rate: Optional[float] = None,
        hidden_activation: Optional[ActivationProtocol] = None,
        hidden_neurons: Optional[int] = None,
        regularization: Optional[RegularizerProtocol] = None,
    ) -> None:
        """
        Установка конфигурации модели после инициализации.
        """

        if target is not None:
            self.target = target
        if features is not None:
            self.features = features
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if hidden_activation is not None:
            self.hidden_activation = hidden_activation
        if hidden_neurons is not None:
            self.hidden_neurons = hidden_neurons
        if regularization is not None:
            self.regularization = regularization

    def load_and_prepare(
        self, source: Path | str, test_rate: float = 0.3
    ) -> pd.DataFrame:
        """
        Загружает данные и применяет подготовку
        :param source: Путь до файла
        :param test_rate: Процент тестовой выборки от общей
        :return: Сырой датафрейм
        """

        self._raw_data = self.loader.load_raw_data(source)

        self.dataset = self.loader.prepare_data(
            df=self._raw_data,
            features=self.features,
            target=self.target,
            test_rate=test_rate,
        )
        return self._raw_data

    @property
    def raw_data(self) -> pd.DataFrame:
        return self._raw_data

    @raw_data.setter
    def raw_data(self, data: pd.DataFrame) -> None:
        self._raw_data = data

    def load_data(self, source: str | Path) -> None:
        self._raw_data = self.loader.load_raw_data(source)
        return

    def prepare_data(self, raw_data: pd.DataFrame, test_rate: float = 0.3) -> "Dataset":
        if raw_data is None:
            raise ValueError("Нет данных для подготовки")

        self.dataset = self.loader.prepare_data(
            df=raw_data,
            features=self.features,
            target=self.target,
            test_rate=test_rate,
        )
        return self.dataset

    def train(self, epochs: int = 1000) -> list[float]:
        """
        Обучение модели

        :param epochs: Количество эпох для обучения
        :return: Список MSE на каждой эпохе
        """

        if self.dataset is None:
            raise RuntimeError(
                "Данные не загружены. Необходимо вызвать load_and_prepare()."
            )

        print("Обучение модели...")

        return self.model.train(
            training=self.dataset.x_train_N,
            targets=self.dataset.y_train_N,
            epochs=epochs,
            verbose=True,
        )

    @dataclasses.dataclass
    class Predictions:
        train_predictions: np.ndarray
        test_predictions: np.ndarray

    def predict(self) -> Predictions:
        """
        Создаем предсказания по обучающей и тестовой выборке
        """

        if self.dataset is None:
            raise RuntimeError("Данные не загружены.")

        print("Создание предсказаний...")

        train_predictions_norm = self.model.predict_sequence(self.dataset.x_train_N)
        test_predictions_norm = self.model.predict_sequence(self.dataset.x_test_N)

        train_predictions = self.loader.denormalize_predictions(
            np.array(train_predictions_norm)
        )
        test_predictions = self.loader.denormalize_predictions(
            np.array(test_predictions_norm)
        )

        return self.Predictions(
            train_predictions=train_predictions,
            test_predictions=test_predictions,
        )

    def plot_predict_graphic(self, predictions: Predictions) -> None:
        """Строим графики прогнозов и обучения"""

        train_indices = range(1, len(predictions.train_predictions) + 1)
        test_indices = range(
            self.dataset.train_size + 1,
            self.dataset.train_size + len(predictions.test_predictions) + 1,
        )

        # График Общий вид
        plt.plot(
            range(len(self.dataset.df[["Close"]])),
            self.dataset.df[["Close"]],
            label="Исходные данные",
            color="blue",
            alpha=0.7,
        )
        plt.plot(
            train_indices,
            predictions.train_predictions,
            label="Предсказания (обучение)",
            color="green",
            linewidth=2,
        )
        plt.plot(
            test_indices,
            predictions.test_predictions,
            label="Предсказания (тест)",
            color="red",
            linewidth=2,
        )

        plt.axvline(
            x=self.dataset.train_size, color="black", linestyle="--", label="Train/Test"
        )

        plt.xlabel("Дни")
        plt.ylabel("Цена Close")
        plt.title("Предсказания цены Close - общий вид")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


rnn_manager = RNNManager()
