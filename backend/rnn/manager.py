import dataclasses
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from rnn.jordan import JordanRNN
from rnn.prepare import FeaturesType, TargetType
from rnn.prepare.loader import Dataset, DataLoader
from rnn.structure.activation import (
    LinearActivation,
    TanhActivation,
    ActivationProtocol,
)
from rnn.structure.layers import HiddenLayer, OutputLayer
from rnn.structure.regularizer import RegularizerProtocol, NoRegularizer


class RNNManager:
    """Класс, объединяющий подготовку данных, обучение и предсказания."""

    def __init__(
        self,
        target: list[TargetType],
        features: list[FeaturesType],
        learning_rate: float = 0.003,
        hidden_activation: ActivationProtocol = TanhActivation(),
        hidden_neurons: int = 128,
        regularization: RegularizerProtocol = NoRegularizer(),
    ):
        self.features = features
        self.target = target
        self.learning_rate = learning_rate
        self.hidden_activation = hidden_activation
        self.hidden_neurons = hidden_neurons
        self.regularization = regularization

        # Компоненты
        self.loader = DataLoader()
        self.dataset: Optional[Dataset] = None

        # Модель
        self.model = JordanRNN(
            hidden_layer=HiddenLayer(
                neurons=hidden_neurons, activation=hidden_activation
            ),
            output_layer=OutputLayer(activation=LinearActivation()),
            learning_rate=learning_rate,
            regularization=regularization,
        )

    def load_and_prepare(
        self, source: Path | str, test_rate: float = 0.3
    ) -> pd.DataFrame:
        """
        Загружает данные и применяет подготовку
        :param source: Путь до файла
        :param test_rate: Процент тестовой выборки от общей
        :return: Сырой датафрейм
        """

        raw_data = self.loader.load_raw_data(source)
        print(f"Загружено данных: {len(raw_data)} строк")

        self.dataset = self.loader.prepare_data(
            raw_data,
            features=self.features,
            target=self.target,
            test_rate=test_rate,
        )
        return raw_data

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
