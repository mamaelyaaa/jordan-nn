from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rnn.jordan import JordanRNN
from rnn.prepare.loader import DataLoader, Dataset
from rnn.structure.activation import (
    LinearActivation,
    SigmoidActivation,
    TanhActivation,
)
from rnn.structure.layers import HiddenLayer, OutputLayer

DATA_DIR = Path(__file__).parent / "data"


if __name__ == "__main__":
    # Инициализация загрузчика данных
    loader = DataLoader()

    # Загрузка и подготовка данных
    raw_data = loader.load_raw_data(DATA_DIR / "apple.csv")
    print(f"Загружено данных: {len(raw_data)} строк")

    # Подготовка данных
    dataset: Dataset = loader.prepare_data(raw_data, test_rate=0.3)

    print(f"Размеры данных:")
    print(f"x_train_N: {dataset.x_train_N.shape}")
    print(f"y_train_N: {dataset.y_train_N.shape}")
    print(f"x_test_N: {dataset.x_test_N.shape}")
    print(f"y_test_N: {dataset.y_test_N.shape}")

    # Создание и обучение модели
    network = JordanRNN(
        HiddenLayer(neurons=12, activation=TanhActivation()),
        OutputLayer(activation=LinearActivation()),
        learning_rate=0.1,
    )

    print("Обучение модели...")

    mse_history = network.train(
        training=dataset.x_train_N,
        targets=dataset.y_train_N,
        epochs=800,
        verbose=True,
    )

    # Предсказания для всей выборки
    print("Создание предсказаний...")

    # Предсказания для обучающей выборки
    train_predictions_N = network.predict_sequence(dataset.x_train_N)
    test_predictions_N = network.predict_sequence(dataset.x_test_N)

    # Обратное преобразование к исходному масштабу
    train_predictions = loader.denormalize_predictions(np.array(train_predictions_N))
    test_predictions = loader.denormalize_predictions(np.array(test_predictions_N))

    # Индексы для графиков
    train_indices = range(1, len(train_predictions) + 1)
    test_indices = range(
        dataset.train_size + 1, dataset.train_size + len(test_predictions) + 1
    )

    # График Общий вид
    plt.plot(
        range(len(raw_data)),
        raw_data["Close"],
        label="Исходные данные",
        color="blue",
        alpha=0.7,
    )
    plt.plot(
        train_indices,
        train_predictions,
        label="Предсказания (обучение)",
        color="green",
        linewidth=2,
    )
    plt.plot(
        test_indices,
        test_predictions,
        label="Предсказания (тест)",
        color="red",
        linewidth=2,
    )
    plt.axvline(
        x=dataset.train_size,
        color="black",
        linestyle="--",
        label="Разделение train/test",
    )
    plt.xlabel("Дни")
    plt.ylabel("Цена Close")
    plt.title("Предсказания цены Close - общий вид")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
