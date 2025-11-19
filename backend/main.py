from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rnn.jordan import JordanRNN
from rnn.prepare.test_loader import DataLoader, Dataset
from rnn.structure.activation import (
    LinearActivation,
    SigmoidActivation,
    TanhActivation,
)
from rnn.structure.layers import HiddenLayer, OutputLayer

DATA_DIR = Path(__file__).parent / "data"


def restore_prices(start_price: float, log_returns: np.ndarray):
    """
    Восстанавливает цены по лог-доходностям.
    """
    prices = [start_price]
    for lr in log_returns:
        prices.append(prices[-1] * np.exp(lr))
    return np.array(prices[1:])


if __name__ == "__main__":
    # Инициализация загрузчика данных
    loader = DataLoader()

    # Загрузка и подготовка данных
    raw_data = loader.load_raw_data(DATA_DIR / "apple.csv")
    print(f"Загружено данных: {len(raw_data)} строк")

    # Подготовка данных
    dataset: Dataset = loader.prepare_data(
        raw_data,
        features=[
            "close_rel",
            "pct_return",
            "low_rel",
            "high_rel",
            "volatility_abs",
            "ema_14",
        ],
        target=["target_close_1d"],
        test_rate=0.3,
    )

    # Создание и обучение модели
    network = JordanRNN(
        HiddenLayer(neurons=64, activation=TanhActivation()),
        OutputLayer(activation=LinearActivation()),
        learning_rate=0.003,
    )

    print("Обучение модели...")

    mse_history = network.train(
        training=dataset.x_train_N,
        targets=dataset.y_train_N,
        epochs=1000,
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
    #
    # # --- Восстановление цен ---
    # train_start = raw_data["Close"].iloc[1]
    # train_close_predict = restore_prices(train_start, train_predictions.flatten())
    #
    # # начальная точка для теста — последняя цена train
    # test_start = raw_data["Close"].iloc[dataset.train_size - 1]
    # test_close_predict = restore_prices(test_start, test_predictions.flatten())
    #
    # print("Нормализованн    ые предсказания (первые 10):", train_predictions_N[:10])
    # print("Денормализованные лог-доходности:", train_predictions[:10])

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

    plt.axvline(x=dataset.train_size, color="black", linestyle="--", label="Train/Test")

    plt.xlabel("Дни")
    plt.ylabel("Цена Close")
    plt.title("Предсказания цены Close - общий вид")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
