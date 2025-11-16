import matplotlib.pyplot as plt
import numpy as np

from network.training.jordan import JordanRNN
from loader import DataLoader
from network.training import SigmoidActivation, TanhActivation, ReLUActivation
from training import HiddenLayer, OutputLayer, LinearActivation

if __name__ == "__main__":
    # Инициализация загрузчика данных
    loader = DataLoader()

    # Загрузка и подготовка данных
    raw_data = loader.load_raw_data("../data.csv")
    print(f"Загружено данных: {len(raw_data)} строк")

    # Подготовка данных
    data = loader.prepare_data(raw_data, test_rate=0.3)

    print(f"Размеры данных:")
    print(f"x_train_N: {data['x_train_N'].shape}")
    print(f"y_train_N: {data['y_train_N'].shape}")
    print(f"x_test_N: {data['x_test_N'].shape}")
    print(f"y_test_N: {data['y_test_N'].shape}")

    # Создание и обучение модели
    hidden_layer = HiddenLayer(neurons=8, activation=TanhActivation())
    output_layer = OutputLayer(neurons=1, activation=LinearActivation())
    network = JordanRNN(hidden_layer, output_layer, learning_rate=0.003)

    print("Обучение модели...")

    mse_history = network.train(
        training=data["x_train_N"], targets=data["y_train_N"], epochs=2000, verbose=True
    )

    # Предсказания для всей выборки
    print("Создание предсказаний...")

    # Предсказания для обучающей выборки
    train_predictions_N = []
    for i in range(len(data["x_train_N"])):
        predict = network.predict(data["x_train_N"][i])
        train_predictions_N.append(predict.flatten()[0])

    # Предсказания для тестовой выборки
    test_predictions_N = []
    for i in range(len(data["x_test_N"])):
        predict = network.predict(data["x_test_N"][i])
        test_predictions_N.append(predict.flatten()[0])

    # Обратное преобразование к исходному масштабу
    train_predictions = loader.denormalize_predictions(np.array(train_predictions_N))
    test_predictions = loader.denormalize_predictions(np.array(test_predictions_N))

    # Индексы для графиков
    train_indices = range(1, len(train_predictions) + 1)
    test_indices = range(
        data["train_size"] + 1, data["train_size"] + len(test_predictions) + 1
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
        train_indices, train_predictions, label="Предсказания (обучение)", color="green"
    )
    plt.plot(test_indices, test_predictions, label="Предсказания (тест)", color="red")
    plt.axvline(
        x=data["train_size"],
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
