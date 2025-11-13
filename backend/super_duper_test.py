import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from network.training.activation import SigmoidActivation, LinearActivation
from network.jordan import JordanRNN
from network.training.layers import HiddenLayer, OutputLayer

# Загрузка данных
data = pd.read_csv("data.csv")
df = data.drop(columns=["OpenInt"])

# Используем ВСЕ колонки как входные данные (Open, High, Low, Close, Volume)
features = df[["Open", "High", "Low", "Close", "Volume"]].values
target = df["Close"].values  # Цена закрытия как целевая переменная

# Нормализация данных
features_min = features.min(axis=0)
features_max = features.max(axis=0)
features_normalized = (features - features_min) / (features_max - features_min)

target_min = target.min()
target_max = target.max()
target_normalized = (target - target_min) / (target_max - target_min)

# Подготовка последовательностей
sequence_length = 3  # Уменьшил для стабильности
X = []
y = []

for i in range(len(features_normalized) - sequence_length):
    X.append(
        features_normalized[i : i + sequence_length].flatten()
    )  # Вытягиваем в один вектор
    y.append(target_normalized[i + sequence_length])

X = np.array(X)
y = np.array(y)

# Разделение на обучающую и тестовую выборки
split_index = int(0.75 * len(X))
X_train = X[:split_index]
y_train = y[:split_index]
X_test = X[split_index:]
y_test = y[split_index:]

print(f"Размерность данных:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# Создание модели с очень маленьким learning rate
hidden_layer = HiddenLayer(neurons=8, activation=SigmoidActivation())
output_layer = OutputLayer(neurons=1, activation=LinearActivation())
network = JordanRNN(hidden_layer, output_layer, learning_rate=0.01)

print("\nОбучение модели...")
mse_history = network.train(X_train, y_train, epochs=2000, verbose=True)

# Прогнозирование
predictions_normalized = []
for i in range(len(X_test)):
    pred = network.predict(X_test[i])
    predictions_normalized.append(pred[0])

# Денормализация
predictions = np.array(predictions_normalized) * (target_max - target_min) + target_min
y_test_denormalized = y_test * (target_max - target_min) + target_min

# Вычисление ошибок
test_mse = np.mean((y_test_denormalized - predictions) ** 2)
errors = y_test_denormalized - predictions

print(f"\nРезультаты:")
print(f"MSE на тестовой выборке: {test_mse:.6f}")
print(f"Средняя ошибка: {np.mean(errors):.4f}")
print(f"Стандартное отклонение ошибок: {np.std(errors):.4f}")

# ГРАФИКИ
plt.figure(figsize=(15, 10))

# 1. История обучения
plt.subplot(2, 2, 1)
plt.plot(mse_history)
plt.title("История обучения (MSE)")
plt.xlabel("Эпоха")
plt.ylabel("MSE")
plt.grid(True)

# 2. Реальные vs Предсказанные значения
plt.subplot(2, 2, 2)
test_dates = range(len(y_test_denormalized))
plt.plot(test_dates, y_test_denormalized, label="Реальные", marker="o", markersize=3)
plt.plot(test_dates, predictions, label="Предсказания", marker="s", markersize=3)
plt.title("Реальные vs Предсказанные значения")
plt.xlabel("Временной шаг")
plt.ylabel("Цена закрытия")
plt.legend()
plt.grid(True)

# 3. Все данные цены закрытия
plt.subplot(2, 2, 3)
all_dates = range(len(target))
train_end = split_index + sequence_length
plt.plot(all_dates, target, label="Все данные", alpha=0.7)
plt.axvline(x=train_end, color="red", linestyle="--", label="Начало теста")
plt.plot(
    range(train_end, len(target)),
    y_test_denormalized,
    label="Тест реальные",
    linewidth=2,
)
plt.plot(
    range(train_end, len(target)),
    predictions,
    label="Тест предсказания",
    linestyle="--",
    linewidth=2,
)
plt.title("Цена закрытия с тестовой выборкой")
plt.xlabel("Временной шаг")
plt.ylabel("Цена закрытия")
plt.legend()
plt.grid(True)

plt.show()
