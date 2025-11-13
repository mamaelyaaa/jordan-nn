import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from network.training.activation import (
    LinearActivation,
    TanhActivation,
)
from network.jordan import JordanRNN
from network.training.layers import HiddenLayer, OutputLayer

# Загрузка данных
data = pd.read_csv("data.csv")
df = data.drop(columns=["OpenInt"])

# Используем ВСЕ колонки как входные данные (Open, High, Low, Close, Volume)
x = df[["Close", "High", "Low", "Volume"]].values
y = df["Close"].values  # Цена закрытия как целевая переменная

# x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
# y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

# Параметры
test_rate = 0.25
test_size = int(len(x) * test_rate)
train_size = len(x) - test_size

# РАЗДЕЛЕНИЕ ДАННЫХ ДО НОРМАЛИЗАЦИИ (исправление утечки данных)
sep_idx = train_size - 1

# Разделяем исходные данные
x_train = x[:sep_idx]
y_train = y[1 : sep_idx + 1]

x_test = x[sep_idx + 1 : len(x)]
y_test = y[sep_idx + 2 : len(x)]


assert len(x_train) + len(x_test) + 1 == len(x)
assert len(y_train) + len(y_test) + 2 == len(y)

# print("Размеры исходных данных:")
# print(f"Обучающие: {x_train.shape}, {y_train.shape}")
# print(f"Тестовые: {x_test.shape}, {y_test.shape}")

# НОРМАЛИЗАЦИЯ ТОЛЬКО НА ОСНОВЕ ОБУЧАЮЩИХ ДАННЫХ
# Нормализация x
# ПРАВИЛЬНО - использовать только обучающие данные
x_min = x_train.min(axis=0)  # минимумы только из обучающей выборки
x_max = x_train.max(axis=0)  # максимумы только из обучающей выборки

# Защита от деления на ноль
x_range = x_max - x_min

x_train_N = (x_train - x_min) / x_range
x_test_N = (x_test - x_min) / x_range

x_N = (x - x_min) / x_range

# Нормализация y
y_min = y.min()
y_max = y.max()
y_range = y_max - y_min

# Защита от деления на ноль
if y_range == 0:
    y_range = 1.0

y_train_N = (y_train - y_min) / y_range
y_test_N = (y_test - y_min) / y_range

print(f"\nПараметры нормализации:")
print(f"x min: {x_min}")
print(f"x max: {x_max}")
print(f"y min: {y_min:.4f}, y max: {y_max:.4f}")


# СОЗДАНИЕ И ОБУЧЕНИЕ МОДЕЛИ
hidden_layer = HiddenLayer(neurons=6, activation=TanhActivation())
output_layer = OutputLayer(neurons=1, activation=LinearActivation())
network = JordanRNN(hidden_layer, output_layer, learning_rate=0.001)

print("\nОбучение модели...")
mse_history = network.train(x_train_N, y_train_N, epochs=5000, verbose=True)

# ПРОГНОЗИРОВАНИЕ
print("\nПрогнозирование на тестовой выборке...")

y_calc_N = []

for i in range(len(x_N) - 1):
    predict = network.predict(x_N[i])
    y_calc_N.append(predict[0])

y_calc_N = np.array(y_calc_N)

# ДЕНОРМАЛИЗАЦИЯ ПРЕДСКАЗАНИЙ И ТЕСТОВЫХ ДАННЫХ
y_calc = y_calc_N * y_range + y_min

# ВЫЧИСЛЕНИЕ ОШИБОК
# test_mse = np.mean((y_test_denormalized - predictions_denormalized) ** 2)
# errors = y_test_denormalized - predictions_denormalized
#
# print(f"\nРезультаты:")
# print(f"MSE на тестовой выборке: {test_mse:.6f}")
# print(f"RMSE на тестовой выборке: {np.sqrt(test_mse):.6f}")
# print(f"MAE на тестовой выборке: {np.mean(np.abs(errors)):.6f}")
# print(f"Средняя ошибка: {np.mean(errors):.6f}")
# print(f"Стандартное отклонение ошибок: {np.std(errors):.6f}")
# print(f"Min ошибка: {np.min(errors):.6f}")
# print(f"Max ошибка: {np.max(errors):.6f}")

# Проверка денормализации
print(f"\nПроверка денормализации:")
print(
    f"Диапазон нормализованных предсказаний: [{y_calc_N.min():.4f}, {y_calc_N.max():.4f}]"
)
print(
    f"Диапазон денормализованных предсказаний: [{y_calc.min():.4f}, {y_calc.max():.4f}]"
)
print(f"Диапазон реальных значений: [{y_test.min():.4f}, {y_test.max():.4f}]")

# Проверка на NaN/Inf
print(f"\nПроверка корректности данных:")
print(f"NaN в предсказаниях: {np.any(np.isnan(y_calc))}")
print(f"Inf в предсказаниях: {np.any(np.isinf(y_calc))}")
print(f"NaN в реальных значениях: {np.any(np.isnan(y_test))}")

# ГРАФИКИ
plt.figure()
plt.subplot()
all_dates = range(len(y))

plt.plot(all_dates, y, label="Все данные", alpha=0.7, linewidth=1)
plt.axvline(
    x=sep_idx,
    color="red",
    linestyle="--",
    label="Разделение train/test",
    alpha=0.7,
)

# # Вычисляем индексы для отображения предсказаний
# plt.plot(
#     range(sep_idx + 2, len(x)),
#     y_test,
#     label="Тест реальные",
#     linewidth=2,
#     color="green",
#     marker="o",
# )

# plt.plot(
#     range(sep_idx + 1, len(x) - 1),
#     x_test,
#     label="Тест реальные",
#     linewidth=2,
#     color="green",
#     marker="o",
# )

plt.plot(
    all_dates[1:],
    y_calc,
    label="Модель",
    linestyle="--",
    linewidth=2,
    color="orange",
    marker="o",
)

plt.title("Общий вид данных с предсказаниями")
plt.xlabel("Временной шаг")
plt.ylabel("Цена закрытия")
plt.legend()
plt.grid(True)


#
# # 4. График ошибок
# plt.subplot(2, 2, 4)
# plt.plot(test_dates, errors, label="Ошибки", color="red", linewidth=1)
# plt.axhline(y=0, color="black", linestyle="-", alpha=0.5)
# plt.axhline(
#     y=np.mean(errors),
#     color="blue",
#     linestyle="--",
#     label=f"Средняя ошибка: {np.mean(errors):.4f}",
# )
# plt.fill_between(test_dates, errors, 0, alpha=0.3, color="red")
# plt.title("График ошибок предсказания")
# plt.xlabel("Временной шаг")
# plt.ylabel("Ошибка")
# plt.legend()
# plt.grid(True)
#
plt.tight_layout()
plt.show()
