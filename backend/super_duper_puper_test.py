import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from jordan.activation import (
    SigmoidActivation,
    LinearActivation,
    TanhActivation,
    ReLUActivation,
)
from jordan.jordan import JordanRNN
from jordan.layers import HiddenLayer, OutputLayer

# Загрузка данных
data = pd.read_csv("data.csv")
df = data.drop(columns=["OpenInt"])

# Используем ВСЕ колонки как входные данные (Open, High, Low, Close, Volume)
features = df[["Open", "High", "Low", "Close", "Volume"]].values
target = df["Close"].values  # Цена закрытия как целевая переменная

# Параметры
sequence_length = 3
test_size = 0.25

# РАЗДЕЛЕНИЕ ДАННЫХ ДО НОРМАЛИЗАЦИИ (исправление утечки данных)
split_index_raw = int((1 - test_size) * len(features))

# Разделяем исходные данные
features_train_raw = features[:split_index_raw]
features_test_raw = features[split_index_raw:]
target_train_raw = target[:split_index_raw]
target_test_raw = target[split_index_raw:]

print("Размеры исходных данных:")
print(f"Обучающие: {features_train_raw.shape}, {target_train_raw.shape}")
print(f"Тестовые: {features_test_raw.shape}, {target_test_raw.shape}")

# НОРМАЛИЗАЦИЯ ТОЛЬКО НА ОСНОВЕ ОБУЧАЮЩИХ ДАННЫХ
# Нормализация features
features_min = features_train_raw.min(axis=0)
features_max = features_train_raw.max(axis=0)

# Защита от деления на ноль
features_range = features_max - features_min
features_range[features_range == 0] = 1.0

features_train_normalized = (features_train_raw - features_min) / features_range
features_test_normalized = (features_test_raw - features_min) / features_range

# Нормализация target
target_min = target_train_raw.min()
target_max = target_train_raw.max()
target_range = target_max - target_min

# Защита от деления на ноль
if target_range == 0:
    target_range = 1.0

target_train_normalized = (target_train_raw - target_min) / target_range
target_test_normalized = (target_test_raw - target_min) / target_range

print(f"\nПараметры нормализации:")
print(f"Features min: {features_min}")
print(f"Features max: {features_max}")
print(f"Target min: {target_min:.4f}, Target max: {target_max:.4f}")


# ПОДГОТОВКА ПОСЛЕДОВАТЕЛЬНОСТЕЙ
def create_sequences(features_data, target_data, seq_length, prediction_horizon=1):
    """
    Создание последовательностей с явным указанием горизонта прогноза

    Parameters:
    - prediction_horizon: на сколько дней вперед предсказываем (по умолчанию 1)
    """
    X = []
    y = []
    for i in range(len(features_data) - seq_length - prediction_horizon + 1):
        # Вход: последовательность из seq_length дней
        X.append(features_data[i : i + seq_length].flatten())
        # Цель: значение через prediction_horizon дней после последовательности
        y.append(target_data[i + seq_length + prediction_horizon - 1])
    return np.array(X), np.array(y)


# Использование:
X_train, y_train = create_sequences(
    features_train_normalized,
    target_train_normalized,
    sequence_length,
    prediction_horizon=1,
)

# Создаем последовательности для тестовых данных
X_test, y_test = create_sequences(
    features_test_normalized, target_test_normalized, sequence_length
)

X_test = X_test[1 : len(X_test) - 1]
y_test = y_test[1 : len(y_test) - 1]

print(f"\nРазмерность данных после создания последовательностей:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# Проверка нормализации
print(f"\nПроверка нормализации:")
print(f"X_train диапазон: [{X_train.min():.4f}, {X_train.max():.4f}]")
print(f"y_train диапазон: [{y_train.min():.4f}, {y_train.max():.4f}]")
print(f"X_test диапазон: [{X_test.min():.4f}, {X_test.max():.4f}]")
print(f"y_test диапазон: [{y_test.min():.4f}, {y_test.max():.4f}]")

# СОЗДАНИЕ И ОБУЧЕНИЕ МОДЕЛИ
hidden_layer = HiddenLayer(neurons=8, activation=SigmoidActivation())
output_layer = OutputLayer(neurons=1, activation=LinearActivation())
network = JordanRNN(hidden_layer, output_layer, learning_rate=0.01)

print("\nОбучение модели...")
mse_history = network.train(X_train, y_train, epochs=1000, verbose=True)

# ПРОГНОЗИРОВАНИЕ
print("\nПрогнозирование на тестовой выборке...")
predictions_normalized = []
for i in range(len(X_test)):
    pred = network.predict(X_test[i])
    predictions_normalized.append(pred[0])

predictions_normalized = np.array(predictions_normalized)

# ДЕНОРМАЛИЗАЦИЯ ПРЕДСКАЗАНИЙ И ТЕСТОВЫХ ДАННЫХ
predictions_denormalized = predictions_normalized * target_range + target_min
y_test_denormalized = y_test * target_range + target_min

# ВЫЧИСЛЕНИЕ ОШИБОК
test_mse = np.mean((y_test_denormalized - predictions_denormalized) ** 2)
errors = y_test_denormalized - predictions_denormalized

print(f"\nРезультаты:")
print(f"MSE на тестовой выборке: {test_mse:.6f}")
print(f"RMSE на тестовой выборке: {np.sqrt(test_mse):.6f}")
print(f"MAE на тестовой выборке: {np.mean(np.abs(errors)):.6f}")
print(f"Средняя ошибка: {np.mean(errors):.6f}")
print(f"Стандартное отклонение ошибок: {np.std(errors):.6f}")
print(f"Min ошибка: {np.min(errors):.6f}")
print(f"Max ошибка: {np.max(errors):.6f}")

# Проверка денормализации
print(f"\nПроверка денормализации:")
print(
    f"Диапазон нормализованных предсказаний: [{predictions_normalized.min():.4f}, {predictions_normalized.max():.4f}]"
)
print(
    f"Диапазон денормализованных предсказаний: [{predictions_denormalized.min():.4f}, {predictions_denormalized.max():.4f}]"
)
print(
    f"Диапазон реальных значений: [{y_test_denormalized.min():.4f}, {y_test_denormalized.max():.4f}]"
)

# Проверка на NaN/Inf
print(f"\nПроверка корректности данных:")
print(f"NaN в предсказаниях: {np.any(np.isnan(predictions_denormalized))}")
print(f"Inf в предсказаниях: {np.any(np.isinf(predictions_denormalized))}")
print(f"NaN в реальных значениях: {np.any(np.isnan(y_test_denormalized))}")

# ГРАФИКИ
plt.figure()

# # 1. История обучения
# plt.subplot(2, 2, 1)
# plt.plot(mse_history)
# plt.title("История обучения (MSE)")
# plt.xlabel("Эпоха")
# plt.ylabel("MSE")
# plt.grid(True)
# plt.yscale("log")  # Логарифмическая шкала для лучшей визуализации

# # 2. Реальные vs Предсказанные значения
# plt.subplot(2, 2, 2)
# test_dates = range(len(y_test_denormalized))
# plt.plot(
#     test_dates,
#     y_test_denormalized,
#     label="Реальные",
#     marker="o",
#     markersize=3,
#     linewidth=1,
# )
# plt.plot(
#     test_dates,
#     predictions_denormalized,
#     label="Предсказания",
#     marker="s",
#     markersize=3,
#     linewidth=1,
# )
# plt.title("Реальные vs Предсказанные значения (тестовая выборка)")
# plt.xlabel("Временной шаг")
# plt.ylabel("Цена закрытия")
# plt.legend()
# plt.grid(True)
#
# 3. Все данные цены закрытия
plt.subplot()
all_dates = range(len(target))
train_end = split_index_raw + sequence_length

plt.plot(all_dates, target, label="Все данные", alpha=0.7, linewidth=1)
plt.axvline(
    x=split_index_raw,
    color="red",
    linestyle="--",
    label="Разделение train/test",
    alpha=0.7,
)

# Вычисляем индексы для отображения предсказаний
test_start_idx = split_index_raw + sequence_length
test_indices = range(test_start_idx, test_start_idx + len(predictions_denormalized))

plt.plot(
    test_indices, y_test_denormalized, label="Тест реальные", linewidth=2, color="green"
)
plt.plot(
    test_indices,
    predictions_denormalized,
    label="Тест предсказания",
    linestyle="--",
    linewidth=2,
    color="orange",
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

# Дополнительная статистика
print(f"\nДополнительная статистика:")
print(f"Средняя цена в тестовой выборке: {np.mean(y_test_denormalized):.4f}")
print(
    f"Относительная ошибка (MSE/средняя цена): {(test_mse / np.mean(y_test_denormalized)) * 100:.2f}%"
)
print(
    f"Точность предсказания: {100 * (1 - np.mean(np.abs(errors)) / np.mean(y_test_denormalized)):.2f}%"
)
