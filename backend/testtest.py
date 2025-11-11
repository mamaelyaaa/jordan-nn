# ------------------------------------------------------
# 1. Загрузка данных
# ------------------------------------------------------
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from jordan.activation import SigmoidActivation
from jordan.jordan import JordanRNN
from jordan.layers import HiddenLayer, OutputLayer

df = pd.read_csv("data.csv")

# удалить OpenInt
df = df.drop(columns=["OpenInt"])

# берём только Close
features = df[["Open", "High", "Low", "Close", "Volume"]].values
target = df["Close"].values  # Цена закрытия как целевая переменная

# нормализация
mean = np.mean(features)
std = np.std(features)
features_norm = (features - mean) / std

# ------------------------------------------------------
# 2. Формирование выборки (каждое X[t] -> Y[t] = close[t+1])
# ------------------------------------------------------
X = features_norm[:-1].reshape(-1, 1)
Y = features_norm[1:].reshape(-1, 1)

# ------------------------------------------------------
# 3. Создание модели
# ------------------------------------------------------
h_layer = HiddenLayer(activation=SigmoidActivation(), neurons=5)
o_layer = OutputLayer(activation=SigmoidActivation(), neurons=1)

model = JordanRNN(h_layer, o_layer, learning_rate=0.01)

# ------------------------------------------------------
# 4. Обучение
# ------------------------------------------------------
mse_hist = model.train(X, Y, epochs=2000)

# ------------------------------------------------------
# 5. Предсказания
# ------------------------------------------------------
pred = model.predict(X)

# денормализация
pred_real = pred * std + mean
close_real = features_norm[:-1]

# ------------------------------------------------------
# 6. Графики
# ------------------------------------------------------
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(close_real, label="Real Close")
plt.plot(pred_real, label="Predicted")
plt.legend()
plt.title("Реальные vs Предсказанные")

plt.subplot(2, 2, 2)
plt.plot(mse_hist)
plt.title("MSE по эпохам")

plt.subplot(2, 2, 3)
plt.plot(pred_real - close_real)
plt.title("Ошибка")

plt.tight_layout()
plt.show()
