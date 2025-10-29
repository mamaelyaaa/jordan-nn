import numpy as np

from jordan.activation import SigmoidActivation

# Входные нейроны
k = 2
# Скрытый слой
m = 2
# Выходной слой (и Контекст)
n = 2

# Скорость обучения
lr = 1

# Функция активации
activation = SigmoidActivation()


X = np.array(
    [
        [0, 0],
        [1, 0.75],
        [0.5, 1],
        [0.75, 0.25],
        [0.25, 0.5],
        [0, 0.25],
        [0.75, 1],
        [1, 0.5],
        [0.5, 0],
        [0.25, 0.75],
    ]
)

y = np.array(
    [
        [0, 1],
        [0.875, 0.75],
        [0.75, 0.5],
        [0.5, 0.375],
        [0.375, 0.5],
        [0.125, 0.75],
        [0.875, 0.75],
        [0.75, 0.5],
        [0.25, 0.5],
        [0.5, 0.375],
    ]
)

# Контекст
c = np.array([0, 0])

# 1. Инициализируем все веса

# От входного до скрытого
W_ih = np.random.uniform(-1, 1, size=(m, k + 1))

# !!! От контекста до скрытого слоя (без фиктивной единицы)
W_ch = np.random.uniform(-1, 1, size=(m, n))

# От скрытого до выходного
W_ho = np.random.uniform(-1, 1, size=(n, m + 1))


X_fictive = np.array([np.append(1, X[i]) for i in range(len(X))])
epoch = 0

for epoch in range(len(X)):
    # Прямой проход
    # print("=== Прямой проход ===")
    # 2. Рассчитываем состояния нейронов скрытого слоя:

    S_h: np.ndarray = np.dot(W_ih, X_fictive[epoch]) + np.dot(W_ch, c)
    assert S_h.shape == (m,)

    # print(f"- Вектор состояний скрытого слоя {S_h = }")

    # 3. Рассчитываем значения выходов нейронов скрытого слоя:

    H_out: np.ndarray = activation.calculate(S_h)
    assert H_out.shape == (m,)

    # print(f"- Вектор выходов скрытого слоя {H_out = }")

    # 4. Расширяем вектор значений выходов нейронов скрытого слоя, добавляя единицу в начало вектора:

    H_fictive = np.append(1, S_h)
    assert H_fictive.shape == (m + 1,)

    # print(f"- Входные данные для выходного слоя {H_fictive = }")

    # 5. Рассчитываем состояния нейронов выходного слоя
    S_y: np.ndarray = np.dot(W_ho, H_fictive)
    assert S_y.shape == (n,)

    # print(f"- Вектор состояний нейронов выходного слоя {S_y = }")

    # 6. Рассчитываем значения выходов нейронов выходного слоя:

    y_exp: np.ndarray = activation.calculate(S_y)
    assert y_exp.shape == (n,)

    # print(f"- Вектор выходов {y_exp = }\n")

    # Обратное распространение ошибки
    # print("=== Обратное распространение ошибки ===")

    # 1. Рассчитываем вектор значений ошибок выходов сети относительно обучающего примера:
    D: np.ndarray = y[epoch] - y_exp
    assert D.shape == (n,)

    # print(f"- Вектор значений ошибок выходов сети: {D = }")

    # 2. Рассчитываем значения невязок нейронов выходного слоя:
    # print("Невязки:")

    R_y: np.ndarray = D * activation.saturation * y_exp * (1 - y_exp)
    assert R_y.shape == (n,)

    # print(f"- Нейронов выходного слоя: {R_y = }")

    # 3. Рассчитываем значения невязок нейронов скрытого слоя:

    R_h: np.ndarray = (
        np.dot(W_ho[:, :-1].T, R_y) * activation.saturation * H_out * (1 - H_out)
    )
    assert R_h.shape == (m,)

    # print(f"- Нейронов скрытого слоя: {R_h = }")

    # 4. Рассчитываются поправки весовых коэффициентов:

    W_ho += lr * np.outer(R_y, H_fictive)
    assert W_ho.shape == (n, m + 1)

    # print(f"Матрица весов между скрытым и выходным слоем:\n {W_ho}")

    W_ch += lr * np.outer(R_h, c)
    assert W_ch.shape == (m, n)

    # print(f"Матрица весов между контекстом и скрытым слоем:\n {W_ch}")

    W_ih += lr * np.outer(R_h, X_fictive[epoch])
    assert W_ih.shape == (m, k + 1)

    # print(f"Матрица весов между входным и скрытым слоем:\n {W_ih}")

    # Обновляем контекст
    c = y_exp

    print(f"Эпоха {epoch + 1} завершена! MAE: {sum(D) / len(D)}")
