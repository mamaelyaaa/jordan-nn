import numpy as np

from network.training.activation import SigmoidActivation

input_neurons = 2
hidden_neurons = 3
output_neurons = 2

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

Y = np.array(
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

sigmoid = SigmoidActivation()
lr = 0.1

# 1. Инициализируем все веса

# От входного до скрытого (3 нейрона в скрытом слое)
W_ih = np.random.uniform(-1, 1, size=(hidden_neurons, input_neurons + 1))

# От скрытого до выходного
W_ho = np.random.uniform(-1, 1, size=(output_neurons, hidden_neurons + 1))

# 2. Рассчитываем состояния в скрытом слое

X_fictive = np.array([np.append(1, X[i]) for i in range(len(X))])

for epoch in range(len(X) - 4):

    S_h: np.ndarray = np.dot(W_ih, X_fictive[epoch])
    assert S_h.shape == (hidden_neurons,)

    # 3. Рассчитываем выходы в скрытом слое

    H_exp: np.ndarray = sigmoid.calculate(S_h)
    assert H_exp.shape == (hidden_neurons,)

    # 4. Рассчитываем состояния в выходном слое (зависят от скрытого)

    Y_inp = np.append(1, S_h)
    assert Y_inp.shape == (hidden_neurons + 1,)

    S_y: np.ndarray = np.dot(W_ho, Y_inp)
    assert S_y.shape == (output_neurons,)

    # 5. Рассчитываем выходы

    Y_exp: np.ndarray = sigmoid.calculate(S_y)
    assert Y_exp.shape == (output_neurons,)

    # 6. Рассчитываем ошибку в выходах

    D: np.ndarray = Y[epoch] - Y_exp
    assert D.shape == (output_neurons,)

    # 7. Рассчитываем невязки выходного слоя

    R_out: np.ndarray = D * sigmoid.saturation * Y_exp * (1 - Y_exp)
    assert R_out.shape == (output_neurons,)

    R_sum: np.ndarray = np.dot(W_ho[:, :-1].T, R_out)
    assert R_sum.shape == (hidden_neurons,)

    # 8. Рассчитываем невязки скрытого слоя

    R_hid: np.ndarray = R_sum * sigmoid.saturation * H_exp * (1 - H_exp)
    assert R_hid.shape == (hidden_neurons,)

    # 9. Делаем поправки весов (сначала для скрытого, потом для выходного)

    W_ih += lr * R_hid * X_fictive[epoch]
    assert W_ih.shape == (hidden_neurons, input_neurons + 1)

    # 10. Продолжаем делать поправки

    W_ho += lr * np.outer(R_out, Y_inp)
    assert W_ho.shape == (output_neurons, hidden_neurons + 1)

    print(f"Эпоха {epoch + 1} завершена! MAE: {np.mean(np.abs(D)):.4f}")
