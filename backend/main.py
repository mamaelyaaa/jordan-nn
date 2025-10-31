import numpy as np

from jordan.activation import SigmoidActivation, ReLUActivation, TanhActivation
from jordan.layers import HiddenLayer, OutputLayer
from jordan.network import JordanNetwork

if __name__ == "__main__":

    x = np.array(
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

    jordan_nn = JordanNetwork(
        hid_layer=HiddenLayer(
            neurons=4,
            activation=TanhActivation(saturation=1.5),
        ),
        out_layer=OutputLayer(
            outputs=len(y[0]),
            activation=SigmoidActivation(),
        ),
        learning_rate=0.2,
    )

    jordan_nn.train(x, y, epochs=5000, verbose=True)

    print("Ожидаемые vs Предсказанные:")
    for i in range(len(x)):
        pred = jordan_nn.predict(x[i : i + 1])[0]
        print(f"Input: {x[i]} | Expected: {y[i]} | Predicted: {pred}")

    total_error = 0
    for i in range(len(x)):
        pred = jordan_nn.predict(x[i : i + 1])[0]
        error = np.mean(np.abs(y[i] - pred))
        total_error += error
    print(f"Средняя ошибка: {total_error/len(x):.4f}")
