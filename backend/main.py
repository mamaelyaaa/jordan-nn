import numpy as np

from jordan.activation import SigmoidActivation
from jordan.network import JordanNetwork, Layer

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
        x,
        y,
        hid_layer=Layer(neurons=5, activation=SigmoidActivation()),
        out_layer=Layer(neurons=2, activation=SigmoidActivation()),
        learning_rate=0.3,
    )

    jordan_nn.train(x, y, epochs=3000)
