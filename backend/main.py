import numpy as np

from jordan.activation import SigmoidActivation
from jordan.layers import HiddenLayer

neurons = 3

X = np.array(
    [
        [0.0, 0.5, 0.5, 0.7],
        [0.0, 0.4, 0.4, 0.5],
        [1.0, 0.55, 0.45, 0.6],
        [1.0, 0.5, 0.4, 0.4],
        [1.0, 0.5, 0.6, 0.8],
        [0.0, 0.4, 0.4, 0.65],
    ]
)

y = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)

W = np.random.uniform(-1, 1, (neurons, X.shape[1] + 1))
X_1 = np.array([np.append(1, X[i]) for i in range(len(X))])

sigmoid = SigmoidActivation()

S = np.dot(X_1, W.T)
print(S)

y_exp = sigmoid.calculate(S[0])
print(y_exp)

err = y[0] - y_exp
print(err)

if __name__ == "__main__":

    # hl = HiddenLayer(
    #     inputs=X,
    #     weights=
    # )
    # l = Layer(
    #     inputs=X[0],
    #     neurons=2,
    #     activation=SigmoidActivation(),
    # )
    pass
    # print(l.neurons)
    # print(l.calculate_statements())
    # print(l.calculate_outputs())
