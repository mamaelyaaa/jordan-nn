import numpy as np
from matplotlib import pyplot as plt

from network.training.activation import (
    SigmoidActivation,
    LinearActivation,
)
from network.jordan import JordanRNN
from network.training.layers import HiddenLayer, OutputLayer

if __name__ == "__main__":
    jnn = JordanRNN(
        hidden_layer=HiddenLayer(activation=SigmoidActivation(), neurons=2),
        output_layer=OutputLayer(activation=LinearActivation(), neurons=2),
        learning_rate=0.03,
    )

    # Example training data
    training_data = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    target_data = np.array([[0.8, 0.7], [0.6, 0.5], [0.4, 0.3]])

    # Train the network
    mse_history = jnn.train(training_data, target_data, epochs=20000, verbose=True)

    # Визуализация
    plt.plot(mse_history)
    plt.title("Training MSE over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.show()
