import numpy as np

from jordan.activation import ActivationProtocol


class Layer:
    inputs: np.ndarray
    states: np.ndarray

    def __init__(self, neurons: int, activation: ActivationProtocol):
        self.neurons = neurons
        self.activation = activation


class HiddenLayer(Layer):

    def __init__(self, neurons: int, activation: ActivationProtocol):
        super().__init__(neurons, activation)


class OutputLayer(Layer):

    def __init__(self, neurons: int, activation: ActivationProtocol):
        super().__init__(neurons, activation)
