from typing import Optional

import numpy as np

from .activation import ActivationProtocol, LinearActivation


class Layer:
    inputs: np.ndarray
    states: np.ndarray

    def __init__(self, activation: ActivationProtocol):
        self.activation = activation


class HiddenLayer(Layer):

    def __init__(self, neurons: int, activation: ActivationProtocol):
        super().__init__(activation)
        self.neurons = neurons


class OutputLayer(Layer):

    def __init__(self, activation: Optional[ActivationProtocol] = None):
        super().__init__(activation or LinearActivation())
