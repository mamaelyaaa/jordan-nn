from jordan.activation import ActivationProtocol


class Layer:
    def __init__(self, activation: ActivationProtocol, fictive: bool):
        self.activation = activation
        self.fictive = fictive


class HiddenLayer(Layer):

    def __init__(
        self, neurons: int, activation: ActivationProtocol, fictive: bool = True
    ):
        super().__init__(activation, fictive)
        self.neurons = neurons


class OutputLayer(Layer):
    def __init__(
        self, outputs: int, activation: ActivationProtocol, fictive: bool = True
    ):
        super().__init__(activation, fictive)
        self.outputs = outputs
