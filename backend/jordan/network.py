from jordan.activation import SigmoidActivation
from jordan.layers import Layer


class NeuralNetwork:

    def __init__(self, layers: list[Layer]):
        self.layers = layers


nn = NeuralNetwork(
    layers=[
        Layer(neurons=2, activation=SigmoidActivation(), inputs=[0.1, 0.56]),
        Layer(neurons=2, activation=SigmoidActivation(), inputs=[0.1, 0.56]),
    ]
)

print(nn.layers)
