from jordan.activation import SigmoidActivation
from jordan.layers import Layer

x: list[list[float]] = [[0.0, 0.4], [0.5, 0.5]]
y: list[float] = [0, 1]

if __name__ == "__main__":
    l = Layer(
        inputs=x[0],
        neurons_count=1,
        activation=SigmoidActivation(),
    )

    print(l.neurons)
    print(l.calculate_statements())
    print(l.calculate_outputs())
