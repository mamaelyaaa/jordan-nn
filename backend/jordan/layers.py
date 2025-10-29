import numpy as np


class HiddenLayer:

    def __init__(self, inputs: np.ndarray, weights: np.ndarray, fictive: bool = True):
        self.inputs = [1] + inputs if fictive else inputs
        self.weights = weights

    def state(self) -> np.ndarray:
        pass

    def outputs(self):
        return


# class Layer:
#     """Класс, имплементирующий базовый слой НС"""
#
#     def __init__(
#         self,
#         inputs: list[float] | np.ndarray,
#         neurons: int,
#         activation: ActivationProtocol,
#         weights: list[np.float64] = None,
#         learning_rate: float = 0.5,
#     ):
#         """
#         :param neurons: Количество нейронов в слое
#         :param inputs: Вектор входных данных
#         :param activation: Функция активации
#         :param learning_rate: Скорость обучения
#         """
#         self.neurons: list[Neuron] = [Neuron(inputs, weights) for _ in range(neurons)]
#         self.weights = np.array([neuron.weights for neuron in self.neurons])
#         self.activation = activation
#         self.learning_rate = learning_rate
#
#     def calculate_statements(self) -> np.ndarray:
#         """Рассчитывает состояния нейронов"""
#
#         # weights = np.array([neuron.weights for neuron in self.neurons])
#
#         return weights
#         # return [
#         #     sum(weights * inp for weights, inp in zip(neuron.weights, neuron.inputs))
#         #     for neuron in self.neurons
#         # ]
#
#     def calculate_outputs(self) -> list[float]:
#         """Рассчитывает выходы нейронов"""
#
#         return [
#             self.activation.calculate(state) for state in self.calculate_statements()
#         ]
