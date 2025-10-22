from typing import Protocol

from .activation import ActivationProtocol
from .neurons import Neuron


class LayerProtocol(Protocol):
    """
    Базовый слой

    @param neurons: *args нейроны
    @param activation: Функция активации
    @param learning_rate: Скорость сходимости
    """

    # Матрица весов
    weights: list[list[float]]

    # Состояния слоя
    statements: list[float]

    # Выходы слоя
    outputs: list[float]


class Layer:
    """Класс, имплементирующий базовый слой НС"""

    def __init__(
        self,
        neurons_count: int,
        inputs: list[float],
        activation: ActivationProtocol,
        weights: list[float] = None,
        learning_rate: float = 0.5,
    ):
        """
        :param neurons_count: Количество нейронов в слое
        :param inputs: Входящие данные для нейронов в слое
        :param activation: Функция активации
        :param learning_rate: Скорость обучения
        """
        self.neurons: list[Neuron] = [
            Neuron(inputs, weights) for _ in range(neurons_count)
        ]
        self.activation = activation
        self.learning_rate = learning_rate

    def calculate_statements(self) -> list[float]:
        """Рассчитывает состояния нейронов"""

        return [
            sum(weights * inp for weights, inp in zip(neuron.weights, neuron.inputs))
            for neuron in self.neurons
        ]

    def calculate_outputs(self) -> list[float]:
        """Рассчитывает выходы нейронов"""

        return [
            self.activation.calculate(state) for state in self.calculate_statements()
        ]
