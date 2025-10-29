import logging

import numpy as np

from jordan.activation import ActivationProtocol

# TODO Вынести в другой файл
logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)


class Layer:

    def __init__(
        self, neurons: int, activation: ActivationProtocol, fictive: bool = True
    ):
        self.neurons = neurons
        self.activation = activation
        self.fictive = fictive


class JordanNetwork:
    """Класс, имплементирующий работу нейронной сети Джордана"""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        hid_layer: Layer,
        out_layer: Layer,
        learning_rate: float = 0.5,
    ):
        self.x = x
        self.y = y
        self.h_layer = hid_layer
        self.o_layer = out_layer
        self.lr = learning_rate
        self._init_weights()
        self.context: np.ndarray = np.zeros(out_layer.neurons)

        # Вспомогательные элементы для обратного распространения ошибки
        self.__h_with_fictive = None
        self.__x_with_fictive = None
        self.__s_y = None
        self.__s_h = None

    def forward(self, training: np.ndarray) -> np.ndarray:
        """
        Прямой проход по нейронной сети

        :param training: Тренировочные данные (одна выборка)
        :return: Выходы нейронной сети
        """

        # Проверка на входные данные
        if training.ndim != 1:
            raise ValueError(f"Ожидается 1D массив, получен {training.ndim}D")

        x_with_fictive = np.append(1, training)
        self.__x_with_fictive = x_with_fictive

        # Рассчитываем состояния нейронов скрытого слоя
        s_h: np.ndarray = np.dot(self.w_ih, x_with_fictive) + np.dot(
            self.w_ch, self.context
        )
        assert s_h.shape == (self.h_layer.neurons,)
        self.__s_h = s_h

        # Рассчитываем значения выходов нейронов скрытого слоя
        h_out: np.ndarray = self.h_layer.activation.calculate(s_h)
        assert h_out.shape == (self.h_layer.neurons,)

        # Расширяем вектор значений выходов нейронов скрытого слоя, добавляя единицу в начало вектора:
        h_with_fictive = np.append(1, h_out)
        assert h_with_fictive.shape == (self.h_layer.neurons + 1,)
        self.__h_with_fictive = h_with_fictive

        # Рассчитываем состояния нейронов выходного слоя
        s_y: np.ndarray = np.dot(self.w_ho, h_with_fictive)
        assert s_y.shape == (self.o_layer.neurons,)
        self.__s_y = s_y

        # Рассчитываем значения выходов нейронов выходного слоя:
        y_exp: np.ndarray = self.o_layer.activation.calculate(s_y)
        assert y_exp.shape == (self.o_layer.neurons,)

        # НОВОЕ!! Обновляем контекст
        self.context = y_exp.copy()

        return y_exp

    def backward(
        self,
        y: np.ndarray,
        y_exp: np.ndarray,
    ) -> float:
        """
        Обратное распространение ошибки
        :param x:
        :param y:
        :param y_exp: Полученные экспериментальные выходные значения
        :return: Возвращает значение ошибки
        """

        # Рассчитываем вектор значений ошибок выходов сети относительно обучающего примера
        delta: np.ndarray = y - y_exp
        assert delta.shape == (self.o_layer.neurons,)

        # Рассчитываем значения невязок нейронов выходного слоя
        # P.S. r - residual (с англ. "невязка")
        r_y: np.ndarray = delta * self.o_layer.activation.derivative(self.__s_y)
        assert r_y.shape == (self.o_layer.neurons,)

        # Рассчитываем значения невязок нейронов скрытого слоя
        r_h: np.ndarray = np.dot(
            self.w_ho[:, :-1].T, r_y
        ) * self.h_layer.activation.derivative(self.__s_h)
        assert r_h.shape == (self.h_layer.neurons,)

        # Рассчитываются поправки весовых коэффициентов
        self._update_weights(x=self.x_with_fictive, residual_out=r_y, residual_hid=r_h)

        return float(np.mean(np.abs(delta)))

    def train(
        self, x: np.ndarray, y: np.ndarray, epochs: int = 100, verbose: bool = True
    ):
        """
        Обучение сети на наборе данных

        :param x: Тестовые данные (выборка)
        :param y: Учитель
        :param epochs: Максимальное количество эпох для обучения
        :param verbose: Показывает процесс обучения
        :return:
        """
        original_context = self.context.copy()
        errors = []

        for epoch in range(epochs):
            total_error = 0
            for i in range(len(x)):
                y_exp: np.ndarray = self.forward(training=x[i])
                total_error += self.backward(y[i], y_exp)

            avg_error = total_error / len(x)
            errors.append(avg_error)

            if verbose and epoch % 10 == 0:
                print(f"Эпоха {epoch}: Средняя ошибка = {avg_error:.6f}")

        self.context = original_context
        return errors

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Предсказание для нескольких примеров"""
        predictions = []
        original_context = self.context.copy()

        for i in range(len(x)):
            prediction = self.forward(x[i])
            predictions.append(prediction)

        # Восстанавливаем контекст
        self.context = original_context
        return np.array(predictions)

    def _update_weights(
        self,
        residual_out: np.ndarray,
        residual_hid: np.ndarray,
        x: np.ndarray,
    ) -> None:
        """
        Обновление весовых матриц

        :param residual_out: Невязки выходного слоя
        :param residual_hid: Невязки скрытого слоя
        :param x: Тестовые данные
        :return:
        """

        # Обновляем веса w_ho (выходной слой)
        self.w_ho += self.lr * np.outer(residual_out, self.__h_with_fictive)

        # Обновляем веса w_ch (контекстные связи)
        self.w_ch += self.lr * np.outer(residual_hid, self.context)

        # Обновляем веса w_ih (входные связи)
        self.w_ih += self.lr * np.outer(residual_hid, x)

    def _init_weights(self) -> None:
        """Инициализирует матрицы весов"""

        input_size = self.x.shape[1]  # количество признаков

        # Веса связей между входами и нейронами скрытого слоя
        self.w_ih: np.ndarray = np.random.uniform(
            -1, 1, size=(self.h_layer.neurons, input_size + 1)
        )

        # Веса связей между контекстом и нейронами скрытого слоя (без фиктивной единицы)
        self.w_ch: np.ndarray = np.random.uniform(
            -1, 1, size=(self.h_layer.neurons, self.o_layer.neurons)
        )

        # Веса связей между нейронами скрытого и выходного слоёв
        self.w_ho: np.ndarray = np.random.uniform(
            -1, 1, size=(self.o_layer.neurons, self.h_layer.neurons + 1)
        )
        logger.debug("Матрица связей между нейронами инициализирована!")
        return
