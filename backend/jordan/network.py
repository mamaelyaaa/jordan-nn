import logging

import numpy as np

from jordan.activation import ActivationProtocol
from jordan.layers import HiddenLayer, OutputLayer

# TODO Вынести в другой файл
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class JordanNetwork:
    """Класс, имплементирующий работу нейронной сети Джордана"""

    def __init__(
        self,
        hid_layer: HiddenLayer,
        out_layer: OutputLayer,
        learning_rate: float = 0.5,
    ):
        """

        :param hid_layer: Скрытый слой
        :param out_layer: Выходной слой
        :param learning_rate: Скорость обучения модели
        """
        self._h_layer = hid_layer
        self._o_layer = out_layer
        self._lr = learning_rate
        self._context: np.ndarray = np.zeros(out_layer.outputs)

    def _forward(
        self, training: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Прямой проход по нейронной сети

        :param training: Тренировочные данные (одна выборка)
        :return: (y_exp, h_out, s_y, s_h)
        """

        # Проверка на входные данные
        if training.ndim != 1:
            raise ValueError(f"Ожидается 1D массив, получен {training.ndim}D")

        # Проверка инициализации весов
        if self.w_ih is None or self.w_ch is None or self.w_ho is None:
            raise ValueError("Веса не инициализированы. Сначала вызовите train()")

        # Рассчитываем состояния нейронов скрытого слоя
        s_h: np.ndarray = np.dot(self.w_ih, training) + np.dot(self.w_ch, self._context)
        assert s_h.shape == (self._h_layer.neurons,)
        self._h_layer.states = s_h

        # Рассчитываем значения выходов нейронов скрытого слоя
        h_out: np.ndarray = self._h_layer.activation.calculate(s_h)
        assert h_out.shape == (self._h_layer.neurons,)

        # Создаем версию с фиктивным нейроном для весов
        if self._h_layer.fictive:
            h_out = np.append(1, h_out)
            assert h_out.shape == (self._h_layer.neurons + 1,)

        # Рассчитываем состояния нейронов выходного слоя
        s_y: np.ndarray = np.dot(self.w_ho, h_out)
        assert s_y.shape == (self._o_layer.neurons,)

        # Рассчитываем значения выходов нейронов выходного слоя:
        y_exp: np.ndarray = self._o_layer.activation.calculate(s_y)
        assert y_exp.shape == (self._o_layer.neurons,)

        # НОВОЕ!! Обновляем контекст
        self._context = y_exp.copy()

        return y_exp, h_out, s_y, s_h

    def _backward(
        self,
        x: np.ndarray,
        y: np.ndarray,
        y_exp: np.ndarray,
        h_out: np.ndarray,
        s_y: np.ndarray,
        s_h: np.ndarray,
    ) -> float:
        """
        Обратное распространение ошибки
        :param x: Входные данные
        :param y: Ожидаемые выходные значения
        :param y_exp: Полученные экспериментальные выходные значения
        :param h_out: Выходы скрытого слоя с фиктивным нейроном
        :param s_y: Состояния выходного слоя
        :param s_h: Состояния скрытого слоя
        :return: Возвращает значение ошибки
        """

        # Рассчитываем вектор значений ошибок выходов сети относительно обучающего примера
        delta: np.ndarray = y - y_exp
        assert delta.shape == (self._o_layer.outputs,)

        # Рассчитываем значения невязок нейронов выходного слоя
        r_y: np.ndarray = delta * self._o_layer.activation.derivative(s_y)
        assert r_y.shape == (self._o_layer.outputs,)

        # Рассчитываем значения невязок нейронов скрытого слоя
        # Используем срез для исключения фиктивного нейрона
        w_ho_effective = self.w_ho[:, 1:] if self._h_layer.fictive else self.w_ho
        r_h: np.ndarray = np.dot(
            w_ho_effective.T, r_y
        ) * self._h_layer.activation.derivative(s_h)
        assert r_h.shape == (self._h_layer.neurons,)

        # Рассчитываются поправки весовых коэффициентов
        self._update_weights(x, residual_out=r_y, residual_hid=r_h, h_out=h_out)

        return float(np.mean(np.abs(delta)))

    def train(
        self, x: np.ndarray, y: np.ndarray, epochs: int = 100, verbose: bool = True
    ) -> list[float]:
        """
        Обучение сети на наборе данных

        :param x: Входные данные (выборка)
        :param y: Ожидаемые выходные значения
        :param epochs: Максимальное количество эпох для обучения
        :param verbose: Показывает процесс обучения
        :return: Список ошибок по эпохам
        """
        # Проверка совместимости размеров
        if len(x) != len(y):
            raise ValueError("x и y должны иметь одинаковую длину")

        # Добавляем фиктивную единицу ко всем примерам если нужно
        if self._h_layer.fictive:
            x = np.array([np.append(1, sample) for sample in x])

        # Инициализируем веса
        self._init_weights(x[0])

        original_context = self._context.copy()
        errors = []

        for epoch in range(epochs):
            total_error = 0
            for i in range(len(x)):
                y_exp, *args = self._forward(training=x[i])
                total_error += self._backward(x[i], y[i], y_exp, *args)

            avg_error = total_error / len(x)
            errors.append(avg_error)

            if verbose and epoch % 10 == 0:
                print(f"Эпоха {epoch}: Средняя ошибка = {avg_error:.6f}")

        self._context = original_context
        return errors

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Предсказание для нескольких примеров"""

        if self.w_ih is None:
            raise ValueError("Сеть не обучена. Сначала вызовите train()")

        # Добавляем фиктивную единицу если нужно
        if self._h_layer.fictive:
            x = np.array([np.append(1, sample) for sample in x])

        predictions = []
        original_context = self._context.copy()

        for i in range(len(x)):
            y_exp, _, _, _ = self._forward(x[i])
            predictions.append(y_exp)

        # Восстанавливаем контекст
        self._context = original_context
        return np.array(predictions)

    def _update_weights(
        self,
        x: np.ndarray,
        residual_out: np.ndarray,
        residual_hid: np.ndarray,
        h_out: np.ndarray,
    ) -> None:
        """
        Обновление весовых матриц

        :param residual_out: Невязки выходного слоя
        :param residual_hid: Невязки скрытого слоя
        :param x: Входные данные (уже с фиктивным нейроном если нужно)
        :param h_out: Выходы скрытого слоя с фиктивным нейроном
        :return:
        """

        # Обновляем веса w_ho (выходной слой)
        self.w_ho += self._lr * np.outer(residual_out, h_out)

        # Обновляем веса w_ch (контекстные связи)
        self.w_ch += self._lr * np.outer(residual_hid, self._context)

        # Обновляем веса w_ih (входные связи)
        self.w_ih += self._lr * np.outer(residual_hid, x)

    def _init_weights(self, x_sample: np.ndarray) -> None:
        """
        Инициализирует матрицы весов
        :param x_sample: Один пример из обучающей выборки (уже с фиктивным нейроном если нужно)
        :return:
        """

        input_size = x_sample.shape[0]  # количество признаков в одном примере

        # Веса связей между входами и нейронами скрытого слоя
        self.w_ih = np.random.uniform(-1, 1, size=(self._h_layer.neurons, input_size))

        # Веса связей между контекстом и нейронами скрытого слоя
        self.w_ch = np.random.uniform(
            -1, 1, size=(self._h_layer.neurons, self._o_layer.outputs)
        )

        # Веса связей между нейронами скрытого и выходного слоёв
        hidden_size = (
            self._h_layer.neurons + 1
            if self._h_layer.fictive
            else self._h_layer.neurons
        )
        self.w_ho = np.random.uniform(-1, 1, size=(self._o_layer.outputs, hidden_size))

        logger.debug("Матрица связей между нейронами инициализирована!")
