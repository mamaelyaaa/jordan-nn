import numpy as np

from network.training import HiddenLayer, OutputLayer


class JordanRNN:
    """Рекуррентная нейронная сеть Джордана"""

    # Веса между слоями
    w_ih: np.ndarray
    w_ch: np.ndarray
    w_ho: np.ndarray

    # Bias'ы
    b_h: np.ndarray
    b_o: np.ndarray

    def __init__(
        self,
        hidden_layer: HiddenLayer,
        output_layer: OutputLayer,
        learning_rate: float,
    ):
        self.h_layer = hidden_layer
        self.o_layer = output_layer
        self.lr = learning_rate
        self.context = np.zeros(self.o_layer.neurons)

    def _initialize_weights(self, x_sample: np.ndarray) -> None:
        """Initialize weights with appropriate dimensions"""
        # Ensure inputs are column vectors
        if len(x_sample.shape) == 1:
            x_sample = x_sample.reshape(-1, 1)

        k = x_sample.shape[0]  # Input size
        m = self.h_layer.neurons  # Hidden layer size
        n = self.o_layer.neurons  # Output layer size

        # Initialize weights with random values
        self.w_ih = np.random.uniform(-1, 1, size=(m, k))
        self.w_ch = np.random.uniform(-1, 1, size=(m, n))
        self.w_ho = np.random.uniform(-1, 1, size=(n, m))

        self.b_h = np.ones((m, 1))
        self.b_o = np.ones((n, 1))

    def _reset_context(self) -> None:
        """Сброс контекста"""
        self.context = np.zeros((self.o_layer.neurons, 1))

    def forward(self, x: np.ndarray):
        """
        Прямой проход по нейронной сети
        """
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)

        self.h_layer.inputs = x

        # Рассчитываем состояния нейронов в скрытом слое
        s_h: np.ndarray = (
            np.dot(self.w_ih, self.h_layer.inputs)
            + np.dot(self.w_ch, self.context)
            + self.b_h
        )
        self.h_layer.states = s_h

        # Рассчитываем значения выходов нейронов скрытого слоя
        h: np.ndarray = self.h_layer.activation.calculate(s_h)
        self.o_layer.inputs = h

        # Рассчитываем состояния нейронов выходного слоя
        s_y: np.ndarray = np.dot(self.w_ho, self.o_layer.inputs) + self.b_o
        self.o_layer.states = s_y

        # Рассчитываем значения выходов нейронов выходного слоя
        y_exp: np.ndarray = self.o_layer.activation.calculate(s_y)
        return y_exp

    def bptt(
        self, y_exp: np.ndarray, y: np.ndarray, next_lg: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Backpropagation Trough Time - Обратное распространение ошибок сквозь время
        """

        # Рассчитываем вектор значений ошибок выходов сети относительно обучающего примера
        delta: np.ndarray = y - y_exp

        # Рассчитываем значения невязок нейронов выходного слоя
        lg_o: np.ndarray = delta * self.o_layer.activation.derivative(
            self.o_layer.states
        )

        # Рассчитываем значения невязок нейронов скрытого слоя
        lg_h: np.ndarray = self.h_layer.activation.derivative(self.h_layer.states) * (
            np.dot(self.w_ho.T, lg_o)
            + np.dot(
                self.w_ho.T,
                np.dot(self.w_ch.T, next_lg)
                * self.o_layer.activation.derivative(self.o_layer.states),
            )
        )

        return lg_o, lg_h

    def train(
        self,
        training: np.ndarray,
        targets: np.ndarray,
        epochs: int = 1000,
        verbose: bool = True,
    ) -> list[float]:
        """Обучение сети Джордана"""
        self._initialize_weights(training[0])
        mse_history = []

        for epoch in range(epochs):
            # Матрица поправок весовых коэффициентов
            diff_w_ho: np.ndarray = np.zeros_like(self.w_ho)
            diff_w_ih: np.ndarray = np.zeros_like(self.w_ih)
            diff_w_ch: np.ndarray = np.zeros_like(self.w_ch)
            diff_b_h: np.ndarray = np.zeros_like(self.b_h)
            diff_b_o: np.ndarray = np.zeros_like(self.b_o)

            # Следующая невязка
            next_lg: np.ndarray = np.zeros((self.h_layer.neurons, 1))
            self._reset_context()

            mse_samples = []

            # Проход по выборке
            for i in range(len(training)):
                y_exp = self.forward(training[i])

                mse_samples.append(np.mean((targets[i] - y_exp) ** 2))

                # Рассчет MSE
                mse = np.mean((targets[i] - y_exp) ** 2)
                mse_samples.append(mse)

                lg_o, lg_h = self.bptt(y_exp, targets[i], next_lg)
                next_lg = lg_h

                diff_w_ho += np.outer(lg_o, self.o_layer.inputs)
                diff_w_ih += np.outer(lg_h, self.h_layer.inputs)
                diff_w_ch += np.outer(lg_h, self.context)
                diff_b_h += lg_h
                diff_b_o += lg_o

                self.context = y_exp.copy()

            self.w_ho += self.lr * diff_w_ho
            self.w_ih += self.lr * diff_w_ih
            self.w_ch += self.lr * diff_w_ch
            self.b_h += self.lr * diff_b_h
            self.b_o += self.lr * diff_b_o

            mse = np.average(mse_samples)
            mse_history.append(mse)

            if verbose:
                print(f"Epoch {epoch + 1}, MSE: {mse:.6f}")

        return mse_history

    def predict(self, x: np.ndarray):
        """Предсказание для одного входного вектора"""
        self._reset_context()
        return self.forward(x)
