import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

from .structure.layers import HiddenLayer, OutputLayer
from .structure.regularizer import RegularizerProtocol, NoRegularizer


class JordanRNN:
    """Рекуррентная нейронная сеть Джордана"""

    @dataclass
    class Gradients:
        w_ih: np.ndarray
        w_ch: np.ndarray
        w_ho: np.ndarray
        b_h: np.ndarray
        b_o: np.ndarray

    def __init__(
        self,
        hidden_layer: HiddenLayer,
        output_layer: OutputLayer,
        learning_rate: float,
        regularization: Optional[RegularizerProtocol] = None,
    ):
        """
        :param hidden_layer: Скрытый слой
        :param output_layer: Выходной слой
        :param learning_rate: Скорость обучения
        :param regularization: Регуляризация
        """
        self.h_layer = hidden_layer
        self.o_layer = output_layer
        self.lr = learning_rate
        self.regularizer = regularization or NoRegularizer()

        # Создаем контекст
        self.context: Optional[np.ndarray] = None
        self.context_size: Optional[int] = None

    def _initialize_weights(self, x_sample: np.ndarray, y_sample: np.ndarray) -> None:
        """Инициализация весов модели"""

        x_sample = x_sample.reshape(-1, 1) if x_sample.ndim == 1 else x_sample
        y_sample = y_sample.reshape(-1, 1) if y_sample.ndim == 1 else y_sample

        k = x_sample.shape[0]  # Количество входов модели
        m = self.h_layer.neurons
        n = y_sample.shape[0]  # Количество выходов модели

        self.context_size = n
        self.context = np.zeros((n, 1))

        self.w_ih = np.random.uniform(-0.5, 0.5, size=(m, k))
        self.w_ch = np.random.uniform(-0.5, 0.5, size=(m, n))
        self.w_ho = np.random.uniform(-0.5, 0.5, size=(n, m))

        self.b_h = np.zeros((m, 1))
        self.b_o = np.zeros((n, 1))

    def _reset_context(self) -> None:
        """Сброс контекста"""
        self.context = np.zeros((self.context_size, 1))

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

        self._initialize_weights(x_sample=training[0], y_sample=targets[0])
        mse_history = []

        for epoch in range(epochs):
            # Инициализируем матрицу градиентов
            gradients = self._init_gradients()

            # Следующая невязка
            next_lg: np.ndarray = np.zeros((self.h_layer.neurons, 1))
            self._reset_context()

            mse_samples = []

            # Проход по выборке
            for i in range(len(training)):
                y_exp = self.forward(training[i])

                # Рассчет MSE
                mse = np.mean((targets[i] - y_exp) ** 2)
                mse_samples.append(mse)

                lg_o, lg_h = self.bptt(y_exp, targets[i], next_lg)
                next_lg = lg_h

                # Накапливание градиентов
                self._accumulate_gradients(gradients, lg_o, lg_h)

                # Обновление контекста для следующего шага
                self.context = y_exp.copy()

            # Нормализация градиентов по размеру выборки
            self._normalize_gradients(gradients, n_samples=len(training))
            # Обновление весов градиентов
            self._update_weights(gradients)

            mse = np.average(mse_samples)
            mse_history.append(mse)

            if verbose:
                print(f"Epoch {epoch + 1}, MSE: {mse:.6f}")

        return mse_history

    def train_with_control(
        self,
        training: np.ndarray,
        targets: np.ndarray,
        epochs: int,
        session_id: str,
        verbose: bool = True,
    ) -> list[float]:
        """
        Обучение сети с контролем паузы/остановки через сессию
        """

        # Инициализация весов
        self._initialize_weights(x_sample=training[0], y_sample=targets[0])
        mse_history = []

        # Импорт
        import asyncio

        # Импорт менеджеров
        from api.training.session import session_manager
        from api.websockets.connection import websocket_manager

        # Создаем event loop для этого потока
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        def run_async(coro):
            return loop.run_until_complete(coro)

        for epoch in range(1, epochs + 1):
            try:
                # Проверка состояния сессии
                session = run_async(session_manager.get_session(session_id))
                if not session:
                    print(f"Session {session_id} not found, stopping training")
                    break

                # Проверка паузы
                if session.is_paused():
                    # Отправляем уведомление о паузе
                    run_async(
                        websocket_manager.send_progress(
                            session_id,
                            {
                                "type": "training_paused",
                                "session_id": session_id,
                                "epoch": epoch - 1,
                            },
                        )
                    )

                    print(f"Training paused at epoch {epoch}")
                    while session.is_paused():
                        session = run_async(session_manager.get_session(session_id))
                        if not session or session.should_stop():
                            break

                    # Отправляем уведомление о возобновлении
                    run_async(
                        websocket_manager.send_progress(
                            session_id,
                            {
                                "type": "training_resumed",
                                "session_id": session_id,
                                "epoch": epoch,
                            },
                        )
                    )

                # Проверка остановки
                if session.should_stop():
                    print(f"Training stopped by user at epoch {epoch}")
                    run_async(
                        session_manager.update_session(
                            session_id,
                            status="stopped",
                            current_epoch=epoch - 1,
                            end_time=datetime.now(),
                        )
                    )
                    run_async(
                        websocket_manager.send_progress(
                            session_id,
                            {
                                "type": "training_stopped",
                                "session_id": session_id,
                                "epoch": epoch - 1,
                            },
                        )
                    )
                    break

                # Обучение на одной эпохе
                gradients = self._init_gradients()
                next_lg = np.zeros((self.h_layer.neurons, 1))
                self._reset_context()

                mse_samples = []

                # Проход по обучающей выборке
                for i in range(len(training)):
                    if i % 50 == 0:
                        session = run_async(session_manager.get_session(session_id))
                        if not session:
                            break

                        if session.is_paused():
                            print(f"Training paused during epoch {epoch} at sample {i}")
                            while session.is_paused():
                                time.sleep(1)
                                session = run_async(
                                    session_manager.get_session(session_id)
                                )
                                if not session or session.should_stop():
                                    break

                        if session and session.should_stop():
                            print(
                                f"Training stopped during epoch {epoch} at sample {i}"
                            )
                            return mse_history

                    # Прямой проход и обратное распространение
                    y_exp = self.forward(training[i])
                    mse = np.mean((targets[i] - y_exp) ** 2)
                    mse_samples.append(mse)

                    lg_o, lg_h = self.bptt(y_exp, targets[i], next_lg)
                    next_lg = lg_h

                    self._accumulate_gradients(gradients, lg_o, lg_h)
                    self.context = y_exp.copy()

                if not session:
                    break

                # Обновление весов
                self._normalize_gradients(gradients, n_samples=len(training))
                self._update_weights(gradients)

                # Вычисление MSE
                mse = np.average(mse_samples)
                mse_history.append(mse)

                # === КРИТИЧЕСКИЙ УЧАСТОК: последовательная отправка ===
                # 1. Сначала обновляем сессию
                run_async(
                    session_manager.update_session(
                        session_id, current_epoch=epoch, loss_history=mse_history.copy()
                    )
                )

                # 2. Ждем завершения обновления
                time.sleep(0.01)  # Небольшая задержка для гарантии

                # 3. Затем отправляем WebSocket сообщение
                run_async(
                    websocket_manager.send_progress(
                        session_id,
                        {
                            "type": "training",
                            "epoch": epoch,
                            "total_epochs": epochs,
                            "loss": mse,
                            "mse_history": mse_history,
                        },
                    )
                )

                # 4. Ждем завершения отправки
                time.sleep(0.01)

                if verbose:
                    print(f"Epoch {epoch}/{epochs}, MSE: {mse:.6f}")

            except Exception as e:
                print(f"Error in epoch {epoch}: {e}")
                try:
                    run_async(
                        websocket_manager.send_progress(
                            session_id,
                            {
                                "type": "training_error",
                                "session_id": session_id,
                                "error": str(e),
                                "epoch": epoch,
                            },
                        )
                    )
                except:
                    pass
                raise

        return mse_history

    def _init_gradients(self) -> "Gradients":
        """Инициализирует градиенты"""

        return self.Gradients(
            w_ho=np.zeros_like(self.w_ho),
            w_ih=np.zeros_like(self.w_ih),
            w_ch=np.zeros_like(self.w_ch),
            b_h=np.zeros_like(self.b_h),
            b_o=np.zeros_like(self.b_o),
        )

    def _accumulate_gradients(
        self,
        gradients: Gradients,
        lg_o: np.ndarray,
        lg_h: np.ndarray,
    ) -> "Gradients":
        gradients.w_ho += np.outer(lg_o, self.o_layer.inputs)
        gradients.w_ih += np.outer(lg_h, self.h_layer.inputs)
        gradients.w_ch += np.outer(lg_h, self.context)
        gradients.b_h += lg_h
        gradients.b_o += lg_o
        return gradients

    @staticmethod
    def _normalize_gradients(g: Gradients, n_samples: int) -> None:
        for key in g.__dict__:
            g.__dict__[key] /= n_samples
        return

    def _update_weights(self, g: Gradients) -> None:
        self.w_ho += self.lr * (g.w_ho - self.regularizer.compute_gradient(self.w_ho))
        self.w_ih += self.lr * (g.w_ih - self.regularizer.compute_gradient(self.w_ih))
        self.w_ch += self.lr * (g.w_ch - self.regularizer.compute_gradient(self.w_ch))
        self.b_h += self.lr * g.b_h
        self.b_o += self.lr * g.b_o
        return

    def predict(self, x: np.ndarray):
        """Предсказание для одного входного вектора"""
        self._reset_context()
        return self.forward(x)

    def predict_sequence(self, x_sequence: np.ndarray):
        """Предсказание для последовательности с сохранением контекста"""
        self._reset_context()
        predictions = []
        for i in range(len(x_sequence)):
            predict = self.forward(x_sequence[i])
            predictions.append(predict.flatten().copy())
            self.context = predict.copy()
        return np.array(predictions)
