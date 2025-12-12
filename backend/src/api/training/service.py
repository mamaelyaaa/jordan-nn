import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, Any, Optional, List
import threading

from api.training.schemas import PredictionsSchema
from api.training.session import session_manager
from api.websockets.connection import websocket_manager
from rnn.manager import rnn_manager
from rnn.structure.activation import (
    TanhActivation,
    SigmoidActivation,
    LinearActivation,
    ReLUActivation,
)
from rnn.structure.regularizer import L1, L2, NoRegularizer


class TrainingService:
    """Сервис выполнения обучения с асинхронным контролем"""

    def __init__(self):
        # Создаем пул потоков для CPU-intensive операций
        self.executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="training_"
        )
        self.active_trainings: Dict[str, asyncio.Task] = {}
        self.training_locks: Dict[str, threading.Lock] = {}

    async def run_training(self, session_id: str, config: dict[str, Any]):
        """Запускает асинхронное обучение"""
        try:
            # Проверяем, не запущено ли уже обучение для этой сессии
            if await self._is_training_active(session_id):
                await websocket_manager.send_progress(
                    session_id,
                    {
                        "type": "training_error",
                        "session_id": session_id,
                        "error": "Training is already running for this session",
                    },
                )
                return

            session = await session_manager.get_session(session_id)
            if not session:
                return

            # Обновляем статус
            await self._update_session_status(
                session_id, "configuring", start_time=datetime.now()
            )

            await websocket_manager.send_progress(
                session_id,
                {
                    "type": "training_started",
                    "session_id": session_id,
                    "status": "configuring",
                },
            )

            # Запускаем асинхронную задачу обучения
            training_task = asyncio.create_task(
                self._train_async(session_id, config, session)
            )
            self.active_trainings[session_id] = training_task

            # Добавляем обработчик завершения задачи
            training_task.add_done_callback(
                lambda t: self._cleanup_training(session_id)
            )

        except Exception as e:
            print(f"Error starting training for session {session_id}: {e}")
            await self._handle_training_error(session_id, e)

    async def _train_async(self, session_id: str, config: Dict[str, Any], session):
        """Асинхронное обучение (не блокирует основной поток)"""
        try:
            # Шаг 1: Быстрая настройка и подготовка
            dataset = await self._setup_training(session_id, config)

            # Шаг 2: Обучение модели в отдельном потоке
            mse_history = await self._execute_training_in_thread(
                session_id, config, dataset
            )

            # Проверяем, было ли обучение остановлено
            if await self._is_training_stopped(session_id):
                return

            # Шаг 3: Завершение и получение результатов
            await self._complete_training(session_id, mse_history, dataset)

        except asyncio.CancelledError:
            print(f"Training cancelled for session {session_id}")
            await self._handle_training_cancelled(session_id)
            raise

        except Exception as e:
            print(f"Training error in session {session_id}: {e}")
            await self._handle_training_error(session_id, e)
            raise

    async def _setup_training(self, session_id: str, config: Dict[str, Any]):
        """Настройка модели и подготовка данных"""
        # Конфигурация активаций и регуляризаций
        activation_map = {
            "tanh": TanhActivation(),
            "sigmoid": SigmoidActivation(),
            "linear": LinearActivation(),
            "relu": ReLUActivation(),
        }

        regularizer_map = {
            "L1": lambda rate: L1(lm=rate),
            "L2": lambda rate: L2(lm=rate),
        }

        # Настройка RNN
        regularizer_config = None
        if config.get("regularizer"):
            regularizer_func = regularizer_map.get(config["regularizer"])
            if regularizer_func:
                regularizer_config = regularizer_func(config["regularizer_rate"])

        rnn_manager.set_config(
            target=config["target"],
            features=config["features"],
            learning_rate=config["learning_rate"],
            hidden_activation=activation_map[config["hidden_activation"]],
            hidden_neurons=config["hidden_neurons"],
            regularization=regularizer_config,  # Может быть None
        )

        # Загрузка и подготовка данных в отдельном потоке
        dataset = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self._prepare_data_sync,
            config["stock_symbol"],
            config["test_rate"],
        )

        # Обновляем размеры данных
        await self._update_session_status(
            session_id,
            "preparing_data",
            train_size=dataset.train_size,
            test_size=dataset.test_size,
        )

        # Отправляем WebSocket сообщение
        await websocket_manager.send_progress(
            session_id,
            {
                "type": "data_prepared",
                "train_size": dataset.train_size,
                "test_size": dataset.test_size,
            },
        )

        # Инициализируем модель
        rnn_manager.initialize_model()

        # Начинаем обучение
        await self._update_session_status(
            session_id, "training", total_epochs=config["epochs"]
        )

        return dataset

    def _prepare_data_sync(self, stock_symbol: str, test_rate: float):
        """Подготовка данных (синхронная)"""
        rnn_manager.load_data(stock_symbol)
        return rnn_manager.prepare_data(test_rate=test_rate)

    async def _execute_training_in_thread(
        self, session_id: str, config: Dict[str, Any], dataset
    ):
        """Запуск обучения модели в отдельном потоке"""
        loop = asyncio.get_event_loop()

        # Запускаем обучение в отдельном потоке
        mse_history = await loop.run_in_executor(
            self.executor,
            self._train_model_sync,
            session_id,
            config,
            dataset,
        )

        return mse_history

    def _train_model_sync(self, session_id: str, config: Dict[str, Any], dataset):
        """Синхронное обучение модели в отдельном потоке"""
        try:
            # Создаем свой event loop для этого потока
            loop = self._get_event_loop_for_thread()

            def run_async(coro):
                return loop.run_until_complete(coro)

            # Получаем сессию для проверки состояния
            from api.training.session import session_manager as sm

            session = run_async(sm.get_session(session_id))
            if not session:
                return []

            # Проверяем состояние перед началом
            if self._should_stop_training(session):
                return []

            # Запускаем обучение с контролем
            mse_history = rnn_manager.model.train_with_control(
                training=dataset.x_train_N,
                targets=dataset.y_train_N,
                epochs=config["epochs"],
                session_id=session_id,
                verbose=False,
            )

            return mse_history

        except Exception as e:
            print(f"Error in training model for session {session_id}: {e}")
            self._send_error_to_websocket(session_id, str(e))
            return []

    async def _complete_training(
        self, session_id: str, mse_history: List[float], dataset
    ):
        """Завершение обучения и получение результатов"""
        if not mse_history:
            print(f"No training history for session {session_id}")
            await self._handle_training_error(
                session_id, Exception("Training produced no results")
            )
            return

        # Обновляем историю ошибок
        await self._update_session_status(
            session_id,
            "completed",
            end_time=datetime.now(),
            loss_history=mse_history,
            current_epoch=len(mse_history),
        )

        # Получаем предсказания в отдельном потоке
        predictions = await asyncio.get_event_loop().run_in_executor(
            self.executor, rnn_manager.predict
        )

        session = await session_manager.get_session(session_id)
        if not session:
            return

        # Проверяем структуру predictions
        if hasattr(predictions, "train_predictions") and hasattr(
            predictions, "test_predictions"
        ):
            session.predictions = PredictionsSchema(
                train_prediction=[train for train in predictions.train_predictions],
                test_prediction=[test for test in predictions.test_predictions],
            )

        # Отправляем финальные результаты
        await self._send_completion_message(session_id, mse_history)

    async def _save_predictions(
        self, session_id: str, predictions, mse_history: List[float]
    ):
        """Сохранение предсказаний в сессию"""
        session = await session_manager.get_session(session_id)
        if not session:
            return

        # Проверяем структуру predictions
        if hasattr(predictions, "train_predictions") and hasattr(
            predictions, "test_predictions"
        ):
            session.predictions = PredictionsSchema(
                train_prediction=[train for train in predictions.train_predictions],
                test_prediction=[test for test in predictions.test_predictions],
            )

            print(f"MSE history length: {len(mse_history)}")
            print(f"Train predictions length: {len(predictions.train_predictions)}")
            print(f"Test predictions length: {len(predictions.test_predictions)}")

        else:
            print(f"Unexpected predictions structure: {type(predictions)}")

    async def _send_completion_message(self, session_id: str, mse_history: List[float]):
        """Отправка сообщения о завершении обучения"""
        session = await session_manager.get_session(session_id)
        if not session:
            return

        await websocket_manager.send_progress(
            session_id,
            {
                "type": "training_completed",
                "session_id": session_id,
                "final_loss": mse_history[-1] if mse_history else None,
                "predictions": (
                    session.predictions
                    if hasattr(session.predictions, "dict")
                    else session.predictions
                ),
                "mse_history": mse_history,
            },
        )

    # Вспомогательные методы

    async def _is_training_active(self, session_id: str) -> bool:
        """Проверяет, активно ли обучение для сессии"""
        if session_id not in self.active_trainings:
            return False
        task = self.active_trainings[session_id]
        return not task.done()

    async def _is_training_stopped(self, session_id: str) -> bool:
        """Проверяет, было ли обучение остановлено"""
        session = await session_manager.get_session(session_id)
        return session and session.status == "stopped"

    async def _update_session_status(self, session_id: str, status: str, **kwargs):
        """Обновляет статус сессии"""
        await session_manager.update_session(session_id, status=status, **kwargs)

    def _get_event_loop_for_thread(self) -> asyncio.AbstractEventLoop:
        """Получает или создает event loop для текущего потока"""
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

    def _should_stop_training(self, session) -> bool:
        """Проверяет, нужно ли остановить обучение"""
        if session.should_stop():
            return True
        if session.is_paused():
            session.wait_if_paused()
        return False

    def _send_error_to_websocket(self, session_id: str, error: str):
        """Отправляет ошибку через WebSocket"""
        try:
            loop = self._get_event_loop_for_thread()

            async def send_error():
                await websocket_manager.send_progress(
                    session_id,
                    {
                        "type": "training_error",
                        "session_id": session_id,
                        "error": error,
                    },
                )

            loop.run_until_complete(send_error())
        except Exception as e:
            print(f"Failed to send error to websocket: {e}")

    async def _handle_training_error(self, session_id: str, error: Exception):
        """Обработка ошибок обучения"""
        await self._update_session_status(
            session_id,
            "error",
            error=str(error),
            end_time=datetime.now(),
        )

        await websocket_manager.send_progress(
            session_id,
            {
                "type": "training_error",
                "session_id": session_id,
                "error": str(error),
            },
        )

        self._cleanup_training(session_id)

    async def _handle_training_cancelled(self, session_id: str):
        """Обработка отмены обучения"""
        await self._update_session_status(
            session_id,
            "stopped",
            end_time=datetime.now(),
            error="Training was cancelled",
        )

        await websocket_manager.send_progress(
            session_id,
            {
                "type": "training_stopped",
                "session_id": session_id,
            },
        )

        self._cleanup_training(session_id)

    def _cleanup_training(self, session_id: str):
        """Очищает ресурсы после завершения обучения"""
        if session_id in self.active_trainings:
            del self.active_trainings[session_id]

        if session_id in self.training_locks:
            del self.training_locks[session_id]


training_service = TrainingService()
