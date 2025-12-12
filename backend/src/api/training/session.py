import asyncio
import uuid
from typing import Optional, Dict, Any
from datetime import datetime
import threading
import time

from api.training.schemas import PredictionsSchema
from rnn.schemas import TrainingRNNConfig


class TrainingSession:
    """Класс сессии обучения с потокобезопасностью"""

    def __init__(self, session_id: str, config: TrainingRNNConfig):
        self.id = session_id
        self.config = config
        self.status = "created"
        self.current_epoch = 0
        self.total_epochs = config.epochs
        self.loss_history: list[float] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.predictions: Optional[dict[str, Any]] = None
        self.error: Optional[str] = None
        self.train_size: Optional[int] = None
        self.test_size: Optional[int] = None

        # Для управления обучением (потокобезопасные флаги)
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Изначально не на паузе
        self._lock = threading.Lock()

    def stop(self):
        """Устанавливает флаг остановки"""
        self._stop_event.set()
        with self._lock:
            self.status = "stopped"

    def pause(self):
        """Ставит на паузу"""
        self._pause_event.clear()
        with self._lock:
            self.status = "paused"

    def resume(self):
        """Снимает с паузы"""
        self._pause_event.set()
        with self._lock:
            self.status = "training"

    def should_stop(self) -> bool:
        """Проверяет, нужно ли остановиться"""
        return self._stop_event.is_set()

    def is_paused(self) -> bool:
        """Проверяет, на паузе ли обучение"""
        return not self._pause_event.is_set()

    def wait_if_paused(self):
        """Ждет, если обучение на паузе"""
        while self.is_paused() and not self.should_stop():
            time.sleep(0.5)

    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует в словарь"""
        return {
            "id": self.id,
            "config": self.config.model_dump(),
            "status": self.status,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "loss_history": self.loss_history,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "predictions": self.predictions,
            "error": self.error,
            "train_size": self.train_size,
            "test_size": self.test_size,
        }


class TrainingSessionManager:
    """Менеджер сессий обучения"""

    def __init__(self):
        self.sessions: dict[str, TrainingSession] = {}
        self.training_tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    async def create_session(self, config: TrainingRNNConfig) -> str:
        """Создает новую сессию обучения"""
        async with self._lock:
            session_id = str(uuid.uuid4())
            session = TrainingSession(session_id, config)
            self.sessions[session_id] = session
            return session_id

    async def get_session(self, session_id: str) -> Optional[TrainingSession]:
        """Получает данные сессии"""
        async with self._lock:
            return self.sessions.get(session_id)

    async def update_session(self, session_id: str, **updates):
        """Обновляет данные сессии"""
        async with self._lock:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                for key, value in updates.items():
                    setattr(session, key, value)

    async def stop_session(self, session_id: str):
        """Останавливает сессию"""
        async with self._lock:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                session.stop()

            if session_id in self.training_tasks:
                self.training_tasks[session_id].cancel()

    async def pause_session(self, session_id: str):
        """Приостанавливает сессию"""
        async with self._lock:
            if session_id in self.sessions:
                self.sessions[session_id].pause()

    async def resume_session(self, session_id: str):
        """Возобновляет сессию"""
        async with self._lock:
            if session_id in self.sessions:
                self.sessions[session_id].resume()


session_manager = TrainingSessionManager()
