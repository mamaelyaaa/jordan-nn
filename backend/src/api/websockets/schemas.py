from pydantic import BaseModel


class TrainingStartResponse(BaseModel):
    """Ответ на запуск обучения"""

    detail: str
    session_id: str
