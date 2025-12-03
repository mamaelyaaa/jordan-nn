from typing import Optional

from pydantic import BaseModel, Field

from rnn.prepare.feature_engine import FeaturesEnum
from rnn.structure.activation import ActivationEnum
from rnn.structure.regularizer import RegularizerEnum


class TrainingRNNConfig(BaseModel):
    stock_symbol: str
    # Выборка
    test_rate: float = Field(default=0.3, ge=0, le=1)
    features: list[FeaturesEnum] = [FeaturesEnum.CLOSE]

    # Настройки сети
    epochs: int = 365
    learning_rate: float = Field(default=0.003, ge=0)
    regularizer: Optional[RegularizerEnum] = None
    regularizer_rate: Optional[float] = Field(default=0.0003, ge=0)

    # Скрытый слой
    hidden_neurons: int = 128
    hidden_activation: ActivationEnum = ActivationEnum.TANH


class TrainingStartResponse(BaseModel):
    """Расширенный ответ для старта обучения"""

    detail: str
    stock_symbol: str
    train_size: Optional[int] = None
    test_size: Optional[int] = None
    task_id: Optional[str] = None
