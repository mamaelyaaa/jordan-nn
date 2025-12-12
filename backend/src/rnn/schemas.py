from typing import Optional

from pydantic import BaseModel, Field

from rnn.prepare.feature_engine import FeaturesEnum
from rnn.prepare.target_engine import TargetEnum
from rnn.structure.activation import ActivationEnum
from rnn.structure.regularizer import RegularizerEnum


class TrainingRNNConfig(BaseModel):
    stock_symbol: str = "AAPL"
    # Выборка
    test_rate: float = Field(default=0.3, ge=0, le=1)
    features: list[FeaturesEnum] = [
        FeaturesEnum.CLOSE,
        FeaturesEnum.EMA14,
        FeaturesEnum.VOLATILITY,
    ]
    target: TargetEnum = TargetEnum.CLOSE_1D

    # Настройки сети
    epochs: int = 1000
    learning_rate: float = Field(default=0.003, ge=0)
    regularizer: Optional[RegularizerEnum] = None
    regularizer_rate: Optional[float] = Field(default=0.0003, ge=0)

    # Скрытый слой
    hidden_neurons: int = 128
    hidden_activation: ActivationEnum = ActivationEnum.TANH
