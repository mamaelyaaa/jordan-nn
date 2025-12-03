from fastapi import APIRouter, HTTPException
from starlette import status

from rnn.manager import rnn_manager
from rnn.prepare.feature_engine import FeaturesEnum
from rnn.prepare.loader import Dataset
from rnn.structure.activation import (
    ActivationEnum,
    ActivationProtocol,
    TanhActivation,
    ReLUActivation,
    LinearActivation,
    SigmoidActivation,
)
from rnn.structure.regularizer import RegularizerProtocol, L1, L2, RegularizerEnum
from .schemas import TrainingRNNConfig, TrainingStartResponse

router = APIRouter(prefix="/training", tags=["Обучение🔰"])


@router.post("/start", response_model=TrainingStartResponse)
async def start_training(train_config: TrainingRNNConfig):
    """Старт обучения нейронной сети"""

    # Маппинг признаков
    features_map: dict[str, str] = {
        FeaturesEnum.CLOSE: "close_rel",
    }

    # Маппинг активационных функций
    activation_map: dict[str, ActivationProtocol] = {
        ActivationEnum.TANH: TanhActivation(),
        ActivationEnum.SIGMOID: SigmoidActivation(),
        ActivationEnum.LINEAR: LinearActivation(),
        ActivationEnum.RELU: ReLUActivation(),
    }

    # Маппинг регуляризатора
    regularizer_map: dict[str, RegularizerProtocol] = {
        RegularizerEnum.L1: L1(lm=train_config.regularizer_rate),
        RegularizerEnum.L2: L2(lm=train_config.regularizer_rate),
    }

    # Устанавливаем необходимые значения для обучения модели
    rnn_manager.set_config(
        target=["target_close_1d"],
        features=[features_map[feature] for feature in train_config.features],
        learning_rate=train_config.learning_rate,
        hidden_activation=activation_map[train_config.hidden_activation],
        hidden_neurons=train_config.hidden_neurons,
        regularization=regularizer_map[train_config.regularizer],
    )

    try:
        dataset: Dataset = rnn_manager.prepare_data(
            raw_data=rnn_manager.raw_data,
            test_rate=train_config.test_rate,
        )

        # TODO Сделать обучение по эпохам с возможностью остановить, а так же отправлять события по типу вебсокетов

        # rnn_manager.train(epochs=train_config.epochs)

        return TrainingStartResponse(
            detail="Данные успешно подготовлены",
            train_size=dataset.train_size,
            test_size=dataset.test_size,
            stock_symbol=train_config.stock_symbol.upper(),
        )

    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Нет данных для начала обучения. Для начала выполните подгрузку данных",
        )


@router.post("/{session_id}/stop")
async def stop_training(session_id: int):
    """Остановка обучения"""
    pass


@router.get("/{session_id}/results")
async def get_predictions(session_id: int):
    """Получение результатов обучения"""
    pass
