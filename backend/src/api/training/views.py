from fastapi import APIRouter, HTTPException, BackgroundTasks
from starlette import status

from api.stocks.schemas import StockDailyData
from api.training.schemas import (
    TrainingResultsSchema,
    SessionResponseSchema,
    SessionSchema,
)
from api.training.service import training_service
from api.training.session import session_manager
from api.websockets.schemas import TrainingStartResponse
from config import settings
from rnn.schemas import TrainingRNNConfig
from schemas import BaseSchemaResponse

router = APIRouter(prefix="/training", tags=["Обучение🔰"])


@router.post("/start", response_model=TrainingStartResponse)
async def start_training(
    train_config: TrainingRNNConfig, background_tasks: BackgroundTasks
):
    """
    Старт обучения нейронной сети

    Выводит уникальный id сессии, который нужно использовать для подключения к вебсокету
    """

    session_id = await session_manager.create_session(train_config)

    config_dict = train_config.model_dump(exclude_none=True, exclude_unset=True)
    background_tasks.add_task(training_service.run_training, session_id, config_dict)

    return TrainingStartResponse(
        session_id=session_id,
        detail="Обучение начато",
    )


@router.get(
    "/{session_id}/results",
    response_model=TrainingResultsSchema,
    responses={
        404: {"model": BaseSchemaResponse, "description": "Сессия не найдена"},
        400: {"model": BaseSchemaResponse, "description": "Обучение еще не завершено"},
    },
)
async def get_training_results(session_id: str):
    """
    Получение результатов обучения с историческими данными

    Запускается после остановки получения результатов от вебсокетов
    """

    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Сессия не найдена"
        )

    if session.status not in ["completed", "stopped", "error"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Обучение еще не завершено"
        )

    # Получаем символ акции из конфига сессии
    stock_symbol = (
        session.config.stock_symbol
        if hasattr(session.config, "stock_symbol")
        else "AAPL"
    )

    # Загружаем исторические данные
    from rnn.manager import rnn_manager

    try:
        # Загружаем данные (они могут быть уже загружены)
        if not hasattr(rnn_manager, "raw_data") or rnn_manager.raw_data is None:
            file_path = f"{settings.files.stocks / stock_symbol.lower()}.us.txt"
            rnn_manager.raw_data = rnn_manager.loader.load_raw_data(source=file_path)

        # Берем последние N дней
        days = session.config.days if hasattr(session.config, "days") else 365
        raw_data = rnn_manager.raw_data.tail(days)

        # Конвертируем в список словарей
        data_list = raw_data.reset_index().to_dict("records")

        # Форматируем для StockDailyData
        formatted_data = []
        for record in data_list:
            # Приводим ключи к нужному формату
            formatted_record = {
                "Date": record.get("Date") if "Date" in record else record.get("index"),
                "High": record.get("High"),
                "Open": record.get("Open"),
                "Close": record.get("Close"),
                "Low": record.get("Low"),
            }
            formatted_data.append(StockDailyData(**formatted_record))

        return {
            "session_id": session_id,
            "status": session.status,
            "raw_data": formatted_data,
            "loss_history": session.loss_history,
            "predictions": session.predictions,
        }

    except Exception as e:
        # Если не удалось загрузить данные, возвращаем без них
        print(f"Не удалось загрузить исторические данные: {e}")
        return {
            "session_id": session_id,
            "status": session.status,
            "raw_data": [],
            "loss_history": session.loss_history,
            "predictions": session.predictions,
        }


@router.get("/sessions", response_model=SessionResponseSchema)
async def list_sessions():
    """Получение списка всех сессий"""

    sessions_list = []
    for session_id, session in session_manager.sessions.items():
        sessions_list.append(
            {
                "session_id": session_id,
                "status": session.status,
                "stock_symbol": (
                    session.config.stock_symbol
                    if hasattr(session.config, "stock_symbol")
                    else "unknown"
                ),
                "created_at": (
                    session.start_time.isoformat() if session.start_time else None
                ),
            }
        )

    return SessionResponseSchema(
        sessions=[SessionSchema.model_validate(session) for session in sessions_list],
        total=len(sessions_list),
    )
