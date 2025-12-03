import os
from typing import Literal, Optional

import pandas as pd
from fastapi import APIRouter, Query, HTTPException, Path
from starlette import status

from config import settings
from rnn.manager import rnn_manager
from schemas import BaseSchemaResponse
from .schemas import StockInfo, StockResponseData, AvailableStocks

router = APIRouter(prefix="/stocks", tags=["Компании🏦"])


@router.get("", response_model=AvailableStocks)
async def get_available_stocks_list():
    """Получение списка акций компаний"""

    stocks = []

    for file in os.listdir(settings.files.stocks):
        if file.endswith(".txt"):
            stocks.append(StockInfo.from_filename(file))
    return AvailableStocks(total=len(stocks), stocks=stocks)


@router.get(
    "/{symbol}/history",
    response_model=StockResponseData,
    responses={
        status.HTTP_404_NOT_FOUND: {
            "model": BaseSchemaResponse,
            "description": "Информация о компании не найдена",
        }
    },
    response_model_exclude_none=True,
)
async def get_stock_history(
    symbol: str = Path(
        description="Уникальный тикер компании на бирже",
        example="AAPL",
    ),
    days: int = Query(
        default=365,
        description="Количество записей об акции компании, заканчивая самыми новыми",
    ),
    fields: Optional[list[Literal["Date", "Close", "Open", "High", "Low"]]] = Query(
        default=None,
        description="Поля для возврата. Если не указано - возвращаются все поля.",
        example="/stocks/AAPL/history?fields=Date&fields=Close",
    ),
):
    """
    Получение данных об акциях компании. Необходимо для построения первичного графика
    """
    file_path = f"{settings.files.stocks / symbol}.us.txt"

    try:
        raw_data = rnn_manager.loader.load_raw_data(source=file_path)
        data = raw_data.tail(days)

        # Сохраняем данные в датасет
        rnn_manager.raw_data = data

        if fields:
            # Проверяем, что запрошенные поля существуют в данных
            available_fields = set(raw_data.columns)
            requested_fields = set(fields)

            # Находим пересечение запрошенных и доступных полей
            valid_fields = list(requested_fields.intersection(available_fields))

            if not valid_fields:
                # Если ни одно поле не валидно, возвращаем все
                data = data.to_dict("records")
            else:
                # Возвращаем только запрошенные поля
                data = data[valid_fields].to_dict("records")
        else:
            # Если поля не указаны, возвращаем все
            data = data.to_dict("records")

        return StockResponseData(symbol=symbol, data=data, days=days)

    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Файл не найден"
        )
