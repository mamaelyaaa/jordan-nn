from datetime import datetime

from pydantic import BaseModel, Field

from api.stocks.schemas import StockDailyData


class PredictionsSchema(BaseModel):
    train_prediction: list[float]
    test_prediction: list[float]


class TrainingResultsSchema(BaseModel):
    session_id: str
    status: str
    loss_history: list[float]
    raw_data: list[StockDailyData]
    predictions: PredictionsSchema


class SessionSchema(BaseModel):
    session_id: str
    status: str
    stock_symbol: str
    created_at: datetime


class SessionResponseSchema(BaseModel):
    sessions: list[SessionSchema]
    total: int
