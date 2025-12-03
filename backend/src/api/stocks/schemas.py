from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class StockInfo(BaseModel):
    symbol: str
    name: str

    @classmethod
    def from_filename(cls, filename: str) -> "StockInfo":
        symbol = filename.split(".")[0]
        company_names = {
            "aapl": "Apple Inc.",
            "googl": "Alphabet Inc. (Google)",
            "msft": "Microsoft Corporation",
            "tsla": "Tesla Inc.",
            "amzn": "Amazon.com Inc.",
            "meta": "Meta Platforms Inc.",
            "nvda": "NVIDIA Corporation",
            "jpm": "JPMorgan Chase & Co.",
            "ibm": "IBM",
        }
        return cls(symbol=symbol.upper(), name=company_names[symbol])


class AvailableStocks(BaseModel):
    total: int
    stocks: list[StockInfo]


class StockDailyData(BaseModel):
    date: Optional[datetime] = Field(default=None, alias="Date")
    high: Optional[float] = Field(default=None, alias="High")
    open: Optional[float] = Field(default=None, alias="Open")
    close: Optional[float] = Field(default=None, alias="Close")
    low: Optional[float] = Field(default=None, alias="Low")


class StockResponseData(BaseModel):
    symbol: str
    days: int
    data: list[StockDailyData]
