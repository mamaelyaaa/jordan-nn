from fastapi import APIRouter
from .stocks.views import router as stocks_router
from .training.views import router as training_router
from .websockets.views import router as websocket_router

router = APIRouter(prefix="/api")

router.include_router(stocks_router)
router.include_router(training_router)
