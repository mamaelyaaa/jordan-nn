import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import JSONResponse

from api import (
    router as main_router,
    websocket_router,
)
from api.exceptions import AppException
from config import settings
from schemas import BaseSchemaResponse

app = FastAPI(
    title=settings.api.title,
    description=settings.api.description,
    debug=settings.api.debug,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(main_router)
app.include_router(websocket_router)


@app.exception_handler(AppException)
async def handle_app_exception(request: Request, exc: AppException):
    return JSONResponse(
        status_code=exc.status_code,
        content=BaseSchemaResponse(detail=exc.message),
    )


if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        port=settings.run.port,
        host=settings.run.host,
        reload=settings.run.reload,
    )
