import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import (
    router as main_router,
    websocket_router,
)
from config import settings

app = FastAPI(
    title=settings.api.title,
    description=settings.api.description,
    debug=settings.api.debug,
)

# CORS для разработки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(main_router)
app.include_router(websocket_router)


if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        port=settings.run.port,
        host=settings.run.host,
        reload=settings.run.reload,
    )
