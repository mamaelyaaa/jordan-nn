from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


@router.websocket("/ws/training")
async def websocket_training(websocket: WebSocket):
    await websocket.accept()

    try:
        await websocket.send_json(
            {"status": "connected", "message": "WebSocket подключен успешно"}
        )

    except WebSocketDisconnect:
        print("🔌 WebSocket отключен")
    except Exception as e:
        print(f"❌ Ошибка WebSocket: {e}")
