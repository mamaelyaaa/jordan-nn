from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.training.session import session_manager
from api.websockets.connection import websocket_manager

router = APIRouter()


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket для получения обновлений обучения"""

    await websocket_manager.connect(websocket, session_id)

    try:
        session = await session_manager.get_session(session_id)

        if session:
            await websocket.send_json({"type": "session_info", **session.to_dict()})

        # Ждем сообщения от клиента
        while True:
            data = await websocket.receive_json()

            if data.get("action") == "stop":
                await session_manager.stop_session(session_id)
                await websocket.send_json(
                    {"type": "action_confirmation", "action": "stop", "success": True}
                )

            elif data.get("action") == "pause":
                await session_manager.pause_session(session_id)
                await websocket.send_json(
                    {"type": "action_confirmation", "action": "pause", "success": True}
                )

            elif data.get("action") == "resume":
                await session_manager.resume_session(session_id)
                await websocket.send_json(
                    {"type": "action_confirmation", "action": "resume", "success": True}
                )

            elif data.get("action") == "ping":
                await websocket.send_json(
                    {"type": "pong", "timestamp": "2024-01-01T00:00:00"}
                )

    except WebSocketDisconnect:
        websocket_manager.disconnect(session_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        websocket_manager.disconnect(session_id)
