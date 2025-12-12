from fastapi.websockets import WebSocket


class WebSocketManager:
    """Простой менеджер WebSocket соединений"""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send_progress(self, session_id: str, data: dict):
        """Отправляет прогресс обучения через WebSocket"""
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_json(data)
            except Exception:
                self.disconnect(session_id)


websocket_manager = WebSocketManager()
