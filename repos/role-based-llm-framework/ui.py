# ui.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from typing import List
import os

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Jinja2Templates
templates = Jinja2Templates(directory="templates")

def create_ui() -> None:
    # Initialize FastAPI for UI
    @app.get("/")
    async def home(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

def setup_dashboards() -> None:
    # Create dashboards for each role
    @app.get("/dashboard/{role}")
    async def dashboard(request: Request, role: str):
        if role in ['pm', 'coder', 'researcher']:
            return templates.TemplateResponse(f"{role}_dashboard.html", {"request": request})
        else:
            raise HTTPException(status_code=404, detail="Role not found")

def setup_notifications() -> None:
    # Implement real-time notifications for task updates and alerts
    class ConnectionManager:
        def __init__(self):
            self.active_connections: List[WebSocket] = []

        async def connect(self, websocket: WebSocket) -> None:
            await websocket.accept()
            self.active_connections.append(websocket)

        def disconnect(self, websocket: WebSocket) -> None:
            self.active_connections.remove(websocket)

        async def send_personal_message(self, message: str, websocket: WebSocket) -> None:
            await websocket.send_text(message)

        async def broadcast(self, message: str) -> None:
            for connection in self.active_connections:
                try:
                    await connection.send_text(message)
                except WebSocketDisconnect:
                    self.disconnect(connection)

    manager = ConnectionManager()

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await manager.connect(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                await manager.broadcast(f"Message text was: {data}")
        except WebSocketDisconnect:
            manager.disconnect(websocket)
        except Exception as e:
            print(f"WebSocket error: {str(e)}")
            manager.disconnect(websocket)

if __name__ == '__main__':
    import uvicorn
    create_ui()
    setup_dashboards()
    setup_notifications()
    uvicorn.run(app, host="0.0.0.0", port=8000)
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from typing import List

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Jinja2Templates
templates = Jinja2Templates(directory="templates")

def create_ui() -> None:
    # Initialize FastAPI for UI
    @app.get("/")
    async def home(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

def setup_dashboards() -> None:
    # Create dashboards for each role
    @app.get("/dashboard/{role}")
    async def dashboard(request: Request, role: str):
        if role in ['pm', 'coder', 'researcher']:
            return templates.TemplateResponse(f"{role}_dashboard.html", {"request": request})
        else:
            raise HTTPException(status_code=404, detail="Role not found")

def setup_notifications() -> None:
    # Implement real-time notifications for task updates and alerts
    class ConnectionManager:
        def __init__(self):
            self.active_connections: List[WebSocket] = []

        async def connect(self, websocket: WebSocket) -> None:
            await websocket.accept()
            self.active_connections.append(websocket)

        def disconnect(self, websocket: WebSocket) -> None:
            self.active_connections.remove(websocket)

        async def broadcast(self, message: str) -> None:
            for connection in self.active_connections:
                try:
                    await connection.send_text(message)
                except WebSocketDisconnect:
                    self.disconnect(connection)

    manager = ConnectionManager()

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await manager.connect(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                await manager.broadcast(f"Message text was: {data}")
        except WebSocketDisconnect:
            manager.disconnect(websocket)
        except Exception as e:
            print(f"WebSocket error: {str(e)}")
            manager.disconnect(websocket)

if __name__ == '__main__':
    import uvicorn
    create_ui()
    setup_dashboards()
    setup_notifications()
    uvicorn.run(app, host="0.0.0.0", port=8000)
