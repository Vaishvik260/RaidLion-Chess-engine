Python 3.12.9 (v3.12.9:fdb81425a9a, Feb  4 2025, 12:21:36) [Clang 13.0.0 (clang-1300.0.29.30)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
>>> from fastapi import FastAPI, WebSocket, WebSocketDisconnect
... from typing import Dict, List
... import uuid
... 
... app = FastAPI()
... 
... # Store game rooms and players
... rooms: Dict[str, Dict] = {}
... 
... @app.post("/create_room/")
... async def create_room():
...     room_id = str(uuid.uuid4())[:8]
...     rooms[room_id] = {
...         "players": [],
...         "board": [""] * 9,
...         "turn": "X"
...     }
...     return {"room_id": room_id}
... 
... 
... @app.websocket("/ws/{room_id}/{player_id}")
... async def websocket_endpoint(websocket: WebSocket, room_id: str, player_id: str):
...     await websocket.accept()
...     
...     if room_id not in rooms:
...         await websocket.send_json({"error": "Room not found"})
...         await websocket.close()
...         return
... 
...     room = rooms[room_id]
...     if len(room["players"]) >= 2:
...         await websocket.send_json({"error": "Room full"})
...         await websocket.close()
...         return
... 
...     room["players"].append({"id": player_id, "ws": websocket})
... 
...     try:
...         while True:
...             data = await websocket.receive_json()
...             move = data.get("move")
...             player_symbol = data.get("symbol")
... 
...             if room["turn"] != player_symbol:
...                 await websocket.send_json({"error": "Not your turn"})
...                 continue
... 
...             if room["board"][move] != "":
...                 await websocket.send_json({"error": "Cell occupied"})
...                 continue
... 
...             room["board"][move] = player_symbol
...             room["turn"] = "O" if player_symbol == "X" else "X"
... 
...             # Broadcast to all players
...             for player in room["players"]:
...                 await player["ws"].send_json({
...                     "board": room["board"],
...                     "turn": room["turn"],
...                     "last_move": move,
...                     "symbol": player_symbol
...                 })
... 
...     except WebSocketDisconnect:
...         print(f"Player {player_id} disconnected")
...         room["players"] = [p for p in room["players"] if p["id"] != player_id]
...         if not room["players"]:
...             del rooms[room_id]
