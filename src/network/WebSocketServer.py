import asyncio
import json
import websockets

from eyeTracking.EyeTracker import EyeTracker

class WebSocketServer:
    def __init__(self, tracker: EyeTracker, host="localhost", port=8765):
        self.tracker = tracker
        self.host = host
        self.port = port
        self._stop = False

    async def handler(self, websocket, path=None):
        try:
            while not self._stop:
                data = self.tracker.get_gaze()
                await websocket.send(json.dumps(data))
                await asyncio.sleep(0.016)
        except websockets.exceptions.ConnectionClosed:
            pass

    async def run(self):
        async with websockets.serve(self.handler, self.host, self.port):
            print(f"WebSocket server running on ws://{self.host}:{self.port}")
            while not self._stop:
                await asyncio.sleep(0.1)
    def stop(self):
        self._stop = True
