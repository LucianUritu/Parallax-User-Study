import asyncio
import cv2
from collections import deque

from eyeTracking.EyeTracker import EyeTracker
from network.WebSocketServer import WebSocketServer
from PreviewRenderer import preview_loop

async def main():
    tracker = EyeTracker()
    server = WebSocketServer(tracker)

    tracker.start()
    try:
        await asyncio.gather(server.run(), preview_loop(tracker))
    finally:
        server.stop()
        tracker.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass