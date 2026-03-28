import cv2
import numpy as np
import asyncio
import websockets
import json
import threading


class PuzzleParallax:
    """Parallax viewer with focus sweet spot puzzle mechanics, supports eye tracking"""

    def __init__(self):
        self.original_img = cv2.imread("image.jpg")
        self.depth_img = cv2.imread("depth.png", 0)

        if self.original_img is None or self.depth_img is None:
            print("Error: Could not load required images")
            return

        # Resize if needed
        max_width = 800
        if self.original_img.shape[1] > max_width:
            scale = max_width / self.original_img.shape[1]
            new_width = int(self.original_img.shape[1] * scale)
            new_height = int(self.original_img.shape[0] * scale)
            self.original_img = cv2.resize(self.original_img, (new_width, new_height))
            self.depth_img = cv2.resize(self.depth_img, (new_width, new_height))

        self.h, self.w = self.original_img.shape[:2]
        self.depth_norm = self.depth_img.astype(np.float32) / 255.0

        # Mouse position (normalized -1 to 1)
        self.mouse_x = 0.0
        self.mouse_y = 0.0

        # Eye tracking variables
        self.eye_x = 0.0
        self.eye_y = 0.0
        self.gaze_connected = False

        # Sweet spot for perfect focus
        self.sweet_spot_x = 0.3
        self.sweet_spot_y = -0.2

        # Focus parameters
        self.focus_radius = 0.4  # How close to sweet spot for good focus

    # Mouse Handling (gets active when eye tracking is not connected)
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x = (x / self.w) * 2.0 - 1.0
            self.mouse_y = (y / self.h) * 2.0 - 1.0

    # Eye Tracking
    async def receive_eye_data(self, uri="ws://localhost:8765/"):
        try:
            async with websockets.connect(uri) as websocket:
                self.gaze_connected = True
                while True:
                    message = await websocket.recv()
                    data = json.loads(message)
                    self.eye_x = data.get("x", 0.0)
                    self.eye_y = data.get("y", 0.0)
        except Exception as e:
            print("Eye-tracking connection error:", e)
            self.gaze_connected = False

    def start_eye_tracking(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.receive_eye_data())

    # Parallax and Focus
    def calculate_focus_quality(self, control_x, control_y):
        distance_x = abs(control_x - self.sweet_spot_x)
        distance_y = abs(control_y - self.sweet_spot_y)
        distance = np.sqrt(distance_x ** 2 + distance_y ** 2)
        focus_quality = max(0.0, 1.0 - (distance / self.focus_radius))
        return focus_quality

    def create_puzzle_parallax(self):
        # Use eye-tracking if connected, else mouse
        control_x = self.eye_x if self.gaze_connected else self.mouse_x
        control_y = -self.eye_y if self.gaze_connected else self.mouse_y

        focus_quality = self.calculate_focus_quality(control_x, control_y)

        # Shifts based on control position and depth
        shift_x = control_x * 30
        shift_y = control_y * 30

        output = np.zeros_like(self.original_img)

        for y in range(self.h):
            for x in range(self.w):
                depth_factor = self.depth_norm[y, x]
                if depth_factor < 0.5:
                    actual_shift_x = int(shift_x * depth_factor * 2)
                    actual_shift_y = int(shift_y * depth_factor * 2)
                else:
                    actual_shift_x = -int(shift_x * (depth_factor - 0.5) * 2)
                    actual_shift_y = -int(shift_y * (depth_factor - 0.5) * 2)

                new_x = x + actual_shift_x
                new_y = y + actual_shift_y

                if 0 <= new_x < self.w and 0 <= new_y < self.h:
                    output[new_y, new_x] = self.original_img[y, x]

        if focus_quality < 1.0:
            blur_amount = int((1.0 - focus_quality) * 15) + 1
            if blur_amount % 2 == 0:
                blur_amount += 1
            output = cv2.GaussianBlur(output, (blur_amount, blur_amount), 0)

        return output, focus_quality, control_x, control_y

    # Main Loop
    def run(self):
        print("Puzzle Parallax Viewer with Eye Tracking")
        print("Move your mouse or use eye tracking to find the sweet spot!")
        print("Press Q to quit, R to reset position")

        cv2.namedWindow('Puzzle Parallax', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Puzzle Parallax', self.mouse_callback)

        # Start eye-tracking thread
        tracking_thread = threading.Thread(target=self.start_eye_tracking, daemon=True)
        tracking_thread.start()

        while True:
            parallax_img, focus_quality, control_x, control_y = self.create_puzzle_parallax()

            # Add UI feedback using actual control positions
            color = (0, 255, 0) if focus_quality > 0.8 else (0, 255, 255) if focus_quality > 0.5 else (0, 0, 255)
            info_text = f"Focus: {focus_quality:.2f} | Position: ({control_x:.2f}, {control_y:.2f})"
            cv2.putText(parallax_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if focus_quality > 0.9:
                cv2.putText(parallax_img, "PERFECT FOCUS!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            elif focus_quality > 0.7:
                cv2.putText(parallax_img, "Getting closer...", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255),
                            2)

            cv2.imshow('Puzzle Parallax', parallax_img)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.mouse_x = 0.0
                self.mouse_y = 0.0

        cv2.destroyAllWindows()


if __name__ == "__main__":
    viewer = PuzzleParallax()
    if viewer.original_img is not None:
        viewer.run()
