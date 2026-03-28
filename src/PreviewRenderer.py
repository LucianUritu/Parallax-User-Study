import asyncio
import cv2
from collections import deque
from eyeTracking.EyeTracker import EyeTracker

async def preview_loop(tracker: EyeTracker):
    cv2.namedWindow("Eye Tracking", cv2.WINDOW_NORMAL)
    pupil_trail = deque(maxlen=120)  #store last N pupil points while recording

    while True:
        frame = tracker.last_frame
        if frame is not None:
            gaze = tracker.get_gaze()

            #pupil trail plotted only while recording
            if tracker.state.recording:
                px = gaze.get("pupil_fx", None)
                py = gaze.get("pupil_fy", None)
                if px is not None and py is not None and px >= 0 and py >= 0:
                    pupil_trail.append((int(px), int(py)))
            else:
                pupil_trail.clear()

            #draw pupil trail + current point
            for i in range(1, len(pupil_trail)):
                cv2.line(frame, pupil_trail[i - 1], pupil_trail[i], (0, 255, 255), 2)
            if pupil_trail:
                cv2.circle(frame, pupil_trail[-1], 6, (0, 0, 255), -1)

            tracker.draw_overlay(frame)
            cv2.imshow("Eye Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            tracker.stop()
            break

        tracker.handle_key(key)
        await asyncio.sleep(0)
