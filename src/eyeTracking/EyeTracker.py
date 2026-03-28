from __future__ import annotations
import csv
import threading
import time
import cv2
import json
from pathlib import Path
from dataclasses import dataclass

from camera.CameraPositionEstimator import CameraPositionEstimator

@dataclass
class TestState:
    testType: int = 0
    distance: int = 1
    recording: bool = False
    trial_id: int = 0
    trial_start_perf: float | None = None

class EyeTracker:
    def __init__(self, camera_index=0):
        self.cam_estimator = CameraPositionEstimator()

        self.camera_index = camera_index

        self._stop_event = threading.Event()
        self._thread = None

        self.current_gaze = {"x": 0.0, "y": 0.0, "direction": "center", "confidence": 0.0}
        self.smoothed_x = 0.0
        self.smoothed_y = 0.0
        self.base_alpha = 0.3

        # cascades
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

        #TEST LOGGING STATE
        self._lock = threading.Lock()  # protect shared state
        self.state = TestState()

        self.samples = []  # raw time-series samples
        self._seq = 0 #sequence counter for outgoing gaze packets

        self.last_frame = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def get_gaze(self):
        with self._lock:
            return self.current_gaze.copy()

    def labels(self):  #single label helper (removes duplicated label logic in main)
        t = ["linearity_horizontal", "linearity_vertical", "distance_sensitivity"][self.state.testType]
        d = ["close", "medium", "far"][self.state.distance]
        return t, d

    def handle_key(self, key: int):  #single entry point for ALL keyboard controls
        if key == 0xFF:
            return

        do_export = False
        do_clear = False

        with self._lock:
            if key == ord('1'):
                self.state.testType = 0
            elif key == ord('2'):
                self.state.testType = 1
            elif key == ord('3'):
                self.state.testType = 2
            elif key == ord('d'):
                self.state.distance = (self.state.distance + 1) % 3
            elif key == ord(' '):
                # toggle recording
                if not self.state.recording:
                    self.state.trial_id += 1
                    self.state.trial_start_perf = time.perf_counter()
                    self.state.recording = True
                else:
                    self.state.recording = False
                    self.state.trial_start_perf = None
            elif key == ord('e'):
                do_export = True
            elif key == ord('c'):
                do_clear = True

        if do_export:
            self.export_logs()
        if do_clear:
            self.clear_logs()

    def draw_overlay(self, frame):
        testType, dist = self.labels()
        rec = "ON" if self.state.recording else "OFF"
        line1 = f"MODE:{testType} | dist:{dist} | trial:{self.state.trial_id} | REC:{rec}"
        line2 = "Keys: 1=LinH 2=LinV 3=Dist  D=dist  SPACE=rec  E=export  C=clear  ESC=quit"
        cv2.putText(frame, line1, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, line2, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)

    def clear_logs(self):
        with self._lock:
            self.samples = []

    def export_logs(self):
        with self._lock:
            data = list(self.samples)

        out_dir = Path("logs")
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        json_path = out_dir / f"eye_tracking_trials_{stamp}.json"
        csv_path = out_dir / f"eye_tracking_trials_{stamp}.csv"

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        if data:
            fieldnames = sorted({k for row in data for k in row.keys()})
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(data)

        print(f"[EyeTracker] Exported {len(data)} samples to:")
        print(f"  - {json_path}")
        print(f"  - {csv_path}")


    def _run(self):
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100), maxSize=(600, 600)
            )

            best_data = self.current_gaze.copy()

            for (fx, fy, fw, fh) in faces:
                roi_gray = gray[fy:fy+fh, fx:fx+fw]
                roi_color = frame[fy:fy+fh, fx:fx+fw]

                eyes = self.eye_cascade.detectMultiScale(
                    roi_gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20), maxSize=(120, 120)
                )
                if len(eyes) == 0:
                    continue

                face_h = fh
                candidates = []
                for (ex, ey, ew, eh) in eyes:
                    # require eye center to be in upper 60% of face ROI
                    if (ey + eh * 0.5) > (face_h * 0.6):
                        continue
                    # basic size filter to avoid tiny false positives
                    if ew < 8 or eh < 6:
                        continue
                    candidates.append((ex, ey, ew, eh))

                if not candidates:
                    continue

                # sort by vertical position (top-most first) then left-to-right and keep two
                candidates.sort(key=lambda r: (r[1], r[0]))
                eyes = candidates[:2]

                per_eye = []
                for (ex, ey, ew, eh) in eyes:
                    pad = 5
                    x1 = max(0, ex - pad)
                    y1 = max(0, ey - pad)
                    x2 = min(roi_color.shape[1], ex + ew + pad)
                    y2 = min(roi_color.shape[0], ey + eh + pad)

                    eye_roi = roi_color[y1:y2, x1:x2]
                    if eye_roi.size == 0:
                        continue
                    
                    eye_center = (ew // 2, eh // 2)
                    eye_origin_in_frame = (fx + x1, fy + y1)

                    pupil_frame_x = fx + x1 + eye_center[0]
                    pupil_frame_y = fy + y1 + eye_center[1]

                    try:
                        nx, ny, conf = self.cam_estimator.estimate(
                            eye_center, eye_origin_in_frame, eye_roi.shape, frame.shape
                        )
                    except Exception:
                        nx, ny, conf = 0.0, 0.0, 0.5
                    nx = -nx
                    ny = -ny

                    per_eye.append({
                        "ex": ex, "ey": ey, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "pupil": eye_center,
                        "pupil_fx": pupil_frame_x,
                        "pupil_fy": pupil_frame_y,
                        "nx": nx, "ny": ny, "conf": float(max(0.0, conf))
                    })
                
                if not per_eye:
                    continue

                # Weighted average by confidence (fallback to simple mean if all conf==0)
                total_w = sum(p["conf"] for p in per_eye)
                if total_w > 1e-6:
                    avg_x = sum(p["nx"] * p["conf"] for p in per_eye) / total_w
                    avg_y = sum(p["ny"] * p["conf"] for p in per_eye) / total_w
                    combined_conf = max(p["conf"] for p in per_eye)
                else:
                    avg_x = sum(p["nx"] for p in per_eye) / len(per_eye)
                    avg_y = sum(p["ny"] for p in per_eye) / len(per_eye)
                    combined_conf = 0.0

                if total_w > 1e-6:
                    avg_pupil_fx = sum(p["pupil_fx"] * p["conf"] for p in per_eye) / total_w
                    avg_pupil_fy = sum(p["pupil_fy"] * p["conf"] for p in per_eye) / total_w
                else:
                    avg_pupil_fx = sum(p["pupil_fx"] for p in per_eye) / len(per_eye)
                    avg_pupil_fy = sum(p["pupil_fy"] for p in per_eye) / len(per_eye)

                adaptive_alpha = self.base_alpha * (0.5 + combined_conf * 0.5)
                self.smoothed_x = (1 - adaptive_alpha) * self.smoothed_x + adaptive_alpha * avg_x
                self.smoothed_y = (1 - adaptive_alpha) * self.smoothed_y + adaptive_alpha * avg_y

                best_data = {
                    "x": self.smoothed_x,
                    "y": self.smoothed_y,
                    "direction": "camera",
                    "confidence": combined_conf,
                    "pupil_fx": float(avg_pupil_fx),
                    "pupil_fy": float(avg_pupil_fy),
                }

                # draw visual feedback for each detected eye (keeps single-window debug)
                for p in per_eye:
                    cv2.rectangle(roi_color, (p["x1"], p["y1"]), (p["x2"], p["y2"]), (255, 0, 0), 1)
                    if p["pupil"] is not None:
                        px = int(p["pupil"][0] + p["x1"])
                        py = int(p["pupil"][1] + p["y1"])
                        cv2.circle(roi_color, (px, py), 3, (0, 0, 255), -1)
                # show camera source (hide numeric confidence in ROI label)
                info = "cam"
                cv2.putText(roi_color, info, (per_eye[0]["ex"], per_eye[0]["ey"] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            # ADD TRACE METADATA
            self._seq += 1
            best_data["seq"] = self._seq
            best_data["t_capture_ms"] = int(time.time() * 1000)

            with self._lock:
                self.current_gaze = best_data

                if self.state.recording and self.state.trial_start_perf is not None:
                    t_ms = (time.perf_counter() - self.state.trial_start_perf) * 1000.0
                    testType, dist = self.labels()

                    self.samples.append({
                        "trial_id": self.state.trial_id,
                        "testType": testType,
                        "distance": dist if self.state.testType == 2 else None,
                        "t_ms": t_ms,
                        "x": float(best_data["x"]),
                        "y": float(best_data["y"]),
                        "confidence": float(best_data.get("confidence", 0.0)),
                        "seq": int(best_data.get("seq", 0)),
                        "t_capture_ms": int(best_data.get("t_capture_ms", 0)),
                    })

            status = f"x:{best_data['x']:.2f} y:{best_data['y']:.2f}"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            self.last_frame = frame
            #cv2.putText(frame, f"Confidence: {self.current_gaze.get('confidence',0):.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cap.release()