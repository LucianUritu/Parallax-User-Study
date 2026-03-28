import numpy as np

class CameraPositionEstimator:
    def __init__(self, pupil_confidence = 0.8, fallback_confidence = 0.3, edge_penality = 0.5):
        self.pupil_confidence = pupil_confidence
        self.fallback_confidence = fallback_confidence
        self.edge_penality = edge_penality
    
    def estimate(self, pupil, eye_roi_origin, eye_roi_shape, frame_shape):
        frame_h, frame_w = frame_shape[0], frame_shape[1]
        origin_x, origin_y = eye_roi_origin
        roi_w = max(1, eye_roi_shape[1])
        roi_h = max(1, eye_roi_shape[0])

        if pupil is not None:
            pupil_x_frame = origin_x + float(pupil[0])
            pupil_y_frame = origin_y + float(pupil[1])
            confidence = self.pupil_confidence
        else:
            pupil_x_frame = origin_x + roi_w / 2.0
            pupil_y_frame = origin_y + roi_h / 2.0
            confidence = self.fallback_confidence
        
        center_x = frame_w / 2.0
        center_y = frame_h / 2.0

        norm_x = (pupil_x_frame - center_x) / center_x
        norm_y = (center_y - pupil_y_frame) / center_y

        norm_x = float(np.clip(norm_x, -1.0, 1.0))
        norm_y = float(np.clip(norm_y, -1.0, 1.0))

        edge_factor = max(abs(norm_x), abs(norm_y))
        confidence = max(0.0, confidence * (1.0 - edge_factor * self.edge_penality))

        return norm_x, norm_y, confidence
