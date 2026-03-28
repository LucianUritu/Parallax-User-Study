import cv2
import numpy as np

class PupilDetector:
    def __init__(self, min_area = 20, max_area = 600, circ_thresh = 0.25):
        self.min_area = min_area
        self.max_area = max_area
        self.circ_thresh = circ_thresh
    
    def detect(self, eye_roi):
        if eye_roi is None or eye_roi.size == 0:
            return None, None

        gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = []
        for c in contours:
            area = cv2.contourArea(c)
            if self.min_area < area < self.max_area:
                per = cv2.arcLength(c, True)
                if per > 0:
                    circ = 4 * np.pi * area / (per * per)
                    if circ > self.circ_thresh:
                        valid.append((c, area))
        if not valid:
            return None, None
    
        best_contour, _ = max(valid, key=lambda x: x[1])
        M = cv2.moments(best_contour)
        if M.get("m00", 0) == 0:
            return None, None
        cx = int(M["m10"] / M ["m00"])
        cy = int(M["m01"] / M ["m00"])
        x, y, w, h = cv2.boundingRect(best_contour)
        return (cx, cy), (x, y, w, h)
    

                    