import cv2
import numpy as np
from ultralytics import YOLO
import os

class FleetDetector:
    def __init__(self, model_path="weights/best.pt"):
        self.model = YOLO(model_path)

        # Define HSV color ranges for fleet colors
        self.color_map = {
            "Red": ((0,15),(160,180),(255,0,0)),
            "Yellow": ((16,35), None, (255,255,0)),
            "Green": ((36,85), None, (0,255,0)),
            "Blue": ((86,135), None, (0,0,255)),
            "Purple": ((136,159), None, (255,0,255)),
        }

    def detect(self, image_path):
        """
        Returns: processed image path, dict of counts per color
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        results = self.model.predict(img, imgsz=640, device="cpu")  # no resizing below 640
        annotated = results[0].plot()  # get annotated image as numpy array

        # Count fleets by color
        counts = {}
        for box, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            crop = img[y1:y2, x1:x2]
            hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            mean_hue = int(np.mean(hsv_crop[:,:,0]))
            detected_color = "Unknown"
            for color, (h_range, _, _) in self.color_map.items():
                if h_range[0] <= mean_hue <= h_range[1]:
                    detected_color = color
                    break
            counts[detected_color] = counts.get(detected_color, 0) + 1

        # Save processed image
        processed_path = os.path.join("/tmp/uploads", f"processed_{os.path.basename(image_path)}")
        cv2.imwrite(processed_path, annotated)

        return processed_path, counts
