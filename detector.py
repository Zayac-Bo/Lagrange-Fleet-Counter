import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter
from sklearn.cluster import KMeans
import os

# --------------------------
# Configuration
# --------------------------

YOLO_MODEL_PATH = "weights/best.pt"
DEFAULT_CLUSTERS = 3  # expected number of main fleet colors

# Standard HSV colors for naming
COLOR_NAMES = {
    "red":    np.array([0, 180, 180]),
    "yellow": np.array([30, 180, 180]),
    "green":  np.array([60, 180, 180]),
    "blue":   np.array([120, 180, 180]),
    "purple": np.array([150, 180, 180]),
    "white":  np.array([0, 0, 200]),
}

# --------------------------
# Helper Functions
# --------------------------

def closest_color_name(hsv_color):
    """Return the name of the closest standard color"""
    min_dist = float('inf')
    best_name = None
    for name, ref in COLOR_NAMES.items():
        dist = np.linalg.norm(hsv_color - ref)
        if dist < min_dist:
            min_dist = dist
            best_name = name
    return best_name

def annotate_image(img, detections):
    """Draw bounding boxes and color names"""
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        label = det['color_name']
        color_rgb = tuple(int(c) for c in det['cluster_color'])
        cv2.rectangle(img, (x1, y1), (x2, y2), color_rgb[::-1], 2)
        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_rgb[::-1], 1)
    return img

# --------------------------
# Main Detection Class
# --------------------------

class FleetDetector:
    def __init__(self, model_path=YOLO_MODEL_PATH, n_clusters=DEFAULT_CLUSTERS):
        self.n_clusters = n_clusters
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            self.model = None
            print("YOLO model not found. Please train or place weights in", model_path)

    def detect(self, img_path, conf_thresh=0.3):
        img = cv2.imread(img_path)
        detections = []

        if self.model:
            results = self.model(img_path, conf=conf_thresh)
            # Collect crops and their mean colors
            colors = []
            boxes = []
            for r in results:
                for box in r.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box)
                    crop = img[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                    mean_color = np.mean(hsv_crop.reshape(-1,3), axis=0)
                    colors.append(mean_color)
                    boxes.append([x1, y1, x2, y2])

            # Apply K-means clustering on colors
            if colors:
                kmeans = KMeans(n_clusters=min(self.n_clusters, len(colors)), random_state=42)
                kmeans.fit(colors)
                labels = kmeans.labels_
                centers = kmeans.cluster_centers_

                # Assign cluster info and name to each detection
                for i, bbox in enumerate(boxes):
                    cluster_label = labels[i]
                    cluster_color = centers[cluster_label]
                    color_name = closest_color_name(cluster_color)
                    detections.append({
                        "bbox": bbox,
                        "cluster_label": cluster_label,
                        "cluster_color": cluster_color,
                        "color_name": color_name
                    })

        return img, detections

    def count_colors(self, detections):
        """Count fleets per color name"""
        return Counter([d['color_name'] for d in detections])

    def save_annotated(self, img, detections, output_path):
        annotated = annotate_image(img.copy(), detections)
        cv2.imwrite(output_path, annotated)
        return output_path

# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    detector = FleetDetector(n_clusters=3)
    img, dets = detector.detect("uploads/test.png")
    counts = detector.count_colors(dets)
    print("Detected fleet counts by color:", counts)
    detector.save_annotated(img, dets, "outputs/test_annotated.png")
