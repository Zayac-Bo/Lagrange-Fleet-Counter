# fleet_detector.py
import os
import cv2
import numpy as np
from collections import Counter, OrderedDict
from ultralytics import YOLO

# -------------------------
# Palette (exact values you provided)
# Keys are the display names used in tables
PALETTE_RGB = OrderedDict([
    # Friendly
    ("Light Blue (Community)", (104, 210, 255)),
    ("Blue (Squad)", (104, 144, 255)),
    ("Own fleets", (255, 255, 255)),
    # Friendly - allies
    ("Turquoise", (68, 192, 194)),
    ("Dark turquoise", (55, 140, 167)),
    ("Lilac", (106, 95, 159)),
    # Enemies
    ("Pink", (204, 64, 166)),
    ("Lemon Yellow", (180, 196, 2)),
    # Neutrals
    ("Orange", (187, 110, 51)),
    # NPC
    ("Red (Pirates)", (197, 77, 77)),
    ("Warm Yellow (Neutral)", (234, 194, 98)),
    ("Green (Friendly)", (98, 191, 96)),
])

# Group mapping for result tables (order preserved)
GROUPS = OrderedDict([
    ("Enemies", ["Pink", "Lemon Yellow"]),
    ("Friendly", ["Light Blue (Community)", "Blue (Squad)", "Own fleets"]),
    ("Friendly - allies", ["Turquoise", "Dark turquoise", "Lilac"]),
    ("NPC", ["Red (Pirates)", "Warm Yellow (Neutral)", "Green (Friendly)"]),
    ("Neutrals", ["Orange"]),
])

# LAB threshold: if distance > this -> classify as Other
LAB_DISTANCE_THRESHOLD = 25.0

# Utility conversions: store palette in LAB (for perceptual distances) and BGR (for cv2 drawing)
def _rgb_to_bgr(rgb):
    # rgb is (R,G,B) -> return BGR tuple for cv2
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))

def _rgb_to_lab(rgb):
    # rgb: (R,G,B)
    arr = np.uint8([[[rgb[0], rgb[1], rgb[2]]]])  # RGB to BGR conversion will be handled by cv2 by swapping channels if needed
    # cv2.cvtColor expects BGR, so convert explicitly: swap to BGR first
    bgr = arr[:, :, ::-1]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(float)[0, 0]
    return lab

PALETTE_LAB = {name: _rgb_to_lab(rgb) for name, rgb in PALETTE_RGB.items()}
PALETTE_BGR = {name: _rgb_to_bgr(rgb) for name, rgb in PALETTE_RGB.items()}

# -------------------------
class FleetDetector:
    def __init__(self, model_path="weights/best.pt", output_folder="output"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at '{model_path}'. Put your best.pt there or provide correct path.")
        self.model = YOLO(model_path)
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def _dominant_bright_color(self, crop_bgr):
        """
        Compute a brightness-weighted dominant color:
        - Convert crop to HSV
        - Use V channel as weights (higher V => brighter pixels)
        - Compute weighted average of RGB using those weights after converting BGR->RGB
        Returns integer RGB tuple (R,G,B)
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return np.array([0,0,0], dtype=int)

        # Convert to HSV (OpenCV uses BGR input)
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV).astype(float)
        v = hsv[:, :, 2]  # 0..255 brightness
        weights = v + 1e-6  # avoid zeros

        # Convert to RGB for consistency with palette (currently crop is BGR)
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(float)

        # Weighted mean for each channel
        wsum = weights.sum()
        if wsum <= 0:
            avg = crop_rgb.reshape(-1, 3).mean(axis=0)
            return avg.astype(int)

        avg_r = (crop_rgb[:, :, 0] * weights).sum() / wsum
        avg_g = (crop_rgb[:, :, 1] * weights).sum() / wsum
        avg_b = (crop_rgb[:, :, 2] * weights).sum() / wsum

        return np.array([int(round(avg_r)), int(round(avg_g)), int(round(avg_b))], dtype=int)

    def _match_palette(self, rgb):
        """Return (best_name, distance). Uses LAB perceptual distance."""
        lab = _rgb_to_lab(tuple(rgb.tolist() if isinstance(rgb, np.ndarray) else rgb))
        best_name = "Other"
        best_dist = float("inf")
        for name, ref_lab in PALETTE_LAB.items():
            d = np.linalg.norm(lab - ref_lab)
            if d < best_dist:
                best_dist = d
                best_name = name
        if best_dist <= LAB_DISTANCE_THRESHOLD:
            return best_name, float(best_dist)
        return "Other", float(best_dist)

    def detect_and_annotate(self, image_path, conf=0.25):
        """
        Run YOLO detection on image_path, classify each bbox color, annotate and save image.
        Returns:
            processed_basename (str),
            grouped_summary (dict),
            detections (list of dicts)
        grouped_summary format:
            {
              "Enemies": {"Pink": 12, "Lemon Yellow": 5, "TOTAL": 17},
              "Friendly": {..., "TOTAL": n},
              ...
            }
        Detections: list of dicts {x1,y1,x2,y2,color_name,dist}
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # run inference
        results = self.model(img, conf=conf)  # returns list-like; single image -> results[0]
        r = results[0]

        # build output image (we'll draw our own boxes/labels)
        try:
            base = r.plot()  # use YOLO plotting as base (optional), then redraw with palette colors
            out_img = base.copy()
        except Exception:
            out_img = img.copy()

        # collect boxes
        boxes = []
        try:
            boxes = r.boxes.xyxy.cpu().numpy()
        except Exception:
            boxes = np.array([])

        counts = Counter()
        detections = []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            # clip
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(img.shape[1]-1, x2); y2 = min(img.shape[0]-1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = img[y1:y2, x1:x2]
            dom_rgb = self._dominant_bright_color(crop)
            color_name, dist = self._match_palette(dom_rgb)

            counts[color_name] += 1
            detections.append({
                "box": (x1, y1, x2, y2),
                "color": color_name,
                "distance": dist,
                "dominant_rgb": tuple(int(x) for x in dom_rgb.tolist())
            })

            # draw rectangle & label with palette color (BGR)
            draw_col = PALETTE_BGR.get(color_name, (200, 200, 200))
            cv2.rectangle(out_img, (x1, y1), (x2, y2), draw_col, 2)
            label = color_name
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(out_img, (x1, y1 - 18), (x1 + tw + 6, y1), draw_col, -1)
            cv2.putText(out_img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

        # build grouped summary (only include groups that have >0 counts)
        grouped = OrderedDict()
        for gname, members in GROUPS.items():
            group_counts = {}
            total = 0
            for m in members:
                cnt = counts.get(m, 0)
                if cnt > 0:
                    group_counts[m] = int(cnt)
                    total += int(cnt)
            if total > 0:
                group_counts["TOTAL"] = int(total)
                grouped[gname] = group_counts

        # include "Other" if present
        other_cnt = counts.get("Other", 0)
        if other_cnt > 0:
            grouped["Other"] = {"Other": int(other_cnt), "TOTAL": int(other_cnt)}

        # save annotated output image
        basename = os.path.basename(image_path)
        out_name = f"processed_{basename}"
        out_path = os.path.join(self.output_folder, out_name)
        cv2.imwrite(out_path, out_img)

        return out_name, grouped, detections
