# fleet_detector.py
import os
import cv2
import numpy as np
from collections import Counter, OrderedDict, defaultdict
from ultralytics import YOLO
from datetime import datetime

# === Palette (RGB) ===
PALETTE_RGB = OrderedDict([
    ("Light Blue (Community)", (104, 210, 255)),
    ("Blue (Squad)", (104, 144, 255)),
    ("Own fleets", (255, 255, 255)),
    ("Turquoise", (68, 192, 194)),
    ("Dark turquoise", (55, 140, 167)),
    ("Lilac", (106, 95, 159)),
    ("Pink", (204, 64, 166)),
    ("Lemon Yellow", (180, 196, 2)),
    ("Orange", (187, 110, 51)),
    ("Red (Pirates)", (197, 77, 77)),
    ("Warm Yellow (Neutral)", (234, 194, 98)),
    ("Green (Friendly)", (98, 191, 96)),
])

# mapping groups for summary tables
GROUPS = OrderedDict([
    ("Enemies", ["Pink", "Lemon Yellow"]),
    ("Friendly", ["Light Blue (Community)", "Blue (Squad)", "Own fleets"]),
    ("Friendly - allies", ["Turquoise", "Dark turquoise", "Lilac"]),
    ("NPC", ["Red (Pirates)", "Warm Yellow (Neutral)", "Green (Friendly)"]),
    ("Neutrals", ["Orange"]),
])

# constants
LAB_DISTANCE_THRESHOLD = 25.0

def _rgb_to_bgr(rgb):
    # rgb (R, G, B) -> cv2 BGR tuple
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))

def _rgb_to_lab(rgb):
    arr = np.uint8([[[rgb[0], rgb[1], rgb[2]]]])
    bgr = arr[:, :, ::-1]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(float)[0, 0]
    return lab

PALETTE_LAB = {name: _rgb_to_lab(rgb) for name, rgb in PALETTE_RGB.items()}
PALETTE_BGR = {name: _rgb_to_bgr(rgb) for name, rgb in PALETTE_RGB.items()}

class FleetDetector:
    def __init__(self, model_path="weights/best.pt", output_folder=None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at '{model_path}'")
        self.model = YOLO(model_path)
        # output folders
        base_out = output_folder or os.path.join(os.getcwd(), "output")
        self.output_folder = base_out
        self.img_out = os.path.join(self.output_folder, "images")
        self.video_out = os.path.join(self.output_folder, "videos")
        os.makedirs(self.img_out, exist_ok=True)
        os.makedirs(self.video_out, exist_ok=True)

    # --- dominant color with 30% bright-region requirement (robust) ---
    def _dominant_bright_color(self, crop_bgr):
        """
        Return RGB tuple (R,G,B). Use top-30%-bright pixels mean if they cover >=30% of crop,
        otherwise fallback to brightness-weighted mean.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return np.array([0,0,0], dtype=int)

        # convert to HSV and RGB
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV).astype(float)
        v = hsv[:, :, 2].flatten()  # brightness
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(float)
        flat_rgb = crop_rgb.reshape(-1, 3)

        total_pixels = flat_rgb.shape[0]
        if total_pixels == 0:
            return np.array([0,0,0], dtype=int)

        # threshold for top 30% brightest
        thresh = np.percentile(v, 70)  # 70th percentile -> top 30%
        bright_mask = v >= thresh
        bright_count = np.count_nonzero(bright_mask)
        bright_ratio = bright_count / float(total_pixels)

        if bright_count > 0 and bright_ratio >= 0.30:
            # use mean of bright pixels
            bright_pixels = flat_rgb[bright_mask]
            mean_rgb = bright_pixels.mean(axis=0)
            return np.array([int(round(x)) for x in mean_rgb], dtype=int)
        else:
            # fallback: brightness-weighted mean across all pixels
            w = v.reshape(-1, 1) + 1e-6
            weighted = (flat_rgb * w).sum(axis=0) / w.sum()
            return np.array([int(round(x)) for x in weighted], dtype=int)

    def _match_palette(self, rgb):
        """Return (best_name, distance) using LAB distance; if distance > threshold return Other"""
        # ensure rgb is iterable of 3 ints
        lab = _rgb_to_lab(tuple(int(x) for x in rgb))
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

    # --- image processing (unchanged public API) ---
    def detect_and_annotate(self, image_path, conf=0.25):
        """
        Process single image:
        returns (processed_basename, grouped_summary, detections_list)
        processed_basename is filename placed into self.img_out (basename only)
        grouped_summary: OrderedDict as before (only non-empty groups)
        detections_list: list of dicts with box, color, distance, dominant_rgb
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # run inference (no plotting)
        results = self.model(img, conf=conf)
        r = results[0]

        # start with original (do not use r.plot())
        out_img = img.copy()

        # boxes
        try:
            boxes = r.boxes.xyxy.cpu().numpy()
        except Exception:
            boxes = np.array([])

        counts = Counter()
        detections = []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
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

        # grouped summary
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

        other_cnt = counts.get("Other", 0)
        if other_cnt > 0:
            grouped["Other"] = {"Other": int(other_cnt), "TOTAL": int(other_cnt)}

        # save annotated image
        basename = os.path.basename(image_path)
        out_name = f"processed_{basename}"
        out_path = os.path.join(self.img_out, out_name)
        cv2.imwrite(out_path, out_img)

        return out_name, grouped, detections

    # --- new: video processing ---
    def process_video(self, video_path, conf=0.25, downscale_width=None):
        """
        Process a video frame-by-frame, annotate frames with our custom rectangles/labels,
        save processed video into self.video_out and return:
          (processed_filename, grouped_summary_for_max_frame, frame_index_of_max, max_counts_per_color)
        grouped_summary_for_max_frame will be OrderedDict same as image case.
        max_counts_per_color is dict mapping color_name->count for the frame with maximum total fleets.
        """

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Cannot open video: " + video_path)

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # optional downscale to save CPU
        if downscale_width and downscale_width < w:
            scale = downscale_width / float(w)
            out_w = downscale_width
            out_h = int(h * scale)
        else:
            out_w, out_h = w, h

        # output file
        basename = os.path.basename(video_path)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_filename = f"processed_{ts}_{basename}"
        out_path = os.path.join(self.video_out, out_filename)

        # VideoWriter (mp4v)
        fourcc = cv2.VideoWriter_fourcc(*"H264")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

        max_total = -1
        max_frame_idx = -1
        max_counts = None  # dict color->count for that frame
        frame_idx = 0

        # iterate frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # optionally resize frame (keep aspect)
            if (out_w, out_h) != (w, h):
                frame_proc = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            else:
                frame_proc = frame.copy()

            # run inference on frame (returns list-like)
            results = self.model(frame_proc, conf=conf)
            r = results[0]
            try:
                boxes = r.boxes.xyxy.cpu().numpy()
            except Exception:
                boxes = np.array([])

            # per-frame color counts
            per_frame_counts = Counter()
            # annotate frame_proc
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(frame_proc.shape[1]-1, x2); y2 = min(frame_proc.shape[0]-1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame_proc[y1:y2, x1:x2]
                dom_rgb = self._dominant_bright_color(crop)
                color_name, dist = self._match_palette(dom_rgb)
                per_frame_counts[color_name] += 1

                draw_col = PALETTE_BGR.get(color_name, (200,200,200))
                cv2.rectangle(frame_proc, (x1, y1), (x2, y2), draw_col, 2)
                label = color_name
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame_proc, (x1, y1 - 18), (x1 + tw + 6, y1), draw_col, -1)
                cv2.putText(frame_proc, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

            total_here = sum(per_frame_counts.values())
            # update max frame info
            if total_here > max_total:
                max_total = total_here
                max_frame_idx = frame_idx
                # store counts mapping color->count for this frame
                max_counts = dict(per_frame_counts)

            # write frame
            writer.write(frame_proc)

        writer.release()
        cap.release()

        # build grouped summary based on max_counts
        grouped = OrderedDict()
        if max_counts is None:
            max_counts = {}

        for gname, members in GROUPS.items():
            group_counts = {}
            total = 0
            for m in members:
                cnt = max_counts.get(m, 0)
                if cnt > 0:
                    group_counts[m] = int(cnt)
                    total += int(cnt)
            if total > 0:
                group_counts["TOTAL"] = int(total)
                grouped[gname] = group_counts

        other_cnt = max_counts.get("Other", 0)
        if other_cnt > 0:
            grouped["Other"] = {"Other": int(other_cnt), "TOTAL": int(other_cnt)}

        return out_filename, grouped, max_frame_idx, max_total
