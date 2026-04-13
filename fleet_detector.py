# fleet_detector.py
import os
import cv2
import numpy as np
from collections import Counter, OrderedDict
from ultralytics import YOLO
from datetime import datetime
import subprocess

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
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))

def _rgb_to_lab(rgb):
    arr = np.uint8([[[rgb[0], rgb[1], rgb[2]]]])
    bgr = arr[:, :, ::-1]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(float)[0, 0]
    return lab

PALETTE_LAB = {name: _rgb_to_lab(rgb) for name, rgb in PALETTE_RGB.items()}
PALETTE_BGR = {name: _rgb_to_bgr(rgb) for name, rgb in PALETTE_RGB.items()}

# Pre-computed arrays for vectorised per-pixel palette voting
_PALETTE_NAMES = list(PALETTE_LAB.keys())
_PALETTE_LAB_ARR = np.array(list(PALETTE_LAB.values()), dtype=float)  # (12, 3)

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

    def _dominant_bright_color(self, crop_bgr):
        """Return dominant fleet-icon RGB using per-pixel palette voting.

        Strategy:
        1. Early-exit for "Own fleets" (white): if >=18% of crop pixels are very
           bright (V>=220) AND near-achromatic (S<=20), the icon is white. This check
           runs before the saturation filter so a crossing travel line cannot hijack
           the primary path and cause white to be missed.
        2. Saturation+brightness filter (S>=50, V>=80): remove stars (S~0) and dark
           background (V~25). All 11 non-white palette colours clear both thresholds.
        3. Per-pixel palette voting on the filtered pixels: each pixel independently
           votes for whichever palette LAB colour it is nearest to (if that distance
           is within LAB_DISTANCE_THRESHOLD). No K-means centroid blending — a pixel
           contaminated by a yellow travel line still votes for "Lemon Yellow" rather
           than dragging the blue cluster's centroid outside all thresholds.
        4. Fallback (V>=180 absolute threshold): catches "Own fleets" when very few
           coloured elements are present, and other low-saturation scenes.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return np.array([0, 0, 0], dtype=int)

        # Resize to max 64×64 for speed; INTER_AREA preserves colour distribution
        h, w = crop_bgr.shape[:2]
        if h > 64 or w > 64:
            scale = min(64.0 / h, 64.0 / w)
            crop_bgr = cv2.resize(crop_bgr, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        s_ch = hsv[:, :, 1].flatten().astype(np.float32)
        v_ch = hsv[:, :, 2].flatten().astype(np.float32)
        pixels_bgr = crop_bgr.reshape(-1, 3)
        n = len(v_ch)

        # --- Step 1: "Own fleets" early exit ---
        # Stars are sparse (<10% of crop), so >=18% very-white pixels → white fleet icon.
        white_mask = (v_ch >= 220) & (s_ch <= 20)
        if np.count_nonzero(white_mask) / n >= 0.18:
            mean_bgr = pixels_bgr[white_mask].astype(float).mean(axis=0)
            return np.array([int(round(mean_bgr[2])), int(round(mean_bgr[1])),
                             int(round(mean_bgr[0]))], dtype=int)

        # --- Step 2: Saturation+brightness filter ---
        sat_mask = (s_ch >= 50) & (v_ch >= 80)
        sat_count = int(np.count_nonzero(sat_mask))

        if sat_count >= 10:
            # Convert filtered pixels (BGR) to LAB in one batch call
            filtered_bgr = pixels_bgr[sat_mask].reshape(-1, 1, 3)
            lab_px = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)

            # --- Step 3: Per-pixel voting ---
            # Distance from each filtered pixel to every palette colour: (N, 12)
            diffs = lab_px[:, np.newaxis, :] - _PALETTE_LAB_ARR[np.newaxis, :, :]
            dists = np.linalg.norm(diffs, axis=2)           # (N, 12)
            nearest_idx = np.argmin(dists, axis=1)           # (N,)
            nearest_dist = dists[np.arange(len(dists)), nearest_idx]  # (N,)

            valid = nearest_dist <= LAB_DISTANCE_THRESHOLD
            if np.any(valid):
                votes = np.bincount(nearest_idx[valid], minlength=len(_PALETTE_NAMES))
                best_name = _PALETTE_NAMES[int(np.argmax(votes))]
                return np.array(list(PALETTE_RGB[best_name]), dtype=int)

            # No pixel within threshold — return mean of filtered pixels so
            # _match_palette can compute the true distance and return "Other"
            mean_bgr = pixels_bgr[sat_mask].astype(float).mean(axis=0)
            return np.array([int(round(mean_bgr[2])), int(round(mean_bgr[1])),
                             int(round(mean_bgr[0]))], dtype=int)

        # --- Fallback ---
        # Handles remaining "Own fleets" crops (when sat_count < 10) and dark scenes.
        # Absolute V threshold avoids the percentile collapsing to near-zero on dark BGs.
        bright_mask = v_ch >= 180
        bright_count = int(np.count_nonzero(bright_mask))
        if bright_count >= 10:
            mean_bgr = pixels_bgr[bright_mask].astype(float).mean(axis=0)
        else:
            w = (s_ch + 1e-6).reshape(-1, 1)
            mean_bgr = (pixels_bgr.astype(float) * w).sum(axis=0) / w.sum()
        return np.array([int(round(mean_bgr[2])), int(round(mean_bgr[1])),
                         int(round(mean_bgr[0]))], dtype=int)

    def _match_palette(self, rgb):
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

    def detect_and_annotate(self, image_path, conf=0.25):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        results = self.model(img, conf=conf)
        r = results[0]
        out_img = img.copy()
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
            draw_col = PALETTE_BGR.get(color_name, (200,200,200))
            cv2.rectangle(out_img, (x1, y1), (x2, y2), draw_col, 2)
            label = color_name
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(out_img, (x1, y1 - 18), (x1 + tw + 6, y1), draw_col, -1)
            cv2.putText(out_img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
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
        basename = os.path.basename(image_path)
        out_name = f"processed_{basename}"
        out_path = os.path.join(self.img_out, out_name)
        cv2.imwrite(out_path, out_img)
        return out_name, grouped, detections

    def process_video(self, video_path, conf=0.25, downscale_width=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Cannot open video: " + video_path)

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames_allowed = int(fps * 10)  # max 10 seconds
        frame_count = min(frame_count, max_frames_allowed)

        if downscale_width and downscale_width < w:
            scale = downscale_width / float(w)
            out_w = downscale_width
            out_h = int(h * scale)
        else:
            out_w, out_h = w, h

        basename = os.path.basename(video_path)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_avi = os.path.join(self.video_out, f"temp_{ts}.avi")
        writer = cv2.VideoWriter(temp_avi, cv2.VideoWriter_fourcc(*'MJPG'), fps, (out_w, out_h))

        max_total = -1
        max_frame_idx = -1
        max_counts = None
        frame_idx = 0

        while frame_idx < frame_count:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if (out_w, out_h) != (w, h):
                frame_proc = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            else:
                frame_proc = frame.copy()
            results = self.model(frame_proc, conf=conf)
            r = results[0]
            try:
                boxes = r.boxes.xyxy.cpu().numpy()
            except Exception:
                boxes = np.array([])
            per_frame_counts = Counter()
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
            if total_here > max_total:
                max_total = total_here
                max_frame_idx = frame_idx
                max_counts = dict(per_frame_counts)
            writer.write(frame_proc)

        writer.release()
        cap.release()

        # convert MJPG AVI -> H264 MP4 using ffmpeg
        final_mp4 = os.path.join(self.video_out, f"processed_{ts}_{basename}.mp4")
        cmd = [
            "ffmpeg", "-y", "-i", temp_avi,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            final_mp4
        ]
        subprocess.run(cmd, check=True)
        os.remove(temp_avi)

        # grouped summary
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

        return os.path.basename(final_mp4), grouped, max_frame_idx, max_total