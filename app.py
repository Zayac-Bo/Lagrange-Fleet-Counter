# app.py
import os
import uuid
from flask import Flask, render_template, request, url_for, send_from_directory, redirect
from werkzeug.utils import secure_filename
from fleet_detector import FleetDetector, PALETTE_RGB
import cv2

ALLOWED_IMG = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
ALLOWED_VID = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}

app = Flask(__name__)

ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(ROOT, "uploads")
OUTPUT_FOLDER = os.path.join(ROOT, "output")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join(ROOT, "weights", "best.pt")
fleet_detector = FleetDetector(model_path=MODEL_PATH, output_folder=OUTPUT_FOLDER)

# simple helper to check extension
def _ext(fname):
    return os.path.splitext(fname)[1].lower()

@app.context_processor
def inject_palette_css():
    def _palette_css(name):
        rgb = PALETTE_RGB.get(name)
        if rgb:
            return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"
        return "#cccccc"
    return dict(_palette_css=_palette_css)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part", 400
        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400

        filename = secure_filename(file.filename)
        uid = str(uuid.uuid4())[:8]
        ext = _ext(filename) or ".png"
        saved_name = f"{uid}{ext}"
        saved_path = os.path.join(UPLOAD_FOLDER, saved_name)
        file.save(saved_path)

        # decide image or video
        if ext in ALLOWED_IMG:
            # image flow
            processed_basename, grouped_summary, detections = fleet_detector.detect_and_annotate(
                saved_path, conf=0.25
            )
            processed_url = url_for("output_image", filename=processed_basename)
            original_url = url_for("uploaded_file", filename=saved_name)
            return render_template(
                "result.html",
                original_image_url=original_url,
                processed_image_url=processed_url,
                fleet_counts=grouped_summary
            )

        elif ext in ALLOWED_VID:

            # ---------------------------------------------------
            # NEW: Limit video duration to 10 seconds
            # ---------------------------------------------------
            cap = cv2.VideoCapture(saved_path)
            if not cap.isOpened():
                return "Cannot open video", 400

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frames / fps
            cap.release()

            if duration > 10.0:
                return "Video is longer than 10 seconds", 400
            # ---------------------------------------------------

            # video flow - process and redirect to video result
            processed_vid_name, grouped_summary, max_frame_idx, max_total = fleet_detector.process_video(
                saved_path, conf=0.25, downscale_width=None
            )
            video_url = url_for("output_video", filename=processed_vid_name)
            return render_template(
                "video_result.html",
                processed_video_url=video_url,
                fleet_counts=grouped_summary,
                max_frame_idx=max_frame_idx,
                max_total=max_total
            )

        else:
            return "Unsupported file type", 400

    return render_template("index.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/output/images/<filename>")
def output_image(filename):
    return send_from_directory(os.path.join(OUTPUT_FOLDER, "images"), filename)

@app.route("/output/videos/<filename>")
def output_video(filename):
    return send_from_directory(os.path.join(OUTPUT_FOLDER, "videos"), filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)