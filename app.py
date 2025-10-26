# app.py
import os
import uuid
from flask import Flask, render_template, request, url_for, send_from_directory
from fleet_detector import FleetDetector, PALETTE_RGB
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(app.root_path, "uploads")
OUTPUT_FOLDER = os.path.join(app.root_path, "output")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# default model path (root of project)
MODEL_PATH = os.path.join(app.root_path, "weights", "best.pt")
fleet_detector = FleetDetector(model_path=MODEL_PATH, output_folder=OUTPUT_FOLDER)


# helper to output CSS color for swatches in template
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
            return "No file uploaded", 400
        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400

        uid = str(uuid.uuid4())
        ext = os.path.splitext(secure_filename(file.filename))[1] or ".png"
        original_name = f"{uid}{ext}"
        original_path = os.path.join(UPLOAD_FOLDER, original_name)
        file.save(original_path)

        processed_basename, grouped_summary, detections = fleet_detector.detect_and_annotate(original_path, conf=0.25)
        processed_url = url_for("output_file", filename=processed_basename)
        original_url = url_for("uploaded_file", filename=original_name)

        return render_template(
            "result.html",
            original_image_url=original_url,
            processed_image_url=processed_url,
            fleet_counts=grouped_summary
        )

    return render_template("index.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/output/<filename>")
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
