from flask import Flask, request, render_template, send_from_directory
import os
import glob
from detector import FleetDetector

# --------------------------
# Configuration
# --------------------------

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

detector = FleetDetector(n_clusters=3)  # auto color detection + naming

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

# --------------------------
# Helpers
# --------------------------

def allowed_file(filename):
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED_EXTENSIONS

def clear_folders():
    """Remove all files in uploads/ and outputs/"""
    for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
        files = glob.glob(f"{folder}/*")
        for f in files:
            os.remove(f)

# --------------------------
# Routes
# --------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    counts = {}
    annotated_filename = None

    if request.method == "POST":
        # Clear old files
        clear_folders()

        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            # Detect fleets and cluster colors
            img, detections = detector.detect(filepath)
            counts = detector.count_colors(detections)  # now uses color names

            # Save annotated image
            annotated_filename = f"annotated_{file.filename}"
            output_path = os.path.join(app.config["OUTPUT_FOLDER"], annotated_filename)
            detector.save_annotated(img, detections, output_path)

    return render_template("index.html", counts=counts, annotated=annotated_filename)

@app.route("/outputs/<filename>")
def output_file(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)

# --------------------------
# Run
# --------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
