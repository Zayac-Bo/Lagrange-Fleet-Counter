from flask import Flask, render_template, request, send_from_directory
from fleet_detector import FleetDetector
import os

app = Flask(__name__)

UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

fleet_detector = FleetDetector(model_path="weights/best.pt")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return "No file uploaded", 400

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        processed_path, fleet_counts = fleet_detector.detect(file_path)

        return render_template(
            "result.html",
            original_image_url=f"/uploads/{file.filename}",
            processed_image_url=f"/uploads/{os.path.basename(processed_path)}",
            fleet_counts=fleet_counts
        )

    return render_template("index.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
