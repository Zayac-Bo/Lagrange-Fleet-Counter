# Infinite Lagrange Fleet Detector (v1)

This is a minimal Python+Flask app that detects "Fleet" icons in screenshots from the game Infinite Lagrange.

Features:
- Web UI: upload PNG/JPG screenshots and get back counts per color (blue/yellow/white) and an annotated image.
- Two detection modes:
  1. **Ultralytics YOLO (recommended if you have trained weights)**: place `weights/best.pt` and the app will use `ultralytics` for inference.
  2. **Template fallback**: a lightweight OpenCV multi-scale & rotation template matcher (works without heavyweight ML libs; good for testing).

## Quick start (on Debian VPS)
1. Create a Python3 venv and activate:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
   If you want to use/train YOLO models, also install `ultralytics` (this is optional and heavy):
   ```bash
   pip install ultralytics
   ```
   See Ultralytics docs: https://docs.ultralytics.com/quickstart/ .  
3. Run the web app:
   ```bash
   python app.py
   ```
   Open `http://server-ip:5000/` and upload a screenshot.

## Training YOLO (high level)
- You already have YOLO-format labels (good). Use Ultralytics YOLOv8 to train a small **yolov8n** model:
  ```bash
  pip install ultralytics
  yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=100 imgsz=640
  ```
  See Ultralytics docs for details: https://docs.ultralytics.com/modes/train/ and https://docs.ultralytics.com/usage/python/.
- After training, put the resulting `best.pt` into `weights/best.pt`.

## Notes and next steps
- The template fallback is intentionally simple and may produce false positives/negatives. It is useful for quick testing on low-resource VPS.
- After you confirm the base flow works, we'll:
  - Integrate your labeled dataset and provide training scripts.
  - Add a small background task queue for async training jobs.
  - Improve detection (non-maximum suppression, size filtering, advanced color heuristics).
  - Add tests and GitHub CI.
