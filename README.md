# Infinite Lagrange Fleet Detector (v0.4b)

<img width="1414" height="759" alt="image" src="https://github.com/user-attachments/assets/18d05db8-e74b-4659-acdb-eb88d969ebc7" />


This is a minimal Python+Flask app that detects fleets in Infinite Lagrange game. 
I only wrote small scripts before and whole this app is coded using ChatGPT and made just for my ingame Org to make fast fleet counting easier and prove that everything can be done with AI novadays.
At the moment i am busy training the model to improve detection quality.

Features:
- Web UI: upload PNG/JPG screenshots or MP4/MOV/AVI videos and get back counts per color and an annotated result.
- **Ultralytics YOLO (recommended if you have trained weights)**: place `weights/best.pt` and the app will use `ultralytics` for inference.
## Quick start
1. Create a Python3 venv and activate:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
   See Ultralytics docs: https://docs.ultralytics.com/quickstart/ .  
3. Run the web app:
   ```bash
   python app.py
   ```
   Open `http://server-ip:5000/` and upload a file.

## Training YOLO (high level)
- train a small **yolov8n** model:
  ```bash
  pip install ultralytics
  yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=100 imgsz=960
  ```
As fleets are relatively small, minimal recommended size for training is 960.
  See Ultralytics docs for details: https://docs.ultralytics.com/modes/train/ and https://docs.ultralytics.com/usage/python/.
- After training, put the resulting `best.pt` into `weights/best.pt`.

## Notes and next steps
- The template fallback is intentionally simple and may produce false positives/negatives. It is useful for quick testing on low-resource VPS.
