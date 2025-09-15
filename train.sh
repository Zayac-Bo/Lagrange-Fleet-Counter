#!/bin/bash
# Example training command for Ultralytics YOLOv8 (requires ultralytics installed)
# Replace data.yaml with your dataset descriptor.
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=100 imgsz=640
