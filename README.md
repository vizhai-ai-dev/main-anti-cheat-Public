# Multi-Person Detection Module

A Flask-compatible Python module that uses OpenCV and YOLO to detect people and faces in images/video frames.

## Features

- Person detection using YOLOv3
- Face detection using Haar Cascade
- Flask API endpoint for integration with web applications
- Returns count of people and faces in the frame

## Prerequisites

- Python 3.6+
- Download YOLOv3 model files:
  - Download YOLOv3 weights: `wget https://pjreddie.com/media/files/yolov3.weights`
  - Download YOLOv3 config: `wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg`
  - Download COCO names: `wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names`

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### As a stand-alone Flask application:

```bash
python multi_person.py
```

The Flask server will start at http://0.0.0.0:5000

### API Usage:

Send a POST request to `/multi_person_check` with JSON payload:

```json
{
  "image": "base64_encoded_image_data"
}
```

Response:

```json
{
  "people_count": 2,
  "face_count": 2,
  "processed_image": "base64_encoded_processed_image"
}
```

### Import as a module:

```python
import cv2
from multi_person import detect_multiple_people

# Read an image or capture from camera
frame = cv2.imread('test_image.jpg')
# Or from camera
# cap = cv2.VideoCapture(0)
# ret, frame = cap.read()

# Detect people and faces
results = detect_multiple_people(frame)
print(f"People detected: {results['people_count']}")
print(f"Faces detected: {results['face_count']}") 