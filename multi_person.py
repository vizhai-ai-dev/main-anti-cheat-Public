import cv2
import numpy as np
from flask import Flask, request, jsonify
import base64

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize YOLO for person detection
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load COCO class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

def detect_multiple_people(frame):
    """
    Detects people in a frame using YOLO and counts them
    
    Args:
        frame: Image frame from camera or uploaded image
        
    Returns:
        dict: Contains count of people detected, face count and processed image
    """
    height, width, _ = frame.shape
    
    # YOLO detection for people
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    # Detecting objects (people)
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Object is a person
            if confidence > 0.5 and class_id == 0:  # Class ID 0 is 'person' in COCO
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Perform non-maximum suppression to remove redundant overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    people_count = len(indexes)
    
    # Draw bounding boxes for people
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = f"Person: {confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    face_count = len(faces)
    
    # Draw bounding boxes for faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Add text showing counts
    cv2.putText(frame, f"People: {people_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Faces: {face_count}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Convert frame to jpg for web display
    _, buffer = cv2.imencode('.jpg', frame)
    img_str = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "people_count": people_count,
        "face_count": face_count,
        "processed_image": img_str
    }

# Function to decode base64 image received from Flask route
def decode_image(encoded_data):
    """Decode base64 image data to OpenCV format"""
    if 'data:image' in encoded_data:
        encoded_data = encoded_data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# Create a Flask app if this module is being used with Flask
app = Flask(__name__)

@app.route('/multi_person_check', methods=['POST'])
def multi_person_check():
    """
    Flask route for multi-person detection
    Expects a JSON with base64 encoded image in the 'image' field
    Returns counts and processed image
    """
    if 'image' not in request.json:
        return jsonify({"error": "No image provided"}), 400
    
    # Decode the image
    image_data = request.json['image']
    frame = decode_image(image_data)
    
    # Get detection results
    results = detect_multiple_people(frame)
    
    return jsonify(results)

# Run the Flask app if this script is executed directly
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000) 