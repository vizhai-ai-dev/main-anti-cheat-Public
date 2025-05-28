import cv2
import numpy as np
import dlib
from flask import Blueprint, request, jsonify
import tempfile
import os
import time

# Initialize blueprint for Flask integration
gaze_blueprint = Blueprint('gaze', __name__)

# Initialize the face detector and facial landmark predictor
face_detector = dlib.get_frontal_face_detector()
try:
    # Pretrained facial landmark predictor (shape_predictor_68_face_landmarks.dat)
    # This file needs to be downloaded separately
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
except RuntimeError:
    print("Error: Missing facial landmark predictor file.")
    print("Download from: https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2")

def shape_to_np(shape, dtype="int"):
    """Convert dlib shape object to numpy array"""
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def get_eye_landmarks(landmarks):
    """Extract eye landmarks from facial landmarks"""
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]
    return left_eye, right_eye

def calculate_eye_ratio(eye):
    """Calculate eye aspect ratio to determine if eye is open"""
    # Calculate the vertical landmarks distances
    vertical_1 = np.linalg.norm(eye[1] - eye[5])
    vertical_2 = np.linalg.norm(eye[2] - eye[4])
    
    # Calculate the horizontal distance
    horizontal = np.linalg.norm(eye[0] - eye[3])
    
    # Calculate eye aspect ratio
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

def calculate_gaze_ratio(eye, frame_gray):
    """Calculate gaze ratio to determine horizontal gaze direction"""
    # Create a mask for the eye
    height, width = frame_gray.shape
    mask = np.zeros((height, width), np.uint8)
    
    # Draw eye region as filled polygon on the mask
    eye_points = eye.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [eye_points], 255)
    
    # Apply mask to get the isolated eye
    eye_region = cv2.bitwise_and(frame_gray, frame_gray, mask=mask)
    
    # Define the minimum and maximum eye region based on eye points
    min_x = np.min(eye[:, 0])
    max_x = np.max(eye[:, 0])
    min_y = np.min(eye[:, 1])
    max_y = np.max(eye[:, 1])
    
    # Crop the eye from the mask
    gray_eye = eye_region[min_y:max_y, min_x:max_x]
    if gray_eye.size == 0:
        return 1.0  # Default value if eye region is empty
    
    # Split eye into left and right parts
    eye_width = max_x - min_x
    left_eye_region = gray_eye[:, :int(eye_width/2)]
    right_eye_region = gray_eye[:, int(eye_width/2):]
    
    # Calculate average intensity for each side
    left_side_avg = np.mean(left_eye_region) if left_eye_region.size > 0 else 0
    right_side_avg = np.mean(right_eye_region) if right_eye_region.size > 0 else 0
    
    # Avoid division by zero
    if right_side_avg == 0:
        return 1.0
    
    # Calculate gaze ratio
    gaze_ratio = left_side_avg / (right_side_avg + 0.00001)
    return gaze_ratio

def calculate_vertical_ratio(eye, frame_gray):
    """Calculate ratio to determine vertical gaze direction using pupil position"""
    # Get eye boundaries
    min_x = np.min(eye[:, 0])
    max_x = np.max(eye[:, 0])
    min_y = np.min(eye[:, 1])
    max_y = np.max(eye[:, 1])
    
    # Ensure we have a valid eye region
    if max_x <= min_x or max_y <= min_y:
        return 1.0
    
    # Extract eye region
    eye_roi = frame_gray[min_y:max_y, min_x:max_x]
    if eye_roi.size == 0:
        return 1.0
    
    # Apply threshold to find the darkest regions (likely pupil/iris)
    # Use adaptive threshold for better results
    _, thresh = cv2.threshold(eye_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_not(thresh)  # Invert so dark areas are white
    
    # Find contours to locate the pupil
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 1.0
    
    # Find the largest contour (likely the pupil)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the centroid of the pupil
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return 1.0
    
    pupil_center_y = M["m01"] / M["m00"]
    eye_height = max_y - min_y
    
    # Calculate relative position (0 = top, 1 = bottom)
    relative_y = pupil_center_y / eye_height
    
    # Convert to vertical ratio where:
    # > 1.0 means looking up
    # < 1.0 means looking down
    # ~1.0 means looking straight
    
    # Center should be around 0.5, adjust the mapping with smoother transitions
    if relative_y < 0.35:  # Upper part of eye (looking up)
        vertical_ratio = 1.0 + (0.35 - relative_y) * 3.0  # Maps to >1.0
    elif relative_y > 0.65:  # Lower part of eye (looking down)
        vertical_ratio = 1.0 - (relative_y - 0.65) * 3.0  # Maps to <1.0
    else:  # Center part (0.35 to 0.65)
        # Smooth transition around center
        deviation = abs(relative_y - 0.5)
        if deviation < 0.1:  # Very center
            vertical_ratio = 1.0
        else:
            # Gentle slope toward edges
            factor = (deviation - 0.1) / 0.05  # 0 to 1 as we move away from center
            if relative_y < 0.5:
                vertical_ratio = 1.0 + factor * 0.2  # Slight upward bias
            else:
                vertical_ratio = 1.0 - factor * 0.2  # Slight downward bias
    
    return vertical_ratio

def get_gaze_direction(frame, debug=False):
    """
    Analyze a frame to determine eye gaze direction
    
    Args:
        frame: OpenCV image frame (BGR format)
        debug: If True, return additional debugging information
    
    Returns:
        String indicating gaze direction: "center", "left", "right", "up", "down" or "unknown"
        If debug=True, returns tuple (direction, debug_info)
    """
    if frame is None:
        result = "unknown"
        return (result, {}) if debug else result
    
    # Convert frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_detector(frame_gray)
    if not faces:
        result = "unknown"
        return (result, {"error": "No faces detected"}) if debug else result
    
    # Get facial landmarks for the first detected face
    face = faces[0]
    landmarks = predictor(frame_gray, face)
    landmarks = shape_to_np(landmarks)
    
    # Get eye landmarks
    left_eye, right_eye = get_eye_landmarks(landmarks)
    
    # Calculate eye aspect ratios
    left_ear = calculate_eye_ratio(left_eye)
    right_ear = calculate_eye_ratio(right_eye)
    
    # Check if eyes are open enough
    if left_ear < 0.15 or right_ear < 0.15:
        result = "eyes_closed"
        debug_info = {
            "left_ear": left_ear,
            "right_ear": right_ear,
            "threshold": 0.15
        }
        return (result, debug_info) if debug else result
    
    # Calculate gaze ratios
    left_gaze_ratio = calculate_gaze_ratio(left_eye, frame_gray)
    right_gaze_ratio = calculate_gaze_ratio(right_eye, frame_gray)
    average_gaze_ratio = (left_gaze_ratio + right_gaze_ratio) / 2
    
    # Calculate vertical ratios
    left_vertical_ratio = calculate_vertical_ratio(left_eye, frame_gray)
    right_vertical_ratio = calculate_vertical_ratio(right_eye, frame_gray)
    average_vertical_ratio = (left_vertical_ratio + right_vertical_ratio) / 2
    
    # Determine horizontal gaze direction with balanced thresholds
    horizontal_direction = "center"
    if average_gaze_ratio > 1.15:  # Balanced threshold for left
        horizontal_direction = "left"
    elif average_gaze_ratio < 0.85:  # Balanced threshold for right
        horizontal_direction = "right"
    
    # Determine vertical gaze direction with balanced thresholds
    vertical_direction = "center"
    if average_vertical_ratio > 1.2:  # Balanced threshold for up
        vertical_direction = "up"
    elif average_vertical_ratio < 0.8:  # Balanced threshold for down
        vertical_direction = "down"
    
    # Combine directions
    if horizontal_direction == "center" and vertical_direction == "center":
        result = "center"
    elif horizontal_direction != "center" and vertical_direction == "center":
        result = horizontal_direction
    elif horizontal_direction == "center" and vertical_direction != "center":
        result = vertical_direction
    else:
        # For diagonal gazes, report the stronger component
        h_strength = abs(average_gaze_ratio - 1.0)
        v_strength = abs(average_vertical_ratio - 1.0)
        result = horizontal_direction if h_strength > v_strength else vertical_direction
    
    if debug:
        debug_info = {
            "left_ear": left_ear,
            "right_ear": right_ear,
            "left_gaze_ratio": left_gaze_ratio,
            "right_gaze_ratio": right_gaze_ratio,
            "average_gaze_ratio": average_gaze_ratio,
            "left_vertical_ratio": left_vertical_ratio,
            "right_vertical_ratio": right_vertical_ratio,
            "average_vertical_ratio": average_vertical_ratio,
            "horizontal_direction": horizontal_direction,
            "vertical_direction": vertical_direction,
            "h_strength": abs(average_gaze_ratio - 1.0),
            "v_strength": abs(average_vertical_ratio - 1.0)
        }
        return (result, debug_info)
    
    return result

def analyze_video_gaze(video_path, max_frames=30):
    """
    Analyze a video file to track gaze direction over multiple frames
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to process
    
    Returns:
        Dictionary with gaze analysis results
    """
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return {"error": "Could not open video file"}
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    # Initialize counters for each gaze direction
    gaze_directions = {
        "center": 0,
        "left": 0,
        "right": 0,
        "up": 0,
        "down": 0,
        "eyes_closed": 0,
        "unknown": 0
    }
    
    # Process frames (sample evenly across the video)
    frames_to_process = min(max_frames, frame_count)
    step = max(1, frame_count // frames_to_process)
    
    frames_processed = 0
    frames_with_face = 0
    
    for i in range(0, frame_count, step):
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Process frame to get gaze direction
        direction = get_gaze_direction(frame)
        gaze_directions[direction] += 1
        
        if direction != "unknown":
            frames_with_face += 1
        
        frames_processed += 1
        if frames_processed >= frames_to_process:
            break
    
    # Clean up
    cap.release()
    
    # Calculate the dominant gaze direction
    valid_directions = {k: v for k, v in gaze_directions.items() 
                      if k not in ["unknown", "eyes_closed"]}
    
    if sum(valid_directions.values()) > 0:
        dominant_direction = max(valid_directions.items(), key=lambda x: x[1])[0]
    else:
        dominant_direction = "unknown"
    
    # Calculate face detection rate
    face_detection_rate = frames_with_face / frames_processed if frames_processed > 0 else 0
    
    return {
        "video_stats": {
            "duration": round(duration, 2),
            "total_frames": frame_count,
            "frames_processed": frames_processed,
            "frames_with_face": frames_with_face,
            "face_detection_rate": round(face_detection_rate * 100, 2)
        },
        "gaze_counts": gaze_directions,
        "dominant_direction": dominant_direction
    }

# Define Flask route for image-based gaze tracking
@gaze_blueprint.route('/gaze', methods=['POST'])
def gaze_endpoint():
    """
    Flask endpoint to analyze eye gaze from uploaded image
    
    Expects: An image file in the request
    Returns: JSON with gaze direction results
    """
    # Check if image file was uploaded
    if 'image' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No image file uploaded'
        }), 400
    
    image_file = request.files['image']
    
    # Read image file
    img_bytes = image_file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return jsonify({
            'success': False,
            'error': 'Invalid image file'
        }), 400
    
    # Process the image to get gaze direction
    gaze_direction = get_gaze_direction(frame)
    
    return jsonify({
        'success': True,
        'gaze': {
            'direction': gaze_direction
        }
    })

# Define Flask route for video-based gaze tracking
@gaze_blueprint.route('/gaze_video', methods=['POST'])
def gaze_video_endpoint():
    """
    Flask endpoint to analyze eye gaze from uploaded video
    
    Expects: A video file in the request
    Returns: JSON with gaze direction statistics over time
    """
    # Check if a video file was uploaded
    if 'video' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No video file uploaded'
        }), 400
    
    video_file = request.files['video']
    
    # Save uploaded file to a temporary location
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    video_file.save(temp_file.name)
    temp_file.close()
    
    try:
        # Process the video
        start_time = time.time()
        results = analyze_video_gaze(temp_file.name)
        processing_time = time.time() - start_time
        
        # Clean up temporary file
        os.unlink(temp_file.name)
        
        if 'error' in results:
            return jsonify({
                'success': False,
                'error': results['error']
            }), 400
        
        return jsonify({
            'success': True,
            'processing_time': round(processing_time, 2),
            'gaze_analysis': results
        })
    
    except Exception as e:
        # Clean up on error
        try:
            os.unlink(temp_file.name)
        except:
            pass
        
        return jsonify({
            'success': False,
            'error': f'Error analyzing video: {str(e)}'
        }), 500

# This module can be used standalone or integrated with Flask
# For Flask integration: app.register_blueprint(gaze_blueprint)