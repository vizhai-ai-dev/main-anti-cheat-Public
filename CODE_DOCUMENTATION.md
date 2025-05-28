# AI Proctoring System - Technical Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [AI Models and Libraries Used](#ai-models-and-libraries-used)
4. [Core Components](#core-components)
5. [Data Flow](#data-flow)
6. [API Endpoints](#api-endpoints)
7. [Scoring System](#scoring-system)
8. [Installation and Setup](#installation-and-setup)
9. [Usage Examples](#usage-examples)

## System Overview

This AI-powered proctoring system is designed to monitor and analyze exam sessions through comprehensive video and audio analysis. The system combines multiple AI models and computer vision techniques to detect potential cheating behaviors including gaze tracking, multiple person detection, lip-sync analysis, and audio anomaly detection.

### Key Features
- **Real-time video analysis** with gaze tracking
- **Multi-person detection** using YOLO
- **Lip-sync detection** for audio-visual synchronization
- **Audio analysis** for multiple speakers and background noise
- **Comprehensive scoring system** with risk assessment
- **Web-based interface** using Streamlit
- **RESTful API** for integration

## Architecture

The system follows a modular architecture with the following main components:

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Web Interface                   │
│                        (app.py)                             │
├─────────────────────────────────────────────────────────────┤
│                   Analysis Orchestrator                      │
│                      (run_all.py)                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────┐ │
│  │ Gaze        │ │ Multi-Person│ │ Lip Sync    │ │ Audio    │ │
│  │ Tracking    │ │ Detection   │ │ Detection   │ │ Analysis │ │
│  │             │ │             │ │             │ │          │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └──────────┘ │
├─────────────────────────────────────────────────────────────┤
│                   Cheat Score Calculator                     │
│                     (cheat_score.py)                        │
└─────────────────────────────────────────────────────────────┘
```

## AI Models and Libraries Used

### 1. Computer Vision Models

#### YOLO (You Only Look Once) v3
- **Purpose**: Real-time object detection for person detection
- **Files**: `yolov3.weights`, `yolov3.cfg`, `coco.names`
- **Model Size**: ~237MB
- **Usage**: Detects multiple people in video frames
- **Accuracy**: Detects persons with confidence threshold > 0.5

#### Dlib Facial Landmark Predictor
- **Purpose**: Facial landmark detection for gaze tracking
- **File**: `shape_predictor_68_face_landmarks.dat` (95MB)
- **Points**: 68 facial landmarks including eye regions
- **Usage**: Eye aspect ratio calculation and gaze direction determination

#### OpenCV Haar Cascades
- **Purpose**: Face detection
- **Model**: `haarcascade_frontalface_default.xml` (built-in OpenCV)
- **Usage**: Quick face detection for person counting verification

#### MediaPipe Face Mesh
- **Purpose**: High-precision facial landmark detection
- **Points**: 468 3D facial landmarks
- **Usage**: Lip landmark extraction for lip-sync analysis
- **Accuracy**: Sub-pixel precision for lip movement tracking

### 2. Audio Processing Libraries

#### Librosa
- **Purpose**: Audio feature extraction and analysis
- **Features Used**:
  - MFCC (Mel-Frequency Cepstral Coefficients)
  - Spectral contrast
  - RMS energy
  - Spectral flatness
- **Usage**: Speaker detection, noise analysis, speech activity detection

#### Face Recognition Library
- **Purpose**: Alternative facial landmark detection
- **Backend**: Dlib-based deep learning models
- **Usage**: Backup method for lip landmark extraction

## Core Components

### 1. Gaze Tracking (`gaze_tracking.py`)

**Models Used:**
- Dlib frontal face detector
- 68-point facial landmark predictor

**How it works:**
1. **Face Detection**: Uses Dlib's HOG-based face detector
2. **Landmark Extraction**: Identifies 68 facial landmarks
3. **Eye Region Isolation**: Extracts eye landmarks (points 36-47)
4. **Gaze Calculation**: 
   - Calculates Eye Aspect Ratio (EAR) to detect blinks
   - Computes horizontal gaze ratio by comparing left/right eye intensities
   - Computes vertical gaze ratio by comparing top/bottom eye intensities
5. **Direction Classification**: Maps ratios to gaze directions (center, left, right, up, down)

**Key Algorithms:**
```python
# Eye Aspect Ratio calculation
def calculate_eye_ratio(eye):
    vertical_1 = np.linalg.norm(eye[1] - eye[5])
    vertical_2 = np.linalg.norm(eye[2] - eye[4])
    horizontal = np.linalg.norm(eye[0] - eye[3])
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

# Gaze ratio for horizontal direction
def calculate_gaze_ratio(eye, frame_gray):
    # Isolates eye region and compares left vs right intensities
    gaze_ratio = left_side_avg / (right_side_avg + 0.00001)
    return gaze_ratio
```

### 2. Multi-Person Detection (`multi_person.py`)

**Models Used:**
- YOLOv3 object detection model
- OpenCV Haar Cascade for face detection

**How it works:**
1. **YOLO Processing**:
   - Creates 416x416 blob from input frame
   - Runs inference through YOLOv3 network
   - Filters detections for 'person' class (ID=0) with confidence > 0.5
   - Applies Non-Maximum Suppression to remove duplicate detections
2. **Face Detection**:
   - Converts frame to grayscale
   - Applies Haar Cascade classifier
   - Counts detected faces
3. **Result Combination**: Returns person count, face count, and annotated image

**YOLO Configuration:**
- Input size: 416x416 pixels
- Confidence threshold: 0.5
- NMS threshold: 0.4
- Classes: 80 COCO classes (focuses on 'person')

### 3. Lip Sync Detection (`lip_sync_detector.py`)

**Models Used:**
- MediaPipe Face Mesh (468 landmarks)
- Face Recognition library (Dlib-based)

**How it works:**
1. **Lip Landmark Extraction**:
   - MediaPipe: Extracts 20 key lip landmarks
   - Face Recognition: Extracts top and bottom lip contours
2. **Lip Distance Calculation**:
   - Measures vertical distance between upper and lower lip
   - Calculates average lip openness per frame
3. **Audio Processing**:
   - Extracts audio using PyDub
   - Detects speech activity using RMS energy
   - Maps audio frames to video frames
4. **Synchronization Analysis**:
   - Correlates lip movement with speech activity
   - Calculates correlation coefficient
   - Determines sync quality score

**Key Lip Landmarks (MediaPipe):**
```python
UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409]
LOWER_LIP = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
```

### 4. Audio Analysis (`audio_analysis.py`)

**Libraries Used:**
- Librosa for feature extraction
- NumPy for signal processing

**Analysis Techniques:**
1. **MFCC Features**: 13-dimensional mel-frequency cepstral coefficients
2. **Multiple Speaker Detection**:
   - Spectral contrast analysis
   - Energy variance calculation
   - Confidence scoring based on variance patterns
3. **Background Noise Detection**:
   - Spectral flatness measurement
   - Signal-to-noise ratio estimation
   - Noise level quantification
4. **Silence Detection**:
   - Amplitude envelope analysis
   - Threshold-based silence identification
   - Duration measurement of silent periods

**Detection Algorithms:**
```python
# Multiple speaker detection using spectral features
def detect_multiple_speakers(self, audio_data):
    contrast = librosa.feature.spectral_contrast(y=audio_data, sr=self.sample_rate)
    energy = librosa.feature.rms(y=audio_data)[0]
    energy_variance = np.var(energy)
    contrast_variance = np.var(contrast)
    confidence_score = min(1.0, (contrast_variance * 10 + energy_variance * 50))
    return confidence_score > 0.6, confidence_score
```

### 5. Cheat Score Calculator (`cheat_score.py`)

**Scoring Methodology:**
- Weighted combination of all analysis results
- Risk-based scoring with configurable weights
- Trend analysis for score history

**Default Weights:**
```python
gaze_weight = 0.25          # 25% - Gaze tracking analysis
lip_sync_weight = 0.20      # 20% - Lip sync analysis  
person_detection_weight = 0.30  # 30% - Multiple person detection
audio_analysis_weight = 0.25    # 25% - Audio analysis
```

**Risk Level Classification:**
- **Low Risk** (0.0 - 0.3): Green - Normal behavior
- **Medium Risk** (0.3 - 0.6): Yellow - Some suspicious activity
- **High Risk** (0.6 - 0.8): Orange - Concerning behavior
- **Critical Risk** (0.8 - 1.0): Red - Likely cheating

## Data Flow

### Real-time Monitoring Flow:
```
Camera Input → Frame Capture → Parallel Analysis
                                     ↓
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ Gaze        │ Person      │ Lip Sync    │ Audio       │
│ Analysis    │ Detection   │ Analysis    │ Analysis    │
└─────────────┴─────────────┴─────────────┴─────────────┘
                                     ↓
              Score Calculation & Risk Assessment
                                     ↓
              Web Interface Display & Alerts
```

### Video Analysis Flow:
```
Video File → Frame Extraction → Batch Processing
                                      ↓
           Audio Extraction → Audio Analysis
                                      ↓
           Combined Results → Comprehensive Report
```

## API Endpoints

### Core Analysis Endpoints:
- `POST /gaze` - Single frame gaze analysis
- `POST /gaze_video` - Video gaze analysis
- `POST /multi_person_check` - Person/face detection
- `POST /lip_sync` - Lip synchronization analysis
- `POST /cheat_score` - Overall risk score calculation

### Configuration Endpoints:
- `GET/POST /cheat_score/config` - Score calculation weights

### Example API Usage:
```python
# Gaze analysis
response = requests.post('/gaze', json={
    'image': base64_encoded_image
})

# Multi-person detection  
response = requests.post('/multi_person_check', json={
    'image': base64_encoded_image
})
```

## Scoring System

### Gaze Tracking Scoring:
- **Center gaze**: 0.0 (normal)
- **Side gazes**: 0.3 (moderate risk)
- **Unknown/No face**: 0.6 (high risk)
- **Off-screen duration penalty**: Progressive based on time away

### Person Detection Scoring:
- **Single person**: 0.0 (normal)
- **Multiple people**: Exponential penalty up to 1.0
- **No person detected**: 0.6 (camera avoidance)
- **Face/person mismatch**: Proportional penalty

### Audio Analysis Scoring:
- **Multiple speakers**: 0.9 penalty
- **Background noise**: 0.4 penalty (scaled by noise level)
- **Prolonged silence**: 0.3 penalty

### Lip Sync Scoring:
- **Good synchronization**: 0.0
- **Poor synchronization**: 0.8 penalty
- **Gradual penalty**: Based on correlation coefficient

## Installation and Setup

### Prerequisites:
```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install python3-dev cmake build-essential
sudo apt-get install libopencv-dev python3-opencv
sudo apt-get install portaudio19-dev

# For macOS
brew install cmake portaudio
```

### Python Dependencies:
```bash
pip install -r requirements.txt
```

### Model Downloads:
```bash
# Download facial landmark predictor (required for gaze tracking)
wget https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2

# Download YOLO models (already included)
# yolov3.weights (237MB)
# yolov3.cfg 
# coco.names
```

## Usage Examples

### 1. Streamlit Web Interface:
```bash
streamlit run app.py
```

### 2. Comprehensive Video Analysis:
```python
from run_all import ProctorAnalyzer

analyzer = ProctorAnalyzer()
results = analyzer.run_comprehensive_analysis('exam_video.mp4')
print(f"Risk Score: {results['overall_score']}")
print(f"Risk Level: {results['risk_level']}")
```

### 3. Individual Module Testing:
```python
# Gaze tracking
from gaze_tracking import get_gaze_direction
direction = get_gaze_direction(frame)

# Person detection  
from multi_person import detect_multiple_people
results = detect_multiple_people(frame)

# Lip sync analysis
from lip_sync_detector import is_lip_synced
sync_results = is_lip_synced('video.mp4')
```

### 4. Real-time Monitoring:
```python
import cv2
from run_all import ProctorAnalyzer

analyzer = ProctorAnalyzer()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret:
        # Analyze current frame
        results = analyzer.analyze_live_frame(frame)
        print(f"Current risk: {results['risk_score']}")
```

## Performance Considerations

### Model Inference Times (typical):
- **Gaze tracking**: ~50ms per frame
- **YOLO person detection**: ~100ms per frame  
- **Face detection (Haar)**: ~20ms per frame
- **Lip sync analysis**: ~200ms per frame
- **Audio analysis**: ~500ms per 10-second segment

### Memory Requirements:
- **YOLO model**: ~800MB GPU/RAM
- **Dlib predictor**: ~100MB RAM
- **MediaPipe**: ~200MB RAM
- **Total system**: ~2GB RAM minimum

### Optimization Tips:
- Use GPU acceleration for YOLO if available
- Reduce video frame rate for real-time processing
- Process audio in chunks for long recordings
- Implement frame skipping for better performance

## Security and Privacy

### Data Handling:
- Video frames processed in memory (not stored)
- Temporary audio files cleaned up automatically
- Only analysis results stored, not raw media
- Base64 encoding for API image transmission

### Model Security:
- Pre-trained models from trusted sources
- No external API calls for inference
- Local processing ensures data privacy
- Deterministic analysis results

This documentation provides a comprehensive overview of the AI proctoring system's architecture, models, and implementation details. The system combines multiple state-of-the-art AI models to provide robust cheating detection capabilities while maintaining privacy and performance. 