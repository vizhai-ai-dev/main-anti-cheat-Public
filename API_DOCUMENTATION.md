# AI Proctoring System - REST API Documentation

## Overview

The AI Proctoring System provides a comprehensive REST API for integrating proctoring capabilities into web applications. The API is designed for React frontends but can be used with any HTTP client.

**Base URL**: `http://localhost:5000`

## Quick Start

1. **Start the API server**:
   ```bash
   python api_server.py
   ```

2. **Test the health endpoint**:
   ```bash
   curl http://localhost:5000/health
   ```

## Authentication

Currently, the API does not require authentication. For production use, implement proper authentication and authorization.

## File Upload Limits

- **Maximum file size**: 100MB
- **Supported video formats**: mp4, avi, mov, mkv
- **Supported audio formats**: wav, mp3, aac, m4a
- **Supported image formats**: jpg, jpeg, png, bmp

## API Endpoints

### Health Check

#### GET /health
Check if the API server is running and healthy.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00.000000",
  "version": "1.0.0",
  "modules": ["gaze_tracking", "lip_sync", "person_detection", "audio_analysis", "cheat_score"]
}
```

### Gaze Tracking

#### POST /api/gaze/analyze_image
Analyze gaze direction from a single image.

**Request**:
- Content-Type: `multipart/form-data`
- Body: `image` (file) - Image file to analyze

**Response**:
```json
{
  "success": true,
  "gaze_direction": "center",
  "processed_image": "base64_encoded_image",
  "timestamp": "2024-01-01T12:00:00.000000"
}
```

**Gaze directions**: `center`, `left`, `right`, `up`, `down`, `unknown`, `eyes_closed`

#### POST /api/gaze/analyze_video
Analyze gaze patterns throughout a video.

**Request**:
- Content-Type: `multipart/form-data`
- Body: `video` (file) - Video file to analyze

**Response**:
```json
{
  "success": true,
  "results": {
    "direction": "center",
    "confidence": 0.85,
    "off_screen_duration": 12.5,
    "gaze_distribution": {
      "center": 0.6,
      "left": 0.2,
      "right": 0.15,
      "down": 0.05
    },
    "suspicious_patterns": 2,
    "processing_time": 15.2
  },
  "timestamp": "2024-01-01T12:00:00.000000"
}
```

### Person Detection

#### POST /api/person/detect_image
Detect people and faces in a single image.

**Request**:
- Content-Type: `multipart/form-data`
- Body: `image` (file) - Image file to analyze

**Response**:
```json
{
  "success": true,
  "results": {
    "people_count": 1,
    "face_count": 1,
    "processed_image": "base64_encoded_image_with_detections",
    "confidence": 0.92
  },
  "timestamp": "2024-01-01T12:00:00.000000"
}
```

### Lip Sync Detection

#### POST /api/lipsync/analyze
Analyze lip synchronization in a video.

**Request**:
- Content-Type: `multipart/form-data`
- Body: `video` (file) - Video file to analyze

**Response**:
```json
{
  "success": true,
  "results": {
    "is_synced": true,
    "sync_score": 0.87,
    "frames_analyzed": 150,
    "video_fps": 30.0,
    "processing_time": 8.5
  },
  "timestamp": "2024-01-01T12:00:00.000000"
}
```

#### POST /api/lipsync/extract_landmarks
Extract lip landmarks from a single image.

**Request**:
- Content-Type: `multipart/form-data`
- Body: `image` (file) - Image file to analyze

**Response**:
```json
{
  "success": true,
  "landmarks": [[x1, y1], [x2, y2], ...],
  "landmarks_count": 20,
  "timestamp": "2024-01-01T12:00:00.000000"
}
```

### Audio Analysis

#### POST /api/audio/analyze
Analyze audio for multiple speakers, background noise, and anomalies.

**Request**:
- Content-Type: `multipart/form-data`
- Body: `audio` (file) - Audio or video file to analyze

**Response**:
```json
{
  "success": true,
  "results": {
    "multiple_speakers": false,
    "speaker_confidence": 0.15,
    "has_background_noise": true,
    "noise_level": 0.3,
    "has_prolonged_silence": false,
    "silence_periods": [[10.5, 15.2], [45.1, 48.3]],
    "audio_duration": 120.5,
    "processing_time": 5.2
  },
  "timestamp": "2024-01-01T12:00:00.000000"
}
```

### Cheat Score Calculation

#### POST /api/cheatscore/calculate
Calculate cheat score from analysis inputs.

**Request**:
- Content-Type: `application/json`
- Body:
```json
{
  "gaze_data": {
    "direction": "left",
    "off_screen_duration": 15.5,
    "suspicious_patterns": 3
  },
  "person_data": {
    "people_count": 2,
    "face_count": 1
  },
  "lip_sync_data": {
    "is_synced": false,
    "sync_score": 0.3
  },
  "audio_data": {
    "multiple_speakers": true,
    "speaker_confidence": 0.8,
    "has_background_noise": true,
    "noise_level": 0.6
  }
}
```

**Response**:
```json
{
  "success": true,
  "cheat_score": 0.75,
  "risk_level": "High",
  "input_data": { ... },
  "timestamp": "2024-01-01T12:00:00.000000"
}
```

**Risk levels**: `Very Low`, `Low`, `Medium`, `High`, `Very High`

#### GET /api/cheatscore/config
Get current cheat score calculation weights.

**Response**:
```json
{
  "gaze_weight": 0.25,
  "lip_sync_weight": 0.20,
  "person_detection_weight": 0.30,
  "audio_analysis_weight": 0.25,
  "gaze_off_screen_penalty": 0.8,
  "multiple_person_penalty": 1.0,
  "multiple_speaker_penalty": 0.9,
  "poor_lip_sync_penalty": 0.8
}
```

### Comprehensive Analysis

#### POST /api/analyze/comprehensive
Run comprehensive analysis on video/audio files using all modules.

**Request**:
- Content-Type: `multipart/form-data`
- Body: 
  - `video` (file) - Video file to analyze
  - `audio` (file, optional) - Separate audio file
  - `session_id` (string, optional) - Session identifier

**Response**:
```json
{
  "success": true,
  "results": {
    "session_info": {
      "session_id": "session_123",
      "start_time": "2024-01-01T12:00:00.000000",
      "total_processing_time": 45.2
    },
    "analysis_results": {
      "gaze_tracking": { ... },
      "person_detection": { ... },
      "lip_sync_detection": { ... },
      "audio_analysis": { ... }
    },
    "cheat_score_analysis": {
      "cheat_score": 0.65,
      "risk_level": "Medium",
      "input_data": { ... }
    },
    "summary": {
      "primary_concerns": ["Multiple people detected", "Poor lip sync"],
      "recommendations": ["Ensure only one person in frame", "Check audio setup"]
    }
  },
  "timestamp": "2024-01-01T12:00:00.000000"
}
```

### Live Monitoring

#### POST /api/live/analyze_frame
Analyze a single frame for live monitoring (optimized for real-time use).

**Request**:
- Content-Type: `multipart/form-data`
- Body: `image` (file) - Image frame to analyze

**Response**:
```json
{
  "success": true,
  "gaze_direction": "center",
  "people_count": 1,
  "face_count": 1,
  "lip_sync_quality": {
    "landmarks_detected": true,
    "is_synced": true,
    "sync_score": 0.9
  },
  "risk_score": 0.15,
  "risk_level": "Low",
  "timestamp": "2024-01-01T12:00:00.000000"
}
```

### Utility Endpoints

#### GET /api/utils/supported_formats
Get supported file formats and limits.

**Response**:
```json
{
  "video_formats": ["mp4", "avi", "mov", "mkv"],
  "audio_formats": ["wav", "mp3", "aac", "m4a"],
  "image_formats": ["jpg", "jpeg", "png", "bmp"],
  "max_file_size_mb": 100
}
```

#### POST /api/utils/test_camera
Test camera access (for debugging).

**Request**:
- Content-Type: `application/json`
- Body:
```json
{
  "camera_id": 0
}
```

**Response**:
```json
{
  "success": true,
  "camera_id": 0,
  "resolution": "640x480",
  "fps": 30.0,
  "frame_captured": true
}
```

## Error Responses

All endpoints return error responses in the following format:

```json
{
  "error": "Error description"
}
```

**Common HTTP status codes**:
- `400` - Bad Request (missing or invalid parameters)
- `413` - File too large (exceeds 100MB limit)
- `404` - Endpoint not found
- `500` - Internal server error

## Integration Examples

### React/JavaScript Example

```javascript
// Upload and analyze an image
const analyzeImage = async (imageFile) => {
  const formData = new FormData();
  formData.append('image', imageFile);
  
  try {
    const response = await fetch('http://localhost:5000/api/gaze/analyze_image', {
      method: 'POST',
      body: formData
    });
    
    const result = await response.json();
    console.log('Gaze direction:', result.gaze_direction);
    return result;
  } catch (error) {
    console.error('Error analyzing image:', error);
  }
};

// Calculate cheat score
const calculateCheatScore = async (analysisData) => {
  try {
    const response = await fetch('http://localhost:5000/api/cheatscore/calculate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(analysisData)
    });
    
    const result = await response.json();
    console.log('Risk level:', result.risk_level);
    return result;
  } catch (error) {
    console.error('Error calculating cheat score:', error);
  }
};
```

### Python Example

```python
import requests

# Analyze gaze in image
def analyze_gaze_image(image_path):
    with open(image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post('http://localhost:5000/api/gaze/analyze_image', files=files)
        return response.json()

# Comprehensive analysis
def comprehensive_analysis(video_path, session_id=None):
    with open(video_path, 'rb') as f:
        files = {'video': f}
        data = {'session_id': session_id} if session_id else {}
        response = requests.post('http://localhost:5000/api/analyze/comprehensive', 
                               files=files, data=data)
        return response.json()

# Calculate cheat score
def calculate_cheat_score(analysis_data):
    response = requests.post('http://localhost:5000/api/cheatscore/calculate', 
                           json=analysis_data)
    return response.json()
```

### cURL Examples

```bash
# Health check
curl http://localhost:5000/health

# Analyze gaze in image
curl -X POST -F "image=@test_image.jpg" \
  http://localhost:5000/api/gaze/analyze_image

# Comprehensive analysis
curl -X POST -F "video=@test_video.mp4" -F "session_id=test_session" \
  http://localhost:5000/api/analyze/comprehensive

# Calculate cheat score
curl -X POST -H "Content-Type: application/json" \
  -d '{"gaze_data":{"direction":"left"},"person_data":{"people_count":1}}' \
  http://localhost:5000/api/cheatscore/calculate
```

## Performance Considerations

1. **File Size**: Keep uploaded files under 100MB for optimal performance
2. **Video Length**: Longer videos take more time to process
3. **Concurrent Requests**: The server can handle multiple requests, but processing is CPU-intensive
4. **Live Monitoring**: Use `/api/live/analyze_frame` for real-time analysis (optimized for speed)

## Production Deployment

For production deployment, consider:

1. **Authentication**: Implement proper API authentication
2. **Rate Limiting**: Add rate limiting to prevent abuse
3. **File Storage**: Use cloud storage for uploaded files
4. **Scaling**: Use multiple server instances behind a load balancer
5. **HTTPS**: Enable SSL/TLS encryption
6. **Monitoring**: Add logging and monitoring for API usage

## Support

For technical support or questions about the API, please refer to the main documentation or contact the development team. 