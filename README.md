# VIZH.AI - Video Integrity Analysis Tool

VIZH.AI is a sophisticated application designed to analyze video integrity, particularly useful for ensuring interview authenticity. The application uses advanced AI techniques to detect potential integrity issues by analyzing various aspects of video content.

## Features

- **Multi-factor Analysis**: Examines numerous aspects of a video to determine integrity:
  - Screen switching detection
  - Gaze tracking analysis
  - Audio analysis (multiple speakers, keyboard typing, etc.)
  - Multi-person detection
  - Lip sync analysis
  - Facial recognition for identity verification

- **Identity Verification**: Analyzes the first detected face in a video and flags any different faces that appear:
  - Tracks the primary participant throughout the video
  - Detects face switching or additional people
  - Provides timestamps of when different faces were detected

- **Comprehensive Reports**: Generates detailed reports with:
  - Overall integrity score
  - Risk assessment (Low, Medium, High, Very High)
  - Detailed findings with visual charts
  - Module-specific scores and metrics

- **Modern UI**: Clean, minimal, and user-friendly interface built with React and Tailwind CSS

## Running with Docker

The easiest way to run VIZH.AI is using Docker and Docker Compose.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

### Starting the Application

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd vizhai
   ```

2. Build and start the containers:
   ```bash
   docker-compose up -d
   ```

3. Access the application:
   - Frontend: http://localhost
   - Backend API: http://localhost/api

### Stopping the Application

```bash
docker-compose down
```

## Development Setup

If you prefer to run the application without Docker for development:

### Backend

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the backend server:
   ```bash
   python backend_server.py
   ```

### Frontend

1. Navigate to the frontend directory:
   ```bash
   cd vizhAI
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

## API Endpoints

- `POST /upload` - Upload a video for analysis
- `GET /analysis/{id}` - Check analysis status
- `GET /analysis/{id}/report` - Get the analysis report
- `POST /demo-analysis` - Generate a demo analysis (for testing)

## API Response Format

The backend provides structured JSON responses to the frontend. Here's the format for the main endpoints:

### Analysis Status Response (`GET /analysis/{id}`)

```json
{
  "id": "unique-analysis-id",
  "status": "processing|completed|failed",
  "created_at": "2023-06-15 10:30:45",
  "completed_at": "2023-06-15 10:32:15",
  "result": { /* Analysis results if completed */ },
  "error": "Error message if status is failed"
}
```

### Analysis Report Response (`GET /analysis/{id}/report`)

```json
{
  "final_score": 92.3,
  "risk": "Low|Medium|High|Very High",
  "reasons": [
    "Reason 1 for the score",
    "Reason 2 for the score"
  ],
  "gaze": {
    "off_screen_count": 2,
    "average_confidence": 0.95,
    "off_screen_time_percentage": 3.2,
    "gaze_direction_timeline": [
      {"timestamp": "00:00:15", "direction": "center"},
      {"timestamp": "00:00:45", "direction": "right"}
    ],
    "score": 96.8
  },
  "audio": {
    "multiple_speakers": false,
    "keyboard_typing_count": 0,
    "silence_percentage": 5.3,
    "background_noise_level": "Low",
    "speaking_timeline": [
      {"start": "00:00:05", "end": "00:00:25", "speaker": "primary"}
    ],
    "score": 98.7
  },
  "multi_person": {
    "max_people_detected": 1,
    "time_with_multiple_people": 0,
    "people_detection_timeline": [
      {"timestamp": "00:00:00", "count": 1},
      {"timestamp": "00:00:30", "count": 1}
    ],
    "different_faces_detected": 0,
    "different_face_timestamps": [],
    "has_different_faces": false,
    "score": 100.0
  },
  "lip_sync": {
    "lip_sync_score": 95.2,
    "major_lip_desync_detected": false,
    "lip_sync_timeline": [
      {"timestamp": "00:00:10", "score": 96.5},
      {"timestamp": "00:00:40", "score": 95.8}
    ],
    "score": 95.2
  },
  "module_scores": {
    "gaze": 96.8,
    "audio": 98.7,
    "multi_person": 100.0,
    "lip_sync": 95.2
  },
  "processing_time": 7.8,
  "video_duration": "02:35"
}
```

## License

This project is licensed under the MIT License. 
 