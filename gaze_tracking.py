import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GazeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye landmarks indices for MediaPipe FaceMesh
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Iris landmarks for more accurate gaze detection
        self.LEFT_IRIS_INDICES = [474, 475, 476, 477]
        self.RIGHT_IRIS_INDICES = [469, 470, 471, 472]
        
        # Gaze thresholds
        self.GAZE_THRESHOLD = 0.15  # Lowered threshold for more sensitive detection
        self.MIN_LOOK_AWAY_DURATION = 1.5  # Minimum duration in seconds to flag as look away
        self.EAR_THRESHOLD = 0.15  # Threshold for eye aspect ratio
        
    def _get_eye_aspect_ratio(self, landmarks, eye_indices: List[int]) -> float:
        """Calculate the eye aspect ratio to determine if eyes are open."""
        points = np.array([(landmarks.landmark[idx].x, landmarks.landmark[idx].y) 
                          for idx in eye_indices])
        
        # Calculate vertical and horizontal distances
        vertical_dist = np.linalg.norm(points[1] - points[5]) + np.linalg.norm(points[2] - points[4])
        horizontal_dist = np.linalg.norm(points[0] - points[3])
        
        # Calculate aspect ratio
        ear = vertical_dist / (2.0 * horizontal_dist)
        return ear
    
    def _get_iris_position(self, landmarks, eye_indices: List[int], iris_indices: List[int]) -> Tuple[float, float]:
        """Calculate the relative position of the iris within the eye."""
        eye_points = np.array([(landmarks.landmark[idx].x, landmarks.landmark[idx].y) 
                              for idx in eye_indices])
        iris_points = np.array([(landmarks.landmark[idx].x, landmarks.landmark[idx].y) 
                               for idx in iris_indices])
        
        # Calculate eye and iris centers
        eye_center = np.mean(eye_points, axis=0)
        iris_center = np.mean(iris_points, axis=0)
        
        # Calculate relative position
        relative_x = (iris_center[0] - eye_center[0]) / (np.max(eye_points[:, 0]) - np.min(eye_points[:, 0]))
        relative_y = (iris_center[1] - eye_center[1]) / (np.max(eye_points[:, 1]) - np.min(eye_points[:, 1]))
        
        return relative_x, relative_y
    
    def _estimate_gaze_direction(self, landmarks) -> Tuple[float, float]:
        """Estimate gaze direction using both eye landmarks and iris positions."""
        # Get iris positions
        left_iris_x, left_iris_y = self._get_iris_position(landmarks, self.LEFT_EYE_INDICES, self.LEFT_IRIS_INDICES)
        right_iris_x, right_iris_y = self._get_iris_position(landmarks, self.RIGHT_EYE_INDICES, self.RIGHT_IRIS_INDICES)
        
        # Calculate average gaze direction
        gaze_x = (left_iris_x + right_iris_x) / 2
        gaze_y = (left_iris_y + right_iris_y) / 2
        
        return gaze_x, gaze_y
    
    def _is_looking_away(self, gaze_x: float, gaze_y: float) -> bool:
        """Determine if the gaze is directed away from the screen."""
        # Check both horizontal and vertical gaze
        horizontal_look_away = abs(gaze_x) > self.GAZE_THRESHOLD
        vertical_look_away = abs(gaze_y) > self.GAZE_THRESHOLD
        
        # Consider looking away if either horizontal or vertical gaze is off
        return horizontal_look_away or vertical_look_away
    
    def detect_gaze_deviation(self, video_path: str) -> Dict:
        """Main function to detect gaze deviations in a video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        # Initialize counters and flags
        frames_tracked = 0
        frames_looked_away = 0
        current_look_away_duration = 0
        max_look_away_duration = 0
        look_away_start_time = None
        recurrent_offscreen = False
        look_away_events = []
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = 1.0 / fps
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                
                # Calculate eye aspect ratios
                left_ear = self._get_eye_aspect_ratio(landmarks, self.LEFT_EYE_INDICES)
                right_ear = self._get_eye_aspect_ratio(landmarks, self.RIGHT_EYE_INDICES)
                
                # Only process if eyes are open
                if left_ear > self.EAR_THRESHOLD and right_ear > self.EAR_THRESHOLD:
                    frames_tracked += 1
                    
                    # Estimate gaze direction
                    gaze_x, gaze_y = self._estimate_gaze_direction(landmarks)
                    
                    # Check if looking away
                    if self._is_looking_away(gaze_x, gaze_y):
                        frames_looked_away += 1
                        if look_away_start_time is None:
                            look_away_start_time = frames_tracked * frame_interval
                        current_look_away_duration = (frames_tracked * frame_interval) - look_away_start_time
                    else:
                        if look_away_start_time is not None:
                            look_away_events.append(current_look_away_duration)
                            current_look_away_duration = 0
                            look_away_start_time = None
                    
                    # Update max look away duration
                    max_look_away_duration = max(max_look_away_duration, current_look_away_duration)
        
        cap.release()
        
        # Check for recurrent offscreen behavior
        if len(look_away_events) > 0:
            recurrent_offscreen = any(duration > self.MIN_LOOK_AWAY_DURATION for duration in look_away_events)
        
        # Calculate suspicion score (0-100)
        score = min(100, (
            (frames_looked_away / max(frames_tracked, 1)) * 50 +  # Percentage of frames looking away
            (max_look_away_duration * 10) +  # Duration of longest look away
            (int(recurrent_offscreen) * 20)  # Bonus for recurrent behavior
        ))
        
        return {
            "frames_tracked": frames_tracked,
            "frames_looked_away": frames_looked_away,
            "max_duration_look_away": round(max_look_away_duration, 1),
            "recurrent_offscreen_behavior": recurrent_offscreen,
            "score": round(score, 1)
        }

# FastAPI implementation
app = FastAPI()

class VideoRequest(BaseModel):
    video_path: str

@app.post("/gaze_tracking")
async def gaze_tracking_endpoint(request: VideoRequest):
    try:
        tracker = GazeTracker()
        results = tracker.detect_gaze_deviation(request.video_path)
        return results
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Using port 8001 