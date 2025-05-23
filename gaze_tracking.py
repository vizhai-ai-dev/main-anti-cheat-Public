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
        
        # Gaze thresholds - Separate for horizontal and vertical
        self.HORIZONTAL_GAZE_THRESHOLD = 0.15  # Threshold for looking left/right
        self.VERTICAL_GAZE_THRESHOLD = 0.25    # Higher threshold for looking up/down
        self.MIN_LOOK_AWAY_DURATION = 1.5      # Minimum duration in seconds to flag as look away
        self.EAR_THRESHOLD = 0.15              # Threshold for eye aspect ratio
        
        # Vertical gaze penalties
        self.LOOKING_UP_PENALTY = 0.7         # 70% penalty for looking up
        self.LOOKING_DOWN_PENALTY = 0.3       # 30% penalty for looking down (typing/coding)
        
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
    
    def _is_looking_away(self, gaze_x: float, gaze_y: float) -> Tuple[bool, float]:
        """Determine if the gaze is directed away from the screen and calculate penalty factor."""
        # Check horizontal gaze (left/right)
        horizontal_look_away = abs(gaze_x) > self.HORIZONTAL_GAZE_THRESHOLD
        
        # Check vertical gaze (up/down)
        vertical_look_away = abs(gaze_y) > self.VERTICAL_GAZE_THRESHOLD
        
        # Calculate penalty factor based on gaze direction
        penalty_factor = 1.0
        
        if vertical_look_away:
            if gaze_y < 0:  # Looking up
                penalty_factor = self.LOOKING_UP_PENALTY
            else:  # Looking down
                penalty_factor = self.LOOKING_DOWN_PENALTY
        
        # Consider looking away if either horizontal gaze is off or vertical gaze is significantly off
        is_looking_away = horizontal_look_away or (vertical_look_away and gaze_y < 0)  # Only consider looking up as suspicious
        
        return is_looking_away, penalty_factor
    
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
        total_penalty = 0.0
        
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
                    
                    # Check if looking away and get penalty factor
                    is_looking_away, penalty_factor = self._is_looking_away(gaze_x, gaze_y)
                    
                    if is_looking_away:
                        frames_looked_away += 1
                        total_penalty += penalty_factor
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
        
        # Calculate average penalty factor
        avg_penalty = total_penalty / max(frames_looked_away, 1)
        
        # Calculate suspicion score (0-100) with penalty factor
        score = min(100, (
            (frames_looked_away / max(frames_tracked, 1)) * 50 * avg_penalty +  # Percentage of frames looking away with penalty
            (max_look_away_duration * 10 * avg_penalty) +  # Duration of longest look away with penalty
            (int(recurrent_offscreen) * 20)  # Bonus for recurrent behavior
        ))
        
        return {
            "frames_tracked": frames_tracked,
            "frames_looked_away": frames_looked_away,
            "max_duration_look_away": round(max_look_away_duration, 1),
            "recurrent_offscreen_behavior": recurrent_offscreen,
            "average_penalty_factor": round(avg_penalty, 2),
            "score": round(score, 1)
        }

    async def analyze(self, video_path: str) -> Dict:
        """
        Analyze a video for gaze tracking.
        This is an async wrapper around detect_gaze_deviation.
        """
        try:
            result = self.detect_gaze_deviation(video_path)
            
            # Calculate confidence based on multiple factors
            tracking_confidence = 0.95  # Base confidence from face mesh tracking
            eye_detection_confidence = min(1.0, result["frames_tracked"] / max(result["frames_tracked"] + result["frames_looked_away"], 1))
            stability_confidence = 1.0 - (result["average_penalty_factor"] * 0.5)  # Penalty factor affects confidence
            
            # Weighted average of confidence factors
            average_confidence = (
                tracking_confidence * 0.4 +  # Base tracking confidence
                eye_detection_confidence * 0.4 +  # Eye detection reliability
                stability_confidence * 0.2  # Gaze stability
            )
            
            return {
                "score": result["score"],
                "off_screen_count": result["frames_looked_away"],
                "average_confidence": round(average_confidence, 3),
                "off_screen_time_percentage": (result["frames_looked_away"] / max(result["frames_tracked"], 1)) * 100,
                "gaze_direction_timeline": [],  # Mock timeline
                "confidence_metrics": {
                    "tracking_confidence": round(tracking_confidence, 3),
                    "eye_detection_confidence": round(eye_detection_confidence, 3),
                    "stability_confidence": round(stability_confidence, 3)
                }
            }
        except Exception as e:
            logger.error(f"Error in gaze analysis: {str(e)}")
            raise

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