import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import face_recognition

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiPersonDetector:
    def __init__(self):
        # Initialize YOLOv8 model
        self.model = YOLO('yolov8n.pt')  # Using nano model for speed, can use larger models for better accuracy
        
        # Detection parameters
        self.CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for person detection
        self.SAMPLE_INTERVAL = 0.5  # Sample frames every 0.5 seconds
        self.MIN_PERSON_HEIGHT = 100  # Minimum height in pixels to consider a detection valid
        self.FACE_RECOGNITION_INTERVAL = 1.0  # Sample frames for face recognition every 1 second
        self.FACE_MATCHING_THRESHOLD = 0.6  # Threshold for face comparison (lower is stricter)
        
        # Store reference face
        self.reference_face_encoding = None
        self.different_faces_detected = 0
        self.different_faces_timestamps = []
        
    def _is_valid_person_detection(self, box: List[float], frame_height: int) -> bool:
        """Check if a detection is a valid person based on size and position."""
        x1, y1, x2, y2 = box
        height = y2 - y1
        
        # Check if detection is tall enough and within frame bounds
        return (height >= self.MIN_PERSON_HEIGHT and 
                y2 <= frame_height and 
                y1 >= 0)
    
    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS format."""
        return str(timedelta(seconds=int(seconds)))
    
    def _extract_face_encodings(self, frame) -> List:
        """Extract face encodings from a frame."""
        # Convert BGR to RGB for face_recognition library
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find face locations
        face_locations = face_recognition.face_locations(rgb_frame)
        
        # If no faces found, return empty list
        if not face_locations:
            return []
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        return face_encodings
    
    def _is_different_face(self, face_encoding) -> bool:
        """Check if the face encoding is different from the reference face."""
        if self.reference_face_encoding is None:
            # This is the first face, set it as reference
            self.reference_face_encoding = face_encoding
            logger.info("Reference face detected and stored")
            return False
        
        # Compare with reference face
        # face_distance returns a numpy array with the distance for each face in the reference
        face_distances = face_recognition.face_distance([self.reference_face_encoding], face_encoding)
        
        # If the distance is greater than the threshold, it's a different face
        return face_distances[0] > self.FACE_MATCHING_THRESHOLD
    
    def detect_multiple_persons(self, video_path: str) -> Dict:
        """Main function to detect multiple persons in a video and track faces."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = 1.0 / fps
        sample_frame_interval = int(self.SAMPLE_INTERVAL * fps)
        face_recognition_interval = int(self.FACE_RECOGNITION_INTERVAL * fps)
        
        logger.info(f"Video FPS: {fps}, Total Frames: {total_frames}")
        
        # Initialize counters and flags
        total_frames_analyzed = 0
        frames_with_extra_people = 0
        first_extra_person_time = None
        current_time = 0
        frame_count = 0
        last_different_face_time = None
        people_detection_timeline = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process for face recognition at specified intervals
            if frame_count % face_recognition_interval == 0:
                face_encodings = self._extract_face_encodings(frame)
                
                # If faces found, check if they're different from reference
                for face_encoding in face_encodings:
                    if self._is_different_face(face_encoding):
                        self.different_faces_detected += 1
                        timestamp = self._format_timestamp(current_time)
                        self.different_faces_timestamps.append(timestamp)
                        logger.info(f"Different face detected at {timestamp}")
                        
                        # Record the first time a different face is detected
                        if last_different_face_time is None:
                            last_different_face_time = current_time
            
            # Sample frames at specified interval for person detection
            if frame_count % sample_frame_interval == 0:
                # Run YOLOv8 detection
                results = self.model(frame, classes=[0])  # class 0 is person in COCO dataset
                
                # Process detections
                valid_persons = 0
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        if box.conf[0] >= self.CONFIDENCE_THRESHOLD:
                            if self._is_valid_person_detection(box.xyxy[0].tolist(), frame.shape[0]):
                                valid_persons += 1
                
                # Update counters
                if valid_persons > 1:  # More than one person detected
                    frames_with_extra_people += 1
                    if first_extra_person_time is None:
                        first_extra_person_time = current_time
                        logger.info(f"First extra person detected at {self._format_timestamp(first_extra_person_time)}")
                
                # Add to timeline
                people_detection_timeline.append({
                    "timestamp": self._format_timestamp(current_time),
                    "count": valid_persons
                })
                
                total_frames_analyzed += 1
                logger.info(f"Processed frame {frame_count}/{total_frames}, Valid persons: {valid_persons}")
            
            current_time += frame_interval
        
        cap.release()
        
        # Calculate time with different faces (in seconds)
        time_with_different_faces = len(self.different_faces_timestamps) * self.FACE_RECOGNITION_INTERVAL
        
        # Calculate suspicion score (0-100)
        score = min(100, (
            (frames_with_extra_people / max(total_frames_analyzed, 1)) * 30 +  # Percentage of frames with extra people
            (int(first_extra_person_time is not None) * 20) +  # Bonus for detecting extra people
            (self.different_faces_detected * 10) +  # Penalty for each different face detected
            (int(time_with_different_faces > 0) * 20)  # Major penalty if different faces detected
        ))
        
        logger.info(f"Analysis complete. Total frames analyzed: {total_frames_analyzed}, "
                   f"Frames with extra people: {frames_with_extra_people}, "
                   f"Different faces detected: {self.different_faces_detected}")
        
        return {
            "total_frames_analyzed": total_frames_analyzed,
            "frames_with_extra_people": frames_with_extra_people,
            "first_extra_person_detected_at": self._format_timestamp(first_extra_person_time) if first_extra_person_time is not None else "None",
            "different_faces_detected": self.different_faces_detected,
            "different_face_timestamps": self.different_faces_timestamps,
            "has_different_faces": self.different_faces_detected > 0,
            "time_with_multiple_people": frames_with_extra_people * self.SAMPLE_INTERVAL,
            "score": round(score, 1),
            "people_detection_timeline": people_detection_timeline
        }

    async def analyze(self, video_path: str) -> Dict:
        """
        Analyze a video for multiple persons.
        This is an async wrapper around detect_multiple_persons.
        """
        try:
            result = self.detect_multiple_persons(video_path)
            
            # Calculate confidence based on multiple factors
            detection_confidence = min(1.0, result["total_frames_analyzed"] / 100)  # More frames analyzed = higher confidence
            face_recognition_confidence = 1.0 - (len(result["different_face_timestamps"]) * 0.1)  # Fewer face changes = higher confidence
            stability_confidence = 1.0 - (result["frames_with_extra_people"] / max(result["total_frames_analyzed"], 1))
            
            # Weighted average of confidence factors
            average_confidence = (
                detection_confidence * 0.4 +  # Detection reliability
                face_recognition_confidence * 0.4 +  # Face recognition accuracy
                stability_confidence * 0.2  # Detection stability
            )
            
            return {
                "score": result["score"],
                "max_people_detected": max(seg["count"] for seg in result["people_detection_timeline"]),
                "time_with_multiple_people": result["time_with_multiple_people"],
                "people_detection_timeline": result["people_detection_timeline"],
                "different_faces_detected": result["different_faces_detected"],
                "different_face_timestamps": result["different_face_timestamps"],
                "has_different_faces": result["has_different_faces"],
                "average_confidence": round(average_confidence, 3),
                "confidence_metrics": {
                    "detection_confidence": round(detection_confidence, 3),
                    "face_recognition_confidence": round(face_recognition_confidence, 3),
                    "stability_confidence": round(stability_confidence, 3)
                }
            }
        except Exception as e:
            logger.error(f"Error in multi-person analysis: {str(e)}")
            raise

# FastAPI implementation
app = FastAPI()

class VideoRequest(BaseModel):
    video_path: str

@app.post("/multi_person")
async def multi_person_endpoint(request: VideoRequest):
    try:
        detector = MultiPersonDetector()
        results = detector.detect_multiple_persons(request.video_path)
        return results
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)  # Using port 8002 to avoid conflicts 