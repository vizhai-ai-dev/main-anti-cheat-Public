import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time

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
    
    def detect_multiple_persons(self, video_path: str) -> Dict:
        """Main function to detect multiple persons in a video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = 1.0 / fps
        sample_frame_interval = int(self.SAMPLE_INTERVAL * fps)
        
        logger.info(f"Video FPS: {fps}, Total Frames: {total_frames}")
        
        # Initialize counters and flags
        total_frames_analyzed = 0
        frames_with_extra_people = 0
        first_extra_person_time = None
        current_time = 0
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Sample frames at specified interval
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
                
                total_frames_analyzed += 1
                logger.info(f"Processed frame {frame_count}/{total_frames}, Valid persons: {valid_persons}")
            
            current_time += frame_interval
        
        cap.release()
        
        # Calculate suspicion score (0-100)
        score = min(100, (
            (frames_with_extra_people / max(total_frames_analyzed, 1)) * 50 +  # Percentage of frames with extra people
            (int(first_extra_person_time is not None) * 20)  # Bonus for detecting extra people
        ))
        
        logger.info(f"Analysis complete. Total frames analyzed: {total_frames_analyzed}, "
                   f"Frames with extra people: {frames_with_extra_people}")
        
        return {
            "total_frames_analyzed": total_frames_analyzed,
            "frames_with_extra_people": frames_with_extra_people,
            "first_extra_person_detected_at": self._format_timestamp(first_extra_person_time) if first_extra_person_time is not None else "None",
            "score": round(score, 1)
        }

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