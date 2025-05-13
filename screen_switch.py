import cv2
import numpy as np
import pytesseract
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from skimage.metrics import structural_similarity as ssim
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScreenSwitchDetector:
    def __init__(self):
        self.scene_threshold = 0.3  # Threshold for scene change detection
        self.overlay_threshold = 0.8  # Threshold for overlay detection
        self.edge_threshold = 50  # Threshold for edge detection
        self.history_size = 5  # Number of frames to keep in history
        
    def _calculate_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Calculate color histogram for a frame."""
        hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        return cv2.normalize(hist, hist).flatten()
    
    def _detect_scene_change(self, frame1: np.ndarray, frame2: np.ndarray) -> bool:
        """Detect significant scene changes between frames."""
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate structural similarity
        score, _ = ssim(gray1, gray2, full=True)
        
        # Calculate histogram difference
        hist1 = self._calculate_histogram(frame1)
        hist2 = self._calculate_histogram(frame2)
        hist_diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        return score < self.scene_threshold or hist_diff < self.scene_threshold
    
    def _detect_overlays(self, frame: np.ndarray) -> bool:
        """Detect UI overlays using edge detection and template matching."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        edges = np.uint8(np.absolute(edges))
        
        # Check for horizontal lines (common in UI overlays)
        horizontal_lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        # Check top and bottom regions for UI elements
        top_region = frame[0:50, :]
        bottom_region = frame[-50:, :]
        
        # Use OCR to detect text in these regions
        top_text = pytesseract.image_to_string(top_region)
        bottom_text = pytesseract.image_to_string(bottom_region)
        
        return (horizontal_lines is not None and len(horizontal_lines) > 0) or \
               (len(top_text.strip()) > 0) or \
               (len(bottom_text.strip()) > 0)
    
    def _detect_fullscreen_violation(self, frame: np.ndarray) -> bool:
        """Detect if the frame is not in fullscreen mode."""
        height, width = frame.shape[:2]
        
        # Check edges for consistent margins
        left_edge = frame[:, 0:5]
        right_edge = frame[:, -5:]
        top_edge = frame[0:5, :]
        bottom_edge = frame[-5:, :]
        
        # Calculate average brightness of edges
        edges = [left_edge, right_edge, top_edge, bottom_edge]
        edge_brightness = [np.mean(edge) for edge in edges]
        
        # If any edge has significantly different brightness, it might be a window border
        return any(brightness > self.edge_threshold for brightness in edge_brightness)
    
    def detect_screen_anomalies(self, video_path: str) -> Dict:
        """Main function to detect screen anomalies in a video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        # Initialize counters and flags
        fullscreen_violations = 0
        overlay_detected = False
        scene_switch_events = 0
        anomaly_start_time = None
        total_anomaly_duration = timedelta()
        
        # Initialize frame history
        frame_history = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            
            # Add frame to history
            frame_history.append(frame)
            if len(frame_history) > self.history_size:
                frame_history.pop(0)
            
            # Detect anomalies
            if len(frame_history) >= 2:
                if self._detect_scene_change(frame_history[-2], frame_history[-1]):
                    scene_switch_events += 1
                    if anomaly_start_time is None:
                        anomaly_start_time = current_time
                
                if self._detect_overlays(frame):
                    overlay_detected = True
                    if anomaly_start_time is None:
                        anomaly_start_time = current_time
                
                if self._detect_fullscreen_violation(frame):
                    fullscreen_violations += 1
                    if anomaly_start_time is None:
                        anomaly_start_time = current_time
            
            # Update anomaly duration
            if anomaly_start_time is not None and not any([
                self._detect_scene_change(frame_history[-2], frame_history[-1]),
                self._detect_overlays(frame),
                self._detect_fullscreen_violation(frame)
            ]):
                total_anomaly_duration += timedelta(seconds=current_time - anomaly_start_time)
                anomaly_start_time = None
        
        cap.release()
        
        # Calculate suspicion score (0-100)
        suspicion_score = min(100, (
            (scene_switch_events * 10) +
            (fullscreen_violations * 15) +
            (int(overlay_detected) * 20) +
            (total_anomaly_duration.total_seconds() * 0.5)
        ))
        
        return {
            "fullscreen_violations": fullscreen_violations,
            "overlay_detected": overlay_detected,
            "scene_switch_events": scene_switch_events,
            "duration_with_anomalies": str(total_anomaly_duration),
            "suspicion_score": round(suspicion_score, 1)
        }

# FastAPI implementation
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class VideoRequest(BaseModel):
    video_path: str

@app.post("/screen_switch")
async def screen_switch_endpoint(request: VideoRequest):
    try:
        detector = ScreenSwitchDetector()
        results = detector.detect_screen_anomalies(request.video_path)
        return results
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 