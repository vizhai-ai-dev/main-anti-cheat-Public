import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import os
from pathlib import Path
import requests
from audio_analysis import AudioAnalyzer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LipNet:
    def __init__(self):
        # Initialize parameters
        self.sequence_length = 30  # Number of frames to process at once
        self.img_height = 50
        self.img_width = 100
        
        # Build and compile model
        self.model = self._build_model()
        
        # Set to evaluation mode (not training)
        self.model.trainable = False
        
    def _build_model(self):
        model = Sequential([
            # First 3D Convolutional Layer
            Conv3D(32, (3, 5, 5), strides=(1, 2, 2), padding='same', activation='relu', input_shape=(self.sequence_length, self.img_height, self.img_width, 1)),
            BatchNormalization(),
            MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)),
            
            # Second 3D Convolutional Layer
            Conv3D(64, (3, 5, 5), strides=(1, 1, 1), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)),
            
            # Third 3D Convolutional Layer
            Conv3D(96, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)),
            
            # Flatten and Dense Layers
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')  # Binary classification: speaking or not
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.0001),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        return model
    
    def preprocess_frame(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize to model input size
        resized = cv2.resize(gray, (self.img_width, self.img_height))
        # Normalize
        normalized = resized / 255.0
        return normalized
    
    def predict_sequence(self, frames):
        try:
            if len(frames) < self.sequence_length:
                # Pad sequence if too short
                padding = [frames[0]] * (self.sequence_length - len(frames))
                frames = padding + frames
            elif len(frames) > self.sequence_length:
                # Truncate if too long
                frames = frames[:self.sequence_length]
            
            # Preprocess frames
            processed_frames = [self.preprocess_frame(frame) for frame in frames]
            # Stack frames
            sequence = np.stack(processed_frames)
            # Add batch and channel dimensions
            sequence = np.expand_dims(sequence, axis=0)
            sequence = np.expand_dims(sequence, axis=-1)
            
            # Since we don't have a trained model, use a simple heuristic
            # This detects mouth movement from the frames directly
            motion = np.mean([np.std(sequence[0, i:i+3]) for i in range(len(sequence[0])-3)])
            return float(motion > 0.05)  # Return speaking probability based on motion
        except Exception as e:
            logger.error(f"Error in LipNet prediction: {str(e)}")
            return 0.0

class LipSyncDetector:
    def __init__(self):
        # Initialize MediaPipe FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize LipNet
        self.lipnet = LipNet()
        
        # Initialize audio analyzer
        self.audio_analyzer = AudioAnalyzer()
        
        # Analysis parameters
        self.FRAME_SAMPLE_RATE = 0.1  # Sample every 100ms
        self.SEQUENCE_WINDOW = 30  # Number of frames to analyze at once
        self.SPEAKING_THRESHOLD = 0.5  # Threshold for LipNet prediction
        self.MISMATCH_THRESHOLD = 0.3  # Threshold for considering a mismatch significant
        self.MIN_MISMATCH_DURATION = 0.2  # Minimum duration for a significant mismatch
        self.MOUTH_MOVEMENT_WINDOW = 0.3  # Window to look for mouth movement around voice activity

    def _extract_frames(self, video_path: str) -> List[Tuple[float, np.ndarray]]:
        """Extract frames from video at specified intervals."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * self.FRAME_SAMPLE_RATE)
        frames = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                frames.append((timestamp, frame))
            
            frame_count += 1
        
        cap.release()
        return frames

    def _detect_speaking(self, frames: List[Tuple[float, np.ndarray]]) -> List[Dict]:
        """Detect speaking using LipNet."""
        speaking_segments = []
        current_sequence = []
        current_timestamps = []
        
        for timestamp, frame in frames:
            current_sequence.append(frame)
            current_timestamps.append(timestamp)
            
            if len(current_sequence) >= self.SEQUENCE_WINDOW:
                # Get prediction for current sequence
                speaking_prob = self.lipnet.predict_sequence(current_sequence)
                
                # Add result for the middle frame of the sequence
                mid_idx = len(current_sequence) // 2
                speaking_segments.append({
                    "timestamp": current_timestamps[mid_idx],
                    "speaking_probability": float(speaking_prob),
                    "is_speaking": speaking_prob > self.SPEAKING_THRESHOLD
                })
                
                # Remove oldest frame
                current_sequence.pop(0)
                current_timestamps.pop(0)
        
        return speaking_segments

    def _detect_mismatches(self, speaking_segments: List[Dict], voice_segments: List[Dict]) -> List[Dict]:
        """Detect mismatches between LipNet predictions and voice activity."""
        mismatches = []
        current_mismatch = None
        
        for segment in speaking_segments:
            timestamp = segment["timestamp"]
            is_speaking = segment["is_speaking"]
            
            # Check if there's voice activity in a window around this timestamp
            is_voice_active = any(
                seg["start"] - self.MOUTH_MOVEMENT_WINDOW <= timestamp <= seg["end"] + self.MOUTH_MOVEMENT_WINDOW
                for seg in voice_segments
            )
            
            # Detect mismatch
            if is_voice_active != is_speaking:
                if current_mismatch is None:
                    current_mismatch = {
                        "start": timestamp,
                        "end": timestamp,
                        "type": "voice_no_lip" if is_voice_active else "lip_no_voice",
                        "severity": segment["speaking_probability"] if is_speaking else 1.0
                    }
                else:
                    current_mismatch["end"] = timestamp
                    current_mismatch["severity"] = max(
                        current_mismatch["severity"],
                        segment["speaking_probability"] if is_speaking else 1.0
                    )
            elif current_mismatch is not None:
                # End current mismatch if duration is significant
                duration = current_mismatch["end"] - current_mismatch["start"]
                if duration >= self.MIN_MISMATCH_DURATION:
                    mismatches.append(current_mismatch)
                current_mismatch = None
        
        # Add final mismatch if exists
        if current_mismatch is not None:
            duration = current_mismatch["end"] - current_mismatch["start"]
            if duration >= self.MIN_MISMATCH_DURATION:
                mismatches.append(current_mismatch)
        
        return mismatches

    def detect_lip_sync_issues(self, video_path: str) -> Dict:
        """Main function to detect lip sync issues in a video."""
        try:
            # Extract frames from video
            frames = self._extract_frames(video_path)
            
            # Detect speaking using LipNet
            speaking_segments = self._detect_speaking(frames)
            
            # Get voice activity from audio
            voice_segments = self.audio_analyzer._detect_voice_activity(video_path)
            
            # Detect mismatches
            mismatches = self._detect_mismatches(speaking_segments, voice_segments)
            
            # Calculate lip sync score
            total_frames = len(speaking_segments)
            mismatch_frames = sum(1 for m in mismatches if m["severity"] > self.MISMATCH_THRESHOLD)
            lip_sync_score = 100 * (1 - (mismatch_frames / max(total_frames, 1)))
            
            return {
                "lip_sync_score": round(lip_sync_score, 1),
                "major_lip_desync_detected": any(m["severity"] > self.MISMATCH_THRESHOLD for m in mismatches),
                "mismatches": mismatches,
                "speaking_segments": speaking_segments,
                "voice_segments": voice_segments
            }
        except Exception as e:
            logger.error(f"Error in lip sync detection: {str(e)}")
            raise

    async def analyze(self, video_path: str) -> Dict:
        """
        Analyze a video for lip sync issues.
        This is an async wrapper around detect_lip_sync_issues.
        """
        try:
            result = self.detect_lip_sync_issues(video_path)
            return {
                "score": result["lip_sync_score"],
                "lip_sync_score": result["lip_sync_score"],
                "major_lip_desync_detected": result["major_lip_desync_detected"],
                "lip_sync_timeline": [
                    {"timestamp": f"00:00:{int(seg['timestamp']):02d}", "score": seg["speaking_probability"] * 100}
                    for seg in result["speaking_segments"]
                ]
            }
        except Exception as e:
            logger.error(f"Error in lip sync analysis: {str(e)}")
            raise

    def __del__(self):
        """Cleanup when the object is destroyed."""
        try:
            if hasattr(self, 'face_mesh'):
                self.face_mesh.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

# FastAPI implementation
app = FastAPI()

class VideoRequest(BaseModel):
    video_path: str

@app.post("/lip_sync")
async def lip_sync_endpoint(request: VideoRequest):
    try:
        detector = LipSyncDetector()
        results = detector.detect_lip_sync_issues(request.video_path)
        return results
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004) 