import cv2
import numpy as np
from typing import Dict, List, Tuple
import mediapipe as mp
from dataclasses import dataclass
import time

@dataclass
class CalibrationData:
    lip_landmarks: List[np.ndarray]
    eye_landmarks: List[np.ndarray]
    lip_mask: np.ndarray
    eye_mask: np.ndarray
    calibration_complete: bool = False

class CalibrationAPI:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.calibration_data = CalibrationData(
            lip_landmarks=[],
            eye_landmarks=[],
            lip_mask=None,
            eye_mask=None
        )
        self.frame_count = 0
        self.total_frames = 250
        self.lip_indices = list(range(61, 68)) + list(range(291, 301))  # Lip landmark indices
        self.eye_indices = list(range(33, 46)) + list(range(133, 146))  # Eye landmark indices
        self.quality_threshold = 0.8
        self.min_landmarks_quality = 0.85
        self.landmark_quality_history = []

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process a single frame for calibration with improved quality checks
        Returns the processed frame and calibration status
        """
        if self.frame_count >= self.total_frames:
            return frame, {"status": "complete", "data": self.calibration_data}

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract landmarks
            h, w = frame.shape[:2]
            landmarks = np.array([[int(lm.x * w), int(lm.y * h)] for lm in face_landmarks.landmark])
            
            # Calculate landmark quality
            landmark_quality = self._calculate_landmark_quality(landmarks, frame.shape[:2])
            self.landmark_quality_history.append(landmark_quality)
            
            # Only store high-quality landmarks
            if landmark_quality >= self.min_landmarks_quality:
                # Store landmarks for calibration
                self.calibration_data.lip_landmarks.append(landmarks[self.lip_indices])
                self.calibration_data.eye_landmarks.append(landmarks[self.eye_indices])

                # Create masks
                lip_mask = self._create_mask(landmarks[self.lip_indices], frame.shape[:2])
                eye_mask = self._create_mask(landmarks[self.eye_indices], frame.shape[:2])

                # Apply masks to frame
                frame = self._apply_masks(frame, lip_mask, eye_mask)

                self.frame_count += 1

        status = {
            "status": "calibrating",
            "progress": (self.frame_count / self.total_frames) * 100,
            "frames_processed": self.frame_count,
            "quality": np.mean(self.landmark_quality_history) if self.landmark_quality_history else 0
        }

        return frame, status

    def _create_mask(self, landmarks: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """Create a binary mask from landmarks"""
        mask = np.zeros(shape, dtype=np.uint8)
        hull = cv2.convexHull(landmarks)
        cv2.fillConvexPoly(mask, hull, 255)
        return mask

    def _apply_masks(self, frame: np.ndarray, lip_mask: np.ndarray, eye_mask: np.ndarray) -> np.ndarray:
        """Apply masks to the frame"""
        # Create a copy of the frame
        masked_frame = frame.copy()
        
        # Apply lip mask
        masked_frame[lip_mask == 0] = 0
        
        # Apply eye mask
        masked_frame[eye_mask == 0] = 0
        
        return masked_frame

    def _calculate_landmark_quality(self, landmarks: np.ndarray, shape: Tuple[int, int]) -> float:
        """Calculate the quality of detected landmarks"""
        h, w = shape
        
        # Check if landmarks are within frame bounds
        if np.any(landmarks < 0) or np.any(landmarks[:, 0] >= w) or np.any(landmarks[:, 1] >= h):
            return 0.0
        
        # Calculate landmark spread
        lip_spread = np.std(landmarks[self.lip_indices], axis=0)
        eye_spread = np.std(landmarks[self.eye_indices], axis=0)
        
        # Calculate landmark symmetry
        lip_symmetry = self._calculate_symmetry(landmarks[self.lip_indices])
        eye_symmetry = self._calculate_symmetry(landmarks[self.eye_indices])
        
        # Combine quality metrics
        quality = (
            0.3 * (1.0 - np.mean(lip_spread) / w) +  # Lip spread
            0.3 * (1.0 - np.mean(eye_spread) / w) +  # Eye spread
            0.2 * lip_symmetry +                      # Lip symmetry
            0.2 * eye_symmetry                        # Eye symmetry
        )
        
        return max(0.0, min(1.0, quality))

    def _calculate_symmetry(self, landmarks: np.ndarray) -> float:
        """Calculate symmetry score for landmarks"""
        if len(landmarks) < 2:
            return 0.0
            
        # Calculate center point
        center = np.mean(landmarks, axis=0)
        
        # Calculate distances from center
        distances = np.linalg.norm(landmarks - center, axis=1)
        
        # Calculate symmetry score based on distance variations
        symmetry = 1.0 - np.std(distances) / np.mean(distances)
        
        return max(0.0, min(1.0, symmetry))

    def get_calibration_results(self) -> Dict:
        """Get the final calibration results with quality checks"""
        if self.frame_count < self.total_frames:
            return {
                "status": "incomplete",
                "progress": (self.frame_count / self.total_frames) * 100,
                "quality": np.mean(self.landmark_quality_history) if self.landmark_quality_history else 0
            }

        # Calculate average landmarks with quality weighting
        if self.landmark_quality_history:
            weights = np.array(self.landmark_quality_history)
            weights = weights / np.sum(weights)  # Normalize weights
            
            avg_lip_landmarks = np.average(self.calibration_data.lip_landmarks, axis=0, weights=weights)
            avg_eye_landmarks = np.average(self.calibration_data.eye_landmarks, axis=0, weights=weights)
        else:
            avg_lip_landmarks = np.mean(self.calibration_data.lip_landmarks, axis=0)
            avg_eye_landmarks = np.mean(self.calibration_data.eye_landmarks, axis=0)

        # Create final masks
        self.calibration_data.lip_mask = self._create_mask(avg_lip_landmarks, (480, 640))
        self.calibration_data.eye_mask = self._create_mask(avg_eye_landmarks, (480, 640))
        self.calibration_data.calibration_complete = True

        # Calculate final quality metrics
        final_quality = np.mean(self.landmark_quality_history) if self.landmark_quality_history else 0
        lip_symmetry = self._calculate_symmetry(avg_lip_landmarks)
        eye_symmetry = self._calculate_symmetry(avg_eye_landmarks)

        return {
            "status": "complete",
            "lip_mask": self.calibration_data.lip_mask,
            "eye_mask": self.calibration_data.eye_mask,
            "avg_lip_landmarks": avg_lip_landmarks.tolist(),
            "avg_eye_landmarks": avg_eye_landmarks.tolist(),
            "quality_metrics": {
                "overall_quality": final_quality,
                "lip_symmetry": lip_symmetry,
                "eye_symmetry": eye_symmetry,
                "frames_used": self.frame_count,
                "average_landmark_quality": np.mean(self.landmark_quality_history) if self.landmark_quality_history else 0
            }
        }

    def reset_calibration(self):
        """Reset the calibration process"""
        self.frame_count = 0
        self.calibration_data = CalibrationData(
            lip_landmarks=[],
            eye_landmarks=[],
            lip_mask=None,
            eye_mask=None
        ) 