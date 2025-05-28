#!/usr/bin/env python3
"""
Calibration API for Gaze Tracking and Lip Sync Detection

This module provides calibration functionality for improved gaze tracking accuracy
and personalized lip movement detection thresholds.
"""

import cv2
import numpy as np
import time
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CalibrationPoint:
    """Represents a calibration point on the screen"""
    x: float
    y: float
    screen_x: int
    screen_y: int
    gaze_data: Optional[Dict] = None

@dataclass
class LipCalibrationData:
    """Represents lip calibration data"""
    silent_distances: List[float]
    speaking_distances: List[float]
    silent_variations: List[float]
    speaking_variations: List[float]
    baseline_distance: float
    baseline_std: float
    speaking_threshold: float
    variation_threshold: float
    accuracy: float
    calibration_time: float

class CalibratedGazeTracker:
    """Enhanced gaze tracker with calibration support"""
    
    def __init__(self):
        """Initialize the calibrated gaze tracker"""
        self.calibration_points: List[CalibrationPoint] = []
        self.is_calibrated = False
        self.calibration_matrix = None
        self.screen_width = 1920  # Default screen width
        self.screen_height = 1080  # Default screen height
        
        # Standard calibration points (normalized coordinates 0-1)
        self.standard_points = [
            (0.1, 0.1),   # Top-left
            (0.5, 0.1),   # Top-center
            (0.9, 0.1),   # Top-right
            (0.1, 0.5),   # Middle-left
            (0.5, 0.5),   # Center
            (0.9, 0.5),   # Middle-right
            (0.1, 0.9),   # Bottom-left
            (0.5, 0.9),   # Bottom-center
            (0.9, 0.9),   # Bottom-right
        ]
    
    def set_screen_resolution(self, width: int, height: int):
        """Set the screen resolution for calibration"""
        self.screen_width = width
        self.screen_height = height
        logger.info(f"Screen resolution set to {width}x{height}")
    
    def get_calibration_points(self) -> List[Tuple[int, int]]:
        """Get the screen coordinates for calibration points"""
        points = []
        for norm_x, norm_y in self.standard_points:
            screen_x = int(norm_x * self.screen_width)
            screen_y = int(norm_y * self.screen_height)
            points.append((screen_x, screen_y))
        return points
    
    def start_calibration(self) -> Dict:
        """Start the calibration process"""
        logger.info("Starting gaze tracking calibration")
        self.calibration_points.clear()
        self.is_calibrated = False
        
        return {
            "status": "started",
            "total_points": len(self.standard_points),
            "screen_resolution": (self.screen_width, self.screen_height),
            "instructions": "Look directly at each calibration point when prompted"
        }
    
    def capture_calibration_point(self, point_index: int, gaze_data: Dict) -> Dict:
        """Capture gaze data for a specific calibration point"""
        if point_index >= len(self.standard_points):
            return {"error": "Invalid calibration point index"}
        
        norm_x, norm_y = self.standard_points[point_index]
        screen_x = int(norm_x * self.screen_width)
        screen_y = int(norm_y * self.screen_height)
        
        calibration_point = CalibrationPoint(
            x=norm_x,
            y=norm_y,
            screen_x=screen_x,
            screen_y=screen_y,
            gaze_data=gaze_data
        )
        
        self.calibration_points.append(calibration_point)
        
        logger.info(f"Captured calibration point {point_index + 1}/{len(self.standard_points)}")
        
        return {
            "status": "captured",
            "point_index": point_index,
            "screen_coordinates": (screen_x, screen_y),
            "points_remaining": len(self.standard_points) - len(self.calibration_points)
        }
    
    def complete_calibration(self) -> Dict:
        """Complete the calibration process and compute transformation matrix"""
        if len(self.calibration_points) < 4:
            return {"error": "Insufficient calibration points (minimum 4 required)"}
        
        try:
            # Extract screen points and gaze points
            screen_points = []
            gaze_points = []
            
            for point in self.calibration_points:
                screen_points.append([point.screen_x, point.screen_y])
                
                # Extract gaze coordinates from gaze_data
                # This is a simplified version - in practice, you'd extract
                # actual gaze coordinates from the eye tracking data
                gaze_x = point.gaze_data.get('gaze_x', point.screen_x)
                gaze_y = point.gaze_data.get('gaze_y', point.screen_y)
                gaze_points.append([gaze_x, gaze_y])
            
            screen_points = np.array(screen_points, dtype=np.float32)
            gaze_points = np.array(gaze_points, dtype=np.float32)
            
            # Compute transformation matrix using perspective transform
            if len(screen_points) >= 4:
                self.calibration_matrix = cv2.getPerspectiveTransform(
                    gaze_points[:4], screen_points[:4]
                )
            else:
                # Use affine transform for fewer points
                self.calibration_matrix = cv2.getAffineTransform(
                    gaze_points[:3], screen_points[:3]
                )
            
            self.is_calibrated = True
            
            # Calculate calibration accuracy
            accuracy = self._calculate_calibration_accuracy()
            
            logger.info("Calibration completed successfully")
            
            return {
                "status": "completed",
                "calibrated": True,
                "accuracy": accuracy,
                "points_used": len(self.calibration_points),
                "transformation_matrix": self.calibration_matrix.tolist()
            }
            
        except Exception as e:
            logger.error(f"Calibration failed: {str(e)}")
            return {"error": f"Calibration computation failed: {str(e)}"}
    
    def _calculate_calibration_accuracy(self) -> float:
        """Calculate the accuracy of the calibration"""
        if not self.is_calibrated or not self.calibration_points:
            return 0.0
        
        total_error = 0.0
        valid_points = 0
        
        for point in self.calibration_points:
            try:
                # Get original gaze coordinates
                gaze_x = point.gaze_data.get('gaze_x', point.screen_x)
                gaze_y = point.gaze_data.get('gaze_y', point.screen_y)
                
                # Transform using calibration matrix
                transformed = self.transform_gaze_point(gaze_x, gaze_y)
                
                if transformed:
                    # Calculate Euclidean distance error
                    error = np.sqrt(
                        (transformed[0] - point.screen_x) ** 2 +
                        (transformed[1] - point.screen_y) ** 2
                    )
                    total_error += error
                    valid_points += 1
                    
            except Exception:
                continue
        
        if valid_points == 0:
            return 0.0
        
        # Return accuracy as percentage (lower error = higher accuracy)
        average_error = total_error / valid_points
        max_screen_distance = np.sqrt(self.screen_width**2 + self.screen_height**2)
        accuracy = max(0, 100 * (1 - average_error / max_screen_distance))
        
        return round(accuracy, 2)
    
    def transform_gaze_point(self, gaze_x: float, gaze_y: float) -> Optional[Tuple[int, int]]:
        """Transform raw gaze coordinates to calibrated screen coordinates"""
        if not self.is_calibrated or self.calibration_matrix is None:
            return None
        
        try:
            # Apply transformation matrix
            if self.calibration_matrix.shape[0] == 3:
                # Perspective transformation
                point = np.array([[[gaze_x, gaze_y]]], dtype=np.float32)
                transformed = cv2.perspectiveTransform(point, self.calibration_matrix)
                screen_x, screen_y = transformed[0][0]
            else:
                # Affine transformation
                point = np.array([[gaze_x, gaze_y, 1]], dtype=np.float32).T
                transformed = self.calibration_matrix @ point
                screen_x, screen_y = transformed[:2, 0]
            
            # Clamp to screen boundaries
            screen_x = max(0, min(int(screen_x), self.screen_width - 1))
            screen_y = max(0, min(int(screen_y), self.screen_height - 1))
            
            return (screen_x, screen_y)
            
        except Exception as e:
            logger.error(f"Gaze transformation failed: {str(e)}")
            return None
    
    def save_calibration(self, filename: str) -> bool:
        """Save calibration data to file"""
        if not self.is_calibrated:
            return False
        
        try:
            calibration_data = {
                "is_calibrated": self.is_calibrated,
                "screen_resolution": (self.screen_width, self.screen_height),
                "calibration_matrix": self.calibration_matrix.tolist(),
                "calibration_points": [
                    {
                        "x": point.x,
                        "y": point.y,
                        "screen_x": point.screen_x,
                        "screen_y": point.screen_y,
                        "gaze_data": point.gaze_data
                    }
                    for point in self.calibration_points
                ],
                "accuracy": self._calculate_calibration_accuracy(),
                "timestamp": time.time()
            }
            
            with open(filename, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            
            logger.info(f"Calibration saved to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save calibration: {str(e)}")
            return False
    
    def load_calibration(self, filename: str) -> bool:
        """Load calibration data from file"""
        try:
            with open(filename, 'r') as f:
                calibration_data = json.load(f)
            
            self.is_calibrated = calibration_data.get("is_calibrated", False)
            self.screen_width, self.screen_height = calibration_data.get(
                "screen_resolution", (1920, 1080)
            )
            
            matrix_data = calibration_data.get("calibration_matrix")
            if matrix_data:
                self.calibration_matrix = np.array(matrix_data, dtype=np.float32)
            
            # Restore calibration points
            self.calibration_points.clear()
            for point_data in calibration_data.get("calibration_points", []):
                point = CalibrationPoint(
                    x=point_data["x"],
                    y=point_data["y"],
                    screen_x=point_data["screen_x"],
                    screen_y=point_data["screen_y"],
                    gaze_data=point_data["gaze_data"]
                )
                self.calibration_points.append(point)
            
            logger.info(f"Calibration loaded from {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load calibration: {str(e)}")
            return False
    
    def get_calibration_status(self) -> Dict:
        """Get current calibration status"""
        return {
            "is_calibrated": self.is_calibrated,
            "screen_resolution": (self.screen_width, self.screen_height),
            "calibration_points_count": len(self.calibration_points),
            "accuracy": self._calculate_calibration_accuracy() if self.is_calibrated else 0.0
        }

# Global calibrated tracker instance
_calibrated_tracker = CalibratedGazeTracker()

def get_calibrated_tracker() -> CalibratedGazeTracker:
    """Get the global calibrated tracker instance"""
    return _calibrated_tracker

def calibrate_gaze_tracking(screen_width: int = 1920, screen_height: int = 1080) -> Dict:
    """Convenience function to start calibration process"""
    tracker = get_calibrated_tracker()
    tracker.set_screen_resolution(screen_width, screen_height)
    return tracker.start_calibration()

def is_system_calibrated() -> bool:
    """Check if the system is currently calibrated"""
    return get_calibrated_tracker().is_calibrated

class LipSyncCalibrator:
    """Calibrator for personalized lip movement detection"""
    
    def __init__(self):
        """Initialize the lip sync calibrator"""
        self.is_calibrated = False
        self.calibration_data: Optional[LipCalibrationData] = None
        self.silent_buffer: List[float] = []
        self.speaking_buffer: List[float] = []
        
        # Default thresholds (will be overridden by calibration)
        self.speaking_threshold = 12.0
        self.variation_threshold = 3.0
        self.baseline_distance = 8.0
        self.baseline_std = 2.0
    
    def start_silent_calibration(self, duration_seconds: float = 5.0) -> Dict:
        """Start silent calibration phase"""
        logger.info(f"Starting silent lip calibration for {duration_seconds} seconds")
        self.silent_buffer.clear()
        
        return {
            "status": "silent_calibration_started",
            "duration": duration_seconds,
            "instructions": "Keep your mouth closed and stay silent. Look at the camera normally.",
            "frames_needed": int(duration_seconds * 30)  # Assume 30 FPS
        }
    
    def add_silent_sample(self, lip_distance: float) -> Dict:
        """Add a lip distance sample during silent calibration"""
        self.silent_buffer.append(lip_distance)
        
        return {
            "status": "silent_sample_added",
            "samples_collected": len(self.silent_buffer),
            "current_distance": lip_distance
        }
    
    def complete_silent_calibration(self) -> Dict:
        """Complete the silent calibration phase"""
        if len(self.silent_buffer) < 30:  # Minimum 1 second of data
            return {"error": "Insufficient silent samples (minimum 30 required)"}
        
        # Calculate baseline statistics
        self.baseline_distance = np.mean(self.silent_buffer)
        self.baseline_std = np.std(self.silent_buffer)
        
        logger.info(f"Silent calibration complete: baseline={self.baseline_distance:.2f}±{self.baseline_std:.2f}")
        
        return {
            "status": "silent_calibration_complete",
            "baseline_distance": self.baseline_distance,
            "baseline_std": self.baseline_std,
            "samples_used": len(self.silent_buffer),
            "next_phase": "speaking_calibration"
        }
    
    def start_speaking_calibration(self, duration_seconds: float = 10.0) -> Dict:
        """Start speaking calibration phase"""
        logger.info(f"Starting speaking lip calibration for {duration_seconds} seconds")
        self.speaking_buffer.clear()
        
        speaking_prompts = [
            "Count from 1 to 10 slowly",
            "Say the alphabet A to Z",
            "Repeat: 'The quick brown fox jumps over the lazy dog'",
            "Say your full name and address",
            "Describe what you see in the room"
        ]
        
        return {
            "status": "speaking_calibration_started",
            "duration": duration_seconds,
            "instructions": "Speak clearly and naturally. Try the following prompts:",
            "prompts": speaking_prompts,
            "frames_needed": int(duration_seconds * 30)  # Assume 30 FPS
        }
    
    def add_speaking_sample(self, lip_distance: float) -> Dict:
        """Add a lip distance sample during speaking calibration"""
        self.speaking_buffer.append(lip_distance)
        
        return {
            "status": "speaking_sample_added",
            "samples_collected": len(self.speaking_buffer),
            "current_distance": lip_distance
        }
    
    def complete_speaking_calibration(self) -> Dict:
        """Complete the speaking calibration phase"""
        if len(self.speaking_buffer) < 60:  # Minimum 2 seconds of data
            return {"error": "Insufficient speaking samples (minimum 60 required)"}
        
        # Calculate speaking statistics
        speaking_mean = np.mean(self.speaking_buffer)
        speaking_std = np.std(self.speaking_buffer)
        speaking_max = np.max(self.speaking_buffer)
        
        logger.info(f"Speaking calibration complete: mean={speaking_mean:.2f}±{speaking_std:.2f}, max={speaking_max:.2f}")
        
        return {
            "status": "speaking_calibration_complete",
            "speaking_mean": speaking_mean,
            "speaking_std": speaking_std,
            "speaking_max": speaking_max,
            "samples_used": len(self.speaking_buffer),
            "next_phase": "threshold_calculation"
        }
    
    def calculate_thresholds(self) -> Dict:
        """Calculate personalized thresholds based on calibration data"""
        if len(self.silent_buffer) < 30 or len(self.speaking_buffer) < 60:
            return {"error": "Insufficient calibration data"}
        
        try:
            # Calculate statistics for both phases
            silent_mean = np.mean(self.silent_buffer)
            silent_std = np.std(self.silent_buffer)
            silent_max = np.max(self.silent_buffer)
            
            speaking_mean = np.mean(self.speaking_buffer)
            speaking_std = np.std(self.speaking_buffer)
            speaking_min = np.min(self.speaking_buffer)
            
            # Calculate variation statistics
            silent_variations = []
            speaking_variations = []
            
            # Calculate rolling variations for silent phase
            for i in range(len(self.silent_buffer) - 10):
                window = self.silent_buffer[i:i+10]
                silent_variations.append(np.std(window))
            
            # Calculate rolling variations for speaking phase
            for i in range(len(self.speaking_buffer) - 10):
                window = self.speaking_buffer[i:i+10]
                speaking_variations.append(np.std(window))
            
            silent_var_mean = np.mean(silent_variations) if silent_variations else 0
            speaking_var_mean = np.mean(speaking_variations) if speaking_variations else 0
            
            # Calculate thresholds using statistical separation
            # Distance threshold: midpoint between silent max and speaking min, with safety margins
            distance_gap = speaking_min - silent_max
            if distance_gap > 2.0:  # Good separation
                self.speaking_threshold = silent_max + (distance_gap * 0.3)  # 30% into the gap
            else:  # Poor separation, use statistical approach
                self.speaking_threshold = silent_mean + (3 * silent_std) + 2.0
            
            # Variation threshold: based on the difference in variation patterns
            var_gap = speaking_var_mean - silent_var_mean
            if var_gap > 1.0:  # Good variation separation
                self.variation_threshold = silent_var_mean + (var_gap * 0.4)
            else:  # Use statistical approach
                self.variation_threshold = silent_var_mean + (2 * np.std(silent_variations)) + 0.5
            
            # Ensure minimum thresholds for safety
            self.speaking_threshold = max(self.speaking_threshold, silent_mean + 3.0)
            self.variation_threshold = max(self.variation_threshold, 1.0)
            
            # Calculate calibration accuracy
            accuracy = self._calculate_lip_calibration_accuracy(
                silent_mean, silent_std, speaking_mean, speaking_std
            )
            
            # Store calibration data
            self.calibration_data = LipCalibrationData(
                silent_distances=self.silent_buffer.copy(),
                speaking_distances=self.speaking_buffer.copy(),
                silent_variations=silent_variations,
                speaking_variations=speaking_variations,
                baseline_distance=silent_mean,
                baseline_std=silent_std,
                speaking_threshold=self.speaking_threshold,
                variation_threshold=self.variation_threshold,
                accuracy=accuracy,
                calibration_time=time.time()
            )
            
            self.is_calibrated = True
            
            logger.info(f"Lip calibration complete - Threshold: {self.speaking_threshold:.2f}, Accuracy: {accuracy:.1f}%")
            
            return {
                "status": "calibration_complete",
                "speaking_threshold": self.speaking_threshold,
                "variation_threshold": self.variation_threshold,
                "baseline_distance": silent_mean,
                "baseline_std": silent_std,
                "accuracy": accuracy,
                "separation_quality": "Good" if distance_gap > 2.0 else "Fair" if distance_gap > 0 else "Poor",
                "distance_gap": distance_gap,
                "recommendations": self._get_calibration_recommendations(distance_gap, accuracy)
            }
            
        except Exception as e:
            logger.error(f"Threshold calculation failed: {str(e)}")
            return {"error": f"Threshold calculation failed: {str(e)}"}
    
    def _calculate_lip_calibration_accuracy(self, silent_mean: float, silent_std: float, 
                                          speaking_mean: float, speaking_std: float) -> float:
        """Calculate the accuracy of lip calibration"""
        try:
            # Test threshold against calibration data
            silent_correct = 0
            speaking_correct = 0
            
            # Test silent samples (should be below threshold)
            for distance in self.silent_buffer:
                if distance < self.speaking_threshold:
                    silent_correct += 1
            
            # Test speaking samples (should be above threshold)
            for distance in self.speaking_buffer:
                if distance >= self.speaking_threshold:
                    speaking_correct += 1
            
            # Calculate accuracy
            total_correct = silent_correct + speaking_correct
            total_samples = len(self.silent_buffer) + len(self.speaking_buffer)
            
            accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
            return round(accuracy, 1)
            
        except Exception:
            return 0.0
    
    def _get_calibration_recommendations(self, distance_gap: float, accuracy: float) -> List[str]:
        """Get recommendations based on calibration results"""
        recommendations = []
        
        if accuracy >= 90:
            recommendations.append("Excellent calibration quality!")
        elif accuracy >= 80:
            recommendations.append("Good calibration quality")
        elif accuracy >= 70:
            recommendations.append("Fair calibration - consider recalibrating")
        else:
            recommendations.append("Poor calibration - please recalibrate")
        
        if distance_gap < 1.0:
            recommendations.append("Speak more clearly during calibration")
            recommendations.append("Ensure good lighting conditions")
        
        if distance_gap < 0:
            recommendations.append("Your speaking lip movement may be subtle - try exaggerating slightly")
        
        return recommendations
    
    def is_speaking(self, lip_distance: float, lip_variation: float = 0.0) -> Dict:
        """Determine if the person is speaking based on calibrated thresholds"""
        if not self.is_calibrated:
            # Use default logic if not calibrated
            return {
                "is_speaking": lip_distance > 12.0 or lip_variation > 3.0,
                "confidence": 0.5,
                "method": "default_thresholds",
                "lip_distance": lip_distance,
                "threshold": 12.0
            }
        
        # Use calibrated thresholds
        distance_speaking = lip_distance > self.speaking_threshold
        variation_speaking = lip_variation > self.variation_threshold
        
        # Additional check: significant deviation from baseline
        baseline_deviation = abs(lip_distance - self.baseline_distance) > (self.baseline_std * 2.5)
        
        is_speaking = distance_speaking or variation_speaking or baseline_deviation
        
        # Calculate confidence based on how far above threshold
        if is_speaking:
            distance_confidence = min(1.0, (lip_distance - self.speaking_threshold) / self.speaking_threshold)
            variation_confidence = min(1.0, lip_variation / self.variation_threshold) if self.variation_threshold > 0 else 0
            confidence = max(distance_confidence, variation_confidence, 0.6)
        else:
            # Confidence is higher when clearly below threshold
            distance_below = max(0, self.speaking_threshold - lip_distance)
            confidence = min(0.9, 0.5 + (distance_below / self.speaking_threshold))
        
        return {
            "is_speaking": is_speaking,
            "confidence": round(confidence, 3),
            "method": "calibrated_thresholds",
            "lip_distance": lip_distance,
            "threshold": self.speaking_threshold,
            "baseline_distance": self.baseline_distance,
            "distance_from_baseline": abs(lip_distance - self.baseline_distance),
            "triggers": {
                "distance": distance_speaking,
                "variation": variation_speaking,
                "baseline_deviation": baseline_deviation
            }
        }
    
    def save_calibration(self, filename: str) -> bool:
        """Save lip calibration data to file"""
        if not self.is_calibrated or not self.calibration_data:
            return False
        
        try:
            calibration_dict = {
                "is_calibrated": self.is_calibrated,
                "speaking_threshold": self.speaking_threshold,
                "variation_threshold": self.variation_threshold,
                "baseline_distance": self.baseline_distance,
                "baseline_std": self.baseline_std,
                "calibration_data": {
                    "silent_distances": self.calibration_data.silent_distances,
                    "speaking_distances": self.calibration_data.speaking_distances,
                    "silent_variations": self.calibration_data.silent_variations,
                    "speaking_variations": self.calibration_data.speaking_variations,
                    "accuracy": self.calibration_data.accuracy,
                    "calibration_time": self.calibration_data.calibration_time
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(calibration_dict, f, indent=2)
            
            logger.info(f"Lip calibration saved to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save lip calibration: {str(e)}")
            return False
    
    def load_calibration(self, filename: str) -> bool:
        """Load lip calibration data from file"""
        try:
            with open(filename, 'r') as f:
                calibration_dict = json.load(f)
            
            self.is_calibrated = calibration_dict.get("is_calibrated", False)
            self.speaking_threshold = calibration_dict.get("speaking_threshold", 12.0)
            self.variation_threshold = calibration_dict.get("variation_threshold", 3.0)
            self.baseline_distance = calibration_dict.get("baseline_distance", 8.0)
            self.baseline_std = calibration_dict.get("baseline_std", 2.0)
            
            # Restore calibration data
            cal_data = calibration_dict.get("calibration_data", {})
            if cal_data:
                self.calibration_data = LipCalibrationData(
                    silent_distances=cal_data.get("silent_distances", []),
                    speaking_distances=cal_data.get("speaking_distances", []),
                    silent_variations=cal_data.get("silent_variations", []),
                    speaking_variations=cal_data.get("speaking_variations", []),
                    baseline_distance=self.baseline_distance,
                    baseline_std=self.baseline_std,
                    speaking_threshold=self.speaking_threshold,
                    variation_threshold=self.variation_threshold,
                    accuracy=cal_data.get("accuracy", 0.0),
                    calibration_time=cal_data.get("calibration_time", time.time())
                )
            
            logger.info(f"Lip calibration loaded from {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load lip calibration: {str(e)}")
            return False
    
    def get_calibration_status(self) -> Dict:
        """Get current lip calibration status"""
        status = {
            "is_calibrated": self.is_calibrated,
            "speaking_threshold": self.speaking_threshold,
            "variation_threshold": self.variation_threshold,
            "baseline_distance": self.baseline_distance,
            "baseline_std": self.baseline_std
        }
        
        if self.calibration_data:
            status.update({
                "accuracy": self.calibration_data.accuracy,
                "calibration_time": self.calibration_data.calibration_time,
                "silent_samples": len(self.calibration_data.silent_distances),
                "speaking_samples": len(self.calibration_data.speaking_distances)
            })
        
        return status
    
    def reset_calibration(self):
        """Reset calibration data"""
        self.is_calibrated = False
        self.calibration_data = None
        self.silent_buffer.clear()
        self.speaking_buffer.clear()
        
        # Reset to default thresholds
        self.speaking_threshold = 12.0
        self.variation_threshold = 3.0
        self.baseline_distance = 8.0
        self.baseline_std = 2.0
        
        logger.info("Lip calibration reset")

# Global lip sync calibrator instance
_lip_sync_calibrator = LipSyncCalibrator()

def get_lip_sync_calibrator() -> LipSyncCalibrator:
    """Get the global lip sync calibrator instance"""
    return _lip_sync_calibrator

def calibrate_lip_sync() -> Dict:
    """Convenience function to start lip sync calibration process"""
    calibrator = get_lip_sync_calibrator()
    calibrator.reset_calibration()
    return calibrator.start_silent_calibration()

def is_lip_sync_calibrated() -> bool:
    """Check if the lip sync system is currently calibrated"""
    return get_lip_sync_calibrator().is_calibrated 