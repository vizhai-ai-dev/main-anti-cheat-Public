#!/usr/bin/env python3
"""
Cheat Score Calculator Module

This module computes a cheating probability score based on inputs from:
- Gaze tracking (eye movement patterns)
- Lip sync detection (audio-visual synchronization)
- Multiple person detection (unauthorized persons)
- Audio analysis (background noise, multiple speakers)

The score ranges from 0 (safe/no cheating detected) to 1 (high risk/likely cheating).
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from flask import Blueprint, request, jsonify
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize blueprint for Flask integration
cheat_score_blueprint = Blueprint('cheat_score', __name__)

@dataclass
class CheatScoreWeights:
    """Configuration class for cheat score calculation weights"""
    gaze_weight: float = 0.25          # Weight for gaze tracking analysis
    lip_sync_weight: float = 0.20      # Weight for lip sync analysis
    person_detection_weight: float = 0.30  # Weight for multiple person detection
    audio_analysis_weight: float = 0.25    # Weight for audio analysis
    
    # Gaze-specific weights
    gaze_off_screen_penalty: float = 0.8   # Penalty for looking away from screen
    gaze_suspicious_pattern_penalty: float = 0.6  # Penalty for suspicious gaze patterns
    
    # Person detection weights
    multiple_person_penalty: float = 1.0   # Maximum penalty for multiple people
    face_person_mismatch_penalty: float = 0.7  # Penalty when face count != person count
    
    # Audio analysis weights
    multiple_speaker_penalty: float = 0.9  # Penalty for multiple speakers detected
    background_noise_penalty: float = 0.4  # Penalty for excessive background noise
    silence_penalty: float = 0.3           # Penalty for prolonged silence
    
    # Lip sync weights
    poor_lip_sync_penalty: float = 0.8     # Penalty for poor audio-visual sync

class CheatScoreCalculator:
    """Main class for calculating cheating probability scores"""
    
    def __init__(self, weights: Optional[CheatScoreWeights] = None):
        """Initialize the cheat score calculator
        
        Args:
            weights: Custom weights configuration. Uses default if None.
        """
        self.weights = weights or CheatScoreWeights()
        self.score_history: List[float] = []
        self.max_history_length = 100
    
    def _normalize_score(self, score: float) -> float:
        """Normalize score to 0-1 range"""
        return max(0.0, min(1.0, score))
    
    def _analyze_gaze_data(self, gaze_data: Dict[str, Any]) -> float:
        """Analyze gaze tracking data and return risk score
        
        Args:
            gaze_data: Dictionary containing gaze analysis results
            Expected format:
            {
                'direction': str,  # 'center', 'left', 'right', 'up', 'down', 'unknown', 'eyes_closed'
                'confidence': float,  # Optional confidence score
                'off_screen_duration': float,  # Optional: seconds looking away
                'suspicious_patterns': int,  # Optional: count of suspicious patterns
            }
        
        Returns:
            Risk score between 0 and 1
        """
        if not gaze_data:
            return 0.5  # Neutral score if no data
        
        risk_score = 0.0
        direction = gaze_data.get('direction', 'unknown')
        
        # Base penalties for different gaze directions
        direction_penalties = {
            'center': 0.0,      # Normal, looking at screen
            'left': 0.3,        # Moderate risk
            'right': 0.3,       # Moderate risk
            'up': 0.2,          # Low risk (thinking)
            'down': 0.1,        # Low risk (normal reading/thinking behavior)
            'unknown': 0.6,     # High risk (face not detected)
            'eyes_closed': 0.3  # Moderate risk (could be thinking or avoiding detection)
        }
        
        risk_score += direction_penalties.get(direction, 0.5)
        
        # Additional penalties
        if 'off_screen_duration' in gaze_data:
            # Penalty increases with time spent looking away
            off_screen_time = gaze_data['off_screen_duration']
            if off_screen_time > 5:  # More than 5 seconds
                risk_score += self.weights.gaze_off_screen_penalty * min(1.0, off_screen_time / 30)
        
        if 'suspicious_patterns' in gaze_data:
            # Penalty for suspicious gaze patterns
            pattern_count = gaze_data['suspicious_patterns']
            if pattern_count > 0:
                risk_score += self.weights.gaze_suspicious_pattern_penalty * min(1.0, pattern_count / 10)
        
        return self._normalize_score(risk_score)
    
    def _analyze_lip_sync_data(self, lip_sync_data: Dict[str, Any]) -> float:
        """Analyze lip sync data and return risk score
        
        Args:
            lip_sync_data: Dictionary containing lip sync analysis results
            Expected format:
            {
                'is_synced': bool,  # Whether audio and video are synchronized
                'sync_score': float,  # Synchronization quality score (0-1)
                'confidence': float,  # Optional confidence in the analysis
            }
        
        Returns:
            Risk score between 0 and 1
        """
        if not lip_sync_data:
            return 0.3  # Moderate risk if no data
        
        risk_score = 0.0
        
        # Check if lip sync is poor
        is_synced = lip_sync_data.get('is_synced', True)
        sync_score = lip_sync_data.get('sync_score', 1.0)
        
        if not is_synced:
            risk_score += self.weights.poor_lip_sync_penalty
        else:
            # Gradual penalty based on sync quality
            risk_score += self.weights.poor_lip_sync_penalty * (1.0 - sync_score)
        
        return self._normalize_score(risk_score)
    
    def _analyze_person_detection_data(self, person_data: Dict[str, Any]) -> float:
        """Analyze person detection data and return risk score
        
        Args:
            person_data: Dictionary containing person detection results
            Expected format:
            {
                'people_count': int,  # Number of people detected
                'face_count': int,    # Number of faces detected
                'confidence': float,  # Optional detection confidence
            }
        
        Returns:
            Risk score between 0 and 1
        """
        if not person_data:
            return 0.4  # Moderate risk if no data
        
        risk_score = 0.0
        people_count = person_data.get('people_count', 1)
        face_count = person_data.get('face_count', 1)
        
        # Penalty for multiple people
        if people_count > 1:
            # Exponential penalty for more people
            excess_people = people_count - 1
            risk_score += self.weights.multiple_person_penalty * min(1.0, excess_people / 3)
        
        # Penalty for mismatch between people and faces
        if abs(people_count - face_count) > 0:
            mismatch_ratio = abs(people_count - face_count) / max(people_count, face_count, 1)
            risk_score += self.weights.face_person_mismatch_penalty * mismatch_ratio
        
        # Penalty if no people detected (camera issues or avoidance)
        if people_count == 0:
            risk_score += 0.6
        
        return self._normalize_score(risk_score)
    
    def _analyze_audio_data(self, audio_data: Dict[str, Any]) -> float:
        """Analyze audio analysis data and return risk score
        
        Args:
            audio_data: Dictionary containing audio analysis results
            Expected format:
            {
                'multiple_speakers': bool,
                'speaker_confidence': float,
                'has_background_noise': bool,
                'noise_level': float,
                'has_prolonged_silence': bool,
                'silence_periods': List[Tuple[float, float]],
                'overall_quality': float,  # Optional overall audio quality score
            }
        
        Returns:
            Risk score between 0 and 1
        """
        if not audio_data:
            return 0.3  # Moderate risk if no data
        
        risk_score = 0.0
        
        # Multiple speakers detection
        if audio_data.get('multiple_speakers', False):
            speaker_confidence = audio_data.get('speaker_confidence', 1.0)
            risk_score += self.weights.multiple_speaker_penalty * speaker_confidence
        
        # Background noise analysis
        if audio_data.get('has_background_noise', False):
            noise_level = audio_data.get('noise_level', 0.5)
            risk_score += self.weights.background_noise_penalty * noise_level
        
        # Prolonged silence analysis
        if audio_data.get('has_prolonged_silence', False):
            silence_periods = audio_data.get('silence_periods', [])
            total_silence_duration = sum(end - start for start, end in silence_periods)
            # Penalty increases with total silence duration
            silence_penalty = min(1.0, total_silence_duration / 60)  # Normalize by 1 minute
            risk_score += self.weights.silence_penalty * silence_penalty
        
        return self._normalize_score(risk_score)
    
    def calculate_cheat_score(self, inputs: Dict[str, Any]) -> float:
        """Calculate overall cheating probability score
        
        Args:
            inputs: Dictionary containing analysis results from all modules
            Expected format:
            {
                'gaze_data': Dict,      # From gaze tracking module
                'lip_sync_data': Dict,  # From lip sync detection module
                'person_data': Dict,    # From person detection module
                'audio_data': Dict,     # From audio analysis module
                'timestamp': float,     # Optional timestamp
                'session_id': str,      # Optional session identifier
            }
        
        Returns:
            Cheating probability score between 0 (safe) and 1 (high risk)
        """
        try:
            # Extract individual analysis results
            gaze_data = inputs.get('gaze_data', {})
            lip_sync_data = inputs.get('lip_sync_data', {})
            person_data = inputs.get('person_data', {})
            audio_data = inputs.get('audio_data', {})
            
            # Calculate individual risk scores
            gaze_risk = self._analyze_gaze_data(gaze_data)
            lip_sync_risk = self._analyze_lip_sync_data(lip_sync_data)
            person_risk = self._analyze_person_detection_data(person_data)
            audio_risk = self._analyze_audio_data(audio_data)
            
            # Calculate weighted overall score
            overall_score = (
                gaze_risk * self.weights.gaze_weight +
                lip_sync_risk * self.weights.lip_sync_weight +
                person_risk * self.weights.person_detection_weight +
                audio_risk * self.weights.audio_analysis_weight
            )
            
            # Normalize the final score
            final_score = self._normalize_score(overall_score)
            
            # Add to history for trend analysis
            self.score_history.append(final_score)
            if len(self.score_history) > self.max_history_length:
                self.score_history.pop(0)
            
            # Log the calculation for debugging
            logger.info(f"Cheat score calculated: {final_score:.3f} "
                       f"(gaze: {gaze_risk:.3f}, lip_sync: {lip_sync_risk:.3f}, "
                       f"person: {person_risk:.3f}, audio: {audio_risk:.3f})")
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error calculating cheat score: {str(e)}")
            return 0.5  # Return neutral score on error
    
    def get_risk_level(self, score: float) -> str:
        """Convert numerical score to risk level description
        
        Args:
            score: Cheat score between 0 and 1
            
        Returns:
            Risk level as string
        """
        if score < 0.2:
            return "Very Low"
        elif score < 0.4:
            return "Low"
        elif score < 0.6:
            return "Medium"
        elif score < 0.8:
            return "High"
        else:
            return "Very High"
    
    def get_score_trend(self, window_size: int = 10) -> Dict[str, float]:
        """Analyze recent score trends
        
        Args:
            window_size: Number of recent scores to analyze
            
        Returns:
            Dictionary with trend analysis
        """
        if len(self.score_history) < 2:
            return {"trend": "insufficient_data", "average": 0.0, "variance": 0.0}
        
        recent_scores = self.score_history[-window_size:]
        average_score = np.mean(recent_scores)
        variance = np.var(recent_scores)
        
        # Determine trend
        if len(recent_scores) >= 3:
            first_half = np.mean(recent_scores[:len(recent_scores)//2])
            second_half = np.mean(recent_scores[len(recent_scores)//2:])
            
            if second_half > first_half + 0.1:
                trend = "increasing"
            elif second_half < first_half - 0.1:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "average": float(average_score),
            "variance": float(variance),
            "sample_size": len(recent_scores)
        }

# Global calculator instance
_calculator = CheatScoreCalculator()

def calculate_cheat_score(inputs: Dict[str, Any]) -> float:
    """Convenience function for calculating cheat score
    
    Args:
        inputs: Dictionary containing analysis results from all modules
        
    Returns:
        Cheating probability score between 0 (safe) and 1 (high risk)
    """
    return _calculator.calculate_cheat_score(inputs)

def get_risk_level(score: float) -> str:
    """Convenience function for getting risk level description"""
    return _calculator.get_risk_level(score)

def get_score_trend(window_size: int = 10) -> Dict[str, float]:
    """Convenience function for getting score trend analysis"""
    return _calculator.get_score_trend(window_size)

# Flask routes for web API integration
@cheat_score_blueprint.route('/cheat_score', methods=['POST'])
def cheat_score_endpoint():
    """Flask route for cheat score calculation
    
    Expects JSON with analysis data from all modules
    Returns cheat score and risk assessment
    """
    try:
        if not request.json:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Calculate cheat score
        score = calculate_cheat_score(request.json)
        risk_level = get_risk_level(score)
        trend_analysis = get_score_trend()
        
        response = {
            "cheat_score": score,
            "risk_level": risk_level,
            "trend_analysis": trend_analysis,
            "timestamp": request.json.get('timestamp'),
            "session_id": request.json.get('session_id')
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in cheat score endpoint: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@cheat_score_blueprint.route('/cheat_score/config', methods=['GET', 'POST'])
def cheat_score_config():
    """Flask route for getting/setting cheat score configuration"""
    global _calculator
    
    if request.method == 'GET':
        # Return current configuration
        weights = _calculator.weights
        config = {
            "gaze_weight": weights.gaze_weight,
            "lip_sync_weight": weights.lip_sync_weight,
            "person_detection_weight": weights.person_detection_weight,
            "audio_analysis_weight": weights.audio_analysis_weight,
            "gaze_off_screen_penalty": weights.gaze_off_screen_penalty,
            "multiple_person_penalty": weights.multiple_person_penalty,
            "multiple_speaker_penalty": weights.multiple_speaker_penalty,
            "poor_lip_sync_penalty": weights.poor_lip_sync_penalty
        }
        return jsonify(config)
    
    elif request.method == 'POST':
        # Update configuration
        try:
            new_weights = CheatScoreWeights(**request.json)
            _calculator.weights = new_weights
            return jsonify({"message": "Configuration updated successfully"})
        except Exception as e:
            return jsonify({"error": f"Invalid configuration: {str(e)}"}), 400

if __name__ == "__main__":
    # This module is designed to be imported and used by other modules
    print("Cheat Score Calculator Module")
    print("Import this module to use calculate_cheat_score() function")
    print("Example: from cheat_score import calculate_cheat_score") 