import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import json
from typing import Dict, List, Optional

# Try to import xgboost, but make it optional
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available, falling back to rule-based scoring only")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CheatScoreCalculator:
    def __init__(self, use_xgboost=False):
        # Check if xgboost is requested but not available
        if use_xgboost and not XGBOOST_AVAILABLE:
            logger.warning("XGBoost requested but not available, falling back to rule-based scoring")
            use_xgboost = False
            
        self.use_xgboost = use_xgboost
        if use_xgboost:
            # Load XGBoost model
            try:
                self.model = xgb.Booster()
                self.model.load_model('models/cheat_score_model.json')
                logger.info("XGBoost model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading XGBoost model: {str(e)}")
                logger.info("Falling back to rule-based scoring")
                self.use_xgboost = False
        
        # Risk thresholds
        self.RISK_THRESHOLDS = {
            "Low": 30,
            "Medium": 60,
            "High": 80
        }
        
        # Feature weights for rule-based scoring
        self.WEIGHTS = {
            "gaze": 0.35,
            "audio": 0.25,
            "multi_person": 0.25,
            "lip_sync": 0.15
        }
    
    def _normalize_score(self, value, min_val=0, max_val=100):
        """Normalize values to 0-100 scale"""
        return max(min(100, float(value)), 0)
    
    def _get_risk_level(self, score):
        """Determine risk level from score"""
        if score < self.RISK_THRESHOLDS["Low"]:
            return "Low"
        elif score < self.RISK_THRESHOLDS["Medium"]:
            return "Medium"
        elif score < self.RISK_THRESHOLDS["High"]:
            return "High"
        else:
            return "Very High"
    
    def _generate_reasons(self, results):
        """Generate plain text reasons for flagging"""
        reasons = []
        
        # Gaze reasons
        if "gaze" in results:
            gaze = results["gaze"]
            if gaze.get("off_screen_count", 0) > 10:
                reasons.append(f"Looked away from screen {gaze.get('off_screen_count')} times")
            if gaze.get("average_confidence", 0) < 0.6:
                reasons.append("Inconsistent gaze tracking detected")
        
        # Audio reasons
        if "audio" in results:
            audio = results["audio"]
            if audio.get("voice_detected", False) and audio.get("multiple_speakers", False):
                reasons.append("Multiple speakers detected")
            if audio.get("keyboard_typing_count", 0) > 20:
                reasons.append(f"Excessive keyboard typing detected ({audio.get('keyboard_typing_count')} instances)")
        
        # Multi-person reasons
        if "multi_person" in results:
            multi = results["multi_person"]
            if multi.get("max_people_detected", 0) > 1:
                reasons.append(f"Maximum of {multi.get('max_people_detected')} people detected")
            if multi.get("time_with_multiple_people", 0) > 10:
                reasons.append(f"Multiple people present for {multi.get('time_with_multiple_people'):.1f} seconds")
            if multi.get("has_different_faces", False):
                diff_faces = multi.get("different_faces_detected", 0)
                reasons.append(f"{diff_faces} different {'face' if diff_faces == 1 else 'faces'} detected (potential identity switching)")
        
        # Lip sync reasons
        if "lip_sync" in results:
            lip_sync = results["lip_sync"]
            if lip_sync.get("major_lip_desync_detected", False):
                reasons.append("Lip movement doesn't match audio (possible voice-over)")
            if lip_sync.get("lip_sync_score", 100) < 70:
                reasons.append("Poor lip-sync detected, potential audio manipulation")
        
        return reasons
    
    def _rule_based_scoring(self, results):
        """Calculate cheating score using rule-based approach"""
        scores = {}
        
        # Calculate gaze score
        if "gaze" in results:
            gaze_data = results["gaze"]
            off_screen_score = 100 - min(100, gaze_data.get("off_screen_count", 0) * 5)
            confidence_score = min(100, gaze_data.get("average_confidence", 0) * 100)
            scores["gaze"] = (off_screen_score * 0.8 + confidence_score * 0.2)
        else:
            scores["gaze"] = 100
        
        # Calculate audio score
        if "audio" in results:
            audio_data = results["audio"]
            speaker_score = 0 if audio_data.get("multiple_speakers", False) else 100
            typing_score = 100 - min(100, audio_data.get("keyboard_typing_count", 0) * 2)
            silence_score = 100 - min(100, audio_data.get("silence_percentage", 0))
            scores["audio"] = (speaker_score * 0.5 + typing_score * 0.3 + silence_score * 0.2)
        else:
            scores["audio"] = 100
        
        # Calculate multi-person score
        if "multi_person" in results:
            multi_data = results["multi_person"]
            people_score = 100 if multi_data.get("max_people_detected", 0) <= 1 else 0
            time_score = 100 - min(100, multi_data.get("time_with_multiple_people", 0) * 5)
            
            # Apply severe penalty for different faces detected
            face_switching_penalty = 0
            if multi_data.get("has_different_faces", False):
                face_switching_penalty = min(100, multi_data.get("different_faces_detected", 0) * 30)
            
            # Combine scores with heavier weight on face switching
            scores["multi_person"] = max(0, (people_score * 0.3 + time_score * 0.2 - face_switching_penalty))
        else:
            scores["multi_person"] = 100
        
        # Calculate lip sync score
        if "lip_sync" in results:
            lip_sync_data = results["lip_sync"]
            lip_sync_score = lip_sync_data.get("lip_sync_score", 100)
            desync_penalty = 30 if lip_sync_data.get("major_lip_desync_detected", False) else 0
            scores["lip_sync"] = max(0, lip_sync_score - desync_penalty)
        else:
            scores["lip_sync"] = 100
        
        # Calculate weighted average
        final_score = 0
        total_weight = 0
        
        for key, score in scores.items():
            weight = self.WEIGHTS.get(key, 0)
            final_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            final_score /= total_weight
        
        # Inverse the score - higher means more likely to be cheating
        cheat_score = 100 - final_score
        
        return cheat_score
    
    def _xgboost_scoring(self, results):
        """Calculate cheating score using XGBoost model"""
        try:
            if not XGBOOST_AVAILABLE:
                logger.warning("XGBoost not available, falling back to rule-based scoring")
                return self._rule_based_scoring(results)
                
            # Extract features from results
            features = []
            
            # Gaze features
            gaze_data = results.get("gaze", {})
            features.append(gaze_data.get("off_screen_count", 0))
            features.append(gaze_data.get("average_confidence", 0))
            
            # Audio features
            audio_data = results.get("audio", {})
            features.append(1 if audio_data.get("multiple_speakers", False) else 0)
            features.append(audio_data.get("keyboard_typing_count", 0))
            features.append(audio_data.get("silence_percentage", 0))
            
            # Multi-person features
            multi_data = results.get("multi_person", {})
            features.append(multi_data.get("max_people_detected", 0))
            features.append(multi_data.get("time_with_multiple_people", 0))
            
            # Lip sync features
            lip_sync_data = results.get("lip_sync", {})
            features.append(lip_sync_data.get("lip_sync_score", 100))
            features.append(1 if lip_sync_data.get("major_lip_desync_detected", False) else 0)
            
            # Convert features to DMatrix
            dmatrix = xgb.DMatrix(np.array([features]))
            
            # Predict score
            cheat_score = float(self.model.predict(dmatrix)[0])
            
            # Ensure score is in range 0-100
            cheat_score = self._normalize_score(cheat_score)
            
            return cheat_score
        except Exception as e:
            logger.error(f"Error in XGBoost scoring: {str(e)}")
            logger.info("Falling back to rule-based scoring")
            return self._rule_based_scoring(results)
    
    def compute_cheating_score(self, results_dict):
        """Main function to compute cheating score"""
        try:
            # Calculate score based on model choice
            if self.use_xgboost:
                final_score = self._xgboost_scoring(results_dict)
            else:
                final_score = self._rule_based_scoring(results_dict)
            
            # Round score to 1 decimal place
            final_score = round(final_score, 1)
            
            # Determine risk level
            risk = self._get_risk_level(final_score)
            
            # Generate reasons
            reasons = self._generate_reasons(results_dict)
            
            return {
                "final_score": final_score,
                "risk": risk,
                "reasons": reasons
            }
        except Exception as e:
            logger.error(f"Error computing cheat score: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise

# FastAPI implementation
app = FastAPI()

class CheatScoreRequest(BaseModel):
    gaze: Optional[Dict] = None
    audio: Optional[Dict] = None
    multi_person: Optional[Dict] = None
    lip_sync: Optional[Dict] = None

@app.post("/cheat_score")
async def cheat_score_endpoint(request: CheatScoreRequest):
    try:
        # Convert request to dictionary
        results_dict = request.dict(exclude_none=True)
        
        # Calculate cheat score
        calculator = CheatScoreCalculator(use_xgboost=False)  # Use rule-based scoring by default
        result = calculator.compute_cheating_score(results_dict)
        
        return result
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005) 