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
        
        # Risk thresholds - Made more granular and strict
        self.RISK_THRESHOLDS = {
            "Very Low": 15,    # New threshold
            "Low": 35,         # Increased from 20
            "Medium": 55,      # Increased from 40
            "High": 75,        # Increased from 60
            "Very High": 85    # New threshold
        }
        
        # Feature weights - Adjusted for better accuracy
        self.WEIGHTS = {
            "gaze": 0.35,      # Decreased from 0.40
            "audio": 0.25,     # Same
            "multi_person": 0.25,  # Increased from 0.20
            "lip_sync": 0.15   # Same
        }
    
    def _normalize_score(self, value, min_val=0, max_val=100):
        """Normalize values to 0-100 scale with improved exponential penalty"""
        normalized = max(min(100, float(value)), 0)
        # Apply improved exponential penalty for lower scores
        if normalized < 50:
            return normalized * (normalized / 50) * (normalized / 50)  # Cubic penalty
        return normalized
    
    def _get_risk_level(self, score):
        """Determine risk level from score with more granular levels"""
        if score < self.RISK_THRESHOLDS["Low"]:
            return "Very Low"
        elif score < self.RISK_THRESHOLDS["Medium"]:
            return "Low"
        elif score < self.RISK_THRESHOLDS["High"]:
            return "Medium"
        elif score < 80:
            return "High"
        else:
            return "Very High"
    
    def _generate_reasons(self, results):
        """Generate detailed reasons for flagging with improved severity levels"""
        reasons = []
        
        # Gaze reasons with improved thresholds
        if "gaze" in results:
            gaze = results["gaze"]
            off_screen_count = gaze.get("off_screen_count", 0)
            if off_screen_count > 3:  # More strict threshold
                severity = "critical" if off_screen_count > 10 else "warning"
                reasons.append({
                    "text": f"Looked away from screen {off_screen_count} times",
                    "severity": severity,
                    "module": "gaze",
                    "confidence": 0.95
                })
            if gaze.get("average_confidence", 0) < 0.8:  # More strict threshold
                reasons.append({
                    "text": "Inconsistent gaze tracking detected",
                    "severity": "warning",
                    "module": "gaze",
                    "confidence": 0.85
                })
            if gaze.get("off_screen_time_percentage", 0) > 15:  # More strict threshold
                reasons.append({
                    "text": f"Spent {gaze.get('off_screen_time_percentage'):.1f}% of time looking away",
                    "severity": "critical",
                    "module": "gaze",
                    "confidence": 0.9
                })
        
        # Audio reasons with improved thresholds
        if "audio" in results:
            audio = results["audio"]
            if audio.get("multiple_speakers", False):
                reasons.append({
                    "text": "Multiple speakers detected",
                    "severity": "critical",
                    "module": "audio",
                    "confidence": 0.95
                })
            keyboard_count = audio.get("keyboard_typing_count", 0)
            if keyboard_count > 3:  # More strict threshold
                severity = "critical" if keyboard_count > 8 else "warning"
                reasons.append({
                    "text": f"Excessive keyboard typing detected ({keyboard_count} instances)",
                    "severity": severity,
                    "module": "audio",
                    "confidence": 0.9
                })
            if audio.get("silence_percentage", 0) > 25:  # More strict threshold
                reasons.append({
                    "text": f"High percentage of silence ({audio.get('silence_percentage'):.1f}%)",
                    "severity": "warning",
                    "module": "audio",
                    "confidence": 0.85
                })
        
        # Multi-person reasons with improved thresholds
        if "multi_person" in results:
            multi = results["multi_person"]
            if multi.get("max_people_detected", 0) > 1:
                reasons.append({
                    "text": f"Multiple people detected (max: {multi.get('max_people_detected')})",
                    "severity": "critical",
                    "module": "multi_person",
                    "confidence": 0.95
                })
            if multi.get("time_with_multiple_people", 0) > 3:  # More strict threshold
                reasons.append({
                    "text": f"Multiple people present for {multi.get('time_with_multiple_people'):.1f} seconds",
                    "severity": "critical",
                    "module": "multi_person",
                    "confidence": 0.9
                })
            if multi.get("has_different_faces", False):
                reasons.append({
                    "text": f"{multi.get('different_faces_detected', 0)} different faces detected",
                    "severity": "critical",
                    "module": "multi_person",
                    "confidence": 0.95
                })
        
        # Lip sync reasons with improved thresholds
        if "lip_sync" in results:
            lip_sync = results["lip_sync"]
            if lip_sync.get("major_lip_desync_detected", False):
                reasons.append({
                    "text": "Major lip-sync desynchronization detected",
                    "severity": "critical",
                    "module": "lip_sync",
                    "confidence": 0.9
                })
            lip_score = lip_sync.get("lip_sync_score", 100)
            if lip_score < 85:  # More strict threshold
                severity = "critical" if lip_score < 70 else "warning"
                reasons.append({
                    "text": f"Poor lip-sync score: {lip_score:.1f}",
                    "severity": severity,
                    "module": "lip_sync",
                    "confidence": 0.85
                })
        
        return reasons
    
    def _rule_based_scoring(self, results):
        """Calculate cheating score using improved rule-based approach"""
        scores = {}
        
        # Calculate gaze score with exponential penalties
        if "gaze" in results:
            gaze_data = results["gaze"]
            off_screen_score = 100 - min(100, gaze_data.get("off_screen_count", 0) * 8)  # Increased penalty
            confidence_score = min(100, gaze_data.get("average_confidence", 0) * 100)
            time_score = 100 - min(100, gaze_data.get("off_screen_time_percentage", 0) * 2)
            scores["gaze"] = self._normalize_score(
                off_screen_score * 0.5 + 
                confidence_score * 0.3 + 
                time_score * 0.2
            )
        else:
            scores["gaze"] = 100
        
        # Calculate audio score with stricter penalties
        if "audio" in results:
            audio_data = results["audio"]
            speaker_score = 0 if audio_data.get("multiple_speakers", False) else 100
            typing_score = 100 - min(100, audio_data.get("keyboard_typing_count", 0) * 5)  # Increased penalty
            silence_score = 100 - min(100, audio_data.get("silence_percentage", 0) * 1.5)  # Increased penalty
            scores["audio"] = self._normalize_score(
                speaker_score * 0.5 + 
                typing_score * 0.3 + 
                silence_score * 0.2
            )
        else:
            scores["audio"] = 100
        
        # Calculate multi-person score with zero tolerance
        if "multi_person" in results:
            multi_data = results["multi_person"]
            people_score = 0 if multi_data.get("max_people_detected", 1) > 1 else 100
            time_score = 100 - min(100, multi_data.get("time_with_multiple_people", 0) * 10)  # Increased penalty
            face_score = 0 if multi_data.get("has_different_faces", False) else 100
            scores["multi_person"] = self._normalize_score(
                people_score * 0.4 + 
                time_score * 0.3 + 
                face_score * 0.3
            )
        else:
            scores["multi_person"] = 100
        
        # Calculate lip sync score with stricter penalties
        if "lip_sync" in results:
            lip_sync_data = results["lip_sync"]
            base_score = lip_sync_data.get("lip_sync_score", 100)
            if lip_sync_data.get("major_lip_desync_detected", False):
                base_score *= 0.5  # 50% penalty for major desync
            scores["lip_sync"] = self._normalize_score(base_score)
        else:
            scores["lip_sync"] = 100
        
        # Calculate final score with adjusted weights
        final_score = (
            scores["gaze"] * self.WEIGHTS["gaze"] +
            scores["audio"] * self.WEIGHTS["audio"] +
            scores["multi_person"] * self.WEIGHTS["multi_person"] +
            scores["lip_sync"] * self.WEIGHTS["lip_sync"]
        )
        
        # Apply exponential penalty to final score
        final_score = self._normalize_score(final_score)
        
        return {
            "final_score": round(final_score, 1),
            "risk": self._get_risk_level(final_score),
            "reasons": self._generate_reasons(results),
            "module_scores": scores
        }
    
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

    async def compute_score(self, results: Dict) -> Dict:
        """
        Compute the final cheat score from all module results.
        This is an async wrapper around _rule_based_scoring.
        """
        try:
            return self._rule_based_scoring(results)
        except Exception as e:
            logger.error(f"Error computing cheat score: {str(e)}")
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