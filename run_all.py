import asyncio
import json
import logging
import os
import time
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, List
import importlib.util
import sys
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoRequest(BaseModel):
    video_path: str

class DirectModuleRunner:
    def __init__(self):
        # Initialize module handlers
        self.modules = {}
        self.load_modules()
        
    def load_module(self, name, file_path):
        """Load a Python module from file path"""
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"Module file not found: {file_path}")
                return False
                
            # Import the module
            spec = importlib.util.spec_from_file_location(name, file_path)
            if spec is None:
                logger.error(f"Could not load spec for module {name} from {file_path}")
                return False
                
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            
            # Store the module
            self.modules[name] = module
            logger.info(f"Successfully loaded module {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading module {name}: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def load_modules(self):
        """Load all required modules"""
        module_files = {
            "screen_switch": "screen_switch.py",
            "gaze": "gaze_tracking.py",
            "audio": "audio_analysis.py",
            "lip_sync": "lip_sync_detector.py",
            "multi_person": "multi_person.py",
            "cheat_score": "cheat_score.py"
        }
        
        for name, file_path in module_files.items():
            self.load_module(name, file_path)
    
    async def run_screen_switch_analysis(self, video_path):
        """Run screen switch analysis"""
        try:
            if "screen_switch" not in self.modules:
                logger.error("Screen switch module not loaded")
                return None
                
            module = self.modules["screen_switch"]
            if hasattr(module, "ScreenSwitchDetector"):
                detector = module.ScreenSwitchDetector()
                result = detector.detect_screen_anomalies(video_path)
                return result
            else:
                logger.error("ScreenSwitchDetector class not found in module")
                return None
        except Exception as e:
            logger.error(f"Error in screen switch analysis: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    async def run_gaze_analysis(self, video_path):
        """Run gaze analysis"""
        try:
            if "gaze" not in self.modules:
                logger.error("Gaze module not loaded")
                return None
                
            module = self.modules["gaze"]
            if hasattr(module, "GazeTracker"):
                detector = module.GazeTracker()
                result = detector.detect_gaze_deviation(video_path)
                return result
            else:
                logger.error("GazeTracker class not found in module")
                return None
        except Exception as e:
            logger.error(f"Error in gaze analysis: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    async def run_audio_analysis(self, video_path):
        """Run audio analysis"""
        try:
            if "audio" not in self.modules:
                logger.error("Audio module not loaded")
                return None
                
            module = self.modules["audio"]
            if hasattr(module, "AudioAnalyzer"):
                analyzer = module.AudioAnalyzer()
                result = analyzer.analyze_audio(video_path)
                return result
            else:
                logger.error("AudioAnalyzer class not found in module")
                return None
        except Exception as e:
            logger.error(f"Error in audio analysis: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    async def run_lip_sync_analysis(self, video_path):
        """Run lip sync analysis"""
        try:
            if "lip_sync" not in self.modules:
                logger.error("Lip sync module not loaded")
                return None
                
            module = self.modules["lip_sync"]
            if hasattr(module, "LipSyncDetector"):
                detector = module.LipSyncDetector()
                result = detector.detect_lip_sync_issues(video_path)
                return result
            else:
                logger.error("LipSyncDetector class not found in module")
                return None
        except Exception as e:
            logger.error(f"Error in lip sync analysis: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    async def run_multi_person_analysis(self, video_path):
        """Run multi-person analysis"""
        try:
            if "multi_person" not in self.modules:
                logger.error("Multi-person module not loaded")
                return None
                
            module = self.modules["multi_person"]
            if hasattr(module, "MultiPersonDetector"):
                detector = module.MultiPersonDetector()
                result = detector.detect_multiple_persons(video_path)
                return result
            else:
                logger.error("MultiPersonDetector class not found in module")
                return None
        except Exception as e:
            logger.error(f"Error in multi-person analysis: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    async def compute_cheat_score(self, module_results):
        """Compute cheating score"""
        try:
            if "cheat_score" not in self.modules:
                logger.error("Cheat score module not loaded")
                return None
                
            module = self.modules["cheat_score"]
            if hasattr(module, "CheatScoreCalculator"):
                calculator = module.CheatScoreCalculator()
                result = calculator.compute_cheating_score(module_results)
                return result
            else:
                logger.error("CheatScoreCalculator class not found in module")
                return None
        except Exception as e:
            logger.error(f"Error computing cheat score: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    async def run_all_modules(self, video_path):
        """Run all detection modules in parallel"""
        # Create tasks for all modules
        tasks = [
            self.run_screen_switch_analysis(video_path),
            self.run_gaze_analysis(video_path),
            self.run_audio_analysis(video_path),
            self.run_lip_sync_analysis(video_path),
            self.run_multi_person_analysis(video_path)
        ]
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        module_results = {}
        module_names = ["screen_switch", "gaze", "audio", "lip_sync", "multi_person"]
        
        for name, result in zip(module_names, results):
            if isinstance(result, Exception):
                logger.error(f"Exception in {name} module: {str(result)}")
            elif result is not None:
                module_results[name] = result
                logger.info(f"{name} analysis completed successfully")
            else:
                logger.warning(f"No result from {name} module")
        
        return module_results
    
    async def run_analysis(self, video_path):
        """Run complete analysis pipeline on video"""
        try:
            # Check if video file exists
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Start timestamp
            start_time = time.time()
            
            # Run all detector modules
            module_results = await self.run_all_modules(video_path)
            
            # Check if we have any results
            if not module_results:
                logger.warning("All modules failed to produce results")
                module_results = {}  # Continue with empty results
            
            # Compute cheat score
            try:
                cheat_score = await self.compute_cheat_score(module_results)
            except Exception as e:
                logger.error(f"Error computing cheat score: {str(e)}")
                logger.error(traceback.format_exc())
                cheat_score = {
                    "final_score": 0.0,
                    "risk": "Unknown",
                    "reasons": ["Failed to compute cheat score"]
                }
            
            # End timestamp
            end_time = time.time()
            processing_time = round(end_time - start_time, 2)
            
            # Combine all results
            results = {
                "video_path": video_path,
                "processing_time_seconds": processing_time,
                "module_results": module_results,
                "cheat_score": cheat_score
            }
            
            return results
            
        except FileNotFoundError as e:
            logger.error(str(e))
            raise
        except Exception as e:
            logger.error(f"Error in analysis pipeline: {str(e)}")
            logger.error(traceback.format_exc())
            raise

# FastAPI implementation
app = FastAPI()
module_runner = DirectModuleRunner()

@app.post("/run_all")
async def run_all_endpoint(request: VideoRequest):
    """
    Master orchestrator endpoint that runs all detector modules on a video
    and aggregates the results into a final cheat score.
    """
    try:
        # Run the complete analysis pipeline
        results = await module_runner.run_analysis(request.video_path)
        return results
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run_all_background")
async def run_all_background_endpoint(request: VideoRequest, background_tasks: BackgroundTasks):
    """
    Run all detector modules in the background and return a job ID.
    Results can be fetched later using the job ID.
    """
    try:
        # Generate a unique job ID
        job_id = f"job_{int(time.time())}"
        
        # Start the analysis in the background
        background_tasks.add_task(module_runner.run_analysis, request.video_path)
        
        return {"job_id": job_id, "status": "processing", "message": "Analysis started in background"}
    except Exception as e:
        logger.error(f"Error starting background task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 