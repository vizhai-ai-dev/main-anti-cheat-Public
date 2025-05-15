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

# Import actual analysis modules
from gaze_tracking import GazeTracker
from lip_sync_detector import LipSyncDetector
from multi_person import MultiPersonDetector
from audio_analysis import AudioAnalyzer
from cheat_score import CheatScoreCalculator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoRequest(BaseModel):
    video_path: str

"""
Simplified DirectModuleRunner for demo purposes
"""

class DirectModuleRunner:
    """
    A class that runs all analysis modules and computes the final cheat score.
    This is a simplified version for demo purposes.
    """
    
    def __init__(self):
        self.gaze_tracker = GazeTracker()
        self.lip_sync_detector = LipSyncDetector()
        self.multi_person_detector = MultiPersonDetector()
        self.audio_analyzer = AudioAnalyzer()
        self.cheat_score_calculator = CheatScoreCalculator()
    
    async def run_all_modules(self, video_path):
        """
        Run all analysis modules on the video
        """
        try:
            # Run all modules in parallel
            gaze_results = await self.gaze_tracker.analyze(video_path)
            lip_sync_results = await self.lip_sync_detector.analyze(video_path)
            multi_person_results = await self.multi_person_detector.analyze(video_path)
            audio_results = await self.audio_analyzer.analyze(video_path)
            
            return {
                "gaze": gaze_results,
                "lip_sync": lip_sync_results,
                "multi_person": multi_person_results,
                "audio": audio_results
            }
        except Exception as e:
            logger.error(f"Error running analysis modules: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    async def compute_cheat_score(self, module_results):
        """
        Compute the final cheat score based on all module results
        """
        try:
            return await self.cheat_score_calculator.compute_score(module_results)
        except Exception as e:
            logger.error(f"Error computing cheat score: {str(e)}")
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
        results = await module_runner.run_all_modules(request.video_path)
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
        background_tasks.add_task(module_runner.run_all_modules, request.video_path)
        
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