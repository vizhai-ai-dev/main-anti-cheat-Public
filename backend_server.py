import os
import shutil
import time
import uuid
from typing import Dict, List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import asyncio
import logging

# Import our analysis modules
from run_all import DirectModuleRunner

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI instance
app = FastAPI(title="VIZH.AI Backend", version="1.0.0")

# Add CORS middleware to allow requests from our frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a temporary directory for uploaded videos
UPLOAD_DIR = "uploaded_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Dictionary to store analysis results
analysis_results = {}

# Create a DirectModuleRunner instance
runner = DirectModuleRunner()

class AnalysisResponse(BaseModel):
    id: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    result: Optional[Dict] = None
    error: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Welcome to VIZH.AI API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/upload")
async def upload_video(video: UploadFile = File(...)):
    try:
        # Create a unique ID for this analysis
        analysis_id = str(uuid.uuid4())
        file_extension = os.path.splitext(video.filename)[1]
        
        # Create a path for the uploaded file
        file_path = os.path.join(UPLOAD_DIR, f"{analysis_id}{file_extension}")
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # Store initial analysis status
        analysis_results[analysis_id] = {
            "id": analysis_id,
            "status": "processing",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "file_path": file_path
        }
        
        # Start analysis in the background
        asyncio.create_task(process_video(analysis_id, file_path))
        
        return {"id": analysis_id}
    
    except Exception as e:
        logger.error(f"Error uploading video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_video(analysis_id: str, file_path: str):
    try:
        # Run the analysis
        logger.info(f"Starting analysis for {analysis_id}")
        
        # Get all module results
        module_results = await runner.run_all_modules(file_path)
        
        # Calculate cheat score
        cheat_score = await runner.compute_cheat_score(module_results)
        
        # Combine all results
        result = {
            **module_results,
            "final_score": cheat_score.get("final_score", 0),
            "risk": cheat_score.get("risk", "Unknown"),
            "reasons": cheat_score.get("reasons", []),
            "module_scores": {
                "screen_switch": module_results.get("screen_switch", {}).get("score", 0),
                "gaze": module_results.get("gaze", {}).get("score", 0),
                "audio": module_results.get("audio", {}).get("score", 0),
                "multi_person": module_results.get("multi_person", {}).get("score", 0),
                "lip_sync": module_results.get("lip_sync", {}).get("score", 0)
            },
            "processing_time": 10.5  # Mock processing time
        }
        
        # Update the analysis result
        analysis_results[analysis_id].update({
            "status": "completed",
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "result": result
        })
        
        logger.info(f"Analysis completed for {analysis_id}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        analysis_results[analysis_id].update({
            "status": "failed",
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error": str(e)
        })

@app.get("/analysis/{analysis_id}")
async def get_analysis_status(analysis_id: str):
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    result = analysis_results[analysis_id].copy()
    
    # Remove the file path from the response
    if "file_path" in result:
        del result["file_path"]
    
    return result

@app.get("/analysis/{analysis_id}/report")
async def get_analysis_report(analysis_id: str):
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    result = analysis_results[analysis_id]
    
    if result["status"] != "completed":
        raise HTTPException(status_code=422, detail="Analysis not completed")
    
    if not result.get("result"):
        raise HTTPException(status_code=500, detail="Analysis result not available")
    
    return result["result"]

# Mock endpoint for demo purposes - creates a sample analysis result
@app.post("/demo-analysis")
async def create_demo_analysis():
    analysis_id = str(uuid.uuid4())
    
    # Create a mock result for demonstration
    mock_result = {
        "final_score": 65.7,
        "risk": "Medium",
        "reasons": [
            "Looked away from screen 12 times",
            "Switched screens 3 times",
            "Multiple speakers detected"
        ],
        "screen_switch": {
            "fullscreen_violations": 2,
            "switch_count": 3,
            "score": 75.5
        },
        "gaze": {
            "off_screen_count": 12,
            "average_confidence": 0.85,
            "score": 70.0
        },
        "audio": {
            "multiple_speakers": True,
            "keyboard_typing_count": 5,
            "silence_percentage": 15,
            "score": 60.2
        },
        "multi_person": {
            "max_people_detected": 1,
            "time_with_multiple_people": 0,
            "score": 95.0
        },
        "lip_sync": {
            "lip_sync_score": 82.5,
            "major_lip_desync_detected": False,
            "score": 82.5
        },
        "module_scores": {
            "screen_switch": 75.5,
            "gaze": 70.0,
            "audio": 60.2,
            "multi_person": 95.0,
            "lip_sync": 82.5
        },
        "processing_time": 8.2
    }
    
    # Store the mock analysis result
    analysis_results[analysis_id] = {
        "id": analysis_id,
        "status": "completed",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "result": mock_result
    }
    
    return {"id": analysis_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 