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
from random import choice
import traceback

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
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],  # Allow React dev server
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
    try:
        analysis_id = str(uuid.uuid4())
        
        # Select one of three demo scenarios
        scenario = choice(["good", "medium", "bad"])
        logger.info(f"Creating demo analysis with ID: {analysis_id}, scenario: {scenario}")
        
        # Create mock results based on the scenario
        if scenario == "good":
            # Good integrity scenario
            mock_result = {
                "final_score": 92.3,
                "risk": "Low",
                "reasons": [
                    "Occasional gaze shifts detected (minimal)",
                    "Excellent lip sync correlation"
                ],
                "gaze": {
                    "off_screen_count": 2,
                    "average_confidence": 0.95,
                    "off_screen_time_percentage": 3.2,
                    "gaze_direction_timeline": [
                        {"timestamp": "00:00:15", "direction": "center"},
                        {"timestamp": "00:00:45", "direction": "right"},
                        {"timestamp": "00:00:48", "direction": "center"},
                        {"timestamp": "00:02:10", "direction": "left"},
                        {"timestamp": "00:02:12", "direction": "center"}
                    ],
                    "score": 96.8
                },
                "audio": {
                    "multiple_speakers": False,
                    "keyboard_typing_count": 0,
                    "silence_percentage": 5.3,
                    "background_noise_level": "Low",
                    "speaking_timeline": [
                        {"start": "00:00:05", "end": "00:00:25", "speaker": "primary"},
                        {"start": "00:00:28", "end": "00:00:55", "speaker": "primary"},
                        {"start": "00:01:02", "end": "00:01:38", "speaker": "primary"},
                        {"start": "00:01:45", "end": "00:02:15", "speaker": "primary"}
                    ],
                    "score": 98.7
                },
                "multi_person": {
                    "max_people_detected": 1,
                    "time_with_multiple_people": 0,
                    "people_detection_timeline": [
                        {"timestamp": "00:00:00", "count": 1},
                        {"timestamp": "00:00:30", "count": 1},
                        {"timestamp": "00:01:00", "count": 1},
                        {"timestamp": "00:01:30", "count": 1},
                        {"timestamp": "00:02:00", "count": 1}
                    ],
                    "different_faces_detected": 0,
                    "different_face_timestamps": [],
                    "has_different_faces": False,
                    "score": 100.0
                },
                "lip_sync": {
                    "lip_sync_score": 95.2,
                    "major_lip_desync_detected": False,
                    "lip_sync_timeline": [
                        {"timestamp": "00:00:10", "score": 96.5},
                        {"timestamp": "00:00:40", "score": 95.8},
                        {"timestamp": "00:01:10", "score": 94.9},
                        {"timestamp": "00:01:40", "score": 97.2},
                        {"timestamp": "00:02:10", "score": 91.6}
                    ],
                    "score": 95.2
                },
                "module_scores": {
                    "gaze": 96.8,
                    "audio": 98.7,
                    "multi_person": 100.0,
                    "lip_sync": 95.2
                },
                "processing_time": 7.8,
                "video_duration": "02:35"
            }
        elif scenario == "medium":
            # Medium risk scenario
            mock_result = {
                "final_score": 65.7,
                "risk": "Medium",
                "reasons": [
                    "Looked away from screen 12 times",
                    "Occasional keyboard typing detected"
                ],
                "gaze": {
                    "off_screen_count": 12,
                    "average_confidence": 0.85,
                    "off_screen_time_percentage": 27.3,
                    "gaze_direction_timeline": [
                        {"timestamp": "00:00:15", "direction": "down"},
                        {"timestamp": "00:00:35", "direction": "center"},
                        {"timestamp": "00:00:55", "direction": "right"},
                        {"timestamp": "00:01:05", "direction": "center"},
                        {"timestamp": "00:01:25", "direction": "left"},
                        {"timestamp": "00:01:35", "direction": "center"},
                        {"timestamp": "00:01:55", "direction": "down"},
                        {"timestamp": "00:02:05", "direction": "center"}
                    ],
                    "score": 70.0
                },
                "audio": {
                    "multiple_speakers": False,
                    "keyboard_typing_count": 5,
                    "silence_percentage": 15.8,
                    "background_noise_level": "Medium",
                    "speaking_timeline": [
                        {"start": "00:00:05", "end": "00:00:22", "speaker": "primary"},
                        {"start": "00:00:28", "end": "00:00:50", "speaker": "primary"},
                        {"start": "00:01:15", "end": "00:01:38", "speaker": "primary"},
                        {"start": "00:01:55", "end": "00:02:10", "speaker": "primary"}
                    ],
                    "score": 60.2
                },
                "multi_person": {
                    "max_people_detected": 1,
                    "time_with_multiple_people": 0,
                    "people_detection_timeline": [
                        {"timestamp": "00:00:00", "count": 1},
                        {"timestamp": "00:00:30", "count": 1},
                        {"timestamp": "00:01:00", "count": 1},
                        {"timestamp": "00:01:30", "count": 1},
                        {"timestamp": "00:02:00", "count": 1}
                    ],
                    "different_faces_detected": 0,
                    "different_face_timestamps": [],
                    "has_different_faces": False,
                    "score": 95.0
                },
                "lip_sync": {
                    "lip_sync_score": 82.5,
                    "major_lip_desync_detected": False,
                    "lip_sync_timeline": [
                        {"timestamp": "00:00:10", "score": 85.3},
                        {"timestamp": "00:00:40", "score": 79.8},
                        {"timestamp": "00:01:10", "score": 84.2},
                        {"timestamp": "00:01:40", "score": 81.5},
                        {"timestamp": "00:02:10", "score": 81.7}
                    ],
                    "score": 82.5
                },
                "module_scores": {
                    "gaze": 70.0,
                    "audio": 60.2,
                    "multi_person": 95.0,
                    "lip_sync": 82.5
                },
                "processing_time": 8.2,
                "video_duration": "02:23"
            }
        else:
            # High risk (bad) scenario
            mock_result = {
                "final_score": 32.1,
                "risk": "Very High",
                "reasons": [
                    "Multiple people detected in frame",
                    "Multiple speakers detected",
                    "Significant gaze shifting away from screen",
                    "Lip sync issues detected"
                ],
                "gaze": {
                    "off_screen_count": 28,
                    "average_confidence": 0.72,
                    "off_screen_time_percentage": 63.7,
                    "gaze_direction_timeline": [
                        {"timestamp": "00:00:05", "direction": "down"},
                        {"timestamp": "00:00:15", "direction": "center"},
                        {"timestamp": "00:00:25", "direction": "right"},
                        {"timestamp": "00:00:35", "direction": "down"},
                        {"timestamp": "00:00:45", "direction": "right"},
                        {"timestamp": "00:00:55", "direction": "center"},
                        {"timestamp": "00:01:05", "direction": "left"},
                        {"timestamp": "00:01:15", "direction": "down"}
                    ],
                    "score": 32.2
                },
                "audio": {
                    "multiple_speakers": True,
                    "keyboard_typing_count": 12,
                    "silence_percentage": 18.5,
                    "background_noise_level": "High",
                    "speaking_timeline": [
                        {"start": "00:00:05", "end": "00:00:18", "speaker": "primary"},
                        {"start": "00:00:21", "end": "00:00:28", "speaker": "secondary"},
                        {"start": "00:00:32", "end": "00:00:45", "speaker": "primary"},
                        {"start": "00:00:48", "end": "00:00:55", "speaker": "secondary"},
                        {"start": "00:01:15", "end": "00:01:25", "speaker": "primary"},
                        {"start": "00:01:28", "end": "00:01:32", "speaker": "secondary"}
                    ],
                    "score": 28.4
                },
                "multi_person": {
                    "max_people_detected": 2,
                    "time_with_multiple_people": 45.2,
                    "people_detection_timeline": [
                        {"timestamp": "00:00:00", "count": 1},
                        {"timestamp": "00:00:15", "count": 2},
                        {"timestamp": "00:00:45", "count": 1},
                        {"timestamp": "00:01:10", "count": 2},
                        {"timestamp": "00:01:40", "count": 2},
                        {"timestamp": "00:02:00", "count": 1}
                    ],
                    "different_faces_detected": 2,
                    "different_face_timestamps": ["00:00:15", "00:01:10"],
                    "has_different_faces": True,
                    "score": 25.4
                },
                "lip_sync": {
                    "lip_sync_score": 62.8,
                    "major_lip_desync_detected": True,
                    "lip_sync_timeline": [
                        {"timestamp": "00:00:10", "score": 75.3},
                        {"timestamp": "00:00:40", "score": 45.8},
                        {"timestamp": "00:01:10", "score": 64.2},
                        {"timestamp": "00:01:40", "score": 58.2},
                        {"timestamp": "00:02:10", "score": 70.5}
                    ],
                    "score": 62.8
                },
                "module_scores": {
                    "gaze": 32.2,
                    "audio": 28.4,
                    "multi_person": 25.4,
                    "lip_sync": 62.8
                },
                "processing_time": 9.5,
                "video_duration": "02:15"
            }
        
        # Store the mock analysis result
        analysis_results[analysis_id] = {
            "id": analysis_id,
            "status": "completed",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "result": mock_result
        }
        
        logger.info(f"Demo analysis created successfully with ID: {analysis_id}")
        return {"id": analysis_id}
    except Exception as e:
        logger.error(f"Error creating demo analysis: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error creating demo analysis: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting VIZH.AI Backend Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000) 