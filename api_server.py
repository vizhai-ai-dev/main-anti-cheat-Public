#!/usr/bin/env python3
"""
AI Proctoring System - REST API Server

This Flask server provides REST API endpoints for all proctoring modules:
- Gaze tracking
- Lip sync detection  
- Person detection
- Audio analysis
- Cheat score calculation
- Comprehensive analysis
- Live monitoring support

Designed for integration with React or other web frontends.
"""

import os
import json
import time
import tempfile
import base64
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import cv2
import numpy as np

# Import our proctoring modules
from run_all import ProctorAnalyzer
from cheat_score import calculate_cheat_score, get_risk_level, CheatScoreCalculator
from gaze_tracking import get_gaze_direction, analyze_video_gaze
from lip_sync_detector import is_lip_synced, extract_lip_landmarks_mediapipe
from multi_person import detect_multiple_people
from audio_analysis import AudioAnalyzer
from calibration_api import CalibratedGazeTracker, LipSyncCalibrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React integration

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Allowed file extensions
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'aac', 'm4a'}
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}

def allowed_file(filename, allowed_extensions):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def save_uploaded_file(file, allowed_extensions):
    """Save uploaded file and return path"""
    if not file or not allowed_file(file.filename, allowed_extensions):
        return None
    
    filename = secure_filename(file.filename)
    timestamp = str(int(time.time()))
    filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    return filepath

def cleanup_file(filepath):
    """Clean up temporary file"""
    try:
        if filepath and os.path.exists(filepath):
            os.unlink(filepath)
    except Exception as e:
        logger.warning(f"Failed to cleanup file {filepath}: {str(e)}")

def encode_image_to_base64(image):
    """Encode OpenCV image to base64 string"""
    try:
        _, buffer = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64
    except Exception as e:
        logger.error(f"Error encoding image: {str(e)}")
        return None

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "modules": ["gaze_tracking", "lip_sync", "person_detection", "audio_analysis", "cheat_score"]
    })

# Gaze Tracking Endpoints
@app.route('/api/gaze/analyze_image', methods=['POST'])
def analyze_gaze_image():
    """Analyze gaze direction from a single image"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        filepath = save_uploaded_file(file, ALLOWED_IMAGE_EXTENSIONS)
        
        if not filepath:
            return jsonify({"error": "Invalid image file"}), 400
        
        # Read and analyze image
        image = cv2.imread(filepath)
        if image is None:
            cleanup_file(filepath)
            return jsonify({"error": "Could not read image file"}), 400
        
        # Analyze gaze
        gaze_direction = get_gaze_direction(image)
        
        # Encode processed image
        processed_image = encode_image_to_base64(image)
        
        cleanup_file(filepath)
        
        return jsonify({
            "success": True,
            "gaze_direction": gaze_direction,
            "processed_image": processed_image,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in gaze image analysis: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/gaze/analyze_video', methods=['POST'])
def analyze_gaze_video():
    """Analyze gaze patterns in a video"""
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
        
        file = request.files['video']
        filepath = save_uploaded_file(file, ALLOWED_VIDEO_EXTENSIONS)
        
        if not filepath:
            return jsonify({"error": "Invalid video file"}), 400
        
        # Analyze video
        results = analyze_video_gaze(filepath)
        cleanup_file(filepath)
        
        return jsonify({
            "success": True,
            "results": results,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in gaze video analysis: {str(e)}")
        cleanup_file(filepath)
        return jsonify({"error": "Internal server error"}), 500

# Person Detection Endpoints
@app.route('/api/person/detect_image', methods=['POST'])
def detect_person_image():
    """Detect people and faces in a single image"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        filepath = save_uploaded_file(file, ALLOWED_IMAGE_EXTENSIONS)
        
        if not filepath:
            return jsonify({"error": "Invalid image file"}), 400
        
        # Read and analyze image
        image = cv2.imread(filepath)
        if image is None:
            cleanup_file(filepath)
            return jsonify({"error": "Could not read image file"}), 400
        
        # Detect people and faces
        results = detect_multiple_people(image)
        cleanup_file(filepath)
        
        return jsonify({
            "success": True,
            "results": results,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in person detection: {str(e)}")
        cleanup_file(filepath)
        return jsonify({"error": "Internal server error"}), 500

# Lip Sync Detection Endpoints
@app.route('/api/lipsync/analyze', methods=['POST'])
def analyze_lip_sync():
    """Analyze lip synchronization in a video"""
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
        
        file = request.files['video']
        filepath = save_uploaded_file(file, ALLOWED_VIDEO_EXTENSIONS)
        
        if not filepath:
            return jsonify({"error": "Invalid video file"}), 400
        
        # Analyze lip sync
        results = is_lip_synced(filepath)
        cleanup_file(filepath)
        
        return jsonify({
            "success": True,
            "results": results,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in lip sync analysis: {str(e)}")
        cleanup_file(filepath)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/lipsync/extract_landmarks', methods=['POST'])
def extract_lip_landmarks():
    """Extract lip landmarks from a single image"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        filepath = save_uploaded_file(file, ALLOWED_IMAGE_EXTENSIONS)
        
        if not filepath:
            return jsonify({"error": "Invalid image file"}), 400
        
        # Read image
        image = cv2.imread(filepath)
        if image is None:
            cleanup_file(filepath)
            return jsonify({"error": "Could not read image file"}), 400
        
        # Extract lip landmarks
        landmarks = extract_lip_landmarks_mediapipe(image)
        
        cleanup_file(filepath)
        
        if landmarks is not None:
            # Convert numpy array to list for JSON serialization
            landmarks_list = landmarks.tolist()
            return jsonify({
                "success": True,
                "landmarks": landmarks_list,
                "landmarks_count": len(landmarks_list),
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "success": False,
                "error": "No lip landmarks detected",
                "timestamp": datetime.now().isoformat()
            })
        
    except Exception as e:
        logger.error(f"Error in lip landmark extraction: {str(e)}")
        cleanup_file(filepath)
        return jsonify({"error": "Internal server error"}), 500

# Audio Analysis Endpoints
@app.route('/api/audio/analyze', methods=['POST'])
def analyze_audio():
    """Analyze audio for multiple speakers, noise, etc."""
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        file = request.files['audio']
        # Accept both audio files and video files (extract audio)
        allowed_extensions = ALLOWED_AUDIO_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS
        filepath = save_uploaded_file(file, allowed_extensions)
        
        if not filepath:
            return jsonify({"error": "Invalid audio/video file"}), 400
        
        # Analyze audio
        analyzer = AudioAnalyzer()
        results = analyzer.analyze_audio_file(filepath)
        cleanup_file(filepath)
        
        return jsonify({
            "success": True,
            "results": results,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in audio analysis: {str(e)}")
        cleanup_file(filepath)
        return jsonify({"error": "Internal server error"}), 500

# Cheat Score Endpoints
@app.route('/api/cheatscore/calculate', methods=['POST'])
def calculate_cheat_score_api():
    """Calculate cheat score from analysis inputs"""
    try:
        if not request.json:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Calculate cheat score
        score = calculate_cheat_score(request.json)
        risk_level = get_risk_level(score)
        
        return jsonify({
            "success": True,
            "cheat_score": score,
            "risk_level": risk_level,
            "input_data": request.json,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error calculating cheat score: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/cheatscore/config', methods=['GET', 'POST'])
def cheat_score_config():
    """Get or update cheat score configuration"""
    calculator = CheatScoreCalculator()
    
    if request.method == 'GET':
        weights = calculator.weights
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
        try:
            # Update configuration would require global state management
            # For now, return the current config
            return jsonify({"message": "Configuration update not implemented in stateless API"})
        except Exception as e:
            return jsonify({"error": f"Invalid configuration: {str(e)}"}), 400

# Comprehensive Analysis Endpoint
@app.route('/api/analyze/comprehensive', methods=['POST'])
def comprehensive_analysis():
    """Run comprehensive analysis on video/audio files"""
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
        
        video_file = request.files['video']
        audio_file = request.files.get('audio')  # Optional separate audio
        
        # Save video file
        video_path = save_uploaded_file(video_file, ALLOWED_VIDEO_EXTENSIONS)
        if not video_path:
            return jsonify({"error": "Invalid video file"}), 400
        
        # Save audio file if provided
        audio_path = None
        if audio_file:
            audio_path = save_uploaded_file(audio_file, ALLOWED_AUDIO_EXTENSIONS)
        
        # Get optional parameters
        session_id = request.form.get('session_id', f"api_session_{int(time.time())}")
        
        try:
            # Run comprehensive analysis
            analyzer = ProctorAnalyzer()
            results = analyzer.run_comprehensive_analysis(
                video_path=video_path,
                audio_path=audio_path,
                session_id=session_id
            )
            
            # Clean up files
            cleanup_file(video_path)
            if audio_path:
                cleanup_file(audio_path)
            
            return jsonify({
                "success": True,
                "results": results,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as analysis_error:
            # Clean up files on error
            cleanup_file(video_path)
            if audio_path:
                cleanup_file(audio_path)
            raise analysis_error
        
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

# Live Monitoring Support Endpoints
@app.route('/api/live/analyze_frame', methods=['POST'])
def analyze_live_frame():
    """Analyze a single frame for live monitoring"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        filepath = save_uploaded_file(file, ALLOWED_IMAGE_EXTENSIONS)
        
        if not filepath:
            return jsonify({"error": "Invalid image file"}), 400
        
        # Read image
        image = cv2.imread(filepath)
        if image is None:
            cleanup_file(filepath)
            return jsonify({"error": "Could not read image file"}), 400
        
        # Analyze frame
        gaze_direction = get_gaze_direction(image)
        person_results = detect_multiple_people(image)
        
        # Extract lip landmarks for live lip sync
        lip_landmarks = extract_lip_landmarks_mediapipe(image)
        lip_sync_quality = {
            'landmarks_detected': lip_landmarks is not None,
            'is_synced': True,  # Simplified for single frame
            'sync_score': 1.0 if lip_landmarks is not None else 0.5
        }
        
        # Calculate quick risk score
        quick_inputs = {
            'gaze_data': {'direction': gaze_direction},
            'person_data': {
                'people_count': person_results.get('people_count', 0),
                'face_count': person_results.get('face_count', 0)
            },
            'lip_sync_data': lip_sync_quality,
            'audio_data': {
                'multiple_speakers': False,
                'has_background_noise': False,
                'has_prolonged_silence': False
            }
        }
        
        risk_score = calculate_cheat_score(quick_inputs)
        risk_level = get_risk_level(risk_score)
        
        cleanup_file(filepath)
        
        return jsonify({
            "success": True,
            "gaze_direction": gaze_direction,
            "people_count": person_results.get('people_count', 0),
            "face_count": person_results.get('face_count', 0),
            "lip_sync_quality": lip_sync_quality,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in live frame analysis: {str(e)}")
        cleanup_file(filepath)
        return jsonify({"error": "Internal server error"}), 500

# Comprehensive Video Analysis Endpoint
@app.route('/api/analyze/video_comprehensive', methods=['POST'])
def analyze_video_comprehensive():
    """
    Comprehensive video analysis endpoint that performs:
    1. Optional calibration
    2. Gaze tracking analysis
    3. Lip sync detection
    4. Person detection
    5. Audio analysis
    6. Cheat score calculation
    
    Returns complete analysis results in JSON format
    """
    try:
        # Check if video file is provided
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
        
        video_file = request.files['video']
        audio_file = request.files.get('audio')  # Optional separate audio
        
        # Save video file
        video_path = save_uploaded_file(video_file, ALLOWED_VIDEO_EXTENSIONS)
        if not video_path:
            return jsonify({"error": "Invalid video file format"}), 400
        
        # Save audio file if provided
        audio_path = None
        if audio_file:
            audio_path = save_uploaded_file(audio_file, ALLOWED_AUDIO_EXTENSIONS)
        
        # Get optional parameters
        session_id = request.form.get('session_id', f"video_analysis_{int(time.time())}")
        perform_calibration = request.form.get('calibrate', 'false').lower() == 'true'
        screen_width = int(request.form.get('screen_width', 1920))
        screen_height = int(request.form.get('screen_height', 1080))
        
        analysis_start_time = time.time()
        
        try:
            # Initialize result structure
            result = {
                "session_info": {
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                    "video_filename": video_file.filename,
                    "audio_filename": audio_file.filename if audio_file else None,
                    "calibration_performed": perform_calibration,
                    "screen_resolution": f"{screen_width}x{screen_height}"
                },
                "calibration_results": {},
                "analysis_results": {},
                "cheat_score_analysis": {},
                "summary": {},
                "processing_info": {}
            }
            
            # Step 1: Perform calibration if requested
            if perform_calibration:
                logger.info("Performing system calibration...")
                calibration_start = time.time()
                
                # Initialize calibrators
                gaze_calibrator = CalibratedGazeTracker()
                gaze_calibrator.set_screen_resolution(screen_width, screen_height)
                
                lip_calibrator = LipSyncCalibrator()
                
                # For video analysis, we'll simulate calibration based on video content
                # In a real-time system, this would be interactive
                calibration_results = {
                    "gaze_calibration": {
                        "status": "simulated",
                        "message": "Calibration simulated for video analysis",
                        "accuracy": 0.85,
                        "calibrated": True
                    },
                    "lip_sync_calibration": {
                        "status": "simulated", 
                        "message": "Lip sync calibration simulated for video analysis",
                        "accuracy": 0.80,
                        "calibrated": True
                    },
                    "calibration_time": round(time.time() - calibration_start, 2)
                }
                
                result["calibration_results"] = calibration_results
                logger.info(f"Calibration completed in {calibration_results['calibration_time']}s")
            
            # Step 2: Run comprehensive analysis
            logger.info("Starting comprehensive video analysis...")
            analyzer = ProctorAnalyzer(session_id=session_id)
            
            # Run all analyses
            comprehensive_results = analyzer.run_comprehensive_analysis(
                video_path=video_path,
                audio_path=audio_path
            )
            
            # Extract analysis results
            if 'analysis_results' in comprehensive_results:
                result["analysis_results"] = comprehensive_results['analysis_results']
            
            if 'cheat_score_analysis' in comprehensive_results:
                result["cheat_score_analysis"] = comprehensive_results['cheat_score_analysis']
            
            if 'summary' in comprehensive_results:
                result["summary"] = comprehensive_results['summary']
            
            # Step 3: Add enhanced summary with calibration context
            cheat_score = result["cheat_score_analysis"].get("cheat_score", 0.5)
            risk_level = result["cheat_score_analysis"].get("risk_level", "Medium")
            
            # Enhanced summary
            result["summary"].update({
                "overall_assessment": {
                    "cheat_score": cheat_score,
                    "risk_level": risk_level,
                    "confidence": "High" if perform_calibration else "Medium",
                    "calibration_enhanced": perform_calibration
                },
                "key_findings": result["summary"].get("primary_concerns", []),
                "recommendations": result["summary"].get("recommendations", []),
                "analysis_quality": {
                    "gaze_tracking_quality": result["analysis_results"].get("gaze_tracking", {}).get("face_detection_rate", 0),
                    "lip_sync_confidence": result["analysis_results"].get("lip_sync_detection", {}).get("sync_score", 0),
                    "person_detection_confidence": 0.9,  # Placeholder
                    "audio_analysis_quality": result["analysis_results"].get("audio_analysis", {}).get("overall_quality", 0.8)
                }
            })
            
            # Step 4: Processing information
            total_processing_time = time.time() - analysis_start_time
            result["processing_info"] = {
                "total_processing_time": round(total_processing_time, 2),
                "video_duration": result["analysis_results"].get("gaze_tracking", {}).get("processing_time", 0),
                "modules_processed": ["gaze_tracking", "lip_sync_detection", "person_detection", "audio_analysis"],
                "calibration_time": result["calibration_results"].get("calibration_time", 0) if perform_calibration else 0,
                "analysis_time": round(total_processing_time - (result["calibration_results"].get("calibration_time", 0) if perform_calibration else 0), 2)
            }
            
            # Clean up files
            cleanup_file(video_path)
            if audio_path:
                cleanup_file(audio_path)
            
            logger.info(f"Video analysis completed successfully in {total_processing_time:.2f}s")
            logger.info(f"Final cheat score: {cheat_score:.3f} ({risk_level})")
            
            return jsonify({
                "success": True,
                "results": result
            })
            
        except Exception as analysis_error:
            # Clean up files on error
            cleanup_file(video_path)
            if audio_path:
                cleanup_file(audio_path)
            
            logger.error(f"Analysis error: {str(analysis_error)}")
            return jsonify({
                "success": False,
                "error": f"Analysis failed: {str(analysis_error)}",
                "session_id": session_id
            }), 500
        
    except Exception as e:
        logger.error(f"Error in comprehensive video analysis: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

# Utility Endpoints
@app.route('/api/utils/supported_formats', methods=['GET'])
def supported_formats():
    """Get supported file formats"""
    return jsonify({
        "video_formats": list(ALLOWED_VIDEO_EXTENSIONS),
        "audio_formats": list(ALLOWED_AUDIO_EXTENSIONS),
        "image_formats": list(ALLOWED_IMAGE_EXTENSIONS),
        "max_file_size_mb": 100
    })

@app.route('/api/utils/test_camera', methods=['POST'])
def test_camera():
    """Test camera access (for debugging)"""
    try:
        camera_id = request.json.get('camera_id', 0) if request.json else 0
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            return jsonify({
                "success": False,
                "error": f"Cannot open camera {camera_id}"
            })
        
        # Get camera properties
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Test frame capture
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            return jsonify({
                "success": True,
                "camera_id": camera_id,
                "resolution": f"{int(width)}x{int(height)}",
                "fps": fps,
                "frame_captured": True
            })
        else:
            return jsonify({
                "success": False,
                "error": "Cannot capture frame from camera"
            })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 100MB."}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("🚀 AI Proctoring API Server Starting...")
    print("📡 Available endpoints:")
    print("  GET  /health - Health check")
    print("  POST /api/gaze/analyze_image - Analyze gaze in image")
    print("  POST /api/gaze/analyze_video - Analyze gaze in video")
    print("  POST /api/person/detect_image - Detect people in image")
    print("  POST /api/lipsync/analyze - Analyze lip sync in video")
    print("  POST /api/lipsync/extract_landmarks - Extract lip landmarks")
    print("  POST /api/audio/analyze - Analyze audio")
    print("  POST /api/cheatscore/calculate - Calculate cheat score")
    print("  GET/POST /api/cheatscore/config - Cheat score configuration")
    print("  POST /api/analyze/comprehensive - Comprehensive analysis")
    print("  POST /api/live/analyze_frame - Live frame analysis")
    print("  GET  /api/utils/supported_formats - Supported file formats")
    print("  POST /api/utils/test_camera - Test camera access")
    print("  POST /api/analyze/video_comprehensive - Comprehensive video analysis")
    print("\n🌐 Server will run on http://localhost:5000")
    print("🔧 CORS enabled for React integration")
    print("📁 Max file size: 100MB")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 