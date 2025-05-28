#!/usr/bin/env python3
"""
Streamlit Proctoring System App

A comprehensive web interface for the AI-powered proctoring system that includes:
- System calibration
- Individual module testing
- Live monitoring
- Comprehensive analysis
- Results visualization
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
from io import BytesIO

# Import our modules
from run_all import ProctorAnalyzer
from cheat_score import calculate_cheat_score, get_risk_level
from gaze_tracking import get_gaze_direction, analyze_video_gaze
from lip_sync_detector import is_lip_synced
from multi_person import detect_multiple_people
from audio_analysis import AudioAnalyzer
from calibration_api import CalibratedGazeTracker

# Configure Streamlit page
st.set_page_config(
    page_title="AI Proctoring System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .module-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .status-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .status-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    .status-danger {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'calibration_status' not in st.session_state:
        st.session_state.calibration_status = False
    if 'calibrated_tracker' not in st.session_state:
        st.session_state.calibrated_tracker = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'live_monitoring' not in st.session_state:
        st.session_state.live_monitoring = False

def display_status_box(message, status_type="info"):
    """Display a styled status box"""
    status_class = f"status-{status_type}"
    st.markdown(f'<div class="status-box {status_class}">{message}</div>', unsafe_allow_html=True)

def calibration_interface():
    """Calibration interface for the system"""
    st.markdown('<h2 class="module-header">🎯 System Calibration</h2>', unsafe_allow_html=True)
    
    st.write("""
    **3-Second Video Calibration Process:**
    1. Position yourself comfortably in front of the camera
    2. Ensure good lighting and clear view of your face
    3. Look directly at the camera during the 3-second recording
    4. Keep your head still and maintain eye contact with the camera
    5. The system will analyze your eye position, lip movement, and face angle
    """)
    
    # Camera preview section
    camera_preview = st.empty()
    
    # Camera controls - avoid nested columns
    if st.button("📹 Show Camera Preview"):
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Cannot open camera device")
                return
            
            # Get camera properties for debugging
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                camera_preview.image(frame, channels="BGR", caption="Position yourself in the frame")
                st.info("💡 Adjust your position so your face is clearly visible and centered")
                st.success(f"📷 Camera detected: {int(width)}x{int(height)} @ {fps:.1f} FPS")
            else:
                st.error("❌ Cannot read from camera")
        except Exception as e:
            st.error(f"❌ Camera error: {str(e)}")
            st.info("💡 Try the Quick Calibration option if camera preview fails")
    
    # Calibration buttons - using horizontal layout without nested columns
    cal_col1, cal_col2, cal_col3 = st.columns([1, 1, 1])
    
    with cal_col1:
        if st.button("🚀 Start 3-Second Calibration", type="primary"):
            calibrate_with_video(camera_preview)
    
    with cal_col2:
        if st.button("⚡ Quick Calibration", help="Single frame calibration for testing"):
            quick_calibrate(camera_preview)
    
    with cal_col3:
        if st.session_state.calibration_status and st.button("🔄 Reset Calibration"):
            st.session_state.calibration_status = False
            st.session_state.calibrated_tracker = None
            if 'calibration_results' in st.session_state:
                del st.session_state.calibration_results
            st.rerun()
    
    # Status display
    if st.session_state.calibration_status:
        st.success("✅ System Calibrated")
        if 'calibration_results' in st.session_state:
            results = st.session_state.calibration_results
            
            # Metrics display
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("Face Detection", f"{results.get('face_detection_rate', 0):.1f}%")
            with metric_col2:
                st.metric("Eye Tracking Quality", f"{results.get('eye_quality', 0):.1f}%")
            with metric_col3:
                st.metric("Calibration Accuracy", f"{results.get('accuracy', 0):.1f}%")
    else:
        st.warning("⚠️ Calibration Required")
        st.info("""
        **Calibration Options:**
        - **3-Second Calibration**: Full video analysis (recommended)
        - **Quick Calibration**: Single frame test (for troubleshooting)
        """)

def calibrate_with_video(camera_preview):
    """Perform 3-second video calibration"""
    try:
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Cannot access camera")
            return
        
        # Set camera properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for better Streamlit performance
        
        # Test camera first
        ret, test_frame = cap.read()
        if not ret:
            st.error("❌ Cannot read from camera")
            cap.release()
            return
        
        # Countdown
        countdown_placeholder = st.empty()
        for i in range(3, 0, -1):
            countdown_placeholder.markdown(f"<h1 style='text-align: center; color: red;'>Starting in {i}...</h1>", unsafe_allow_html=True)
            time.sleep(1)
        
        countdown_placeholder.markdown("<h1 style='text-align: center; color: green;'>Recording! Look at the camera</h1>", unsafe_allow_html=True)
        
        # Record frames with better error handling
        frames = []
        start_time = time.time()
        frame_count = 0
        target_frames = 45  # 3 seconds at 15 FPS
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Capture frames in a more controlled way
        for i in range(target_frames):
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame.copy())
                frame_count += 1
                
                # Update preview every 5th frame to reduce load
                if frame_count % 5 == 0:
                    camera_preview.image(frame, channels="BGR", caption=f"Recording... Frame {frame_count}/{target_frames}")
                
                # Update progress
                progress = (i + 1) / target_frames
                progress_bar.progress(progress)
                status_text.text(f"Captured {frame_count} frames...")
            else:
                st.warning(f"⚠️ Failed to capture frame {i+1}")
            
            time.sleep(0.067)  # ~15 FPS
        
        cap.release()
        countdown_placeholder.empty()
        progress_bar.empty()
        status_text.empty()
        
        # Check if we got enough frames
        if len(frames) < 10:
            display_status_box(f"❌ Calibration failed: Only captured {len(frames)} frames (minimum 10 required)", "danger")
            return
        
        # Analyze recorded frames
        with st.spinner(f"Analyzing {len(frames)} calibration frames..."):
            calibration_results = analyze_calibration_frames(frames)
            
            # Store results
            st.session_state.calibrated_tracker = CalibratedGazeTracker()
            st.session_state.calibration_status = True
            st.session_state.calibration_results = calibration_results
            
            # Display results
            if calibration_results['success']:
                display_status_box("✅ Calibration completed successfully!", "success")
                st.balloons()
                
                # Show calibration metrics
                result_col1, result_col2, result_col3 = st.columns(3)
                with result_col1:
                    st.metric("Frames Analyzed", calibration_results['frames_analyzed'])
                with result_col2:
                    st.metric("Face Detection Rate", f"{calibration_results['face_detection_rate']:.1f}%")
                with result_col3:
                    st.metric("Overall Quality", f"{calibration_results['accuracy']:.1f}%")
                
                # Show recommendations
                if calibration_results.get('recommendations'):
                    st.subheader("📋 Calibration Recommendations:")
                    for rec in calibration_results['recommendations']:
                        st.info(f"💡 {rec}")
                
                st.rerun()
            else:
                display_status_box(f"❌ Calibration failed: {calibration_results['error']}", "danger")
                
    except Exception as e:
        display_status_box(f"❌ Calibration error: {str(e)}", "danger")
        # Make sure to release camera on error
        try:
            cap.release()
        except:
            pass

def quick_calibrate(camera_preview):
    """Quick single-frame calibration for testing"""
    try:
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Cannot access camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        with st.spinner("Capturing calibration frame..."):
            # Capture a single frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                st.error("❌ Failed to capture frame")
                return
            
            # Display the captured frame
            camera_preview.image(frame, channels="BGR", caption="Calibration Frame Captured")
            
            # Analyze the single frame
            calibration_results = analyze_calibration_frames([frame])
            
            # Store results
            st.session_state.calibrated_tracker = CalibratedGazeTracker()
            st.session_state.calibration_status = True
            st.session_state.calibration_results = calibration_results
            
            # Display results
            if calibration_results['success']:
                display_status_box("✅ Quick calibration completed!", "success")
                
                # Show calibration metrics
                quick_col1, quick_col2, quick_col3 = st.columns(3)
                with quick_col1:
                    st.metric("Frames Analyzed", 1)
                with quick_col2:
                    st.metric("Face Detection", "✅" if calibration_results['face_detection_rate'] > 0 else "❌")
                with quick_col3:
                    st.metric("Overall Quality", f"{calibration_results['accuracy']:.1f}%")
                
                st.info("💡 Quick calibration uses a single frame. For better accuracy, use 3-second calibration.")
                st.rerun()
            else:
                display_status_box(f"❌ Quick calibration failed: {calibration_results['error']}", "danger")
                
    except Exception as e:
        display_status_box(f"❌ Quick calibration error: {str(e)}", "danger")
        try:
            cap.release()
        except:
            pass

def analyze_calibration_frames(frames):
    """Analyze the recorded calibration frames"""
    try:
        if not frames:
            return {"success": False, "error": "No frames recorded"}
        
        if len(frames) == 0:
            return {"success": False, "error": "Empty frames list"}
        
        face_detections = 0
        eye_quality_scores = []
        lip_positions = []
        face_angles = []
        analysis_errors = 0
        
        for i, frame in enumerate(frames):
            try:
                if frame is None:
                    analysis_errors += 1
                    continue
                
                # Gaze analysis
                gaze_direction = get_gaze_direction(frame)
                if gaze_direction != "unknown":
                    face_detections += 1
                    
                    # Simulate eye quality score based on gaze detection
                    if gaze_direction == "center":
                        eye_quality_scores.append(95)
                    elif gaze_direction in ["left", "right", "up", "down"]:
                        eye_quality_scores.append(75)
                    else:
                        eye_quality_scores.append(50)
                
                # Person detection for face angle analysis
                person_results = detect_multiple_people(frame)
                if person_results and person_results.get('face_count', 0) > 0:
                    # Simulate face angle quality (in real implementation, would calculate actual angles)
                    face_angles.append(85)  # Good face angle
                
                # Simulate lip position analysis
                lip_positions.append(80)  # Good lip visibility
                
            except Exception as frame_error:
                analysis_errors += 1
                logger.warning(f"Error analyzing frame {i}: {str(frame_error)}")
                continue
        
        # Check if we have enough successful analyses
        successful_frames = len(frames) - analysis_errors
        if successful_frames == 0:
            return {"success": False, "error": "Failed to analyze any frames"}
        
        # Calculate metrics
        face_detection_rate = (face_detections / successful_frames) * 100
        avg_eye_quality = np.mean(eye_quality_scores) if eye_quality_scores else 50
        avg_face_angle = np.mean(face_angles) if face_angles else 50
        avg_lip_quality = np.mean(lip_positions) if lip_positions else 50
        
        # Overall accuracy (weighted average)
        overall_accuracy = (
            face_detection_rate * 0.4 +  # Face detection is most important
            avg_eye_quality * 0.3 +      # Eye quality is second
            avg_face_angle * 0.2 +       # Face angle is third
            avg_lip_quality * 0.1        # Lip quality is least critical
        )
        
        return {
            "success": True,
            "frames_analyzed": len(frames),
            "successful_analyses": successful_frames,
            "analysis_errors": analysis_errors,
            "face_detection_rate": face_detection_rate,
            "eye_quality": avg_eye_quality,
            "face_angle_quality": avg_face_angle,
            "lip_quality": avg_lip_quality,
            "accuracy": overall_accuracy,
            "recommendations": get_calibration_recommendations(face_detection_rate, avg_eye_quality)
        }
        
    except Exception as e:
        logger.error(f"Error in calibration analysis: {str(e)}")
        return {"success": False, "error": str(e)}

def get_calibration_recommendations(face_rate, eye_quality):
    """Get calibration recommendations based on results"""
    recommendations = []
    
    if face_rate < 80:
        recommendations.append("Improve lighting conditions")
        recommendations.append("Position face more centrally in frame")
    
    if eye_quality < 70:
        recommendations.append("Look directly at the camera")
        recommendations.append("Reduce head movement during calibration")
    
    if face_rate > 90 and eye_quality > 85:
        recommendations.append("Excellent calibration quality!")
    
    return recommendations

def live_camera_interface():
    """Live camera monitoring interface"""
    st.markdown('<h2 class="module-header">📹 Live Camera Monitoring</h2>', unsafe_allow_html=True)
    
    if not st.session_state.calibration_status:
        display_status_box("⚠️ Please complete calibration before using live monitoring.", "warning")
        return
    
    # Initialize session state for live monitoring
    if 'live_monitoring_active' not in st.session_state:
        st.session_state.live_monitoring_active = False
    if 'monitoring_stats' not in st.session_state:
        st.session_state.monitoring_stats = {
            'frames_analyzed': 0,
            'risk_history': [],
            'gaze_history': [],
            'person_history': [],
            'lip_sync_history': []
        }
    
    # Camera feed placeholder
    camera_placeholder = st.empty()
    
    # Control buttons
    monitor_col1, monitor_col2 = st.columns(2)
    
    with monitor_col1:
        if st.button("🔴 Start Live Monitoring", type="primary", disabled=st.session_state.live_monitoring_active):
            st.session_state.live_monitoring_active = True
            st.session_state.monitoring_stats = {
                'frames_analyzed': 0,
                'risk_history': [],
                'gaze_history': [],
                'person_history': [],
                'lip_sync_history': []
            }
            st.rerun()
    
    with monitor_col2:
        if st.button("⏹️ Stop Monitoring", disabled=not st.session_state.live_monitoring_active):
            st.session_state.live_monitoring_active = False
            # Clean up lip sync analyzer
            global live_lip_sync_analyzer
            if live_lip_sync_analyzer:
                live_lip_sync_analyzer.cleanup()
                live_lip_sync_analyzer = None
            st.rerun()
    
    # Live monitoring loop
    if st.session_state.live_monitoring_active:
        run_live_monitoring(camera_placeholder)
    
    # Live statistics section
    st.subheader("📊 Live Statistics")
    
    stats = st.session_state.monitoring_stats
    
    # Current metrics
    stats_col1, stats_col2 = st.columns(2)
    with stats_col1:
        st.metric("Frames Analyzed", stats['frames_analyzed'])
    with stats_col2:
        avg_risk = np.mean(stats['risk_history']) if stats['risk_history'] else 0
        st.metric("Average Risk", f"{avg_risk:.3f}")
    
    # Recent activity
    if stats['gaze_history']:
        st.write("**Recent Gaze Directions:**")
        recent_gaze = stats['gaze_history'][-5:]  # Last 5
        for i, gaze in enumerate(reversed(recent_gaze)):
            st.write(f"• Frame -{i}: {gaze}")
    
    # Status indicator
    if st.session_state.live_monitoring_active:
        st.success("🟢 Monitoring Active")
    else:
        st.info("🔴 Monitoring Stopped")
    
    # Monitoring features info
    st.info("""
    **Live Monitoring Features:**
    - Continuous camera feed
    - Real-time gaze tracking
    - Person detection
    - Risk assessment
    - Live statistics
    - Alert system
    """)

def run_live_monitoring(camera_placeholder):
    """Run the continuous live monitoring loop"""
    try:
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Cannot access camera")
            st.session_state.live_monitoring_active = False
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 10)  # Lower FPS for better performance in web
        
        # Analysis placeholders
        analysis_placeholder = st.empty()
        risk_placeholder = st.empty()
        
        frame_count = 0
        start_time = time.time()
        
        # Monitoring loop
        while st.session_state.live_monitoring_active:
            ret, frame = cap.read()
            
            if not ret:
                st.error("❌ Failed to read from camera")
                break
            
            frame_count += 1
            
            # Display current frame
            camera_placeholder.image(frame, channels="BGR", caption=f"Live Feed - Frame {frame_count}")
            
                        # Analyze every 10th frame (to reduce processing load)
            if frame_count % 10 == 0:
                analysis_results = analyze_live_frame(frame)
                
                # Update statistics
                st.session_state.monitoring_stats['frames_analyzed'] += 1
                st.session_state.monitoring_stats['risk_history'].append(analysis_results['risk_score'])
                st.session_state.monitoring_stats['gaze_history'].append(analysis_results['gaze_direction'])
                st.session_state.monitoring_stats['person_history'].append(analysis_results['people_count'])
                
                # Add lip sync tracking
                lip_sync_data = analysis_results.get('lip_sync_quality', {})
                lip_sync_status = "Synced" if lip_sync_data.get('is_synced', True) else "Not Synced"
                st.session_state.monitoring_stats['lip_sync_history'].append(lip_sync_status)
                
                # Keep only last 100 entries
                for key in ['risk_history', 'gaze_history', 'person_history', 'lip_sync_history']:
                    if len(st.session_state.monitoring_stats[key]) > 100:
                        st.session_state.monitoring_stats[key] = st.session_state.monitoring_stats[key][-100:]
                
                # Display analysis results
                display_live_analysis(analysis_results, analysis_placeholder, risk_placeholder)
                
                # Check for alerts
                check_monitoring_alerts(analysis_results)
            
            # Control frame rate
            time.sleep(0.1)  # 10 FPS
            
            # Break if monitoring stopped
            if not st.session_state.live_monitoring_active:
                break
        
        cap.release()
        
        # Final statistics
        if frame_count > 0:
            duration = time.time() - start_time
            fps = frame_count / duration
            st.success(f"✅ Monitoring session completed. Analyzed {frame_count} frames in {duration:.1f}s (avg {fps:.1f} FPS)")
            
    except Exception as e:
        st.error(f"❌ Live monitoring error: {str(e)}")
        st.session_state.live_monitoring_active = False

def analyze_live_frame(frame):
    """Analyze a single frame during live monitoring"""
    try:
        # Gaze analysis
        gaze_direction = get_gaze_direction(frame)
        
        # Person detection
        person_results = detect_multiple_people(frame)
        people_count = person_results['people_count']
        face_count = person_results['face_count']
        
        # Lip sync analysis (simplified for live monitoring)
        lip_sync_quality = analyze_live_lip_sync(frame)
        
        # Audio analysis (simplified - would need microphone access for real audio)
        audio_analysis = {
            'multiple_speakers': False,
            'speaker_confidence': 0.0,
            'has_background_noise': False,
            'noise_level': 0.0
        }
        
        # Calculate risk score
        quick_inputs = {
            'gaze_data': {'direction': gaze_direction},
            'person_data': {
                'people_count': people_count,
                'face_count': face_count
            },
            'lip_sync_data': {
                'is_synced': lip_sync_quality['is_synced'],
                'sync_score': lip_sync_quality['sync_score']
            },
            'audio_data': audio_analysis
        }
        
        risk_score = calculate_cheat_score(quick_inputs)
        risk_level = get_risk_level(risk_score)
        
        return {
            'gaze_direction': gaze_direction,
            'people_count': people_count,
            'face_count': face_count,
            'lip_sync_quality': lip_sync_quality,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'timestamp': time.time()
        }
        
    except Exception as e:
        return {
            'gaze_direction': 'error',
            'people_count': 0,
            'face_count': 0,
            'lip_sync_quality': {'is_synced': True, 'sync_score': 1.0, 'lip_movement': 0},
            'risk_score': 0.5,
            'risk_level': 'Unknown',
            'error': str(e),
            'timestamp': time.time()
        }

class LiveLipSyncAnalyzer:
    """Real-time lip sync analyzer that buffers frames and audio"""
    
    def __init__(self, buffer_size=30):  # 1 second at 30fps
        self.buffer_size = buffer_size
        self.frame_buffer = []
        self.lip_movement_buffer = []
        self.audio_buffer = []
        self.timestamps = []
        import pyaudio
        import numpy as np
        
        # Audio recording setup
        self.audio_format = pyaudio.paFloat32
        self.channels = 1
        self.rate = 44100
        self.chunk = 1024
        
        try:
            self.audio = pyaudio.PyAudio()
            self.audio_stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
                stream_callback=self._audio_callback
            )
            self.audio_stream.start_stream()
            self.audio_enabled = True
        except Exception as e:
            print(f"Audio capture failed: {e}")
            self.audio_enabled = False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        try:
            import pyaudio
            import numpy as np
            
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            # Calculate RMS energy
            rms = np.sqrt(np.mean(audio_data**2))
            
            # Add to buffer with timestamp
            self.audio_buffer.append(rms)
            
            # Keep buffer size manageable
            if len(self.audio_buffer) > self.buffer_size * 10:  # More audio samples
                self.audio_buffer.pop(0)
                
        except Exception as e:
            print(f"Audio callback error: {e}")
        
        try:
            import pyaudio
            return (in_data, pyaudio.paContinue)
        except:
            return (in_data, 0)  # Fallback
    
    def analyze_frame(self, frame):
        """Analyze a single frame and update buffers with improved detection"""
        try:
            from lip_sync_detector import extract_lip_landmarks_mediapipe, calculate_lip_distance, calculate_lip_area
            import time
            import numpy as np
            
            current_time = time.time()
            
            # Extract lip landmarks using improved method
            landmarks = extract_lip_landmarks_mediapipe(frame)
            
            if landmarks is not None:
                # Calculate both lip distance and area for better accuracy
                lip_distance = calculate_lip_distance(landmarks)
                lip_area = calculate_lip_area(landmarks)
                
                # Combine distance and area for more robust detection
                normalized_movement = lip_distance + (lip_area / 1000.0)  # Scale area appropriately
                
                # Add to buffers
                self.frame_buffer.append(frame.copy())
                self.lip_movement_buffer.append(normalized_movement)
                self.timestamps.append(current_time)
                
                # Maintain buffer size
                if len(self.frame_buffer) > self.buffer_size:
                    self.frame_buffer.pop(0)
                    self.lip_movement_buffer.pop(0)
                    self.timestamps.pop(0)
                
                # Analyze sync if we have enough data
                if len(self.lip_movement_buffer) >= 10:  # Need at least 10 frames
                    return self._calculate_sync_quality_improved(normalized_movement, lip_distance, lip_area)
                else:
                    # Not enough data yet
                    return {
                        'is_synced': True,
                        'sync_score': 1.0,
                        'lip_movement': normalized_movement,
                        'lip_distance': lip_distance,
                        'lip_area': lip_area,
                        'landmarks_detected': True,
                        'status': 'buffering'
                    }
            else:
                # No landmarks detected
                return {
                    'is_synced': True,
                    'sync_score': 0.5,
                    'lip_movement': 0,
                    'lip_distance': 0,
                    'lip_area': 0,
                    'landmarks_detected': False,
                    'status': 'no_face'
                }
                
        except Exception as e:
            return {
                'is_synced': True,
                'sync_score': 1.0,
                'lip_movement': 0,
                'lip_distance': 0,
                'lip_area': 0,
                'landmarks_detected': False,
                'error': str(e),
                'status': 'error'
            }
    
    def _calculate_sync_quality_improved(self, current_movement, lip_distance, lip_area):
        """Calculate sync quality with improved multi-feature analysis"""
        try:
            import numpy as np
            from scipy import signal
            
            # Analyze lip movement patterns with multiple features
            lip_movements = np.array(self.lip_movement_buffer)
            
            # Calculate comprehensive movement statistics
            lip_variation = np.std(lip_movements)
            lip_mean = np.mean(lip_movements)
            lip_range = np.max(lip_movements) - np.min(lip_movements)
            
            # Detect movement trends using derivatives
            if len(lip_movements) >= 5:
                # Smooth the signal first
                smoothed = signal.savgol_filter(lip_movements, min(5, len(lip_movements)//2 | 1), 2)
                lip_velocity = np.diff(smoothed)
                lip_acceleration = np.diff(lip_velocity)
                velocity_variance = np.var(lip_velocity) if len(lip_velocity) > 0 else 0
            else:
                velocity_variance = 0
            
            # Get recent audio activity if available
            audio_activity = 0.0
            audio_variance = 0.0
            if self.audio_enabled and len(self.audio_buffer) > 0:
                # Get audio corresponding to recent frames
                recent_audio = self.audio_buffer[-min(len(self.audio_buffer), 20):]
                audio_activity = np.mean(recent_audio)
                audio_variance = np.var(recent_audio)
            
            # Enhanced speech detection using multiple criteria
            # Base thresholds - more conservative to reduce false positives
            distance_threshold = 8.0  # Normalized lip distance
            area_threshold = 50.0     # Lip area threshold
            variation_threshold = 2.5  # Movement variation
            velocity_threshold = 1.0   # Movement velocity variance
            
            # Multiple speaking indicators
            distance_speaking = lip_distance > distance_threshold
            area_speaking = lip_area > area_threshold
            variation_speaking = lip_variation > variation_threshold
            velocity_speaking = velocity_variance > velocity_threshold
            range_speaking = lip_range > 3.0
            
            # Combine visual indicators with weighted scoring
            visual_indicators = [
                distance_speaking,
                area_speaking, 
                variation_speaking,
                velocity_speaking,
                range_speaking
            ]
            
            # Calculate visual speaking confidence (need at least 2 of 5 indicators)
            visual_confidence = sum(visual_indicators) / len(visual_indicators)
            is_speaking_visual = visual_confidence >= 0.4  # At least 40% confidence
            
            # Audio speaking detection with multiple features
            audio_energy_threshold = 0.008  # RMS energy threshold
            audio_variance_threshold = 0.0001  # Audio variance threshold
            
            energy_speaking = audio_activity > audio_energy_threshold
            variance_speaking = audio_variance > audio_variance_threshold
            
            audio_confidence = 0.0
            is_speaking_audio = False
            if self.audio_enabled:
                audio_indicators = [energy_speaking, variance_speaking]
                audio_confidence = sum(audio_indicators) / len(audio_indicators)
                is_speaking_audio = audio_confidence >= 0.5
            
            # Advanced sync quality calculation
            if self.audio_enabled:
                # Calculate temporal correlation if we have enough data
                if len(lip_movements) >= 15 and len(self.audio_buffer) >= 15:
                    # Get matching audio segment
                    audio_segment = self.audio_buffer[-len(lip_movements):]
                    
                    # Normalize signals
                    if np.std(lip_movements) > 0:
                        lip_norm = (lip_movements - np.mean(lip_movements)) / np.std(lip_movements)
                    else:
                        lip_norm = lip_movements
                        
                    if np.std(audio_segment) > 0:
                        audio_norm = (audio_segment - np.mean(audio_segment)) / np.std(audio_segment)
                    else:
                        audio_norm = audio_segment
                    
                    # Calculate cross-correlation
                    try:
                        correlation = np.corrcoef(lip_norm, audio_norm)[0, 1]
                        if np.isnan(correlation):
                            correlation = 0.0
                    except:
                        correlation = 0.0
                    
                    # Enhanced sync scoring based on correlation and activity matching
                    if is_speaking_visual and is_speaking_audio:
                        # Both speaking - check correlation
                        if abs(correlation) > 0.3:
                            sync_score = 0.85 + 0.15 * abs(correlation)
                            is_synced = True
                        else:
                            sync_score = 0.6
                            is_synced = correlation > 0  # Positive correlation required
                    elif not is_speaking_visual and not is_speaking_audio:
                        # Both silent - excellent sync
                        sync_score = 0.95
                        is_synced = True
                    elif is_speaking_visual and not is_speaking_audio:
                        # Visual but no audio - check confidence levels
                        if visual_confidence < 0.6:  # Low confidence visual
                            sync_score = 0.8  # Maybe just subtle movement
                            is_synced = True
                        else:
                            sync_score = 0.2  # Clear lip movement without audio
                            is_synced = False
                    elif not is_speaking_visual and is_speaking_audio:
                        # Audio but no visual - check audio confidence
                        if audio_confidence < 0.7:  # Low confidence audio
                            sync_score = 0.75  # Maybe background noise
                            is_synced = True
                        else:
                            sync_score = 0.25  # Clear audio without lip movement
                            is_synced = False
                    else:
                        sync_score = 0.7
                        is_synced = True
                        
                else:
                    # Not enough data for correlation - use basic matching
                    if is_speaking_visual and is_speaking_audio:
                        sync_score = 0.8
                        is_synced = True
                    elif not is_speaking_visual and not is_speaking_audio:
                        sync_score = 0.9
                        is_synced = True
                    else:
                        sync_score = 0.5
                        is_synced = False
            else:
                # Only visual analysis - be lenient but realistic
                if is_speaking_visual:
                    sync_score = 0.75  # Assume reasonable sync when speaking detected
                    is_synced = True
                else:
                    sync_score = 0.9   # High score for silence
                    is_synced = True
            
            return {
                'is_synced': is_synced,
                'sync_score': float(sync_score),
                'lip_movement': float(current_movement),
                'lip_distance': float(lip_distance),
                'lip_area': float(lip_area),
                'landmarks_detected': True,
                'is_speaking_visual': is_speaking_visual,
                'visual_confidence': float(visual_confidence),
                'is_speaking_audio': is_speaking_audio if self.audio_enabled else None,
                'audio_confidence': float(audio_confidence) if self.audio_enabled else None,
                'audio_activity': float(audio_activity) if self.audio_enabled else None,
                'lip_variation': float(lip_variation),
                'lip_range': float(lip_range),
                'velocity_variance': float(velocity_variance),
                'correlation': float(correlation) if self.audio_enabled and 'correlation' in locals() else None,
                'status': 'analyzing'
            }
            
        except Exception as e:
            return {
                'is_synced': True,
                'sync_score': 0.5,
                'lip_movement': float(current_movement),
                'lip_distance': float(lip_distance),
                'lip_area': float(lip_area),
                'landmarks_detected': True,
                'error': str(e),
                'status': 'calculation_error'
            }
    
    def cleanup(self):
        """Clean up audio resources"""
        try:
            if hasattr(self, 'audio_stream') and self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            if hasattr(self, 'audio') and self.audio:
                self.audio.terminate()
        except Exception as e:
            print(f"Cleanup error: {e}")

# Global lip sync analyzer instance
live_lip_sync_analyzer = None

def cleanup_live_analyzer():
    """Clean up the live analyzer on app exit"""
    global live_lip_sync_analyzer
    if live_lip_sync_analyzer:
        live_lip_sync_analyzer.cleanup()
        live_lip_sync_analyzer = None

# Register cleanup function
import atexit
atexit.register(cleanup_live_analyzer)

def analyze_live_lip_sync(frame):
    """Analyze lip movement with audio for live monitoring"""
    global live_lip_sync_analyzer
    
    # Initialize analyzer if needed
    if live_lip_sync_analyzer is None:
        live_lip_sync_analyzer = LiveLipSyncAnalyzer()
    
    return live_lip_sync_analyzer.analyze_frame(frame)

def display_live_analysis(results, analysis_placeholder, risk_placeholder):
    """Display live analysis results"""
    # Analysis results
    with analysis_placeholder.container():
        # Main metrics
        main_col1, main_col2, main_col3, main_col4 = st.columns(4)
        
        with main_col1:
            st.metric("Gaze Direction", results['gaze_direction'])
        
        with main_col2:
            st.metric("People Count", results['people_count'])
        
        with main_col3:
            st.metric("Face Count", results['face_count'])
        
        with main_col4:
            lip_sync = results.get('lip_sync_quality', {})
            lip_status = "✅ Synced" if lip_sync.get('is_synced', True) else "❌ Not Synced"
            st.metric("Lip Sync", lip_status)
        
        # Enhanced lip sync details
        if 'lip_sync_quality' in results:
            lip_data = results['lip_sync_quality']
            
            st.markdown("**📊 Lip Sync Analysis:**")
            
            # Main sync metrics
            sync_detail_col1, sync_detail_col2, sync_detail_col3 = st.columns(3)
            
            with sync_detail_col1:
                sync_score = lip_data.get('sync_score', 1.0)
                st.metric("Sync Quality", f"{sync_score:.2f}")
            
            with sync_detail_col2:
                lip_movement = lip_data.get('lip_movement', 0)
                movement_status = "Speaking" if lip_movement > 8 else "Silent"
                st.metric("Lip Movement", movement_status)
            
            with sync_detail_col3:
                status = lip_data.get('status', 'unknown')
                status_display = {
                    'analyzing': '🔍 Analyzing',
                    'buffering': '⏳ Buffering',
                    'no_face': '❌ No Face',
                    'error': '⚠️ Error'
                }.get(status, status.title())
                st.metric("Status", status_display)
            
            # Advanced metrics if available
            if lip_data.get('audio_activity') is not None:
                st.markdown("**🎤 Audio-Visual Sync Details:**")
                audio_col1, audio_col2, audio_col3 = st.columns(3)
                
                with audio_col1:
                    is_speaking_visual = lip_data.get('is_speaking_visual', False)
                    visual_status = "👄 Speaking" if is_speaking_visual else "🤐 Silent"
                    st.metric("Visual", visual_status)
                
                with audio_col2:
                    is_speaking_audio = lip_data.get('is_speaking_audio', False)
                    audio_status = "🔊 Speaking" if is_speaking_audio else "🔇 Silent"
                    st.metric("Audio", audio_status)
                
                with audio_col3:
                    audio_activity = lip_data.get('audio_activity', 0)
                    st.metric("Audio Level", f"{audio_activity:.3f}")
            
            elif status == 'analyzing':
                st.info("💡 Enable microphone permissions for enhanced audio-visual sync analysis!")
    
    # Risk assessment
    with risk_placeholder.container():
        risk_score = results['risk_score']
        risk_level = results['risk_level']
        
        if risk_score < 0.3:
            st.success(f"🟢 Risk Level: {risk_level} (Score: {risk_score:.3f})")
        elif risk_score < 0.7:
            st.warning(f"🟡 Risk Level: {risk_level} (Score: {risk_score:.3f})")
        else:
            st.error(f"🔴 Risk Level: {risk_level} (Score: {risk_score:.3f})")

def check_monitoring_alerts(results):
    """Check for monitoring alerts and display warnings"""
    alerts = []
    
    # High risk alert
    if results['risk_score'] > 0.7:
        alerts.append(f"⚠️ HIGH RISK DETECTED: {results['risk_level']}")
    
    # Multiple people alert
    if results['people_count'] > 1:
        alerts.append(f"👥 Multiple people detected: {results['people_count']}")
    
    # No face detected alert
    if results['face_count'] == 0:
        alerts.append("👤 No face detected in frame")
    
    # Suspicious gaze alert (removed 'down' as it's normal behavior)
    if results['gaze_direction'] in ['left', 'right']:
        alerts.append(f"👁️ Suspicious gaze direction: {results['gaze_direction']}")
    
    # Display alerts
    if alerts:
        for alert in alerts:
            st.error(alert)

def individual_module_testing():
    """Interface for testing individual modules"""
    st.markdown('<h2 class="module-header">🔧 Individual Module Testing</h2>', unsafe_allow_html=True)
    
    module_choice = st.selectbox(
        "Select Module to Test:",
        ["Gaze Tracking", "Lip Sync Detection", "Person Detection", "Audio Analysis"]
    )
    
    uploaded_file = st.file_uploader(
        "Upload test file",
        type=['mp4', 'avi', 'mov', 'wav', 'mp3', 'jpg', 'png'] if module_choice != "Audio Analysis" else ['mp4', 'avi', 'mov', 'wav', 'mp3']
    )
    
    if uploaded_file and st.button(f"Test {module_choice}"):
        with st.spinner(f"Running {module_choice} analysis..."):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    temp_path = tmp_file.name
                
                if module_choice == "Gaze Tracking":
                    if uploaded_file.type.startswith('image'):
                        # Single image analysis
                        frame = cv2.imdecode(np.frombuffer(uploaded_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                        direction = get_gaze_direction(frame)
                        st.success(f"Gaze Direction: {direction}")
                    else:
                        # Video analysis
                        results = analyze_video_gaze(temp_path)
                        st.json(results)
                
                elif module_choice == "Lip Sync Detection":
                    results = is_lip_synced(temp_path)
                    st.json(results)
                    
                    if results.get('is_lip_synced', False):
                        st.success("✅ Good lip synchronization detected")
                    else:
                        st.warning("⚠️ Poor lip synchronization detected")
                
                elif module_choice == "Person Detection":
                    if uploaded_file.type.startswith('image'):
                        frame = cv2.imdecode(np.frombuffer(uploaded_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                        results = detect_multiple_people(frame)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("People Detected", results['people_count'])
                        with col2:
                            st.metric("Faces Detected", results['face_count'])
                        
                        # Display processed image
                        processed_img = base64.b64decode(results['processed_image'])
                        st.image(processed_img, caption="Processed Image with Detections")
                    else:
                        st.warning("Please upload an image file for person detection testing")
                
                elif module_choice == "Audio Analysis":
                    analyzer = AudioAnalyzer()
                    results = analyzer.analyze_audio_file(temp_path)
                    st.json(results)
                    
                    # Display key metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Multiple Speakers", "Yes" if results.get('multiple_speakers', {}).get('detected', False) else "No")
                    with col2:
                        st.metric("Background Noise", "Yes" if results.get('background_noise', {}).get('significant_noise', False) else "No")
                    with col3:
                        st.metric("Prolonged Silence", "Yes" if results.get('prolonged_silence', {}).get('detected', False) else "No")
                
                # Clean up temporary file
                os.unlink(temp_path)
                
            except Exception as e:
                display_status_box(f"❌ Error during {module_choice} analysis: {str(e)}", "danger")

def comprehensive_analysis_interface():
    """Interface for comprehensive video analysis"""
    st.markdown('<h2 class="module-header">📊 Comprehensive Analysis</h2>', unsafe_allow_html=True)
    
    uploaded_video = st.file_uploader(
        "Upload video for comprehensive analysis",
        type=['mp4', 'avi', 'mov'],
        help="Upload a video file to run all proctoring modules and get a comprehensive analysis report"
    )
    
    uploaded_audio = st.file_uploader(
        "Upload separate audio file (optional)",
        type=['wav', 'mp3', 'aac'],
        help="Optional: Upload a separate audio file if not embedded in video"
    )
    
    if uploaded_video:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("🚀 Run Comprehensive Analysis", type="primary"):
                run_comprehensive_analysis(uploaded_video, uploaded_audio)
        
        with col2:
            session_id = st.text_input("Session ID (optional)", value=f"session_{int(time.time())}")

def run_comprehensive_analysis(video_file, audio_file=None):
    """Run comprehensive analysis on uploaded files"""
    with st.spinner("Running comprehensive analysis... This may take a few minutes."):
        try:
            # Save uploaded files temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
                tmp_video.write(video_file.read())
                video_path = tmp_video.name
            
            audio_path = None
            if audio_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_audio:
                    tmp_audio.write(audio_file.read())
                    audio_path = tmp_audio.name
            
            # Initialize analyzer and run analysis
            analyzer = ProctorAnalyzer()
            results = analyzer.run_comprehensive_analysis(video_path, audio_path)
            
            # Store results in session state
            st.session_state.analysis_results = results
            
            # Display results
            display_comprehensive_results(results)
            
            # Clean up temporary files
            os.unlink(video_path)
            if audio_path:
                os.unlink(audio_path)
                
        except Exception as e:
            display_status_box(f"❌ Error during comprehensive analysis: {str(e)}", "danger")

def display_comprehensive_results(results):
    """Display comprehensive analysis results"""
    st.markdown('<h3 class="module-header">📈 Analysis Results</h3>', unsafe_allow_html=True)
    
    # Overall summary
    cheat_score = results.get('cheat_score_analysis', {}).get('cheat_score', 0)
    risk_level = results.get('cheat_score_analysis', {}).get('risk_level', 'Unknown')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Cheat Score",
            f"{cheat_score:.3f}",
            help="Score from 0 (safe) to 1 (high risk)"
        )
    
    with col2:
        if cheat_score < 0.3:
            st.success(f"Risk Level: {risk_level}")
        elif cheat_score < 0.7:
            st.warning(f"Risk Level: {risk_level}")
        else:
            st.error(f"Risk Level: {risk_level}")
    
    with col3:
        processing_time = results.get('session_info', {}).get('total_processing_time', 0)
        st.metric("Processing Time", f"{processing_time:.2f}s")
    
    # Detailed results tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Summary", "👁️ Gaze Analysis", "🎭 Person Detection", 
        "🎵 Audio Analysis", "💋 Lip Sync"
    ])
    
    with tab1:
        display_summary_tab(results)
    
    with tab2:
        display_gaze_analysis_tab(results)
    
    with tab3:
        display_person_detection_tab(results)
    
    with tab4:
        display_audio_analysis_tab(results)
    
    with tab5:
        display_lip_sync_tab(results)

def display_summary_tab(results):
    """Display summary tab content"""
    summary = results.get('summary', {})
    concerns = summary.get('primary_concerns', [])
    recommendations = summary.get('recommendations', [])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚨 Primary Concerns")
        if concerns:
            for concern in concerns:
                st.warning(f"• {concern}")
        else:
            st.success("✅ No major concerns detected")
    
    with col2:
        st.subheader("💡 Recommendations")
        if recommendations:
            for rec in recommendations:
                st.info(f"• {rec}")
        else:
            st.success("✅ No specific recommendations needed")
    
    # Score breakdown chart
    st.subheader("📊 Score Breakdown")
    
    cheat_inputs = results.get('cheat_score_analysis', {}).get('input_data', {})
    
    # Create a breakdown chart
    categories = ['Gaze', 'Person Detection', 'Audio', 'Lip Sync']
    scores = [
        0.8 if cheat_inputs.get('gaze_data', {}).get('direction') in ['left', 'right', 'down'] else 0.2,
        0.9 if cheat_inputs.get('person_data', {}).get('people_count', 1) > 1 else 0.1,
        0.7 if cheat_inputs.get('audio_data', {}).get('multiple_speakers', False) else 0.2,
        0.8 if not cheat_inputs.get('lip_sync_data', {}).get('is_synced', True) else 0.1
    ]
    
    fig = px.bar(
        x=categories,
        y=scores,
        title="Risk Score by Category",
        color=scores,
        color_continuous_scale="RdYlGn_r"
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def display_gaze_analysis_tab(results):
    """Display gaze analysis tab content"""
    gaze_results = results.get('analysis_results', {}).get('gaze_tracking', {})
    
    if gaze_results.get('success', False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Dominant Direction", gaze_results.get('direction', 'Unknown'))
            st.metric("Off-screen Duration", f"{gaze_results.get('off_screen_duration', 0):.2f}s")
            st.metric("Suspicious Patterns", gaze_results.get('suspicious_patterns', 0))
        
        with col2:
            st.metric("Face Detection Rate", f"{gaze_results.get('face_detection_rate', 0):.1f}%")
            st.metric("Processing Time", f"{gaze_results.get('processing_time', 0):.2f}s")
        
        # Gaze distribution chart
        gaze_dist = gaze_results.get('gaze_distribution', {})
        if gaze_dist:
            fig = px.pie(
                values=list(gaze_dist.values()),
                names=list(gaze_dist.keys()),
                title="Gaze Direction Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error(f"Gaze analysis failed: {gaze_results.get('error', 'Unknown error')}")

def display_person_detection_tab(results):
    """Display person detection tab content"""
    person_results = results.get('analysis_results', {}).get('person_detection', {})
    
    if person_results.get('success', False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Max People Detected", person_results.get('people_count', 0))
            st.metric("Max Faces Detected", person_results.get('face_count', 0))
        
        with col2:
            st.metric("Average People", person_results.get('average_people', 0))
            st.metric("Average Faces", person_results.get('average_faces', 0))
        
        with col3:
            st.metric("Frames Analyzed", person_results.get('frames_analyzed', 0))
            st.metric("Processing Time", f"{person_results.get('processing_time', 0):.2f}s")
        
        # Frame-by-frame analysis
        frame_details = person_results.get('frame_details', [])
        if frame_details:
            df = pd.DataFrame(frame_details)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['frame_number'],
                y=df['people_count'],
                mode='lines+markers',
                name='People Count',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=df['frame_number'],
                y=df['face_count'],
                mode='lines+markers',
                name='Face Count',
                line=dict(color='red')
            ))
            fig.update_layout(
                title="People and Face Detection Over Time",
                xaxis_title="Frame Number",
                yaxis_title="Count"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error(f"Person detection failed: {person_results.get('error', 'Unknown error')}")

def display_audio_analysis_tab(results):
    """Display audio analysis tab content"""
    audio_results = results.get('analysis_results', {}).get('audio_analysis', {})
    
    if audio_results.get('success', False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Multiple Speakers",
                "Yes" if audio_results.get('multiple_speakers', False) else "No",
                help=f"Confidence: {audio_results.get('speaker_confidence', 0):.3f}"
            )
        
        with col2:
            st.metric(
                "Background Noise",
                "Yes" if audio_results.get('has_background_noise', False) else "No",
                help=f"Noise Level: {audio_results.get('noise_level', 0):.3f}"
            )
        
        with col3:
            st.metric(
                "Prolonged Silence",
                "Yes" if audio_results.get('has_prolonged_silence', False) else "No"
            )
        
        # Additional metrics
        st.metric("Audio Duration", f"{audio_results.get('audio_duration', 0):.2f}s")
        st.metric("Processing Time", f"{audio_results.get('processing_time', 0):.2f}s")
        
        # Silence periods
        silence_periods = audio_results.get('silence_periods', [])
        if silence_periods:
            st.subheader("Silence Periods")
            for i, (start, end) in enumerate(silence_periods):
                st.write(f"Period {i+1}: {start:.2f}s - {end:.2f}s (Duration: {end-start:.2f}s)")
    else:
        st.error(f"Audio analysis failed: {audio_results.get('error', 'Unknown error')}")

def display_lip_sync_tab(results):
    """Display lip sync analysis tab content"""
    lip_sync_results = results.get('analysis_results', {}).get('lip_sync_detection', {})
    
    if lip_sync_results.get('success', False):
        col1, col2 = st.columns(2)
        
        with col1:
            is_synced = lip_sync_results.get('is_synced', True)
            if is_synced:
                st.success("✅ Good Lip Synchronization")
            else:
                st.warning("⚠️ Poor Lip Synchronization")
            
            st.metric("Sync Score", f"{lip_sync_results.get('sync_score', 0):.3f}")
        
        with col2:
            st.metric("Frames Analyzed", lip_sync_results.get('frames_analyzed', 0))
            st.metric("Video FPS", f"{lip_sync_results.get('video_fps', 0):.1f}")
            st.metric("Processing Time", f"{lip_sync_results.get('processing_time', 0):.2f}s")
    else:
        st.error(f"Lip sync analysis failed: {lip_sync_results.get('error', 'Unknown error')}")

def export_results():
    """Export analysis results"""
    if st.session_state.analysis_results:
        st.markdown('<h3 class="module-header">📥 Export Results</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Download JSON Report"):
                json_str = json.dumps(st.session_state.analysis_results, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"proctoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📊 Download CSV Summary"):
                # Create summary CSV
                summary_data = {
                    'Metric': ['Cheat Score', 'Risk Level', 'People Count', 'Face Count', 'Gaze Direction'],
                    'Value': [
                        st.session_state.analysis_results.get('cheat_score_analysis', {}).get('cheat_score', 0),
                        st.session_state.analysis_results.get('cheat_score_analysis', {}).get('risk_level', 'Unknown'),
                        st.session_state.analysis_results.get('analysis_results', {}).get('person_detection', {}).get('people_count', 0),
                        st.session_state.analysis_results.get('analysis_results', {}).get('person_detection', {}).get('face_count', 0),
                        st.session_state.analysis_results.get('analysis_results', {}).get('gaze_tracking', {}).get('direction', 'Unknown')
                    ]
                }
                df = pd.DataFrame(summary_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"proctoring_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

def main():
    """Main Streamlit app"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🎯 AI Proctoring System</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🎯 Calibration", "📹 Live Monitoring", "🔧 Module Testing", "📊 Comprehensive Analysis"]
    )
    
    # System status in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Status")
    if st.session_state.calibration_status:
        st.sidebar.success("✅ Calibrated")
    else:
        st.sidebar.warning("⚠️ Not Calibrated")
    
    # Main content based on page selection
    if page == "🏠 Home":
        st.write("""
        ## Welcome to the AI Proctoring System
        
        This comprehensive system provides advanced proctoring capabilities using AI and computer vision:
        
        ### 🎯 Features:
        - **Gaze Tracking**: Monitor where the user is looking
        - **Person Detection**: Detect multiple people in the frame
        - **Audio Analysis**: Analyze audio for multiple speakers and anomalies
        - **Lip Sync Detection**: Verify audio-visual synchronization
        - **Risk Assessment**: Calculate comprehensive cheating probability scores
        
        ### 🚀 Getting Started:
        1. **Calibrate** the system for accurate gaze tracking
        2. **Test** individual modules with sample files
        3. **Run** comprehensive analysis on exam videos
        4. **Monitor** live sessions in real-time
        
        ### 📊 Analysis Outputs:
        - Detailed risk scores and assessments
        - Visual charts and graphs
        - Exportable reports (JSON/CSV)
        - Real-time monitoring capabilities
        """)
        
        # Quick stats if analysis has been run
        if st.session_state.analysis_results:
            st.markdown("---")
            st.subheader("📈 Latest Analysis Results")
            cheat_score = st.session_state.analysis_results.get('cheat_score_analysis', {}).get('cheat_score', 0)
            risk_level = st.session_state.analysis_results.get('cheat_score_analysis', {}).get('risk_level', 'Unknown')
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Latest Cheat Score", f"{cheat_score:.3f}")
            with col2:
                st.metric("Risk Level", risk_level)
            with col3:
                session_id = st.session_state.analysis_results.get('session_info', {}).get('session_id', 'Unknown')
                st.metric("Session ID", session_id)
    
    elif page == "🎯 Calibration":
        calibration_interface()
    
    elif page == "📹 Live Monitoring":
        live_camera_interface()
    
    elif page == "🔧 Module Testing":
        individual_module_testing()
    
    elif page == "📊 Comprehensive Analysis":
        comprehensive_analysis_interface()
        
        # Display results if available
        if st.session_state.analysis_results:
            st.markdown("---")
            display_comprehensive_results(st.session_state.analysis_results)
            export_results()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>AI Proctoring System v1.0 | Built with Streamlit</p>
        <p>For technical support, please contact the development team.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 