import cv2
import numpy as np
import mediapipe as mp
import face_recognition
import librosa
import soundfile as sf
import tempfile
import os
from scipy.spatial import distance
from scipy import signal
from flask import Blueprint, request, jsonify
import time
from pydub import AudioSegment

# Initialize blueprint for Flask integration
lip_sync_blueprint = Blueprint('lip_sync', __name__)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Define improved lip landmark indices for MediaPipe Face Mesh
# More precise lip contour points
UPPER_LIP_OUTER = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
LOWER_LIP_OUTER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 324]
UPPER_LIP_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324]
LOWER_LIP_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 324]

# Key points for accurate lip distance measurement
LIP_TOP_CENTER = 13      # Top center of upper lip
LIP_BOTTOM_CENTER = 14   # Bottom center of lower lip
LIP_LEFT_CORNER = 61     # Left corner of mouth
LIP_RIGHT_CORNER = 291   # Right corner of mouth

def extract_lip_landmarks_mediapipe(frame):
    """
    Extract lip landmarks using MediaPipe Face Mesh with improved accuracy
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if not results.multi_face_landmarks:
        return None
    
    face_landmarks = results.multi_face_landmarks[0]
    h, w = frame.shape[:2]
    
    # Extract key lip points for more accurate measurement
    lip_points = {}
    key_points = {
        'top_center': LIP_TOP_CENTER,
        'bottom_center': LIP_BOTTOM_CENTER,
        'left_corner': LIP_LEFT_CORNER,
        'right_corner': LIP_RIGHT_CORNER
    }
    
    for name, idx in key_points.items():
        landmark = face_landmarks.landmark[idx]
        lip_points[name] = (int(landmark.x * w), int(landmark.y * h))
    
    # Also get full lip contour for advanced analysis
    outer_lip = []
    for idx in UPPER_LIP_OUTER + LOWER_LIP_OUTER:
        landmark = face_landmarks.landmark[idx]
        x, y = int(landmark.x * w), int(landmark.y * h)
        outer_lip.append((x, y))
    
    return {
        'key_points': lip_points,
        'outer_contour': np.array(outer_lip)
    }

def calculate_lip_distance(landmarks):
    """
    Calculate accurate lip distance using key landmark points
    """
    if landmarks is None or 'key_points' not in landmarks:
        return 0
    
    key_points = landmarks['key_points']
    
    # Calculate vertical distance between lip centers
    top_center = key_points['top_center']
    bottom_center = key_points['bottom_center']
    
    vertical_distance = distance.euclidean(top_center, bottom_center)
    
    # Calculate horizontal width for normalization
    left_corner = key_points['left_corner']
    right_corner = key_points['right_corner']
    horizontal_width = distance.euclidean(left_corner, right_corner)
    
    # Normalize by mouth width to account for different face sizes
    if horizontal_width > 0:
        normalized_distance = (vertical_distance / horizontal_width) * 100
    else:
        normalized_distance = vertical_distance
    
    return normalized_distance

def calculate_lip_area(landmarks):
    """
    Calculate the area inside the lip contour for more comprehensive analysis
    """
    if landmarks is None or 'outer_contour' not in landmarks:
        return 0
    
    contour = landmarks['outer_contour']
    if len(contour) < 3:
        return 0
    
    # Calculate area using the shoelace formula
    area = 0.5 * abs(sum(x0*y1 - x1*y0 for ((x0, y0), (x1, y1)) in zip(contour, contour[1:] + contour[:1])))
    return area

def detect_speech_from_audio(audio_path, frame_rate=30):
    """
    Advanced speech detection using multiple audio features
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050)
        
        # Calculate frame duration
        frame_duration = 1.0 / frame_rate
        hop_length = int(sr * frame_duration)
        
        # Extract multiple audio features
        # 1. RMS Energy
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        
        # 2. Spectral Centroid (brightness)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        
        # 3. Zero Crossing Rate (speech vs silence)
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
        
        # 4. MFCC features (speech characteristics)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=4, hop_length=hop_length)
        mfcc_mean = np.mean(mfccs, axis=0)
        
        # Combine features for better speech detection
        # Normalize features
        rms_norm = (rms - np.mean(rms)) / (np.std(rms) + 1e-8)
        centroid_norm = (spectral_centroid - np.mean(spectral_centroid)) / (np.std(spectral_centroid) + 1e-8)
        zcr_norm = (zcr - np.mean(zcr)) / (np.std(zcr) + 1e-8)
        
        # Adaptive thresholding
        rms_threshold = np.percentile(rms, 30)  # Bottom 30% is likely silence
        centroid_threshold = np.percentile(spectral_centroid, 25)
        zcr_threshold = np.percentile(zcr, 40)
        
        # Speech activity detection
        speech_activity = []
        for i in range(len(rms)):
            # Multiple criteria for speech detection
            energy_speech = rms[i] > rms_threshold
            spectral_speech = spectral_centroid[i] > centroid_threshold
            zcr_speech = zcr[i] > zcr_threshold
            
            # Combine criteria (at least 2 out of 3 should indicate speech)
            speech_score = sum([energy_speech, spectral_speech, zcr_speech])
            is_speech = speech_score >= 2
            
            speech_activity.append({
                'is_speech': is_speech,
                'energy': rms[i],
                'spectral_centroid': spectral_centroid[i],
                'zcr': zcr[i],
                'confidence': speech_score / 3.0
            })
        
        return speech_activity
        
    except Exception as e:
        print(f"Error in speech detection: {e}")
        return []

def analyze_lip_movement_advanced(video_path):
    """
    Advanced lip movement analysis with temporal smoothing
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return []
    
    lip_data = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        landmarks = extract_lip_landmarks_mediapipe(frame)
        
        if landmarks is not None:
            lip_distance = calculate_lip_distance(landmarks)
            lip_area = calculate_lip_area(landmarks)
            
            lip_data.append({
                'frame': frame_count,
                'distance': lip_distance,
                'area': lip_area,
                'landmarks_detected': True
            })
        else:
            # No face detected - interpolate or use zero
            lip_data.append({
                'frame': frame_count,
                'distance': 0,
                'area': 0,
                'landmarks_detected': False
            })
    
    cap.release()
    
    # Apply temporal smoothing to reduce noise
    if len(lip_data) > 0:
        distances = [d['distance'] for d in lip_data]
        areas = [d['area'] for d in lip_data]
        
        # Smooth the signals
        if len(distances) > 5:
            window_size = min(5, len(distances) // 4)
            distances_smooth = signal.savgol_filter(distances, window_size, 2)
            areas_smooth = signal.savgol_filter(areas, window_size, 2)
            
            for i, data in enumerate(lip_data):
                data['distance_smooth'] = distances_smooth[i]
                data['area_smooth'] = areas_smooth[i]
        else:
            for data in lip_data:
                data['distance_smooth'] = data['distance']
                data['area_smooth'] = data['area']
    
    return lip_data

def calculate_sync_correlation(lip_data, speech_data):
    """
    Calculate synchronization correlation with advanced temporal analysis
    """
    if not lip_data or not speech_data:
        return {'correlation': 0, 'lag': 0, 'confidence': 0}
    
    min_length = min(len(lip_data), len(speech_data))
    
    # Extract signals
    lip_signal = np.array([d['distance_smooth'] for d in lip_data[:min_length]])
    speech_signal = np.array([d['energy'] for d in speech_data[:min_length]])
    
    # Handle edge cases
    if len(lip_signal) < 10 or len(speech_signal) < 10:
        return {'correlation': 0, 'lag': 0, 'confidence': 0}
    
    # Normalize signals
    if np.std(lip_signal) > 0:
        lip_signal = (lip_signal - np.mean(lip_signal)) / np.std(lip_signal)
    if np.std(speech_signal) > 0:
        speech_signal = (speech_signal - np.mean(speech_signal)) / np.std(speech_signal)
    
    # Calculate cross-correlation with different lags
    max_lag = min(10, len(lip_signal) // 4)  # Maximum 10 frames or 25% of signal
    correlations = []
    lags = range(-max_lag, max_lag + 1)
    
    for lag in lags:
        try:
            if lag < 0:
                # Speech leads lip movement
                lip_seg = lip_signal[:lag]
                speech_seg = speech_signal[-lag:]
            elif lag > 0:
                # Lip movement leads speech
                lip_seg = lip_signal[lag:]
                speech_seg = speech_signal[:-lag]
            else:
                # No lag
                lip_seg = lip_signal
                speech_seg = speech_signal
            
            if len(lip_seg) > 5 and len(speech_seg) > 5:
                corr = np.corrcoef(lip_seg, speech_seg)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
                else:
                    correlations.append(0)
            else:
                correlations.append(0)
        except:
            correlations.append(0)
    
    # Find best correlation
    if correlations:
        best_corr_idx = np.argmax(np.abs(correlations))
        best_correlation = correlations[best_corr_idx]
        best_lag = lags[best_corr_idx]
        
        # Calculate confidence based on correlation strength and consistency
        confidence = abs(best_correlation)
        
        return {
            'correlation': best_correlation,
            'lag': best_lag,
            'confidence': confidence,
            'all_correlations': correlations
        }
    else:
        return {'correlation': 0, 'lag': 0, 'confidence': 0}

def is_lip_synced(video_path, audio_path=None, threshold=0.3):
    """
    Improved lip sync detection with comprehensive analysis
    """
    # Extract audio if not provided
    if audio_path is None:
        audio_path = extract_audio_from_video(video_path)
        audio_extracted = True
    else:
        audio_extracted = False
    
    try:
        # Get video properties
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        # Analyze video for lip movement
        print("Analyzing lip movement...")
        lip_data = analyze_lip_movement_advanced(video_path)
        
        # Analyze audio for speech
        print("Analyzing speech patterns...")
        speech_data = detect_speech_from_audio(audio_path, frame_rate=fps)
        
        # Calculate synchronization
        print("Calculating synchronization...")
        sync_result = calculate_sync_correlation(lip_data, speech_data)
        
        # Determine if lip-synced based on correlation and confidence
        correlation = sync_result['correlation']
        confidence = sync_result['confidence']
        
        # More nuanced sync detection
        is_synced = False
        sync_quality = "Poor"
        
        if confidence > 0.6 and abs(correlation) > threshold:
            is_synced = True
            if abs(correlation) > 0.7:
                sync_quality = "Excellent"
            elif abs(correlation) > 0.5:
                sync_quality = "Good"
            else:
                sync_quality = "Fair"
        elif confidence > 0.4 and abs(correlation) > threshold * 0.8:
            is_synced = True
            sync_quality = "Fair"
        
        # Additional checks for very short videos
        if duration < 2.0 and len(lip_data) > 0:
            # For short videos, be more lenient
            avg_lip_movement = np.mean([d['distance'] for d in lip_data])
            if avg_lip_movement > 5:  # Some lip movement detected
                is_synced = True
                sync_quality = "Fair"
        
        result = {
            "is_lip_synced": is_synced,
            "sync_quality": sync_quality,
            "correlation": float(correlation),
            "confidence": float(confidence),
            "lag_frames": int(sync_result['lag']),
            "frames_analyzed": len(lip_data),
            "speech_frames": len(speech_data),
            "video_fps": float(fps),
            "duration": float(duration),
            "analysis_details": {
                "avg_lip_distance": float(np.mean([d['distance'] for d in lip_data])) if lip_data else 0,
                "lip_movement_variance": float(np.var([d['distance'] for d in lip_data])) if lip_data else 0,
                "speech_activity_ratio": float(np.mean([d['is_speech'] for d in speech_data])) if speech_data else 0,
                "landmarks_detection_rate": float(np.mean([d['landmarks_detected'] for d in lip_data])) if lip_data else 0
            }
        }
        
        return result
        
    except Exception as e:
        return {
            "is_lip_synced": False,
            "sync_quality": "Error",
            "correlation": 0.0,
            "confidence": 0.0,
            "error": str(e)
        }
        
    finally:
        # Clean up temporary audio file if it was extracted
        if audio_extracted and audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except:
                pass

def extract_audio_from_video(video_path):
    """
    Extract audio from video file using pydub
    """
    temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_audio.close()
    
    try:
        video = AudioSegment.from_file(video_path)
        video.export(temp_audio.name, format="wav")
        return temp_audio.name
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

@lip_sync_blueprint.route('/lip_sync', methods=['POST'])
def lip_sync_endpoint():
    """
    Flask endpoint to analyze lip sync from uploaded video
    """
    if 'video' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No video file uploaded'
        }), 400
    
    video_file = request.files['video']
    
    # Check if separate audio was uploaded
    audio_file = None
    audio_path = None
    if 'audio' in request.files and request.files['audio'].filename:
        audio_file = request.files['audio']
    
    # Save uploaded files to temporary locations
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    video_file.save(temp_video.name)
    temp_video.close()
    
    if audio_file:
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        audio_file.save(temp_audio.name)
        temp_audio.close()
        audio_path = temp_audio.name
    
    try:
        start_time = time.time()
        results = is_lip_synced(temp_video.name, audio_path)
        processing_time = time.time() - start_time
        
        results['processing_time'] = round(processing_time, 2)
        
        return jsonify({
            'success': True,
            'lip_sync_analysis': results
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error analyzing lip sync: {str(e)}'
        }), 500
    
    finally:
        # Clean up temporary files
        try:
            os.unlink(temp_video.name)
            if audio_path:
                os.unlink(audio_path)
        except:
            pass

# This module can be used standalone or integrated with Flask
# For Flask integration: app.register_blueprint(lip_sync_blueprint)