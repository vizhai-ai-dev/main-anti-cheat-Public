#!/usr/bin/env python3
"""
Master Orchestrator Module - run_all.py

This module orchestrates all proctoring checks on a single video/audio input:
- Gaze tracking analysis
- Lip sync detection
- Multiple person detection
- Audio analysis
- Cheat score calculation

The script processes video and audio files and outputs a comprehensive JSON report.
"""

import os
import sys
import json
import time
import tempfile
import argparse
import logging
from typing import Dict, Any, Optional, Tuple
import cv2
import librosa
import numpy as np

# Import analysis modules
from gaze_tracking import analyze_video_gaze, get_gaze_direction
from lip_sync_detector import is_lip_synced, extract_audio_from_video
from multi_person import detect_multiple_people
from audio_analysis import AudioAnalyzer
from cheat_score import calculate_cheat_score, get_risk_level, get_score_trend

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProctorAnalyzer:
    """Main class for orchestrating all proctoring analyses"""
    
    def __init__(self, session_id: Optional[str] = None):
        """Initialize the analyzer
        
        Args:
            session_id: Optional session identifier for tracking
        """
        self.session_id = session_id or f"session_{int(time.time())}"
        self.audio_analyzer = AudioAnalyzer()
        self.temp_files = []  # Track temporary files for cleanup
    
    def cleanup_temp_files(self):
        """Clean up any temporary files created during analysis"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {temp_file}: {e}")
        self.temp_files.clear()
    
    def extract_video_frames_for_person_detection(self, video_path: str, max_frames: int = 10) -> list:
        """Extract frames from video for person detection analysis
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to analyze
            
        Returns:
            List of person detection results for each frame
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return []
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, frame_count // max_frames)
        
        person_results = []
        frames_processed = 0
        
        for i in range(0, frame_count, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            try:
                # Analyze frame for people/faces
                result = detect_multiple_people(frame)
                person_results.append({
                    'frame_number': i,
                    'people_count': result['people_count'],
                    'face_count': result['face_count']
                })
                frames_processed += 1
                
                if frames_processed >= max_frames:
                    break
                    
            except Exception as e:
                logger.warning(f"Error processing frame {i}: {e}")
                continue
        
        cap.release()
        return person_results
    
    def analyze_gaze_tracking(self, video_path: str) -> Dict[str, Any]:
        """Perform gaze tracking analysis on video
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with gaze analysis results
        """
        logger.info("Starting gaze tracking analysis...")
        start_time = time.time()
        
        try:
            # Analyze video for gaze patterns
            gaze_results = analyze_video_gaze(video_path, max_frames=30)
            
            if 'error' in gaze_results:
                logger.error(f"Gaze analysis error: {gaze_results['error']}")
                return {
                    'success': False,
                    'error': gaze_results['error'],
                    'direction': 'unknown',
                    'processing_time': time.time() - start_time
                }
            
            # Calculate suspicious patterns based on gaze distribution
            gaze_counts = gaze_results.get('gaze_counts', {})
            total_frames = sum(gaze_counts.values())
            
            # Calculate off-screen duration (approximate)
            off_screen_frames = gaze_counts.get('left', 0) + gaze_counts.get('right', 0) + gaze_counts.get('down', 0)
            off_screen_ratio = off_screen_frames / total_frames if total_frames > 0 else 0
            
            # Estimate off-screen duration in seconds (assuming ~1 frame per second sampling)
            video_stats = gaze_results.get('video_stats', {})
            duration = video_stats.get('duration', 30)
            frames_processed = video_stats.get('frames_processed', 30)
            time_per_frame = duration / frames_processed if frames_processed > 0 else 1
            off_screen_duration = off_screen_frames * time_per_frame
            
            # Count suspicious patterns (frequent direction changes, excessive looking away)
            suspicious_patterns = 0
            if off_screen_ratio > 0.3:  # More than 30% looking away
                suspicious_patterns += 1
            if gaze_counts.get('unknown', 0) > total_frames * 0.2:  # More than 20% undetected
                suspicious_patterns += 1
            
            result = {
                'success': True,
                'direction': gaze_results.get('dominant_direction', 'unknown'),
                'gaze_distribution': gaze_counts,
                'off_screen_duration': round(off_screen_duration, 2),
                'suspicious_patterns': suspicious_patterns,
                'face_detection_rate': gaze_results.get('video_stats', {}).get('face_detection_rate', 0),
                'processing_time': round(time.time() - start_time, 2)
            }
            
            logger.info(f"Gaze analysis completed in {result['processing_time']}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in gaze tracking analysis: {e}")
            return {
                'success': False,
                'error': str(e),
                'direction': 'unknown',
                'processing_time': time.time() - start_time
            }
    
    def analyze_lip_sync(self, video_path: str, audio_path: Optional[str] = None) -> Dict[str, Any]:
        """Perform lip sync analysis on video
        
        Args:
            video_path: Path to video file
            audio_path: Optional path to separate audio file
            
        Returns:
            Dictionary with lip sync analysis results
        """
        logger.info("Starting lip sync analysis...")
        start_time = time.time()
        
        try:
            # Perform lip sync analysis
            lip_sync_results = is_lip_synced(video_path, audio_path)
            
            result = {
                'success': True,
                'is_synced': lip_sync_results.get('is_lip_synced', True),
                'sync_score': lip_sync_results.get('confidence_score', 1.0),
                'frames_analyzed': lip_sync_results.get('frames_analyzed', 0),
                'video_fps': lip_sync_results.get('video_fps', 30.0),
                'processing_time': round(time.time() - start_time, 2)
            }
            
            logger.info(f"Lip sync analysis completed in {result['processing_time']}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in lip sync analysis: {e}")
            return {
                'success': False,
                'error': str(e),
                'is_synced': True,  # Default to synced if analysis fails
                'sync_score': 0.5,
                'processing_time': time.time() - start_time
            }
    
    def analyze_person_detection(self, video_path: str) -> Dict[str, Any]:
        """Perform person detection analysis on video
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with person detection analysis results
        """
        logger.info("Starting person detection analysis...")
        start_time = time.time()
        
        try:
            # Extract and analyze frames
            frame_results = self.extract_video_frames_for_person_detection(video_path)
            
            if not frame_results:
                return {
                    'success': False,
                    'error': 'No frames could be processed',
                    'people_count': 0,
                    'face_count': 0,
                    'processing_time': time.time() - start_time
                }
            
            # Calculate statistics across all frames
            people_counts = [r['people_count'] for r in frame_results]
            face_counts = [r['face_count'] for r in frame_results]
            
            # Use the most common counts (mode) or average
            avg_people = np.mean(people_counts)
            avg_faces = np.mean(face_counts)
            max_people = max(people_counts)
            max_faces = max(face_counts)
            
            # Determine final counts (use max to be conservative about multiple people)
            final_people_count = int(max_people)
            final_face_count = int(max_faces)
            
            result = {
                'success': True,
                'people_count': final_people_count,
                'face_count': final_face_count,
                'average_people': round(avg_people, 2),
                'average_faces': round(avg_faces, 2),
                'frames_analyzed': len(frame_results),
                'frame_details': frame_results,
                'processing_time': round(time.time() - start_time, 2)
            }
            
            logger.info(f"Person detection completed in {result['processing_time']}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in person detection analysis: {e}")
            return {
                'success': False,
                'error': str(e),
                'people_count': 1,  # Default assumption
                'face_count': 1,
                'processing_time': time.time() - start_time
            }
    
    def analyze_audio(self, video_path: str, audio_path: Optional[str] = None) -> Dict[str, Any]:
        """Perform audio analysis
        
        Args:
            video_path: Path to video file
            audio_path: Optional path to separate audio file
            
        Returns:
            Dictionary with audio analysis results
        """
        logger.info("Starting audio analysis...")
        start_time = time.time()
        
        try:
            # Extract audio if not provided separately
            if audio_path is None:
                audio_path = extract_audio_from_video(video_path)
                self.temp_files.append(audio_path)  # Track for cleanup
            
            # Load and analyze audio
            audio_data, sample_rate = librosa.load(audio_path, sr=22050, mono=True)
            
            # Perform comprehensive audio analysis
            analysis_results = self.audio_analyzer.analyze_audio(audio_data)
            
            # Extract key information for cheat score calculation
            multiple_speakers = analysis_results.get('multiple_speakers', {})
            background_noise = analysis_results.get('background_noise', {})
            prolonged_silence = analysis_results.get('prolonged_silence', {})
            
            result = {
                'success': True,
                'multiple_speakers': multiple_speakers.get('detected', False),
                'speaker_confidence': multiple_speakers.get('confidence', 0.0),
                'has_background_noise': background_noise.get('significant_noise', False),
                'noise_level': background_noise.get('noise_level', 0.0),
                'has_prolonged_silence': prolonged_silence.get('detected', False),
                'silence_periods': [(p['start'], p['end']) for p in prolonged_silence.get('periods', [])],
                'audio_duration': analysis_results.get('audio_duration', 0.0),
                'anomalies_detected': analysis_results.get('anomalies_detected', False),
                'full_analysis': analysis_results,
                'processing_time': round(time.time() - start_time, 2)
            }
            
            logger.info(f"Audio analysis completed in {result['processing_time']}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in audio analysis: {e}")
            return {
                'success': False,
                'error': str(e),
                'multiple_speakers': False,
                'speaker_confidence': 0.0,
                'has_background_noise': False,
                'noise_level': 0.0,
                'has_prolonged_silence': False,
                'silence_periods': [],
                'processing_time': time.time() - start_time
            }
    
    def run_comprehensive_analysis(self, video_path: str, audio_path: Optional[str] = None) -> Dict[str, Any]:
        """Run all analyses and compile results with cheat score
        
        Args:
            video_path: Path to video file
            audio_path: Optional path to separate audio file
            
        Returns:
            Comprehensive analysis results with cheat score
        """
        logger.info(f"Starting comprehensive analysis for session: {self.session_id}")
        overall_start_time = time.time()
        
        try:
            # Validate input files
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            if audio_path and not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Run all analyses
            gaze_results = self.analyze_gaze_tracking(video_path)
            lip_sync_results = self.analyze_lip_sync(video_path, audio_path)
            person_results = self.analyze_person_detection(video_path)
            audio_results = self.analyze_audio(video_path, audio_path)
            
            # Prepare data for cheat score calculation
            cheat_score_inputs = {
                'gaze_data': {
                    'direction': gaze_results.get('direction', 'unknown'),
                    'off_screen_duration': gaze_results.get('off_screen_duration', 0),
                    'suspicious_patterns': gaze_results.get('suspicious_patterns', 0)
                },
                'lip_sync_data': {
                    'is_synced': lip_sync_results.get('is_synced', True),
                    'sync_score': lip_sync_results.get('sync_score', 1.0)
                },
                'person_data': {
                    'people_count': person_results.get('people_count', 1),
                    'face_count': person_results.get('face_count', 1)
                },
                'audio_data': {
                    'multiple_speakers': audio_results.get('multiple_speakers', False),
                    'speaker_confidence': audio_results.get('speaker_confidence', 0.0),
                    'has_background_noise': audio_results.get('has_background_noise', False),
                    'noise_level': audio_results.get('noise_level', 0.0),
                    'has_prolonged_silence': audio_results.get('has_prolonged_silence', False),
                    'silence_periods': audio_results.get('silence_periods', [])
                },
                'timestamp': time.time(),
                'session_id': self.session_id
            }
            
            # Calculate cheat score
            cheat_score = calculate_cheat_score(cheat_score_inputs)
            risk_level = get_risk_level(cheat_score)
            trend_analysis = get_score_trend()
            
            # Compile comprehensive results
            comprehensive_results = {
                'session_info': {
                    'session_id': self.session_id,
                    'timestamp': time.time(),
                    'video_path': video_path,
                    'audio_path': audio_path,
                    'total_processing_time': round(time.time() - overall_start_time, 2)
                },
                'analysis_results': {
                    'gaze_tracking': gaze_results,
                    'lip_sync_detection': lip_sync_results,
                    'person_detection': person_results,
                    'audio_analysis': audio_results
                },
                'cheat_score_analysis': {
                    'cheat_score': cheat_score,
                    'risk_level': risk_level,
                    'trend_analysis': trend_analysis,
                    'input_data': cheat_score_inputs
                },
                'summary': {
                    'overall_risk': risk_level,
                    'primary_concerns': self._identify_primary_concerns(cheat_score_inputs, cheat_score),
                    'recommendations': self._generate_recommendations(cheat_score_inputs, cheat_score)
                }
            }
            
            logger.info(f"Comprehensive analysis completed in {comprehensive_results['session_info']['total_processing_time']}s")
            logger.info(f"Final cheat score: {cheat_score:.3f} ({risk_level})")
            
            return comprehensive_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return {
                'session_info': {
                    'session_id': self.session_id,
                    'timestamp': time.time(),
                    'error': str(e),
                    'total_processing_time': time.time() - overall_start_time
                },
                'analysis_results': {},
                'cheat_score_analysis': {
                    'cheat_score': 0.5,
                    'risk_level': 'Medium',
                    'error': 'Analysis failed'
                }
            }
        
        finally:
            # Clean up temporary files
            self.cleanup_temp_files()
    
    def _identify_primary_concerns(self, inputs: Dict[str, Any], score: float) -> list:
        """Identify the primary concerns based on analysis results"""
        concerns = []
        
        # Check gaze issues
        gaze_data = inputs.get('gaze_data', {})
        if gaze_data.get('direction') in ['left', 'right', 'down']:
            concerns.append(f"Suspicious gaze direction: {gaze_data.get('direction')}")
        if gaze_data.get('off_screen_duration', 0) > 10:
            concerns.append(f"Extended off-screen time: {gaze_data.get('off_screen_duration')}s")
        
        # Check person detection issues
        person_data = inputs.get('person_data', {})
        if person_data.get('people_count', 1) > 1:
            concerns.append(f"Multiple people detected: {person_data.get('people_count')}")
        
        # Check audio issues
        audio_data = inputs.get('audio_data', {})
        if audio_data.get('multiple_speakers'):
            concerns.append("Multiple speakers detected")
        if audio_data.get('has_background_noise'):
            concerns.append(f"Significant background noise: {audio_data.get('noise_level', 0):.2f}")
        
        # Check lip sync issues
        lip_sync_data = inputs.get('lip_sync_data', {})
        if not lip_sync_data.get('is_synced', True):
            concerns.append(f"Poor lip synchronization: {lip_sync_data.get('sync_score', 0):.2f}")
        
        return concerns
    
    def _generate_recommendations(self, inputs: Dict[str, Any], score: float) -> list:
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        if score > 0.7:
            recommendations.append("High risk detected - immediate review recommended")
        elif score > 0.5:
            recommendations.append("Medium risk detected - manual verification suggested")
        
        # Specific recommendations based on issues
        gaze_data = inputs.get('gaze_data', {})
        if gaze_data.get('off_screen_duration', 0) > 5:
            recommendations.append("Monitor for unauthorized materials or assistance")
        
        person_data = inputs.get('person_data', {})
        if person_data.get('people_count', 1) > 1:
            recommendations.append("Verify identity and ensure no unauthorized assistance")
        
        audio_data = inputs.get('audio_data', {})
        if audio_data.get('multiple_speakers'):
            recommendations.append("Check for unauthorized communication or assistance")
        
        return recommendations

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Run comprehensive proctoring analysis')
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('--audio', help='Path to separate audio file (optional)')
    parser.add_argument('--output', help='Output JSON file path (optional)')
    parser.add_argument('--session-id', help='Session identifier (optional)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize analyzer
    analyzer = ProctorAnalyzer(session_id=args.session_id)
    
    try:
        # Run comprehensive analysis
        results = analyzer.run_comprehensive_analysis(args.video_path, args.audio)
        
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {args.output}")
        else:
            print(json.dumps(results, indent=2))
        
        # Print summary
        cheat_score = results.get('cheat_score_analysis', {}).get('cheat_score', 0)
        risk_level = results.get('cheat_score_analysis', {}).get('risk_level', 'Unknown')
        print(f"\n=== ANALYSIS SUMMARY ===")
        print(f"Cheat Score: {cheat_score:.3f}")
        print(f"Risk Level: {risk_level}")
        
        concerns = results.get('summary', {}).get('primary_concerns', [])
        if concerns:
            print(f"Primary Concerns:")
            for concern in concerns:
                print(f"  - {concern}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 