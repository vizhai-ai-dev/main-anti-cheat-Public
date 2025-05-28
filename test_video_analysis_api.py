#!/usr/bin/env python3
"""
Test Script for Comprehensive Video Analysis API

This script demonstrates how to use the new /api/analyze/video_comprehensive endpoint
to perform complete proctoring analysis on a video file.
"""

import requests
import json
import time
import os
from typing import Optional

class VideoAnalysisAPIClient:
    """Client for the comprehensive video analysis API"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        """Initialize the API client
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> dict:
        """Check if the API server is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "healthy": False}
    
    def analyze_video_comprehensive(
        self,
        video_path: str,
        audio_path: Optional[str] = None,
        session_id: Optional[str] = None,
        calibrate: bool = False,
        screen_width: int = 1920,
        screen_height: int = 1080
    ) -> dict:
        """
        Perform comprehensive video analysis
        
        Args:
            video_path: Path to the video file
            audio_path: Optional path to separate audio file
            session_id: Optional session identifier
            calibrate: Whether to perform calibration
            screen_width: Screen width for calibration
            screen_height: Screen height for calibration
            
        Returns:
            Analysis results dictionary
        """
        if not os.path.exists(video_path):
            return {"error": f"Video file not found: {video_path}"}
        
        if audio_path and not os.path.exists(audio_path):
            return {"error": f"Audio file not found: {audio_path}"}
        
        # Prepare files
        files = {}
        with open(video_path, 'rb') as video_file:
            files['video'] = video_file
            
            if audio_path:
                with open(audio_path, 'rb') as audio_file:
                    files['audio'] = audio_file
            
            # Prepare form data
            data = {
                'calibrate': str(calibrate).lower(),
                'screen_width': str(screen_width),
                'screen_height': str(screen_height)
            }
            
            if session_id:
                data['session_id'] = session_id
            
            try:
                print(f"🚀 Starting comprehensive analysis of: {video_path}")
                print(f"📊 Calibration: {'Enabled' if calibrate else 'Disabled'}")
                print(f"🖥️  Screen resolution: {screen_width}x{screen_height}")
                
                start_time = time.time()
                
                response = self.session.post(
                    f"{self.base_url}/api/analyze/video_comprehensive",
                    files=files,
                    data=data,
                    timeout=300  # 5 minute timeout
                )
                
                processing_time = time.time() - start_time
                print(f"⏱️  Request completed in {processing_time:.2f}s")
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.Timeout:
                return {"error": "Request timed out (>5 minutes)"}
            except requests.exceptions.RequestException as e:
                return {"error": f"Request failed: {str(e)}"}
            except Exception as e:
                return {"error": f"Unexpected error: {str(e)}"}
    
    def print_analysis_results(self, results: dict):
        """Print analysis results in a formatted way"""
        if not results.get("success"):
            print(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")
            return
        
        data = results.get("results", {})
        
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE VIDEO ANALYSIS RESULTS")
        print("="*80)
        
        # Session info
        session_info = data.get("session_info", {})
        print(f"\n📋 Session Information:")
        print(f"   Session ID: {session_info.get('session_id', 'N/A')}")
        print(f"   Timestamp: {session_info.get('timestamp', 'N/A')}")
        print(f"   Video: {session_info.get('video_filename', 'N/A')}")
        print(f"   Audio: {session_info.get('audio_filename', 'N/A')}")
        print(f"   Calibration: {session_info.get('calibration_performed', False)}")
        print(f"   Screen Resolution: {session_info.get('screen_resolution', 'N/A')}")
        
        # Calibration results
        if data.get("calibration_results"):
            cal_results = data["calibration_results"]
            print(f"\n🎯 Calibration Results:")
            print(f"   Gaze Calibration: {cal_results.get('gaze_calibration', {}).get('status', 'N/A')}")
            print(f"   Lip Sync Calibration: {cal_results.get('lip_sync_calibration', {}).get('status', 'N/A')}")
            print(f"   Calibration Time: {cal_results.get('calibration_time', 0)}s")
        
        # Cheat score analysis
        cheat_analysis = data.get("cheat_score_analysis", {})
        print(f"\n🚨 CHEAT SCORE ANALYSIS:")
        print(f"   📊 Cheat Score: {cheat_analysis.get('cheat_score', 0):.3f}")
        print(f"   ⚠️  Risk Level: {cheat_analysis.get('risk_level', 'Unknown')}")
        
        # Summary
        summary = data.get("summary", {})
        overall = summary.get("overall_assessment", {})
        print(f"\n📈 Overall Assessment:")
        print(f"   Risk Level: {overall.get('risk_level', 'Unknown')}")
        print(f"   Confidence: {overall.get('confidence', 'Unknown')}")
        print(f"   Calibration Enhanced: {overall.get('calibration_enhanced', False)}")
        
        # Key findings
        key_findings = summary.get("key_findings", [])
        if key_findings:
            print(f"\n🔍 Key Findings:")
            for finding in key_findings:
                print(f"   • {finding}")
        
        # Recommendations
        recommendations = summary.get("recommendations", [])
        if recommendations:
            print(f"\n💡 Recommendations:")
            for rec in recommendations:
                print(f"   • {rec}")
        
        # Analysis quality
        quality = summary.get("analysis_quality", {})
        if quality:
            print(f"\n📊 Analysis Quality:")
            print(f"   Gaze Tracking: {quality.get('gaze_tracking_quality', 0):.2f}")
            print(f"   Lip Sync Confidence: {quality.get('lip_sync_confidence', 0):.2f}")
            print(f"   Person Detection: {quality.get('person_detection_confidence', 0):.2f}")
            print(f"   Audio Analysis: {quality.get('audio_analysis_quality', 0):.2f}")
        
        # Processing info
        processing = data.get("processing_info", {})
        print(f"\n⏱️  Processing Information:")
        print(f"   Total Time: {processing.get('total_processing_time', 0)}s")
        print(f"   Analysis Time: {processing.get('analysis_time', 0)}s")
        print(f"   Modules: {', '.join(processing.get('modules_processed', []))}")
        
        print("\n" + "="*80)

def main():
    """Main function for testing the API"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test comprehensive video analysis API')
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('--audio', help='Path to separate audio file (optional)')
    parser.add_argument('--session-id', help='Session identifier (optional)')
    parser.add_argument('--calibrate', action='store_true', help='Enable calibration')
    parser.add_argument('--screen-width', type=int, default=1920, help='Screen width for calibration')
    parser.add_argument('--screen-height', type=int, default=1080, help='Screen height for calibration')
    parser.add_argument('--server', default='http://localhost:5000', help='API server URL')
    parser.add_argument('--output', help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Initialize client
    client = VideoAnalysisAPIClient(args.server)
    
    # Check server health
    print("🔍 Checking API server health...")
    health = client.health_check()
    if not health.get("status") == "healthy":
        print(f"❌ Server health check failed: {health}")
        return 1
    
    print(f"✅ Server is healthy (version: {health.get('version', 'unknown')})")
    
    # Perform analysis
    results = client.analyze_video_comprehensive(
        video_path=args.video_path,
        audio_path=args.audio,
        session_id=args.session_id,
        calibrate=args.calibrate,
        screen_width=args.screen_width,
        screen_height=args.screen_height
    )
    
    # Print results
    client.print_analysis_results(results)
    
    # Save results if requested
    if args.output:
        try:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to: {args.output}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
    
    # Return appropriate exit code
    if results.get("success"):
        cheat_score = results.get("results", {}).get("cheat_score_analysis", {}).get("cheat_score", 0)
        if cheat_score > 0.7:
            print(f"\n🚨 HIGH RISK DETECTED (Score: {cheat_score:.3f})")
            return 2
        elif cheat_score > 0.4:
            print(f"\n⚠️  MEDIUM RISK DETECTED (Score: {cheat_score:.3f})")
            return 1
        else:
            print(f"\n✅ LOW RISK (Score: {cheat_score:.3f})")
            return 0
    else:
        return 1

if __name__ == "__main__":
    exit(main()) 