#!/usr/bin/env python3
"""
Startup script for the AI Proctoring System Streamlit App

This script launches the Streamlit web interface with proper configuration.
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import streamlit
        import cv2
        import numpy
        import plotly
        import pandas
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def check_model_files():
    """Check if required model files exist"""
    required_files = [
        'yolov3.weights',
        'yolov3.cfg',
        'coco.names',
        'shape_predictor_68_face_landmarks.dat'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing model files: {missing_files}")
        print("Please run: python download_models.py")
        return False
    
    print("✅ All model files are present")
    return True

def main():
    """Main function to start the Streamlit app"""
    print("🎯 AI Proctoring System - Starting...")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check model files
    if not check_model_files():
        sys.exit(1)
    
    # Start Streamlit app
    print("🚀 Launching Streamlit app...")
    print("📱 The app will open in your default web browser")
    print("🔗 URL: http://localhost:8502")
    print("⏹️  Press Ctrl+C to stop the server")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8502",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down the AI Proctoring System")
    except Exception as e:
        print(f"❌ Error starting the app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 