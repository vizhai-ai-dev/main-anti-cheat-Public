#!/usr/bin/env python3
"""
API Server Startup Script

Simple script to start the AI Proctoring API server with dependency checking.
"""

import sys
import subprocess
import importlib

def check_dependencies():
    """Check if required dependencies are installed"""
    required_modules = [
        'flask',
        'flask_cors',
        'cv2',
        'numpy',
        'requests'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            importlib.import_module(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print("❌ Missing required dependencies:")
        for module in missing_modules:
            print(f"   - {module}")
        print("\n💡 Install missing dependencies with:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main startup function"""
    print("🚀 AI Proctoring API Server - Starting...")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    print("✅ All dependencies are installed")
    
    # Import and start the API server
    try:
        from api_server import app
        print("📡 Starting Flask API server...")
        print("🌐 Server will be available at: http://localhost:5001")
        print("📖 API Documentation: See API_DOCUMENTATION.md")
        print("⏹️  Press Ctrl+C to stop the server\n")
        
        app.run(debug=True, host='0.0.0.0', port=5001)
        
    except KeyboardInterrupt:
        print("\n⏹️  Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 