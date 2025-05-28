# AI Proctoring System - Streamlit Web Interface

A comprehensive web-based interface for the AI-powered proctoring system built with Streamlit. This application provides an intuitive way to calibrate, test, and run comprehensive proctoring analysis.

## 🎯 Features

### Core Functionality
- **System Calibration**: Interactive gaze tracking calibration for improved accuracy
- **Live Camera Monitoring**: Real-time frame capture and analysis
- **Individual Module Testing**: Test each proctoring module separately
- **Comprehensive Analysis**: Run all modules on uploaded videos
- **Results Visualization**: Interactive charts and detailed analysis reports
- **Export Capabilities**: Download results as JSON or CSV files

### Proctoring Modules
- **Gaze Tracking**: Monitor eye movement and attention direction
- **Person Detection**: Detect multiple people in the frame
- **Audio Analysis**: Analyze audio for multiple speakers and anomalies
- **Lip Sync Detection**: Verify audio-visual synchronization
- **Risk Assessment**: Calculate comprehensive cheating probability scores

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Webcam (for live monitoring and calibration)
- Required model files (automatically downloaded)

### Installation

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Model Files**
   ```bash
   python download_models.py
   ```

3. **Launch the Application**
   ```bash
   python run_app.py
   ```
   
   Or directly with Streamlit:
   ```bash
   streamlit run app.py
   ```

4. **Access the Web Interface**
   - Open your browser to: http://localhost:8501
   - The app will automatically open in your default browser

## 📱 User Interface Guide

### Navigation
The app uses a sidebar navigation with the following pages:

#### 🏠 Home
- Overview of system features
- Quick access to latest analysis results
- System status information

#### 🎯 Calibration
- **Purpose**: Improve gaze tracking accuracy
- **Process**: 
  1. Click "Start Calibration"
  2. Follow on-screen instructions
  3. Look at calibration points when prompted
  4. System automatically completes calibration
- **Requirements**: Good lighting and clear face visibility

#### 📹 Live Monitoring
- **Prerequisites**: System must be calibrated first
- **Features**:
  - Real-time frame capture from webcam
  - Instant gaze direction analysis
  - Person detection
  - Quick risk assessment
- **Usage**: Click "Capture and Analyze Frame" for instant analysis

#### 🔧 Module Testing
- **Purpose**: Test individual proctoring modules
- **Supported Modules**:
  - Gaze Tracking (images/videos)
  - Lip Sync Detection (videos)
  - Person Detection (images)
  - Audio Analysis (audio/video files)
- **File Support**: MP4, AVI, MOV, WAV, MP3, JPG, PNG

#### 📊 Comprehensive Analysis
- **Purpose**: Complete proctoring analysis on exam videos
- **Features**:
  - Upload video files for analysis
  - Optional separate audio file upload
  - Runs all proctoring modules
  - Generates detailed reports
  - Interactive visualizations
  - Export capabilities

## 📈 Analysis Results

### Risk Assessment
- **Cheat Score**: 0.0 (safe) to 1.0 (high risk)
- **Risk Levels**: Very Low, Low, Medium, High, Very High
- **Color Coding**: Green (safe), Yellow (moderate), Red (high risk)

### Detailed Reports
- **Summary Tab**: Primary concerns and recommendations
- **Gaze Analysis**: Direction distribution and off-screen time
- **Person Detection**: People/face counts over time
- **Audio Analysis**: Speaker detection and noise analysis
- **Lip Sync**: Synchronization quality assessment

### Visualizations
- **Interactive Charts**: Plotly-powered visualizations
- **Time Series**: Frame-by-frame analysis
- **Distribution Charts**: Gaze direction pie charts
- **Risk Breakdown**: Category-wise risk scores

## 🔧 Configuration

### System Settings
- **Screen Resolution**: Automatically detected, can be manually set
- **Processing Parameters**: Configurable in individual modules
- **Export Formats**: JSON (detailed) and CSV (summary)

### Calibration Settings
- **Calibration Points**: 9-point grid system
- **Accuracy Threshold**: Configurable calibration quality
- **Save/Load**: Persistent calibration data

## 📁 File Structure

```
├── app.py                    # Main Streamlit application
├── run_app.py               # Application launcher
├── calibration_api.py       # Calibration functionality
├── run_all.py              # Master orchestrator
├── cheat_score.py          # Risk assessment module
├── gaze_tracking.py        # Gaze analysis module
├── lip_sync_detector.py    # Lip sync analysis module
├── multi_person.py         # Person detection module
├── audio_analysis.py       # Audio analysis module
├── requirements.txt        # Python dependencies
└── README_streamlit.md     # This file
```

## 🎮 Usage Examples

### Basic Workflow
1. **Start Application**: `python run_app.py`
2. **Calibrate System**: Go to Calibration page, click "Start Calibration"
3. **Test Modules**: Upload sample files to test individual modules
4. **Run Analysis**: Upload exam video for comprehensive analysis
5. **Review Results**: Examine detailed reports and visualizations
6. **Export Data**: Download JSON/CSV reports

### Live Monitoring Session
1. Complete system calibration
2. Navigate to Live Monitoring page
3. Click "Capture and Analyze Frame" repeatedly
4. Monitor real-time risk assessments
5. Review gaze direction and person detection results

### Comprehensive Video Analysis
1. Navigate to Comprehensive Analysis page
2. Upload video file (MP4, AVI, MOV)
3. Optionally upload separate audio file
4. Click "Run Comprehensive Analysis"
5. Wait for processing (may take several minutes)
6. Review detailed results in multiple tabs
7. Export reports as needed

## 🔍 Troubleshooting

### Common Issues

#### Camera Access Problems
- **Issue**: "Failed to capture frame from camera"
- **Solution**: 
  - Check camera permissions
  - Ensure no other applications are using the camera
  - Try different camera indices (0, 1, 2)

#### Calibration Failures
- **Issue**: Calibration not completing
- **Solution**:
  - Ensure good lighting conditions
  - Keep face clearly visible to camera
  - Avoid moving during calibration
  - Restart calibration if needed

#### Module Analysis Errors
- **Issue**: Individual modules failing
- **Solution**:
  - Check file format compatibility
  - Ensure model files are downloaded
  - Verify file is not corrupted
  - Check file size limitations

#### Performance Issues
- **Issue**: Slow processing or analysis
- **Solution**:
  - Use smaller video files for testing
  - Ensure sufficient system resources
  - Close other resource-intensive applications
  - Consider reducing video resolution

### Error Messages

#### "Missing model files"
- Run: `python download_models.py`
- Ensure internet connection for downloads

#### "Calibration Required"
- Complete system calibration before using live monitoring
- Check calibration status in sidebar

#### "Processing failed"
- Check file format and size
- Verify all dependencies are installed
- Review console output for detailed error messages

## 🔒 Privacy and Security

### Data Handling
- **Local Processing**: All analysis performed locally
- **No Data Upload**: Files are not sent to external servers
- **Temporary Files**: Automatically cleaned up after processing
- **Session Data**: Stored only in browser session

### Camera Access
- **Permission Required**: Browser will request camera access
- **Local Only**: Camera feed processed locally
- **No Recording**: Live monitoring doesn't save video data

## 🛠️ Development

### Adding New Features
1. Create new module in appropriate file
2. Add import to `app.py`
3. Create new interface function
4. Add to navigation menu
5. Test functionality

### Customizing UI
- Modify CSS in `app.py` for styling changes
- Add new Streamlit components as needed
- Update navigation structure in main function

### Performance Optimization
- Implement caching for expensive operations
- Use Streamlit's session state for data persistence
- Optimize image/video processing pipelines

## 📞 Support

For technical support or questions:
- Check troubleshooting section above
- Review console output for error details
- Ensure all dependencies are properly installed
- Verify model files are present and valid

## 🔄 Updates

To update the system:
1. Pull latest code changes
2. Update dependencies: `pip install -r requirements.txt`
3. Re-download models if needed: `python download_models.py`
4. Restart the application

---

 