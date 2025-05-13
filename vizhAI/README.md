# VIZH.AI - Video Integrity Analysis Tool

![VIZH.AI Logo](public/vizhai-logo.png)

VIZH.AI is a sophisticated web application designed to analyze video integrity, particularly useful for ensuring interview authenticity. The application uses advanced AI techniques to detect potential integrity issues by analyzing various aspects of video content.

## Features

- **Multi-factor Analysis**: Examines numerous aspects of a video to determine integrity:
  - Screen switching detection
  - Gaze tracking analysis
  - Audio analysis (multiple speakers, keyboard typing, etc.)
  - Multi-person detection
  - Lip sync analysis

- **Comprehensive Reports**: Generates detailed reports with:
  - Overall integrity score
  - Risk assessment (Low, Medium, High, Very High)
  - Detailed findings with visual charts
  - Module-specific scores and metrics

- **Modern UI**: Clean, minimal, and user-friendly interface built with React and Tailwind CSS

## Getting Started

### Prerequisites

- Node.js 14.x or higher
- npm or yarn

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/vizhai.git
   cd vizhai
   ```

2. Install dependencies:
   ```bash
   npm install
   # or
   yarn install
   ```

3. Start the development server:
   ```bash
   npm start
   # or
   yarn start
   ```

4. Build for production:
   ```bash
   npm run build
   # or
   yarn build
   ```

## Backend Integration

This frontend application is designed to work with a Python FastAPI backend that performs the actual video analysis. The backend should provide the following endpoints:

- `POST /upload` - For uploading video files
- `GET /analysis/{id}` - For checking analysis status
- `GET /analysis/{id}/report` - For retrieving the final analysis report

Refer to the API service in `src/services/api.ts` for the expected request/response formats.

## Technologies Used

- **Frontend**:
  - React (with TypeScript)
  - React Router for navigation
  - Tailwind CSS for styling
  - Chart.js for data visualization
  - Axios for API communication

- **Suggested Backend** (not included in this repo):
  - Python FastAPI
  - OpenCV for video processing
  - TensorFlow/PyTorch for AI models
  - FFmpeg for video manipulation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by the [Honest Hire App](https://github.com/Honest-Hire/Honest-Hire-App)
- Uses open-source AI models for various detection tasks 