# VIZH.AI - Video Integrity Analysis Tool

VIZH.AI is a sophisticated application designed to analyze video integrity, particularly useful for ensuring interview authenticity. The application uses advanced AI techniques to detect potential integrity issues by analyzing various aspects of video content.

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

## Running with Docker

The easiest way to run VIZH.AI is using Docker and Docker Compose.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

### Starting the Application

1. Clone this repository:
   ```bash
   git clone https://github.com/darshanjijhu/main-anti-cheat-Public.git
   cd vizhai
   ```

2. Build and start the containers:
   ```bash
   docker-compose up -d
   ```

3. Access the application:
   - Frontend: http://localhost
   - Backend API: http://localhost/api

### Stopping the Application

```bash
docker-compose down
```

## Development Setup

If you prefer to run the application without Docker for development:

### Backend

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the backend server:
   ```bash
   python backend_server.py
   ```

### Frontend

1. Navigate to the frontend directory:
   ```bash
   cd vizhAI
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

## API Endpoints

- `POST /upload` - Upload a video for analysis
- `GET /analysis/{id}` - Check analysis status
- `GET /analysis/{id}/report` - Get the analysis report
- `POST /demo-analysis` - Generate a demo analysis (for testing)

## License

This project is licensed under the MIT License. 
