import axios from 'axios';

// Create axios instance with base URL and default configs
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Types for the API responses
export interface AnalysisResult {
  id: string;
  status: 'processing' | 'completed' | 'failed';
  result?: AnalysisReport;
  error?: string;
  created_at: string;
  completed_at?: string;
}

export interface AnalysisReport {
  final_score: number;
  risk: 'Low' | 'Medium' | 'High' | 'Very High';
  reasons: string[];
  gaze?: {
    off_screen_count: number;
    average_confidence: number;
    off_screen_time_percentage?: number;
    gaze_direction_timeline?: Array<{
      timestamp: string;
      direction: string;
    }>;
    score: number;
  };
  audio?: {
    multiple_speakers: boolean;
    keyboard_typing_count: number;
    silence_percentage: number;
    background_noise_level?: string;
    speaking_timeline?: Array<{
      start: string;
      end: string;
      speaker: string;
    }>;
    score: number;
  };
  multi_person?: {
    max_people_detected: number;
    time_with_multiple_people: number;
    people_detection_timeline?: Array<{
      timestamp: string;
      count: number;
    }>;
    different_faces_detected: number;
    different_face_timestamps: string[];
    has_different_faces: boolean;
    score: number;
  };
  lip_sync?: {
    lip_sync_score: number;
    major_lip_desync_detected: boolean;
    lip_sync_timeline?: Array<{
      timestamp: string;
      score: number;
    }>;
    score: number;
  };
  module_scores: {
    gaze: number;
    audio: number;
    multi_person: number;
    lip_sync: number;
  };
  processing_time: number;
  video_duration?: string;
}

// API functions
export const uploadVideo = async (file: File): Promise<{ id: string }> => {
  const formData = new FormData();
  formData.append('video', file);
  
  const response = await api.post('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  
  return response.data;
};

export const getAnalysisStatus = async (id: string): Promise<AnalysisResult> => {
  const response = await api.get(`/analysis/${id}`);
  return response.data;
};

export const getAnalysisReport = async (id: string): Promise<AnalysisReport> => {
  const response = await api.get(`/analysis/${id}/report`);
  return response.data;
};

export default api; 