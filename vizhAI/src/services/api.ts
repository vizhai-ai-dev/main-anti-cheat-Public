import axios from 'axios';

// Create axios instance with base URL and default configs
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || '/api',
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
  screen_switch?: {
    fullscreen_violations: number;
    switch_count: number;
  };
  gaze?: {
    off_screen_count: number;
    average_confidence: number;
  };
  audio?: {
    multiple_speakers: boolean;
    keyboard_typing_count: number;
    silence_percentage: number;
  };
  multi_person?: {
    max_people_detected: number;
    time_with_multiple_people: number;
  };
  lip_sync?: {
    lip_sync_score: number;
    major_lip_desync_detected: boolean;
  };
  module_scores: {
    screen_switch: number;
    gaze: number;
    audio: number;
    multi_person: number;
    lip_sync: number;
  };
  processing_time: number;
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