import React, { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import { uploadVideo } from '../services/api';

const UploadPage: React.FC = () => {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const videoFile = acceptedFiles[0];
    if (videoFile) {
      setFile(videoFile);
      setError(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ 
    onDrop,
    accept: {
      'video/*': ['.mp4', '.mov', '.avi', '.mkv', '.webm']
    },
    maxFiles: 1,
    maxSize: 500 * 1024 * 1024, // 500MB max size
  });

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a video file first');
      return;
    }

    try {
      setUploading(true);
      setError(null);
      
      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 95) {
            clearInterval(progressInterval);
            return 95;
          }
          return prev + 5;
        });
      }, 500);

      // Upload the file
      const response = await uploadVideo(file);
      
      clearInterval(progressInterval);
      setUploadProgress(100);
      
      // Navigate to the report page
      setTimeout(() => {
        navigate(`/report/${response.id}`);
      }, 500);
      
    } catch (err) {
      setError('An error occurred while uploading. Please try again.');
      setUploading(false);
      setUploadProgress(0);
      console.error('Upload error:', err);
    }
  };

  const cancelUpload = () => {
    setFile(null);
    setUploading(false);
    setUploadProgress(0);
    setError(null);
  };

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <div className="text-center mb-12">
        <h1 className="text-3xl font-extrabold text-dark-800 sm:text-4xl">
          Upload Video for Analysis
        </h1>
        <p className="mt-4 text-lg text-dark-500">
          Drag and drop your video file below or click to browse. We support MP4, MOV, AVI, MKV, and WebM formats.
        </p>
      </div>

      <div className="mb-8">
        <div 
          {...getRootProps()} 
          className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-colors
            ${isDragActive ? 'border-primary-500 bg-primary-50' : 'border-gray-300 hover:border-primary-400'}
            ${file && !uploading ? 'bg-green-50 border-green-300' : ''}
            ${uploading ? 'pointer-events-none' : ''}
          `}
        >
          <input {...getInputProps()} />
          
          {file && !uploading ? (
            <div className="text-center">
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-12 h-12 text-green-500 mx-auto mb-4">
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p className="text-xl font-medium text-dark-800">{file.name}</p>
              <p className="text-sm text-dark-500 mt-1">
                {(file.size / (1024 * 1024)).toFixed(2)} MB
              </p>
            </div>
          ) : uploading ? (
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-600 mx-auto mb-4"></div>
              <p className="text-xl font-medium text-dark-800">Uploading...</p>
              <div className="w-full bg-gray-200 rounded-full h-2.5 mt-4 mb-2">
                <div 
                  className="bg-primary-600 h-2.5 rounded-full transition-all duration-300 ease-out" 
                  style={{ width: `${uploadProgress}%` }}
                ></div>
              </div>
              <p className="text-sm text-dark-500">{uploadProgress}% complete</p>
            </div>
          ) : (
            <div>
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-12 h-12 text-dark-400 mx-auto mb-4">
                <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
              </svg>
              <p className="text-xl font-medium text-dark-700">
                {isDragActive ? 'Drop the video here' : 'Drag & drop your video here'}
              </p>
              <p className="text-sm text-dark-500 mt-1">
                or <span className="text-primary-600">browse files</span>
              </p>
              <p className="text-xs text-dark-400 mt-4">
                Maximum file size: 500MB
              </p>
            </div>
          )}
        </div>

        {error && (
          <div className="mt-4 bg-red-50 text-red-800 p-3 rounded-md text-sm">
            {error}
          </div>
        )}
      </div>

      <div className="flex justify-center gap-4">
        {file && !uploading && (
          <>
            <button 
              className="btn-outline" 
              onClick={cancelUpload}
            >
              Cancel
            </button>
            <button 
              className="btn-primary" 
              onClick={handleUpload}
            >
              Analyze Video
            </button>
          </>
        )}
        
        {uploading && (
          <button 
            className="btn-outline opacity-50 cursor-not-allowed" 
            disabled
          >
            Uploading...
          </button>
        )}
      </div>

      <div className="mt-16 bg-primary-50 rounded-lg p-6">
        <h2 className="text-xl font-medium text-dark-800 mb-4">
          Tips for optimal analysis
        </h2>
        <ul className="space-y-2 text-dark-600">
          <li className="flex items-start">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5 text-primary-600 mr-2 mt-0.5 flex-shrink-0">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Ensure the video captures the person's face clearly throughout the recording
          </li>
          <li className="flex items-start">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5 text-primary-600 mr-2 mt-0.5 flex-shrink-0">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Good lighting helps improve detection accuracy
          </li>
          <li className="flex items-start">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5 text-primary-600 mr-2 mt-0.5 flex-shrink-0">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Clear audio improves speech and lip sync analysis
          </li>
          <li className="flex items-start">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5 text-primary-600 mr-2 mt-0.5 flex-shrink-0">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Videos between 5-30 minutes provide the most accurate results
          </li>
        </ul>
      </div>
    </div>
  );
};

export default UploadPage; 