import React from 'react';
import { Link } from 'react-router-dom';

const HomePage: React.FC = () => {
  return (
    <div className="w-full">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-primary-600 to-secondary-600 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
          <div className="max-w-3xl">
            <h1 className="text-4xl font-extrabold tracking-tight text-white sm:text-5xl md:text-6xl">
              Video Integrity Analysis with VIZH.AI
            </h1>
            <p className="mt-6 text-xl text-primary-50 max-w-3xl">
              Detect potential integrity issues in video interviews with our AI-powered video analysis platform. Get comprehensive reports on multiple factors.
            </p>
            <div className="mt-10">
              <Link
                to="/upload"
                className="btn bg-white text-primary-700 hover:bg-primary-50 shadow-md"
              >
                Upload a Video
              </Link>
            </div>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="lg:text-center">
            <h2 className="text-base text-secondary-600 font-semibold tracking-wide uppercase">Features</h2>
            <p className="mt-2 text-3xl leading-8 font-extrabold tracking-tight text-dark-800 sm:text-4xl">
              Comprehensive Video Analysis
            </p>
            <p className="mt-4 max-w-2xl text-xl text-dark-500 lg:mx-auto">
              Our platform analyzes multiple aspects of your video to provide a thorough assessment.
            </p>
          </div>

          <div className="mt-16">
            <div className="grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-3">
              <div className="card">
                <div className="p-2 rounded-md bg-primary-100 inline-block">
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6 text-primary-600">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 17.25v1.007a3 3 0 01-.879 2.122L7.5 21h9l-.621-.621A3 3 0 0115 18.257V17.25m6-12V15a2.25 2.25 0 01-2.25 2.25H5.25A2.25 2.25 0 013 15V5.25m18 0A2.25 2.25 0 0018.75 3H5.25A2.25 2.25 0 003 5.25m18 0V12a2.25 2.25 0 01-2.25 2.25H5.25A2.25 2.25 0 013 12V5.25" />
                  </svg>
                </div>
                <h3 className="mt-4 text-lg font-medium text-dark-900">Screen Analysis</h3>
                <p className="mt-2 text-base text-dark-500">
                  Detects screen switching, fullscreen violations, and other anomalies that may indicate unauthorized resource access.
                </p>
              </div>

              <div className="card">
                <div className="p-2 rounded-md bg-primary-100 inline-block">
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6 text-primary-600">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178z" />
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                </div>
                <h3 className="mt-4 text-lg font-medium text-dark-900">Gaze Tracking</h3>
                <p className="mt-2 text-base text-dark-500">
                  Analyzes eye movements to identify when the subject looks away from the screen excessively.
                </p>
              </div>

              <div className="card">
                <div className="p-2 rounded-md bg-primary-100 inline-block">
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6 text-primary-600">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z" />
                  </svg>
                </div>
                <h3 className="mt-4 text-lg font-medium text-dark-900">Audio Analysis</h3>
                <p className="mt-2 text-base text-dark-500">
                  Detects multiple speakers, keyboard typing, and analyzes speech patterns.
                </p>
              </div>

              <div className="card">
                <div className="p-2 rounded-md bg-primary-100 inline-block">
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6 text-primary-600">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" />
                  </svg>
                </div>
                <h3 className="mt-4 text-lg font-medium text-dark-900">Multi-Person Detection</h3>
                <p className="mt-2 text-base text-dark-500">
                  Identifies if multiple people are present in the video, which may indicate collaboration.
                </p>
              </div>

              <div className="card">
                <div className="p-2 rounded-md bg-primary-100 inline-block">
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6 text-primary-600">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />
                  </svg>
                </div>
                <h3 className="mt-4 text-lg font-medium text-dark-900">Lip Sync Analysis</h3>
                <p className="mt-2 text-base text-dark-500">
                  Checks if lip movements match audio to detect voice-overs or audio manipulation.
                </p>
              </div>

              <div className="card">
                <div className="p-2 rounded-md bg-primary-100 inline-block">
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6 text-primary-600">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0112 15a9.065 9.065 0 00-6.23-.693L5 14.5m14.8.8l1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0 0112 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5" />
                  </svg>
                </div>
                <h3 className="mt-4 text-lg font-medium text-dark-900">Integrity Score</h3>
                <p className="mt-2 text-base text-dark-500">
                  Combines all analysis factors to provide an overall integrity assessment with detailed explanations.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="bg-primary-50">
        <div className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:py-16 lg:px-8 lg:flex lg:items-center lg:justify-between">
          <h2 className="text-3xl font-extrabold tracking-tight text-dark-900 sm:text-4xl">
            <span className="block">Ready to analyze your videos?</span>
            <span className="block text-primary-600">Start using VIZH.AI today.</span>
          </h2>
          <div className="mt-8 flex lg:mt-0 lg:flex-shrink-0">
            <div className="inline-flex rounded-md shadow">
              <Link
                to="/upload"
                className="btn-primary px-6 py-3"
              >
                Get started
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HomePage; 