import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Bar, Doughnut, Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  PointElement,
  LineElement,
} from 'chart.js';
import { getAnalysisStatus, AnalysisReport } from '../services/api';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  PointElement,
  LineElement
);

const ReportPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [report, setReport] = useState<AnalysisReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pollingCount, setPollingCount] = useState(0);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    const fetchReport = async () => {
      if (!id) {
        setError('No analysis ID provided');
        setLoading(false);
        return;
      }

      try {
        const response = await getAnalysisStatus(id);
        
        if (response.status === 'completed' && response.result) {
          setReport(response.result);
          setLoading(false);
        } else if (response.status === 'failed') {
          setError('Analysis failed. Please try again with a different video.');
          setLoading(false);
        } else if (response.status === 'processing') {
          // If still processing and we haven't polled too many times, poll again
          if (pollingCount < 60) { // Poll for up to 5 minutes (60 * 5s = 300s)
            setTimeout(() => {
              setPollingCount(prev => prev + 1);
            }, 5000);
          } else {
            setError('Analysis is taking longer than expected. Please check back later.');
            setLoading(false);
          }
        }
      } catch (err) {
        console.error('Error fetching report:', err);
        setError('Failed to load analysis report. Please try again.');
        setLoading(false);
      }
    };

    if (loading) {
      fetchReport();
    }
  }, [id, loading, pollingCount]);

  // Generate colors for the risk indicator
  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'Low':
        return 'bg-green-500';
      case 'Medium':
        return 'bg-yellow-500';
      case 'High':
        return 'bg-orange-500';
      case 'Very High':
        return 'bg-red-500';
      default:
        return 'bg-gray-500';
    }
  };

  // Chart data
  const generateBarChartData = () => {
    if (!report) return null;
    
    return {
      labels: ['Gaze Analysis', 'Audio Analysis', 'Multi-Person', 'Lip Sync'],
      datasets: [
        {
          label: 'Module Scores',
          data: [
            report.module_scores.gaze,
            report.module_scores.audio,
            report.module_scores.multi_person,
            report.module_scores.lip_sync
          ],
          backgroundColor: [
            'rgba(54, 162, 235, 0.5)',
            'rgba(255, 206, 86, 0.5)',
            'rgba(75, 192, 192, 0.5)',
            'rgba(153, 102, 255, 0.5)'
          ],
          borderColor: [
            'rgba(54, 162, 235, 1)',
            'rgba(255, 206, 86, 1)',
            'rgba(75, 192, 192, 1)',
            'rgba(153, 102, 255, 1)'
          ],
          borderWidth: 1,
        },
      ],
    };
  };

  const generateDoughnutChartData = () => {
    if (!report) return null;
    
    return {
      labels: ['Gaze', 'Audio', 'Multi-Person', 'Lip Sync'],
      datasets: [
        {
          data: [
            report.module_scores.gaze,
            report.module_scores.audio,
            report.module_scores.multi_person,
            report.module_scores.lip_sync
          ],
          backgroundColor: [
            'rgba(54, 162, 235, 0.7)',
            'rgba(255, 206, 86, 0.7)',
            'rgba(75, 192, 192, 0.7)',
            'rgba(153, 102, 255, 0.7)'
          ],
          borderColor: [
            'rgba(54, 162, 235, 1)',
            'rgba(255, 206, 86, 1)',
            'rgba(75, 192, 192, 1)',
            'rgba(153, 102, 255, 1)'
          ],
          borderWidth: 1,
        },
      ],
    };
  };

  // Add new chart generation functions
  const generateLipSyncTimelineChart = () => {
    if (!report?.lip_sync?.lip_sync_timeline) return null;
    
    return {
      labels: report.lip_sync.lip_sync_timeline.map(item => item.timestamp),
      datasets: [
        {
          label: 'Lip Sync Score',
          data: report.lip_sync.lip_sync_timeline.map(item => item.score),
          borderColor: 'rgba(236, 72, 153, 1)',
          backgroundColor: 'rgba(236, 72, 153, 0.2)',
          tension: 0.3,
          fill: true,
        }
      ]
    };
  };
  
  const generatePeopleDetectionTimelineChart = () => {
    if (!report?.multi_person?.people_detection_timeline) return null;
    
    return {
      labels: report.multi_person.people_detection_timeline.map(item => item.timestamp),
      datasets: [
        {
          label: 'People Detected',
          data: report.multi_person.people_detection_timeline.map(item => item.count),
          borderColor: 'rgba(168, 85, 247, 1)',
          backgroundColor: 'rgba(168, 85, 247, 0.2)',
          stepped: true,
        }
      ]
    };
  };
  
  const lineChartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      tooltip: {
        callbacks: {
          label: function(context: any) {
            return `${context.dataset.label}: ${context.parsed.y}`;
          }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
      }
    },
  };
  
  const lipSyncChartOptions = {
    ...lineChartOptions,
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        ticks: {
          callback: function(value: any) {
            return value + '%';
          }
        }
      }
    },
  };

  const barChartOptions = {
    responsive: true,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        callbacks: {
          label: function(context: any) {
            return `Score: ${context.parsed.y}`;
          }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        ticks: {
          callback: function(value: any) {
            return value + '%';
          }
        }
      }
    },
  };
  
  const doughnutChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom' as const,
      },
    },
  };

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="text-center py-20">
          <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-primary-600 mx-auto mb-6"></div>
          <h2 className="text-2xl font-semibold text-dark-800">Analyzing video...</h2>
          <p className="mt-2 text-dark-500">
            This may take a few minutes depending on the video length.
          </p>
          <p className="mt-1 text-dark-400 text-sm">
            Attempt {pollingCount + 1} of 60
          </p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="bg-red-50 p-6 rounded-lg shadow-sm">
          <h2 className="text-xl font-medium text-red-800">Error</h2>
          <p className="mt-2 text-red-700">{error}</p>
          <button 
            className="mt-4 btn-primary"
            onClick={() => navigate('/upload')}
          >
            Upload a New Video
          </button>
        </div>
      </div>
    );
  }

  if (!report) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="bg-yellow-50 p-6 rounded-lg shadow-sm">
          <h2 className="text-xl font-medium text-yellow-800">No Data Available</h2>
          <p className="mt-2 text-yellow-700">No report data is available for this analysis.</p>
          <button 
            className="mt-4 btn-primary"
            onClick={() => navigate('/upload')}
          >
            Upload a New Video
          </button>
        </div>
      </div>
    );
  }

  // Generate chart data
  const barChartData = generateBarChartData();
  const doughnutChartData = generateDoughnutChartData();
  const lipSyncTimelineChart = generateLipSyncTimelineChart();
  const peopleDetectionTimelineChart = generatePeopleDetectionTimelineChart();

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-extrabold text-dark-800">Video Analysis Report</h1>
        <p className="mt-2 text-dark-500">
          Analysis completed in {report.processing_time.toFixed(1)} seconds
        </p>
      </div>

      {/* Score Summary Card */}
      <div className="card mb-8">
        <div className="flex flex-col md:flex-row items-center justify-between">
          <div className="mb-6 md:mb-0">
            <p className="text-dark-500 text-sm uppercase font-medium">Integrity Score</p>
            <div className="flex items-center">
              <div className="text-5xl font-bold mr-2 text-dark-800">{report.final_score.toFixed(1)}%</div>
              <div className={`px-3 py-1 rounded-full text-white text-sm font-medium ${getRiskColor(report.risk)}`}>
                {report.risk} Risk
              </div>
            </div>
          </div>
          <div className="h-40 w-40">
            {doughnutChartData && (
              <Doughnut data={doughnutChartData} options={doughnutChartOptions} />
            )}
          </div>
        </div>
        
        {/* Reasons section */}
        <div className="mt-6 pt-6 border-t border-gray-200">
          <h3 className="text-lg font-medium text-dark-800 mb-3">Key Findings</h3>
          {report.reasons.length > 0 ? (
            <ul className="space-y-2">
              {report.reasons.map((reason, idx) => (
                <li key={idx} className="flex items-start">
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5 text-red-500 mr-2 mt-0.5 flex-shrink-0">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
                  </svg>
                  {reason}
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-green-600">No integrity issues detected.</p>
          )}
        </div>
      </div>

      {/* Tabs */}
      <div className="mb-6 border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          <button
            className={`py-4 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'overview'
                ? 'border-primary-600 text-primary-600'
                : 'border-transparent text-dark-500 hover:text-dark-700 hover:border-gray-300'
            }`}
            onClick={() => setActiveTab('overview')}
          >
            Overview
          </button>
          <button
            className={`py-4 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'detailed'
                ? 'border-primary-600 text-primary-600'
                : 'border-transparent text-dark-500 hover:text-dark-700 hover:border-gray-300'
            }`}
            onClick={() => setActiveTab('detailed')}
          >
            Detailed Analysis
          </button>
        </nav>
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' ? (
        <div>
          <div className="card mb-8">
            <h2 className="text-xl font-semibold text-dark-800 mb-4">Module Scores</h2>
            <div className="h-80 w-full">
              {barChartData && (
                <Bar data={barChartData} options={barChartOptions} />
              )}
            </div>
            <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className="text-lg font-medium text-dark-800 mb-3">Strong Points</h3>
                <ul className="space-y-2">
                  {Object.entries(report.module_scores)
                    .filter(([, score]) => score >= 80)
                    .map(([module, score]) => (
                      <li key={module} className="flex items-start">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5 text-green-500 mr-2 mt-0.5 flex-shrink-0">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <span>
                          <span className="font-medium">{module.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}:</span> {score.toFixed(1)}% integrity
                        </span>
                      </li>
                    ))}
                </ul>
              </div>
              <div>
                <h3 className="text-lg font-medium text-dark-800 mb-3">Areas of Concern</h3>
                <ul className="space-y-2">
                  {Object.entries(report.module_scores)
                    .filter(([, score]) => score < 80)
                    .map(([module, score]) => (
                      <li key={module} className="flex items-start">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5 text-red-500 mr-2 mt-0.5 flex-shrink-0">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
                        </svg>
                        <span>
                          <span className="font-medium">{module.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}:</span> {score.toFixed(1)}% integrity
                        </span>
                      </li>
                    ))}
                </ul>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div>
          {/* Module Scores Section */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            {/* Gaze Analysis */}
            <div className="card">
              <div className="flex items-center">
                <div className="p-2 rounded-md bg-primary-100 mr-4">
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6 text-primary-600">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178z" />
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                </div>
                <div>
                  <h3 className="text-lg font-medium text-dark-800">Gaze Tracking</h3>
                  <p className="text-dark-500 text-sm">Score: {report.module_scores.gaze.toFixed(1)}%</p>
                </div>
              </div>
              <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-gray-50 p-4 rounded-md">
                  <div className="text-sm font-medium text-dark-700 mb-1">Off-Screen Looks</div>
                  <div className="text-2xl font-bold text-dark-800">{report.gaze?.off_screen_count || 0}</div>
                </div>
                <div className="bg-gray-50 p-4 rounded-md">
                  <div className="text-sm font-medium text-dark-700 mb-1">Gaze Confidence</div>
                  <div className="text-2xl font-bold text-dark-800">{(report.gaze?.average_confidence || 0) * 100}%</div>
                </div>
              </div>
            </div>

            {/* Audio Analysis */}
            <div className="card">
              <div className="flex items-center">
                <div className="p-2 rounded-md bg-primary-100 mr-4">
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6 text-primary-600">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z" />
                  </svg>
                </div>
                <div>
                  <h3 className="text-lg font-medium text-dark-800">Audio Analysis</h3>
                  <p className="text-dark-500 text-sm">Score: {report.module_scores.audio.toFixed(1)}%</p>
                </div>
              </div>
              <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-gray-50 p-4 rounded-md">
                  <div className="text-sm font-medium text-dark-700 mb-1">Multiple Speakers</div>
                  <div className="text-2xl font-bold text-dark-800">{report.audio?.multiple_speakers ? 'Yes' : 'No'}</div>
                </div>
                <div className="bg-gray-50 p-4 rounded-md">
                  <div className="text-sm font-medium text-dark-700 mb-1">Keyboard Typing</div>
                  <div className="text-2xl font-bold text-dark-800">{report.audio?.keyboard_typing_count || 0} instances</div>
                </div>
                <div className="bg-gray-50 p-4 rounded-md">
                  <div className="text-sm font-medium text-dark-700 mb-1">Silence</div>
                  <div className="text-2xl font-bold text-dark-800">{report.audio?.silence_percentage || 0}%</div>
                </div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            {/* Multi-Person Analysis */}
            <div className="card">
              <div className="flex items-center">
                <div className="p-2 rounded-md bg-primary-100 mr-4">
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6 text-primary-600">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" />
                  </svg>
                </div>
                <div>
                  <h3 className="text-lg font-medium text-dark-800">Multi-Person Detection</h3>
                  <p className="text-dark-500 text-sm">Score: {report.module_scores.multi_person.toFixed(1)}%</p>
                </div>
              </div>
              <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-gray-50 p-4 rounded-md">
                  <div className="text-sm font-medium text-dark-700 mb-1">Max People Detected</div>
                  <div className="text-2xl font-bold text-dark-800">{report.multi_person?.max_people_detected || 0}</div>
                </div>
                <div className="bg-gray-50 p-4 rounded-md">
                  <div className="text-sm font-medium text-dark-700 mb-1">Time with Multiple People</div>
                  <div className="text-2xl font-bold text-dark-800">{report.multi_person?.time_with_multiple_people || 0}s</div>
                </div>
              </div>
              
              {/* Add Face Recognition Information */}
              {report.multi_person?.has_different_faces && (
                <div className="mt-4">
                  <div className="p-3 bg-red-50 rounded-md border border-red-200">
                    <div className="flex items-center">
                      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5 text-red-600 mr-2">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
                      </svg>
                      <div className="text-sm font-medium text-red-700">
                        {report.multi_person.different_faces_detected} different {report.multi_person.different_faces_detected === 1 ? 'face' : 'faces'} detected
                      </div>
                    </div>
                    
                    {report.multi_person.different_face_timestamps.length > 0 && (
                      <div className="mt-2 ml-7">
                        <div className="text-xs text-red-600 font-medium mb-1">Timestamps:</div>
                        <div className="flex flex-wrap gap-1">
                          {report.multi_person.different_face_timestamps.map((timestamp, idx) => (
                            <span key={idx} className="px-2 py-1 bg-red-100 text-red-800 text-xs rounded-full">
                              {timestamp}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Lip Sync Analysis */}
            <div className="card">
              <div className="flex items-center">
                <div className="p-2 rounded-md bg-primary-100 mr-4">
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6 text-primary-600">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />
                  </svg>
                </div>
                <div>
                  <h3 className="text-lg font-medium text-dark-800">Lip Sync Analysis</h3>
                  <p className="text-dark-500 text-sm">Score: {report.module_scores.lip_sync.toFixed(1)}%</p>
                </div>
              </div>
              <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-gray-50 p-4 rounded-md">
                  <div className="text-sm font-medium text-dark-700 mb-1">Lip Sync Score</div>
                  <div className="text-2xl font-bold text-dark-800">{report.lip_sync?.lip_sync_score || 0}%</div>
                </div>
                <div className="bg-gray-50 p-4 rounded-md">
                  <div className="text-sm font-medium text-dark-700 mb-1">Major Desync Detected</div>
                  <div className="text-2xl font-bold text-dark-800">{report.lip_sync?.major_lip_desync_detected ? 'Yes' : 'No'}</div>
                </div>
              </div>
            </div>
          </div>
          
          {/* Additional Timeline Visualizations */}
          {/* Gaze Direction Timeline */}
          {report.gaze?.gaze_direction_timeline && (
            <div className="card mb-6">
              <h3 className="text-lg font-medium text-dark-800 mb-4">Gaze Direction Timeline</h3>
              <div className="overflow-x-auto">
                <div className="min-w-full">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead>
                      <tr>
                        <th className="px-6 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Timestamp</th>
                        <th className="px-6 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Direction</th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {report.gaze.gaze_direction_timeline.map((item, idx) => (
                        <tr key={idx} className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{item.timestamp}</td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <span className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${
                              item.direction === 'center' ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'
                            }`}>
                              {item.direction}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
          
          {/* People Detection Timeline Chart */}
          {report.multi_person?.people_detection_timeline && (
            <div className="card mb-6">
              <h3 className="text-lg font-medium text-dark-800 mb-4">People Detection Timeline</h3>
              <div className="h-60">
                {peopleDetectionTimelineChart && (
                  <Line data={peopleDetectionTimelineChart} options={lineChartOptions} />
                )}
              </div>
            </div>
          )}
          
          {/* Speaking Timeline */}
          {report.audio?.speaking_timeline && (
            <div className="card mb-6">
              <h3 className="text-lg font-medium text-dark-800 mb-4">Speaking Timeline</h3>
              <div className="overflow-x-auto">
                <div className="min-w-full">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead>
                      <tr>
                        <th className="px-6 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Start</th>
                        <th className="px-6 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">End</th>
                        <th className="px-6 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Speaker</th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {report.audio.speaking_timeline.map((item, idx) => (
                        <tr key={idx} className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{item.start}</td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{item.end}</td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <span className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${
                              item.speaker === 'primary' ? 'bg-blue-100 text-blue-800' : 'bg-red-100 text-red-800'
                            }`}>
                              {item.speaker}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
          
          {/* Lip Sync Timeline Chart */}
          {report.lip_sync?.lip_sync_timeline && (
            <div className="card mb-6">
              <h3 className="text-lg font-medium text-dark-800 mb-4">Lip Sync Timeline</h3>
              <div className="h-60">
                {lipSyncTimelineChart && (
                  <Line data={lipSyncTimelineChart} options={lipSyncChartOptions} />
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Action buttons */}
      <div className="mt-8 flex justify-center gap-4">
        <button 
          className="btn-outline" 
          onClick={() => window.print()}
        >
          Download Report
        </button>
        <button 
          className="btn-primary" 
          onClick={() => navigate('/upload')}
        >
          Analyze Another Video
        </button>
      </div>
    </div>
  );
};

export default ReportPage; 