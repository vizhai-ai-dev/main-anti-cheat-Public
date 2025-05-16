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
  ChartData,
} from 'chart.js';
import { getAnalysisStatus, AnalysisReport as ApiAnalysisReport } from '../services/api';

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

interface Reason {
  text: string;
  severity: 'critical' | 'warning';
  module: string;
}

// Extend the API's AnalysisReport type with our additional properties
interface AnalysisReport extends Omit<ApiAnalysisReport, 'reasons' | 'video_duration'> {
  status: string;
  video_duration: number;
  reasons: Reason[];
}

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
          // Transform the API response to match our extended type
          const transformedReport: AnalysisReport = {
            ...response.result,
            status: response.status,
            reasons: response.result.reasons.map((reason: any) => ({
              text: reason.text || reason,
              severity: reason.severity || 'warning',
              module: reason.module || 'general'
            })),
            video_duration: typeof response.result.video_duration === 'string' 
              ? parseFloat(response.result.video_duration) 
              : 0
          };
          setReport(transformedReport);
          setLoading(false);
        } else if (response.status === 'failed') {
          setError('Analysis failed. Please try again with a different video.');
          setLoading(false);
        } else if (response.status === 'processing') {
          if (pollingCount < 60) {
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

  // Updated risk color function with new risk levels
  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'Very Low':
        return 'bg-green-600';
      case 'Low':
        return 'bg-green-400';
      case 'Medium':
        return 'bg-yellow-500';
      case 'High':
        return 'bg-orange-500';
      case 'Very High':
        return 'bg-red-600';
      default:
        return 'bg-gray-500';
    }
  };

  // Updated risk text color function
  const getRiskTextColor = (risk: string) => {
    switch (risk) {
      case 'Very Low':
      case 'Low':
        return 'text-green-800';
      case 'Medium':
        return 'text-yellow-800';
      case 'High':
        return 'text-orange-800';
      case 'Very High':
        return 'text-red-800';
      default:
        return 'text-gray-800';
    }
  };

  // New function to get severity color
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'text-red-600';
      case 'warning':
        return 'text-yellow-600';
      default:
        return 'text-gray-600';
    }
  };

  // Updated chart data with proper type definitions and null checks
  const generateBarChartData = (): ChartData<'bar', number[], string> | null => {
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
            'rgba(34, 197, 94, 0.5)',  // Green
            'rgba(234, 179, 8, 0.5)',  // Yellow
            'rgba(249, 115, 22, 0.5)', // Orange
            'rgba(239, 68, 68, 0.5)'   // Red
          ],
          borderColor: [
            'rgba(34, 197, 94, 1)',
            'rgba(234, 179, 8, 1)',
            'rgba(249, 115, 22, 1)',
            'rgba(239, 68, 68, 1)'
          ],
          borderWidth: 2,
        },
      ],
    };
  };

  const generateDoughnutChartData = (): ChartData<'doughnut', number[], string> | null => {
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
            'rgba(34, 197, 94, 0.7)',
            'rgba(234, 179, 8, 0.7)',
            'rgba(249, 115, 22, 0.7)',
            'rgba(239, 68, 68, 0.7)'
          ],
          borderColor: [
            'rgba(34, 197, 94, 1)',
            'rgba(234, 179, 8, 1)',
            'rgba(249, 115, 22, 1)',
            'rgba(239, 68, 68, 1)'
          ],
          borderWidth: 2,
        },
      ],
    };
  };

  const generateLipSyncTimelineChart = (): ChartData<'line', number[], string> | null => {
    if (!report?.lip_sync?.lip_sync_timeline) return null;
    
    return {
      labels: report.lip_sync.lip_sync_timeline.map(item => item.timestamp),
      datasets: [
        {
          label: 'Lip Sync Score',
          data: report.lip_sync.lip_sync_timeline.map(item => item.score),
          borderColor: 'rgba(239, 68, 68, 1)',
          backgroundColor: 'rgba(239, 68, 68, 0.2)',
          tension: 0.3,
          fill: true,
        }
      ]
    };
  };
  
  const generatePeopleDetectionTimelineChart = (): ChartData<'line', number[], string> | null => {
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
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {loading ? (
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
            <p className="mt-4 text-gray-600">Loading analysis report...</p>
          </div>
        ) : error ? (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <p className="text-red-600">{error}</p>
            <button
              onClick={() => navigate('/')}
              className="mt-4 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
            >
              Return to Home
            </button>
          </div>
        ) : report ? (
          <div className="space-y-8">
            {/* Overview Section */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <div className="flex items-center justify-between mb-6">
                <h1 className="text-2xl font-bold text-gray-900">Analysis Report</h1>
                <div className="flex items-center space-x-4">
                  <span className={`px-4 py-2 rounded-full ${getRiskColor(report.risk)} ${getRiskTextColor(report.risk)} font-semibold`}>
                    {report.risk} Risk
                  </span>
                  <span className="text-2xl font-bold text-gray-900">
                    Score: {report.final_score.toFixed(1)}
                  </span>
                </div>
              </div>

              {/* Module Scores */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                <div className="bg-gray-50 rounded-lg p-4">
                  <h3 className="text-lg font-semibold mb-4">Module Scores</h3>
                  <div className="h-64">
                    {barChartData && <Bar data={barChartData} options={barChartOptions} />}
                  </div>
                </div>
                <div className="bg-gray-50 rounded-lg p-4">
                  <h3 className="text-lg font-semibold mb-4">Score Distribution</h3>
                  <div className="h-64">
                    {doughnutChartData && <Doughnut data={doughnutChartData} options={doughnutChartOptions} />}
                  </div>
                </div>
              </div>

              {/* Issues and Warnings */}
              <div className="mt-8">
                <h3 className="text-lg font-semibold mb-4">Detected Issues</h3>
                <div className="space-y-4">
                  {report.reasons.map((reason, index) => (
                    <div key={index} className="flex items-start space-x-3 p-4 bg-gray-50 rounded-lg">
                      <div className={`flex-shrink-0 w-2 h-2 mt-2 rounded-full ${
                        reason.severity === 'critical' ? 'bg-red-500' : 'bg-yellow-500'
                      }`}></div>
                      <div>
                        <p className={`font-medium ${getSeverityColor(reason.severity)}`}>
                          {reason.text}
                        </p>
                        <p className="text-sm text-gray-500 mt-1">
                          Module: {reason.module}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Detailed Analysis Tabs */}
            <div className="bg-white rounded-lg shadow-lg">
              <div className="border-b border-gray-200">
                <nav className="flex -mb-px">
                  {['overview', 'gaze', 'audio', 'multi-person', 'lip-sync'].map((tab) => (
                    <button
                      key={tab}
                      onClick={() => setActiveTab(tab)}
                      className={`px-6 py-3 text-sm font-medium ${
                        activeTab === tab
                          ? 'border-b-2 border-blue-500 text-blue-600'
                          : 'text-gray-500 hover:text-gray-700'
                      }`}
                    >
                      {tab.split('-').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
                    </button>
                  ))}
                </nav>
              </div>

              {/* Tab Content */}
              <div className="p-6">
                {activeTab === 'overview' && (
                  <div className="space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="bg-gray-50 rounded-lg p-4">
                        <h4 className="text-lg font-semibold mb-3">Processing Information</h4>
                        <div className="space-y-2">
                          <p className="text-sm text-gray-600">
                            <span className="font-medium">Processing Time:</span> {report.processing_time.toFixed(1)} seconds
                          </p>
                          <p className="text-sm text-gray-600">
                            <span className="font-medium">Video Duration:</span> {typeof report.video_duration === 'number' ? report.video_duration.toFixed(1) : 'N/A'} seconds
                          </p>
                        </div>
                      </div>
                      <div className="bg-gray-50 rounded-lg p-4">
                        <h4 className="text-lg font-semibold mb-3">Overall Assessment</h4>
                        <div className="space-y-2">
                          <p className="text-sm text-gray-600">
                            <span className="font-medium">Final Score:</span> {report.final_score.toFixed(1)}%
                          </p>
                          <p className="text-sm text-gray-600">
                            <span className="font-medium">Risk Level:</span> {report.risk}
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
                {activeTab === 'gaze' && (
                  <div className="space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="bg-gray-50 rounded-lg p-4">
                        <h4 className="text-lg font-semibold mb-3">Gaze Statistics</h4>
                        <div className="space-y-2">
                          <p className="text-sm text-gray-600">
                            <span className="font-medium">Off-Screen Looks:</span> {report.gaze?.off_screen_count || 0}
                          </p>
                          <p className="text-sm text-gray-600">
                            <span className="font-medium">Average Confidence:</span> {(report.gaze?.average_confidence || 0) * 100}%
                          </p>
                          <p className="text-sm text-gray-600">
                            <span className="font-medium">Module Score:</span> {report.module_scores.gaze.toFixed(1)}%
                          </p>
                        </div>
                      </div>
                      {report.gaze?.gaze_direction_timeline && (
                        <div className="bg-gray-50 rounded-lg p-4">
                          <h4 className="text-lg font-semibold mb-3">Gaze Direction Timeline</h4>
                          <div className="h-64">
                            <Line 
                              data={{
                                labels: report.gaze.gaze_direction_timeline.map(item => item.timestamp),
                                datasets: [{
                                  label: 'Gaze Direction',
                                  data: report.gaze.gaze_direction_timeline.map(item => 
                                    item.direction === 'center' ? 100 : 0
                                  ),
                                  borderColor: 'rgba(34, 197, 94, 1)',
                                  backgroundColor: 'rgba(34, 197, 94, 0.2)',
                                  stepped: true,
                                }]
                              }}
                              options={lineChartOptions}
                            />
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}
                {activeTab === 'audio' && (
                  <div className="space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="bg-gray-50 rounded-lg p-4">
                        <h4 className="text-lg font-semibold mb-3">Audio Analysis</h4>
                        <div className="space-y-2">
                          <p className="text-sm text-gray-600">
                            <span className="font-medium">Multiple Speakers:</span> {report.audio?.multiple_speakers ? 'Yes' : 'No'}
                          </p>
                          <p className="text-sm text-gray-600">
                            <span className="font-medium">Keyboard Typing:</span> {report.audio?.keyboard_typing_count || 0} instances
                          </p>
                          <p className="text-sm text-gray-600">
                            <span className="font-medium">Silence Percentage:</span> {report.audio?.silence_percentage || 0}%
                          </p>
                          <p className="text-sm text-gray-600">
                            <span className="font-medium">Module Score:</span> {report.module_scores.audio.toFixed(1)}%
                          </p>
                        </div>
                      </div>
                      {report.audio?.speaking_timeline && (
                        <div className="bg-gray-50 rounded-lg p-4">
                          <h4 className="text-lg font-semibold mb-3">Speaking Timeline</h4>
                          <div className="overflow-x-auto">
                            <table className="min-w-full divide-y divide-gray-200">
                              <thead>
                                <tr>
                                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Start</th>
                                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">End</th>
                                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Speaker</th>
                                </tr>
                              </thead>
                              <tbody className="divide-y divide-gray-200">
                                {report.audio.speaking_timeline.map((item, idx) => (
                                  <tr key={idx}>
                                    <td className="px-4 py-2 text-sm text-gray-600">{item.start}</td>
                                    <td className="px-4 py-2 text-sm text-gray-600">{item.end}</td>
                                    <td className="px-4 py-2">
                                      <span className={`px-2 py-1 text-xs rounded-full ${
                                        item.speaker === 'primary' 
                                          ? 'bg-blue-100 text-blue-800' 
                                          : 'bg-red-100 text-red-800'
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
                      )}
                    </div>
                  </div>
                )}
                {activeTab === 'multi-person' && (
                  <div className="space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="bg-gray-50 rounded-lg p-4">
                        <h4 className="text-lg font-semibold mb-3">People Detection</h4>
                        <div className="space-y-2">
                          <p className="text-sm text-gray-600">
                            <span className="font-medium">Max People Detected:</span> {report.multi_person?.max_people_detected || 0}
                          </p>
                          <p className="text-sm text-gray-600">
                            <span className="font-medium">Time with Multiple People:</span> {report.multi_person?.time_with_multiple_people || 0}s
                          </p>
                          <p className="text-sm text-gray-600">
                            <span className="font-medium">Different Faces Detected:</span> {report.multi_person?.different_faces_detected || 0}
                          </p>
                          <p className="text-sm text-gray-600">
                            <span className="font-medium">Module Score:</span> {report.module_scores.multi_person.toFixed(1)}%
                          </p>
                        </div>
                      </div>
                      {report.multi_person?.people_detection_timeline && peopleDetectionTimelineChart && (
                        <div className="bg-gray-50 rounded-lg p-4">
                          <h4 className="text-lg font-semibold mb-3">People Detection Timeline</h4>
                          <div className="h-64">
                            <Line 
                              data={peopleDetectionTimelineChart}
                              options={lineChartOptions}
                            />
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}
                {activeTab === 'lip-sync' && (
                  <div className="space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="bg-gray-50 rounded-lg p-4">
                        <h4 className="text-lg font-semibold mb-3">Lip Sync Analysis</h4>
                        <div className="space-y-2">
                          <p className="text-sm text-gray-600">
                            <span className="font-medium">Lip Sync Score:</span> {report.lip_sync?.lip_sync_score || 0}%
                          </p>
                          <p className="text-sm text-gray-600">
                            <span className="font-medium">Major Desync Detected:</span> {report.lip_sync?.major_lip_desync_detected ? 'Yes' : 'No'}
                          </p>
                          <p className="text-sm text-gray-600">
                            <span className="font-medium">Module Score:</span> {report.module_scores.lip_sync.toFixed(1)}%
                          </p>
                        </div>
                      </div>
                      {report.lip_sync?.lip_sync_timeline && lipSyncTimelineChart && (
                        <div className="bg-gray-50 rounded-lg p-4">
                          <h4 className="text-lg font-semibold mb-3">Lip Sync Timeline</h4>
                          <div className="h-64">
                            <Line 
                              data={lipSyncTimelineChart}
                              options={lipSyncChartOptions}
                            />
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
};

export default ReportPage; 