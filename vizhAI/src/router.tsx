import { createBrowserRouter } from 'react-router-dom';
import App from './App';
import HomePage from './pages/HomePage';
import UploadPage from './pages/UploadPage';
import ReportPage from './pages/ReportPage';
import NotFoundPage from './pages/NotFoundPage';
import Layout from './components/Layout';
import { Outlet } from 'react-router-dom';

export const router = createBrowserRouter(
  [
    {
      element: <Layout><Outlet /></Layout>,
      children: [
        {
          path: '/',
          element: <HomePage />,
        },
        {
          path: '/upload',
          element: <UploadPage />,
        },
        {
          path: '/report/:id',
          element: <ReportPage />,
        },
        {
          path: '*',
          element: <NotFoundPage />,
        },
      ],
    },
  ]
); 