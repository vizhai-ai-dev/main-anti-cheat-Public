import React from 'react';
import { Link } from 'react-router-dom';

const NotFoundPage: React.FC = () => {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 text-center">
      <h1 className="text-6xl font-extrabold text-primary-600">404</h1>
      <p className="mt-2 text-3xl font-bold text-dark-800">Page not found</p>
      <p className="mt-4 text-lg text-dark-500">
        Sorry, we couldn't find the page you're looking for.
      </p>
      <div className="mt-10">
        <Link to="/" className="btn-primary">
          Go back home
        </Link>
      </div>
    </div>
  );
};

export default NotFoundPage; 