import React, { ReactNode } from 'react';
import { Link, useLocation } from 'react-router-dom';

interface LayoutProps {
  children: ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const location = useLocation();
  
  const isActiveRoute = (path: string) => {
    return location.pathname === path ? 
      'text-primary-600 border-primary-600' : 
      'text-dark-500 hover:text-primary-600 border-transparent';
  };

  return (
    <div className="min-h-screen flex flex-col">
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16 items-center">
            <div className="flex items-center">
              <Link to="/" className="flex items-center">
                <span className="text-2xl font-bold text-primary-600">VIZH<span className="text-secondary-500">.AI</span></span>
              </Link>
              <nav className="ml-10 flex space-x-8">
                <Link 
                  to="/" 
                  className={`${isActiveRoute('/')} inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium`}
                >
                  Home
                </Link>
                <Link 
                  to="/upload" 
                  className={`${isActiveRoute('/upload')} inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium`}
                >
                  Upload
                </Link>
              </nav>
            </div>
            <div className="flex items-center">
              <button className="btn-outline text-sm">
                Help
              </button>
            </div>
          </div>
        </div>
      </header>
      
      <main className="flex-grow">
        {children}
      </main>
      
      <footer className="bg-white border-t border-gray-200 py-4">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <p className="text-sm text-center text-dark-400">
            © {new Date().getFullYear()} VIZH.AI · All rights reserved
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Layout; 