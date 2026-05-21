import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Home from './pages/Home';
import Register from './pages/Register';
import Recognize from './pages/Recognize';
import Dashboard from './pages/Dashboard';
import { Camera, Users, UserPlus, ScanFace, Menu } from 'lucide-react';
import { Toaster } from 'react-hot-toast';

function App() {
  const [isMenuOpen, setIsMenuOpen] = React.useState(false);

  return (
    <Router>
      <div className="min-h-screen flex flex-col">
        <nav className="bg-white shadow-sm border-b border-gray-200">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between h-16">
              <div className="flex justify-between h-16 w-full">
                <div className="flex-shrink-0 flex items-center">
                  <Camera className="h-8 w-8 text-blue-600" />
                  <span className="ml-2 text-xl font-bold text-gray-900">FaceAuth</span>
                </div>
                {/* Desktop Menu */}
                <div className="hidden sm:ml-6 sm:flex sm:space-x-8">
                  <Link to="/" className="border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700 inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium">
                    Home
                  </Link>
                  <Link to="/register" className="border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700 inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium">
                    <UserPlus className="w-4 h-4 mr-1"/> Register
                  </Link>
                  <Link to="/recognize" className="border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700 inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium">
                    <ScanFace className="w-4 h-4 mr-1"/> Recognize
                  </Link>
                  <Link to="/dashboard" className="border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700 inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium">
                    <Users className="w-4 h-4 mr-1"/> Users
                  </Link>
                </div>
                {/* Mobile menu button */}
                <div className="flex items-center sm:hidden">
                  <button onClick={() => setIsMenuOpen(!isMenuOpen)} className="text-gray-500 hover:text-gray-700">
                    <Menu className="h-6 w-6" />
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Mobile Menu */}
          {isMenuOpen && (
            <div className="sm:hidden border-t border-gray-200 pt-2 pb-4 space-y-1">
              <Link to="/" onClick={() => setIsMenuOpen(false)} className="block pl-3 pr-4 py-2 border-l-4 border-transparent text-base font-medium text-gray-600 hover:text-gray-800 hover:bg-gray-50 hover:border-gray-300">Home</Link>
              <Link to="/register" onClick={() => setIsMenuOpen(false)} className="block pl-3 pr-4 py-2 border-l-4 border-transparent text-base font-medium text-gray-600 hover:text-gray-800 hover:bg-gray-50 hover:border-gray-300">Register</Link>
              <Link to="/recognize" onClick={() => setIsMenuOpen(false)} className="block pl-3 pr-4 py-2 border-l-4 border-transparent text-base font-medium text-gray-600 hover:text-gray-800 hover:bg-gray-50 hover:border-gray-300">Recognize</Link>
              <Link to="/dashboard" onClick={() => setIsMenuOpen(false)} className="block pl-3 pr-4 py-2 border-l-4 border-transparent text-base font-medium text-gray-600 hover:text-gray-800 hover:bg-gray-50 hover:border-gray-300">Users</Link>
            </div>
          )}
        </nav>

        <main className="flex-grow max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-8">
          <Toaster position="top-right" />
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/register" element={<Register />} />
            <Route path="/recognize" element={<Recognize />} />
            <Route path="/dashboard" element={<Dashboard />} />
          </Routes>
        </main>
        
        <footer className="bg-white border-t border-gray-200 py-6">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-sm text-gray-500">
            &copy; 2026 FaceAuth AI. All rights reserved.
          </div>
        </footer>
      </div>
    </Router>
  );
}

export default App;
