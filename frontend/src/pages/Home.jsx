import React from 'react';
import { Link } from 'react-router-dom';
import { ScanFace, UserPlus } from 'lucide-react';

export default function Home() {
  return (
    <div className="flex flex-col items-center justify-center py-20 text-center">
      <h1 className="text-5xl font-extrabold text-gray-900 tracking-tight mb-4">
        AI-Powered Face Recognition
      </h1>
      <p className="text-xl text-gray-600 max-w-2xl mb-12">
        Securely register and authenticate users using state-of-the-art facial recognition technology built with FastAPI and React.
      </p>
      
      <div className="flex flex-col sm:flex-row space-y-4 sm:space-y-0 sm:space-x-6">
        <Link to="/register" className="flex items-center justify-center px-8 py-4 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition shadow-lg text-lg font-medium">
          <UserPlus className="w-6 h-6 mr-3" />
          Register New Face
        </Link>
        <Link to="/recognize" className="flex items-center justify-center px-8 py-4 bg-white text-gray-800 border-2 border-gray-200 rounded-lg hover:border-blue-600 hover:text-blue-600 transition shadow-sm text-lg font-medium">
          <ScanFace className="w-6 h-6 mr-3" />
          Recognize User
        </Link>
      </div>
    </div>
  );
}
