import React, { useState } from 'react';

import { useNavigate } from 'react-router-dom';

import WebcamCapture from '../components/WebcamCapture';

import {
  api,
  dataURLtoBlob
} from '../services/api';


export default function Recognize() {

  const navigate = useNavigate();

  const [isActive, setIsActive] = useState(false);

  const [loading, setLoading] = useState(false);

  const [result, setResult] = useState(null);

  const [error, setError] = useState(null);

  // START RECOGNITION
  const handleStartRecognition = () => {

    setResult(null);

    setError(null);

    setIsActive(true);
  };

  // LIVE FRAME RECOGNITION
  const handleRecognizeFrame = async (
    screenshot
  ) => {

    // PREVENT SPAM REQUESTS
    if (loading) return;

    try {

      setLoading(true);

      setError(null);

      const blob = dataURLtoBlob(
        screenshot
      );

      const formData = new FormData();

      formData.append(
        'image',
        blob,
        'capture.jpg'
      );

      const response = await api.post(
        '/recognize',
        formData,
        {
          headers: {
            'Content-Type':
              'multipart/form-data',
          },
        }
      );

      console.log(
        "RECOGNITION RESULT:",
        response.data
      );

      setResult(response.data);

    } catch (err) {

      console.error(
        "Recognition error:",
        err
      );

      const errorMessage =
        err.response?.data?.detail ||
        err.message ||
        'Recognition failed';

      setError(errorMessage);

    } finally {

      setLoading(false);
    }
  };

  return (

    <div className="max-w-5xl mx-auto bg-slate-900 border border-slate-800 p-8 md:p-12 rounded-3xl shadow-2xl">

      {/* TITLE */}
      <h2 className="text-3xl font-extrabold text-center text-white mb-2">

        Multi-User Biometric Verification

      </h2>

      <p className="text-center text-slate-400 text-sm mb-10">

        Real-time multi-face recognition system

      </p>

      {/* CAMERA */}
      <div className="bg-slate-950/40 p-6 rounded-3xl border border-slate-800/50 flex flex-col items-center justify-center mb-8 w-full">

        <WebcamCapture
          mode="recognition"
          isActive={isActive}
          onRecognizeFrame={handleRecognizeFrame}
          faces={result?.faces || []}
          imageWidth={result?.width}
          imageHeight={result?.height}
        />

        {!isActive && (

          <button
            type="button"
            onClick={
              handleStartRecognition
            }
            className="mt-6 px-8 py-3 bg-sky-500 hover:bg-sky-600 text-white rounded-xl font-bold"
          >

            Start Recognition

          </button>
        )}

      </div>

      {/* LOADING */}
      {loading && (

        <div className="text-center py-2">

          <div className="inline-flex items-center px-4 py-2 bg-green-500/10 border border-green-500/20 text-green-400 rounded-full text-xs font-semibold animate-pulse">

            Verifying Biometric Signatures...

          </div>

        </div>
      )}

      {/* ERROR */}
      {error && (

        <div className="p-4 mt-6 bg-red-950/30 border border-red-500/30 rounded-2xl text-center text-red-400 text-sm font-medium">

          {error}

        </div>
      )}

      {/* DETECTED USERS GRID */}
      {result && result.faces && result.faces.length > 0 && (
        <div className="mt-6">
          <h3 className="text-lg font-bold text-white mb-4">Detected Faces</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {result.faces.map((face, index) => {
              const isMatched = face.status === 'matched';
              return (
                <div 
                  key={index} 
                  className={`p-4 rounded-2xl border flex items-center justify-between transition-all duration-300 ${
                    isMatched 
                      ? 'border-emerald-500/30 bg-emerald-950/20 text-emerald-400 shadow-emerald-950/10' 
                      : 'border-rose-500/30 bg-rose-950/20 text-rose-400 shadow-rose-950/10'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <span className="text-xl">{isMatched ? '👤' : '❓'}</span>
                    <div>
                      <h4 className="font-bold text-white text-base leading-tight">
                        {face.name}
                      </h4>
                      <p className="text-xs text-slate-400 mt-1">
                        Status: <span className={isMatched ? 'text-emerald-400 font-medium' : 'text-rose-400 font-medium'}>
                          {isMatched ? 'Verified' : 'Unknown'}
                        </span>
                      </p>
                    </div>
                  </div>
                  {face.confidence > 0 && (
                    <div className="text-right">
                      <span className="font-mono font-bold text-sm bg-slate-950/60 px-2 py-1 rounded-lg border border-slate-800">
                        {face.confidence.toFixed(1)}%
                      </span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

    </div>
  );
}