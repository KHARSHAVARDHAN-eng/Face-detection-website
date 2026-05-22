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
        <div className="mt-8">
          <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
            <span>🔍</span> Detected Faces ({result.faces.length})
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {result.faces.map((face, index) => {
              const isMatched = face.status === 'matched' || face.authentication_status === 'verified';
              const isSpoof = face.status === 'spoof' || face.spoof_detected === true || face.authentication_status === 'denied';
              const isUnknown = face.status === 'unknown' || face.authentication_status === 'unregistered' || face.name === 'Invalid User';

              let cardStyle = '';
              let statusText = '';
              let statusColor = '';
              let icon = '';

              if (isMatched) {
                cardStyle = 'border-emerald-500/30 bg-emerald-950/20 text-emerald-400 shadow-emerald-950/10';
                statusText = 'Live Person Verified';
                statusColor = 'text-emerald-400';
                icon = '👤';
              } else if (isSpoof) {
                cardStyle = 'border-rose-500/40 bg-rose-950/30 text-rose-400 shadow-rose-950/20 animate-pulse border-2';
                statusText = '';
                statusColor = '';
                icon = '🚨';
              } else {
                cardStyle = 'border-amber-500/30 bg-amber-950/20 text-amber-400 shadow-amber-950/10';
                statusText = 'Invalid / Unregistered User';
                statusColor = 'text-amber-400';
                icon = '❓';
              }

              return (
                <div 
                  key={index} 
                  className={`p-5 rounded-2xl border flex items-center justify-between transition-all duration-300 ${cardStyle}`}
                >
                  <div className="flex items-center gap-3">
                    <span className="text-2xl">{icon}</span>
                    <div>
                      <h4 className="font-bold text-white text-base leading-tight">
                        {face.name}
                      </h4>
                      {isSpoof ? (
                        <div className="text-xs text-rose-400 mt-2 flex flex-col gap-0.5 font-semibold">
                          <span className="text-rose-400 font-extrabold uppercase tracking-wider text-[10px]">
                            ⚠️ SPOOF / PROXY ATTEMPT DETECTED
                          </span>
                          <span className="text-rose-300">
                            📱 Phone Screen Replay Attack
                          </span>
                          <span className="text-rose-500 font-bold uppercase text-[11px] tracking-widest mt-1">
                            🚫 Access Denied
                          </span>
                        </div>
                      ) : (
                        <p className="text-xs text-slate-400 mt-1">
                          Status: <span className={`${statusColor} font-medium`}>
                            {statusText}
                          </span>
                        </p>
                      )}
                    </div>
                  </div>
                  {face.confidence !== undefined && (
                    <div className="text-right flex flex-col items-end gap-1">
                      <span className="font-mono font-bold text-xs bg-slate-950/60 px-2.5 py-1 rounded-lg border border-slate-800 text-slate-300">
                        {face.confidence.toFixed(1)}% Match
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