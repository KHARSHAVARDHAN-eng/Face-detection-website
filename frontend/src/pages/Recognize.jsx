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

    <div className="max-w-2xl mx-auto bg-slate-900 border border-slate-800 p-8 md:p-12 rounded-3xl shadow-2xl">

      {/* TITLE */}
      <h2 className="text-3xl font-extrabold text-center text-white mb-2">

        Biometric Verification

      </h2>

      <p className="text-center text-slate-400 text-sm mb-10">

        Real-time facial recognition system

      </p>

      {/* CAMERA */}
      <div className="bg-slate-950/40 p-6 rounded-3xl border border-slate-800/50 flex flex-col items-center justify-center mb-8">

        <WebcamCapture
          mode="recognition"
          isActive={isActive}
          onRecognizeFrame={
            handleRecognizeFrame
          }
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

            Verifying Biometric Signature...

          </div>

        </div>
      )}

      {/* ERROR */}
      {error && (

        <div className="p-4 mt-6 bg-red-950/30 border border-red-500/30 rounded-2xl text-center text-red-400 text-sm font-medium">

          {error}

        </div>
      )}

      {/* MATCHED */}
      {result &&
        result.status === 'matched' && (

          <div className="mt-6 p-8 rounded-3xl border border-green-500/30 bg-green-950/20 text-center">

            <div className="text-4xl mb-3">
              ✅
            </div>

            <h3 className="text-2xl font-bold text-green-400 mb-2">

              Identity Verified

            </h3>

            <p className="text-white text-xl">

              Welcome,
              {' '}

              <span className="font-black">

                {result.name}

              </span>

            </p>

            {result.confidence !== undefined && result.confidence !== null && (
              <p className="text-slate-400 text-xs mt-4">
                Confidence Score: <span className="text-green-400 font-mono font-bold">{(result.confidence * 100).toFixed(2)}%</span>
              </p>
            )}

          </div>
        )}

      {/* INVALID */}
      {result &&
        result.status !== 'matched' && (

          <div className="mt-6 p-8 rounded-3xl border border-red-500/40 bg-red-950/20 text-center">

            <div className="text-4xl mb-3">
              🚨
            </div>

            <h3 className="text-2xl font-black text-red-500 mb-2">

              Invalid Person

            </h3>

            <p className="text-slate-300 text-sm mb-4">

              Please Register

            </p>

            {result.confidence !== undefined && result.confidence !== null && (
              <p className="text-slate-400 text-xs mb-6">
                Highest Similarity: <span className="text-red-400 font-mono font-bold">{(result.confidence * 100).toFixed(2)}%</span>
              </p>
            )}

            <button
              onClick={() =>
                navigate('/register')
              }
              className="px-6 py-2.5 bg-red-600 hover:bg-red-700 text-white font-bold rounded-xl"
            >

              Go to Registration

            </button>

          </div>
        )}

    </div>
  );
}