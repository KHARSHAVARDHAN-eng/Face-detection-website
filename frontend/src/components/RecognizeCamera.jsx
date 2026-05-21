import React, { useEffect, useRef, useState } from 'react';
import Webcam from 'react-webcam';

export default function RecognizeCamera({ onCapture, isDisabled }) {
  const webcamRef = useRef(null);
  const [cameraState, setCameraState] = useState('initializing'); // 'initializing' | 'ready' | 'denied'
  const [isLoaded, setIsLoaded] = useState(false);

  // Wrap onCapture in a ref to prevent recreation of interval on every render
  const onCaptureRef = useRef(onCapture);
  useEffect(() => {
    onCaptureRef.current = onCapture;
  }, [onCapture]);

  const handleUserMedia = () => {
    setCameraState('ready');
    setIsLoaded(true);
  };

  const handleUserMediaError = (error) => {
    console.error("Camera access error:", error);
    setCameraState('denied');
  };

  useEffect(() => {
    if (isDisabled || cameraState !== 'ready') return;

    const interval = setInterval(() => {
      if (webcamRef.current && isLoaded) {
        const screenshot = webcamRef.current.getScreenshot();
        if (screenshot) {
          if (onCaptureRef.current) {
            onCaptureRef.current(screenshot);
          }
        }
      }
    }, 1500); // Real-time capture every 1.5 seconds

    return () => clearInterval(interval);
  }, [isLoaded, isDisabled, cameraState]);

  return (
    <div className="flex flex-col items-center select-none">
      <style>{`
        @keyframes pulse-glow {
          0%, 100% { box-shadow: 0 0 20px rgba(34, 197, 94, 0.2); }
          50% { box-shadow: 0 0 35px rgba(34, 197, 94, 0.5); }
        }
        @keyframes scan-line {
          0% { top: 0%; opacity: 0; }
          10% { opacity: 1; }
          90% { opacity: 1; }
          100% { top: 100%; opacity: 0; }
        }
        .pulse-border {
          animation: pulse-glow 2s infinite ease-in-out;
        }
        .scan-laser {
          animation: scan-line 3s infinite linear;
        }
      `}</style>

      <div className="relative w-[280px] h-[280px] flex items-center justify-center">
        {/* Outer green pulsing glow border ring */}
        {cameraState === 'ready' && (
          <div className="absolute inset-0 rounded-full border-4 border-green-500 pulse-border pointer-events-none z-10"></div>
        )}

        {/* Webcam Mask */}
        <div className="w-[260px] h-[260px] rounded-full overflow-hidden bg-black flex items-center justify-center relative border border-slate-800">
          {cameraState === 'denied' ? (
            <div className="absolute inset-0 bg-slate-950 flex flex-col items-center justify-center text-center p-6 text-white text-xs">
              <span className="text-2xl mb-2">📷❌</span>
              <span className="font-semibold text-red-400 mb-1">Camera Access Blocked</span>
              <span className="text-slate-400">Please enable camera permission in your browser and refresh.</span>
            </div>
          ) : (
            <Webcam
              ref={webcamRef}
              audio={false}
              mirrored
              screenshotFormat="image/jpeg"
              onUserMedia={handleUserMedia}
              onUserMediaError={handleUserMediaError}
              videoConstraints={{
                width: 640,
                height: 640,
                facingMode: "user"
              }}
              className="w-full h-full object-cover rounded-full"
            />
          )}

          {/* Laser scanning sweep line */}
          {isLoaded && cameraState === 'ready' && (
            <div className="absolute left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-green-400 to-transparent shadow-[0_0_10px_#22c55e] scan-laser pointer-events-none"></div>
          )}

          {/* Initializing State */}
          {cameraState === 'initializing' && (
            <div className="absolute inset-0 bg-slate-950/80 flex flex-col items-center justify-center text-white text-sm backdrop-blur-md">
              <div className="w-8 h-8 border-4 border-t-green-500 border-slate-700 rounded-full animate-spin mb-2"></div>
              <span className="font-medium tracking-wide">Starting Face Authentication...</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
