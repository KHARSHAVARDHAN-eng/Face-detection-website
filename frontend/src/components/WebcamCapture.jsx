import React, { useEffect, useRef, useState } from 'react';
import Webcam from 'react-webcam';
import { FaceMesh } from '@mediapipe/face_mesh';
import { Camera } from '@mediapipe/camera_utils';
import { api, dataURLtoBlob } from '../services/api';

export default function WebcamCapture({
  mode = 'enrollment',
  isActive = false,
  onScanComplete,
  onRecognizeFrame,
  registrationSuccess = false,
  faces = [],
  imageWidth = 1280,
  imageHeight = 720
}) {

  const webcamRef = useRef(null);

  const faceMeshRef = useRef(null);

  const cameraRef = useRef(null);

  const embeddingsRef = useRef([]);

  const thumbnailRef = useRef(null);

  const processingRef = useRef(false);

  const recognizeProcessingRef = useRef(false);

  const currentPoseRef = useRef(null);

  const consecutivePoseRef = useRef(0);

  const lastMovementRef = useRef(Date.now());

  const scanStartRef = useRef(Date.now());

  const guideIntervalRef = useRef(null);

  const capturedPosesRef = useRef({
    straight: false,
    left: false,
    right: false,
    up: false,
    down: false,
  });

  const [capturedPoses, setCapturedPoses] = useState({
    straight: false,
    left: false,
    right: false,
    up: false,
    down: false,
  });

  const [status, setStatus] = useState('IDLE');
  const statusRef = useRef('IDLE');

  const [instruction, setInstruction] = useState(
    'Awaiting scanner activation...'
  );

  // RESET
  const resetScanner = () => {

    embeddingsRef.current = [];

    thumbnailRef.current = null;

    processingRef.current = false;

    recognizeProcessingRef.current = false;

    currentPoseRef.current = null;

    consecutivePoseRef.current = 0;

    const resetState = {
      straight: false,
      left: false,
      right: false,
      up: false,
      down: false,
    };

    capturedPosesRef.current = resetState;
    setCapturedPoses(resetState);

    statusRef.current = 'IDLE';
    setStatus('IDLE');

    setInstruction(
      mode === 'enrollment'
        ? 'Awaiting scanner activation...'
        : 'Ready for recognition'
    );

    if (cameraRef.current) {
      cameraRef.current.stop();
      cameraRef.current = null;
    }

    if (guideIntervalRef.current) {
      clearInterval(guideIntervalRef.current);
      guideIntervalRef.current = null;
    }
  };

  // START
  useEffect(() => {

    if (!isActive) {
      resetScanner();
      return;
    }

    if (registrationSuccess) return;

    initializeFaceMesh();

    if (mode === 'enrollment') {
      scanStartRef.current = Date.now();
      lastMovementRef.current = Date.now();
      statusRef.current = 'SCANNING';
      setStatus('SCANNING');
      setInstruction('Please move your face slowly to complete enrollment.');

      guideIntervalRef.current = setInterval(() => {
        const now = Date.now();
        const elapsed = now - scanStartRef.current;
        const idle = now - lastMovementRef.current;
        const missingLR = !capturedPosesRef.current.left || !capturedPosesRef.current.right;
        const missingUD = !capturedPosesRef.current.up || !capturedPosesRef.current.down;

        if (statusRef.current !== 'SCANNING') {
          return;
        }

        if (idle > 5000) {
          setInstruction('Please move your face slowly.');
          return;
        }

        if (elapsed > 20000 && missingLR) {
          setInstruction('Please turn your face left and right slowly.');
          return;
        }

        if (elapsed > 20000 && missingUD) {
          setInstruction('Please tilt your face up and down slightly.');
          return;
        }

        setInstruction('Please move your face slowly to complete enrollment.');
      }, 500);
    }

    return () => {
      if (cameraRef.current) {
        cameraRef.current.stop();
      }
      if (guideIntervalRef.current) {
        clearInterval(guideIntervalRef.current);
        guideIntervalRef.current = null;
      }
    };

  }, [isActive, registrationSuccess]);

  const poseThresholds = {
    yaw: 0.02,
    pitch: 0.02,
  };

  const getPoseFromLandmarks = (landmarks) => {
    const nose = landmarks[1];
    const leftEye = landmarks[33];
    const rightEye = landmarks[263];
    const chin = landmarks[152];

    if (!nose || !leftEye || !rightEye || !chin) {
      return null;
    }

    const faceCenterX = (leftEye.x + rightEye.x) / 2;
    const faceCenterY = (leftEye.y + rightEye.y + chin.y) / 3;
    const yaw = nose.x - faceCenterX;
    const pitch = nose.y - faceCenterY;

    if (Math.abs(pitch) > Math.abs(yaw) * 0.8 && Math.abs(pitch) > poseThresholds.pitch) {
      return pitch < 0 ? 'up' : 'down';
    }

    if (Math.abs(yaw) > poseThresholds.yaw) {
      return yaw < 0 ? 'left' : 'right';
    }

    return 'straight';
  };

  const allPosesCaptured = (poses) =>
    Object.values(poses).every(Boolean);

  const completeEnrollment = () => {
    statusRef.current = 'COMPLETE';
    setStatus('COMPLETE');
    setInstruction('All required poses captured. Finalizing enrollment...');
    if (guideIntervalRef.current) {
      clearInterval(guideIntervalRef.current);
      guideIntervalRef.current = null;
    }
    if (cameraRef.current) {
      cameraRef.current.stop();
      cameraRef.current = null;
    }
    if (onScanComplete) {
      onScanComplete(embeddingsRef.current, thumbnailRef.current);
    }
  };

  const capturePoseEmbedding = async (pose) => {
    if (processingRef.current) return;

    processingRef.current = true;

    try {
      const screenshot = webcamRef.current?.getScreenshot();
      if (!screenshot) {
        return;
      }

      if (!thumbnailRef.current) {
        thumbnailRef.current = screenshot;
      }

      setInstruction('Recording a good enrollment frame...');

      const blob = dataURLtoBlob(screenshot);
      const formData = new FormData();
      formData.append('image', blob, 'frame.jpg');

      const response = await api.post('/process-frame', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data && response.data.success) {
        embeddingsRef.current.push(response.data.embedding);
        const nextCaptured = {
          ...capturedPosesRef.current,
          [pose]: true,
        };

        capturedPosesRef.current = nextCaptured;
        setCapturedPoses(nextCaptured);

        const nextCount = Object.values(nextCaptured).filter(Boolean).length;
        const remaining = 5 - nextCount;
        setInstruction(
          remaining > 0
            ? 'Good frame captured. Continue moving slowly.'
            : 'Great job — finishing enrollment now.'
        );

        if (allPosesCaptured(nextCaptured)) {
          completeEnrollment();
        }
      } else {
        setInstruction('Could not capture a clear frame. Please continue moving slowly.');
      }
    } catch (err) {
      console.error('Embedding failed', err);
      setInstruction('Embedding failed. Please try that pose again.');
    } finally {
      processingRef.current = false;
    }
  };

  const handlePoseDetection = (landmarks) => {
    const pose = getPoseFromLandmarks(landmarks);
    if (!pose || statusRef.current !== 'SCANNING') return;

    const now = Date.now();
    lastMovementRef.current = now;

    if (currentPoseRef.current !== pose) {
      currentPoseRef.current = pose;
      consecutivePoseRef.current = 1;
      setInstruction('Please move your face slowly to complete enrollment.');
      return;
    }

    consecutivePoseRef.current += 1;
    const poseAlreadyCaptured = capturedPosesRef.current[pose];

    if (!poseAlreadyCaptured && consecutivePoseRef.current >= 2) {
      capturePoseEmbedding(pose);
      return;
    }

    if (!poseAlreadyCaptured) {
      setInstruction('Please move your face slowly to complete enrollment.');
    }
  };

  // FACEMESH
  const initializeFaceMesh = () => {

    if (faceMeshRef.current) return;

    const faceMesh = new FaceMesh({

      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`

    });

    faceMesh.setOptions({

      maxNumFaces: mode === 'enrollment' ? 1 : 10,

      refineLandmarks: true,

      minDetectionConfidence: 0.5,

      minTrackingConfidence: 0.5,
    });

    faceMesh.onResults(async (results) => {

      if (
        !results.multiFaceLandmarks ||
        results.multiFaceLandmarks.length === 0
      ) {
        return;
      }

      // ENROLLMENT
      if (mode === 'enrollment') {
        if (!results.multiFaceLandmarks) {
          return;
        }

        handlePoseDetection(results.multiFaceLandmarks[0]);
      }

      // RECOGNITION
      if (mode === 'recognition') {

        if (recognizeProcessingRef.current) return;

        recognizeProcessingRef.current = true;

        try {

          const screenshot =
            webcamRef.current?.getScreenshot();

          if (
            screenshot &&
            onRecognizeFrame
          ) {

            await onRecognizeFrame(
              screenshot
            );
          }

        } catch (err) {

          console.error(
            "Recognition frame failed",
            err
          );

        } finally {

          setTimeout(() => {

            recognizeProcessingRef.current = false;

          }, 800);
        }
      }

    });

    faceMeshRef.current = faceMesh;
  };

  // CAMERA READY
  const handleUserMedia = () => {

    const video =
      webcamRef.current?.video;

    if (!video) return;

    const videoWidth = mode === 'enrollment' ? 640 : 1280;
    const videoHeight = mode === 'enrollment' ? 640 : 720;

    const camera = new Camera(video, {

      onFrame: async () => {

        if (
          faceMeshRef.current &&
          video.readyState === 4
        ) {

          await faceMeshRef.current.send({
            image: video
          });
        }
      },

      width: videoWidth,

      height: videoHeight,
    });

    cameraRef.current = camera;

    camera.start();
  };

  // PROGRESS
  const capturedCount = Object.values(capturedPoses).filter(Boolean).length;
  const progress =
    mode === 'enrollment'
      ? (capturedCount / 5) * 100
      : 100;

  if (mode === 'recognition') {
    const hasSpoof = faces.some(face => face.status === 'spoof' || face.spoof_detected === true || face.authentication_status === 'denied');
    
    return (
      <div className="flex flex-col items-center w-full">
        {/* CAMERA CARD */}
        <div className="w-full max-w-4xl aspect-[16/9] rounded-3xl overflow-hidden border border-slate-800 bg-black relative shadow-2xl">
          <Webcam
            ref={webcamRef}
            audio={false}
            mirrored
            screenshotFormat="image/jpeg"
            onUserMedia={handleUserMedia}
            videoConstraints={{
              width: 1280,
              height: 720,
              facingMode: 'user'
            }}
            style={{
              width: '100%',
              height: '100%',
              objectFit: 'cover'
            }}
          />

          {/* FLASHING RED OVERLAY FOR SPOOF DETECTED */}
          {hasSpoof && (
            <>
              <div className="absolute inset-0 border-[6px] border-rose-600/80 pointer-events-none bg-rose-950/15 animate-[pulse_1s_infinite] z-20 rounded-3xl" />
              <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-30 bg-rose-950/95 border border-rose-500/50 backdrop-blur-sm px-6 py-2 rounded-2xl flex items-center gap-3 shadow-2xl animate-bounce">
                <span className="relative flex h-3 w-3">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-rose-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-3 w-3 bg-rose-500"></span>
                </span>
                <span className="text-rose-100 font-extrabold text-xs uppercase tracking-wider">
                  ⚠️ SPOOF / PROXY ATTEMPT DETECTED
                </span>
              </div>
            </>
          )}
          
          {/* FACES OVERLAY */}
          {isActive && faces && faces.map((face, idx) => {
            const imgWidth = imageWidth || 1280;
            const imgHeight = imageHeight || 720;
            const left = ((imgWidth - face.box[2]) / imgWidth) * 100;
            const top = (face.box[1] / imgHeight) * 100;
            const width = ((face.box[2] - face.box[0]) / imgWidth) * 100;
            const height = ((face.box[3] - face.box[1]) / imgHeight) * 100;
            
            const isSpoof = face.status === 'spoof' || face.spoof_detected === true || face.authentication_status === 'denied';
            const isMatched = face.status === 'matched' || face.authentication_status === 'verified';
            const isUnknown = face.status === 'unknown' || face.authentication_status === 'unregistered' || face.name === 'Invalid User';
            
            let labelBgClass = 'bg-amber-500/90 border border-amber-400/30 text-white shadow-amber-950/20';
            let boxClass = 'border-2 border-amber-500 shadow-[0_0_15px_rgba(245,158,11,0.5)]';
            let icon = '❓';
            
            if (isSpoof) {
              labelBgClass = 'bg-rose-600/95 border border-rose-500/40 text-white shadow-rose-950/40 animate-pulse';
              boxClass = 'border-2 border-rose-500 shadow-[0_0_20px_rgba(244,63,94,0.8),inset_0_0_10px_rgba(244,63,94,0.4)] animate-pulse';
              icon = '🚨';
            } else if (isMatched) {
              labelBgClass = 'bg-emerald-500/90 border border-emerald-400/30 text-white shadow-emerald-950/20';
              boxClass = 'border-2 border-emerald-500 shadow-[0_0_15px_rgba(16,185,129,0.6)]';
              icon = '👤';
            }

            return (
              <div 
                key={idx}
                className={`absolute transition-all duration-150 rounded-xl ${boxClass}`}
                style={{
                  left: `${left}%`,
                  top: `${top}%`,
                  width: `${width}%`,
                  height: `${height}%`,
                }}
              >
                {/* FLOATING LABEL */}
                <div className={`absolute -top-10 left-1/2 transform -translate-x-1/2 px-3 py-1 rounded-full text-[10px] font-semibold whitespace-nowrap shadow-xl backdrop-blur-sm flex flex-col items-center gap-0.5 ${labelBgClass}`}>
                  <div className="flex items-center gap-1">
                    <span>{icon}</span>
                    <span>{face.name}</span>
                    {face.confidence !== undefined && face.confidence >= 0 && (
                      <span className="font-mono font-bold opacity-90 border-l border-white/20 pl-1">
                        {face.confidence.toFixed(1)}%
                      </span>
                    )}
                  </div>
                  {isSpoof ? (
                    <span className="text-[8px] opacity-90 font-bold uppercase tracking-wider text-rose-200">
                      SPOOF / PROXY ATTEMPT DETECTED
                    </span>
                  ) : isUnknown ? (
                    <span className="text-[8px] opacity-90 font-medium text-amber-200">
                      Invalid / Unregistered User
                    </span>
                  ) : (
                    <span className="text-[8px] opacity-90 font-medium text-emerald-100">
                      Live Person Verified
                    </span>
                  )}
                </div>
              </div>
            );
          })}

          {!isActive && !registrationSuccess && (
            <div className="absolute inset-0 bg-black/80 flex items-center justify-center text-slate-500 text-sm font-bold">
              Scanner Inactive
            </div>
          )}
        </div>

        <div className="mt-6 text-slate-300 text-sm text-center">
          {instruction}
        </div>
      </div>
    );
  }

  return (

    <div className="flex flex-col items-center w-full">

      <div className="relative w-[340px] h-[340px] flex items-center justify-center">

        {/* CAMERA */}
        <div className="w-[260px] h-[260px] rounded-full overflow-hidden border border-slate-700 bg-black relative">

          <Webcam
            ref={webcamRef}
            audio={false}
            mirrored
            screenshotFormat="image/jpeg"
            onUserMedia={handleUserMedia}
            videoConstraints={{
              width: 640,
              height: 640,
              facingMode: 'user'
            }}
            style={{
              width: '100%',
              height: '100%',
              objectFit: 'cover'
            }}
          />

          {!isActive &&
            !registrationSuccess && (

              <div className="absolute inset-0 bg-black/80 flex items-center justify-center text-slate-500 text-sm font-bold">

                Scanner Inactive

              </div>
            )}

          {registrationSuccess && (

            <div className="absolute inset-0 bg-black/80 flex flex-col items-center justify-center text-emerald-400 font-bold">

              <div className="text-3xl mb-2">
                ✅
              </div>

              Enrollment Successful

            </div>
          )}

        </div>

        {/* RING */}
        <svg
          width="340"
          height="340"
          className="absolute"
        >

          <circle
            cx="170"
            cy="170"
            r="140"
            fill="transparent"
            stroke="#ffffff"
            strokeWidth="8"
          />

          {(isActive || registrationSuccess) && (

            <circle
              cx="170"
              cy="170"
              r="140"
              fill="transparent"
              stroke={
                registrationSuccess
                  ? '#10b981'
                  : '#38bdf8'
              }
              strokeWidth="8"
              strokeDasharray={880}
              strokeDashoffset={
                880 - (progress / 100) * 880
              }
              transform="rotate(-90 170 170)"
              strokeLinecap="round"
              style={{
                transition:
                  'stroke-dashoffset 0.3s linear'
              }}
            />
          )}

        </svg>

      </div>

      {/* STATUS */}
      <div className="mt-6 text-center">

        {status === 'SCANNING' &&
          mode === 'enrollment' && (

            <div className="text-amber-400 text-2xl font-bold mb-3">

              Enrollment {Math.round(progress)}% complete

            </div>
          )}

        {status === 'COMPLETE' && (

          <div className="text-emerald-400 text-2xl font-bold mb-3">

            Scan Complete ✓

          </div>
        )}

        {status === 'FAILED' && (

          <div className="text-red-400 text-xl font-bold mb-3">

            Scan Failed

          </div>
        )}

        <div className="text-slate-300 text-sm mb-4">
          {instruction}
        </div>

      </div>

    </div>
  );
}