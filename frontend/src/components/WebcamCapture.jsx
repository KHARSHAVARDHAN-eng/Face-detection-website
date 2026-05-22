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