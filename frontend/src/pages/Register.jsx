import React, { useState } from 'react';
import WebcamCapture from '../components/WebcamCapture';
import { api } from '../services/api';
import toast from 'react-hot-toast';
import { useNavigate } from 'react-router-dom';

export default function Register() {

  const [name, setName] = useState('');
  const [embeddings, setEmbeddings] = useState([]);
  const [imageBase64, setImageBase64] = useState(null);

  const [isScanning, setIsScanning] = useState(false);
  const [scanComplete, setScanComplete] = useState(false);
  const [registrationSuccess, setRegistrationSuccess] = useState(false);

  const [loading, setLoading] = useState(false);

  const navigate = useNavigate();

  // START SCAN
  const handleStartScan = () => {

    if (!name.trim()) {
      toast.error('Please enter your name first');
      return;
    }

    // RESET EVERYTHING
    setEmbeddings([]);
    setImageBase64(null);

    setScanComplete(false);
    setRegistrationSuccess(false);

    setIsScanning(true);
  };

  // SCAN COMPLETE CALLBACK
  const handleScanComplete = (
    collectedEmbeddings,
    thumbnailBase64
  ) => {

    console.log(
      'Embeddings received:',
      collectedEmbeddings?.length
    );

    setIsScanning(false);

    // IMPORTANT FIX
    // LOWERED STRICT REQUIREMENT
    if (
      collectedEmbeddings &&
      collectedEmbeddings.length >= 5
    ) {

      setEmbeddings(collectedEmbeddings);

      setImageBase64(thumbnailBase64);

      setScanComplete(true);

      toast.success('Face scan completed');

    } else {

      setEmbeddings([]);

      setImageBase64(null);

      setScanComplete(false);

      toast.error(
        'Unable to capture enough face embeddings'
      );
    }
  };

  // FINAL SAVE
  const handleSubmit = async (e) => {

    e.preventDefault();

    if (!name.trim()) {
      toast.error('Please enter your name');
      return;
    }

    if (embeddings.length < 5) {
      toast.error(
        'Please complete face scan first'
      );
      return;
    }

    try {

      setLoading(true);

      const payload = {
        name: name.trim(),
        embeddings,
        image_base64: imageBase64,
      };

      console.log(
        'Sending registration payload'
      );

      console.log(
        'Embeddings:',
        embeddings.length
      );

      const response = await api.post(
        '/register',
        payload,
        {
          headers: {
            'Content-Type': 'application/json',
          },
        }
      );

      console.log(response.data);

      toast.success(
        'Enrollment completed successfully'
      );

      setRegistrationSuccess(true);

      setScanComplete(true);

      setIsScanning(false);

    } catch (err) {

      console.error(err);

      toast.error(
        err.response?.data?.detail ||
        'Registration failed'
      );

    } finally {

      setLoading(false);

    }
  };

  return (

    <div className="max-w-2xl mx-auto bg-slate-900 border border-slate-800 p-8 md:p-12 rounded-3xl shadow-2xl">

      <h2 className="text-3xl font-extrabold text-center text-white mb-2">

        Biometric Enrollment

      </h2>

      <p className="text-center text-slate-400 text-sm mb-10">

        Register your face biometrics

      </p>

      <form
        onSubmit={handleSubmit}
        className="space-y-10"
      >

        {/* NAME */}
        <div>

          <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">

            Full Name

          </label>

          <input
            type="text"
            placeholder="Enter your name"
            value={name}
            onChange={(e) =>
              setName(e.target.value)
            }
            disabled={
              loading ||
              isScanning ||
              registrationSuccess
            }
            className="w-full px-5 py-4 bg-slate-950 border border-slate-800 rounded-2xl text-white text-lg placeholder-slate-600 focus:outline-none"
          />

        </div>

        {/* CAMERA */}
        <div className="bg-slate-950/40 p-6 rounded-3xl border border-slate-800/50 flex flex-col items-center justify-center">

          <WebcamCapture
            mode="enrollment"
            isActive={isScanning}
            onScanComplete={handleScanComplete}
            registrationSuccess={
              registrationSuccess
            }
          />

          {/* START BUTTON */}
          {!isScanning &&
            !scanComplete &&
            !registrationSuccess && (

              <button
                type="button"
                onClick={handleStartScan}
                className="mt-6 px-8 py-3 bg-sky-500 hover:bg-sky-600 text-white rounded-xl font-bold"
              >

                Start Face Scan

              </button>
            )}

          {/* RETRY */}
          {!isScanning &&
            scanComplete &&
            !registrationSuccess && (

              <div className="mt-6 text-emerald-400 font-bold">

                ✅ Scan Complete

              </div>
            )}

          {/* SUCCESS */}
          {registrationSuccess && (

            <div className="mt-6 flex flex-col items-center">

              <div className="text-emerald-400 text-lg font-bold">

                ✅ Enrollment Complete

              </div>

              <button
                type="button"
                onClick={() =>
                  navigate('/dashboard')
                }
                className="mt-4 px-6 py-2 bg-emerald-600 hover:bg-emerald-700 rounded-xl text-white font-bold"
              >

                Go To Dashboard

              </button>

            </div>
          )}

        </div>

        {/* SAVE BUTTON */}
        {!registrationSuccess && (

          <button
            type="submit"
            disabled={
              loading ||
              !scanComplete ||
              embeddings.length < 5
            }
            className="w-full py-5 rounded-2xl bg-sky-500 hover:bg-sky-600 disabled:bg-slate-800 text-white text-lg font-bold"
          >

            {loading
              ? 'Saving Biometrics...'
              : 'Complete Enrollment'}

          </button>
        )}

      </form>

    </div>
  );
}