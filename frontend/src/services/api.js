import axios from 'axios';

// =========================================
// API URLS
// =========================================
const API_URL =
  import.meta.env.VITE_API_URL !== undefined
    ? import.meta.env.VITE_API_URL
    : 'http://localhost:8000/api';

export const STATIC_URL =
  import.meta.env.VITE_STATIC_URL !== undefined
    ? import.meta.env.VITE_STATIC_URL
    : 'http://localhost:8000';


// =========================================
// AXIOS INSTANCE
// =========================================
export const api = axios.create({

  baseURL: API_URL,

  timeout: 30000,
});


// =========================================
// REQUEST LOGGER
// =========================================
api.interceptors.request.use(

  (config) => {

    console.log(
      `API REQUEST -> ${config.method?.toUpperCase()} ${config.url}`
    );

    return config;
  },

  (error) => {

    console.error(
      "REQUEST ERROR:",
      error
    );

    return Promise.reject(error);
  }
);


// =========================================
// RESPONSE LOGGER
// =========================================
api.interceptors.response.use(

  (response) => {

    console.log(
      `API RESPONSE <- ${response.status} ${response.config.url}`
    );

    return response;
  },

  (error) => {

    console.error(
      "API ERROR:",
      error?.response?.data || error.message
    );

    return Promise.reject(error);
  }
);


// =========================================
// BASE64 -> BLOB
// =========================================
export const dataURLtoBlob = (
  dataurl
) => {

  const arr = dataurl.split(',');

  const mime = arr[0].match(/:(.*?);/)[1];

  const bstr = atob(arr[1]);

  let n = bstr.length;

  const u8arr = new Uint8Array(n);

  while (n--) {

    u8arr[n] = bstr.charCodeAt(n);
  }

  return new Blob(
    [u8arr],
    { type: mime }
  );
};